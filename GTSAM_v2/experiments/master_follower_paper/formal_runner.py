"""Strictly causal estimators for the controlled and radial experiments."""

from __future__ import annotations

import heapq
import math
import time
from typing import Any

import gtsam  # type: ignore
import numpy as np
from scipy.stats import chi2  # type: ignore

from estimator.communication import (
    MASTER_NAVIGATION_PROVENANCE,
    SOURCE_ODOMETRY_PROVENANCE,
    MasterNavigationPublisher,
    MasterPoseMessage,
    SourceOdomIncrement,
)
from estimator.numerics import quadratic_form, small_matmul
from .formal_protocol import (
    PROTOCOL_BASELINE,
    PROTOCOL_JOINT,
    PROTOCOL_JOINT_V2,
    FormalWorldDataset,
    PhysicalWave,
    SharedJointMeasurement,
    SharedTimestampMasterMeasurement,
    TwoWaveMeasurement,
    aggregate_odom_segments_for_protocol,
)
from .graphs import (
    CYCLE_SHARED_TIMESTAMP,
    FormalCentralizedOracle,
    MasterFollowerDualGraphEstimator,
)
from .model import OdomSegment


METHOD_DR = "dr_only"
METHOD_BASELINE = "dual_master_equal_wave_budget"
METHOD_CONTROLLED_BASELINE = "dual_master_2s"
METHOD_PROPOSED = "proposed_joint_shared_boundary"
METHOD_BACKBONE_V2 = "joint_master_backbone_only"
METHOD_PROPOSED_V2 = "proposed_joint_bidirectional_v2"
METHOD_ORACLE = "centralized_oracle"
RADIAL_METHODS = (METHOD_DR, METHOD_BASELINE, METHOD_PROPOSED, METHOD_ORACLE)
RADIAL_V2_METHODS = (METHOD_DR, METHOD_BASELINE, METHOD_PROPOSED_V2, METHOD_ORACLE)
RADIAL_V2_MECHANISM_METHODS = (
    METHOD_DR,
    METHOD_BASELINE,
    METHOD_BACKBONE_V2,
    METHOD_PROPOSED_V2,
    METHOD_ORACLE,
)
CONTROLLED_METHODS = (METHOD_DR, METHOD_CONTROLLED_BASELINE)


def _world_position_covariance(
    pose: gtsam.Pose2,
    covariance: np.ndarray,
) -> np.ndarray:
    cosine = math.cos(float(pose.theta()))
    sine = math.sin(float(pose.theta()))
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=float)
    result = small_matmul(
        small_matmul(rotation, np.asarray(covariance[:2, :2], dtype=float)),
        rotation.T,
    )
    return 0.5 * (result + result.T)


def _largest_eigenvalue_2x2(matrix: np.ndarray) -> float:
    a = float(matrix[0, 0])
    b = 0.5 * float(matrix[0, 1] + matrix[1, 0])
    d = float(matrix[1, 1])
    return 0.5 * (a + d + math.sqrt(max((a - d) ** 2 + 4.0 * b * b, 0.0)))


def metric_row(
    dataset: FormalWorldDataset,
    method: str,
    time_s: float,
    follower_id: str,
    estimate: gtsam.Pose2,
    covariance: np.ndarray,
) -> dict[str, Any]:
    truth = dataset.truth_pose(follower_id, time_s)
    error_xy = np.asarray(
        [estimate.x() - truth.x(), estimate.y() - truth.y()], dtype=float
    )
    covariance_xy = _world_position_covariance(estimate, covariance)
    heading = float(truth.theta())
    parallel = np.asarray([math.cos(heading), math.sin(heading)], dtype=float)
    perpendicular = np.asarray([-math.sin(heading), math.cos(heading)], dtype=float)
    error_parallel = float(parallel @ error_xy)
    error_perpendicular = float(perpendicular @ error_xy)
    p_parallel = float(parallel @ covariance_xy @ parallel)
    p_perpendicular = float(perpendicular @ covariance_xy @ perpendicular)
    nees_perpendicular = error_perpendicular**2 / max(p_perpendicular, 1e-15)
    nees_xy = quadratic_form(error_xy, covariance_xy)
    error_se2 = np.asarray(estimate.localCoordinates(truth), dtype=float)
    nees_se2 = quadratic_form(error_se2, covariance)
    return {
        "scenario": dataset.spec.name,
        "seed": int(dataset.seed),
        "method": method,
        "time_s": float(time_s),
        "agent_id": follower_id,
        "estimate_x_m": float(estimate.x()),
        "estimate_y_m": float(estimate.y()),
        "error_parallel_m": error_parallel,
        "error_perpendicular_m": error_perpendicular,
        "squared_error_parallel_m2": error_parallel**2,
        "squared_error_perpendicular_m2": error_perpendicular**2,
        "squared_error_xy_m2": float(error_xy @ error_xy),
        "p_parallel_m2": p_parallel,
        "p_perpendicular_m2": p_perpendicular,
        "trace_p_xy_m2": float(np.trace(covariance_xy)),
        "lambda_max_p_xy_m2": _largest_eigenvalue_2x2(covariance_xy),
        "nees_perpendicular": nees_perpendicular,
        "nees_xy": nees_xy,
        "nees_se2": nees_se2,
        "covered_perpendicular_95": int(
            nees_perpendicular <= float(chi2.ppf(0.95, 1))
        ),
        "covered_xy_95": int(nees_xy <= float(chi2.ppf(0.95, 2))),
        "covered_se2_95": int(nees_se2 <= float(chi2.ppf(0.95, 3))),
    }


def _selected_protocol(dataset: FormalWorldDataset, method: str) -> str | None:
    if method in {METHOD_BASELINE, METHOD_CONTROLLED_BASELINE}:
        return PROTOCOL_BASELINE
    if method == METHOD_PROPOSED:
        return PROTOCOL_JOINT
    if method in {METHOD_BACKBONE_V2, METHOD_PROPOSED_V2}:
        return PROTOCOL_JOINT_V2
    if method == METHOD_ORACLE:
        return dataset.spec.joint_protocol
    if method == METHOD_DR:
        return None
    raise ValueError(f"unknown formal method: {method}")


def _event_queue(
    dataset: FormalWorldDataset,
    method: str,
    odom_segments: dict[str, tuple[OdomSegment, ...]],
) -> list[tuple[float, int, int, str, Any]]:
    events: list[tuple[float, int, int, str, Any]] = []
    serial = 0
    for follower_id in dataset.config.follower_ids:
        for segment in odom_segments[follower_id]:
            heapq.heappush(
                events,
                (segment.end_time, 0, serial, "segment_complete", segment),
            )
            serial += 1
    protocol = _selected_protocol(dataset, method)
    if protocol is not None:
        selected_waves = (
            dataset.baseline_waves
            if protocol == PROTOCOL_BASELINE
            else dataset.joint_waves
        )
        for wave in selected_waves:
            heapq.heappush(
                events, (wave.send_time, 1, serial, "wave_send", wave)
            )
            serial += 1
            for receiver_id, receive_time in wave.receive_times:
                if receiver_id not in dataset.config.follower_ids:
                    continue
                heapq.heappush(
                    events,
                    (
                        receive_time,
                        2,
                        serial,
                        "wave_receive",
                        (wave, receiver_id),
                    ),
                )
                serial += 1
        measurements: tuple[Any, ...]
        if protocol == PROTOCOL_BASELINE:
            measurements = dataset.baseline_measurements
        elif protocol == PROTOCOL_JOINT:
            measurements = dataset.joint_measurements
        elif method == METHOD_BACKBONE_V2:
            measurements = dataset.joint_master_measurements
        else:
            relay_master_rows = tuple(
                measurement
                for measurement in dataset.joint_master_measurements
                if measurement.is_relay_row
            )
            measurements = dataset.joint_measurements + relay_master_rows
        for measurement in measurements:
            heapq.heappush(
                events,
                (
                    measurement.completion_time,
                    3,
                    serial,
                    "measurement_complete",
                    measurement,
                ),
            )
            serial += 1
    for grid_index in range(1, dataset.config.step_count + 1):
        evaluation_time = grid_index * dataset.config.sensor_dt_s
        ratio = evaluation_time / dataset.config.evaluation_period_s
        if abs(ratio - round(ratio)) > 1e-9:
            continue
        heapq.heappush(
            events, (evaluation_time, 4, serial, "evaluate", grid_index)
        )
        serial += 1
    return events


def run_formal_method(
    dataset: FormalWorldDataset,
    method: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    protocol = _selected_protocol(dataset, method)
    if method in {
        METHOD_PROPOSED,
        METHOD_BACKBONE_V2,
        METHOD_PROPOSED_V2,
        METHOD_ORACLE,
    } and not dataset.joint_cycles:
        raise ValueError("joint methods require a joint-protocol scenario")
    if protocol in {PROTOCOL_JOINT, PROTOCOL_JOINT_V2} and (
        protocol != dataset.spec.joint_protocol
    ):
        raise ValueError("method and scenario joint protocols do not match")
    start_wall = time.perf_counter()
    active_odom_segments = aggregate_odom_segments_for_protocol(
        dataset, protocol
    )
    oracle = FormalCentralizedOracle(dataset) if method == METHOD_ORACLE else None
    distributed = (
        {
            follower_id: MasterFollowerDualGraphEstimator(
                dataset,
                follower_id,
                include_remote_initial_states=(method != METHOD_BACKBONE_V2),
            )
            for follower_id in dataset.config.follower_ids
        }
        if oracle is None
        else None
    )
    packets: dict[int, MasterPoseMessage | SourceOdomIncrement | None] = {}
    rows: list[dict[str, Any]] = []
    shadow_created_before_arrival = 0
    transmitted_wave_ids: set[int] = set()
    for event_time, _, _, event_type, payload in sorted(
        _event_queue(dataset, method, active_odom_segments)
    ):
        if event_type == "segment_complete":
            segment: OdomSegment = payload
            if oracle is not None:
                oracle.add_segment(segment)
            else:
                distributed[segment.agent_id].add_local_segment(segment)
        elif event_type == "wave_send":
            wave: PhysicalWave = payload
            if wave.wave_id in transmitted_wave_ids:
                raise RuntimeError("physical wave was transmitted more than once")
            transmitted_wave_ids.add(wave.wave_id)
            if wave.sender_id in dataset.config.master_ids:
                message = MasterNavigationPublisher.publish(
                    dataset.master_messages[wave.wave_id]
                )
                packets[wave.wave_id] = message
                if oracle is not None:
                    oracle.ingest_master_pose(message)
            elif distributed is not None and method in {
                METHOD_PROPOSED,
                METHOD_PROPOSED_V2,
            }:
                increment = distributed[wave.sender_id].make_source_increment(
                    wave.sender_event.event_id
                )
                if increment is None:
                    raise RuntimeError("relay send produced no source-only increment")
                if increment.end_time > wave.send_time + 1e-10:
                    raise RuntimeError("source summary includes future odometry")
                packets[wave.wave_id] = increment
            else:
                packets[wave.wave_id] = None
        elif event_type == "wave_receive":
            if distributed is None:
                continue
            wave, receiver_id = payload
            packet = packets.get(wave.wave_id)
            if isinstance(packet, MasterPoseMessage):
                distributed[receiver_id].cooperative.ingest_master_pose(packet)
            elif isinstance(packet, SourceOdomIncrement) and method in {
                METHOD_PROPOSED,
                METHOD_PROPOSED_V2,
            }:
                end_key = int(
                    gtsam.symbol(packet.source_id, packet.end_event_id)
                )
                if end_key in distributed[receiver_id].cooperative.graph.known_keys:
                    shadow_created_before_arrival += 1
                distributed[receiver_id].cooperative.ingest_follower_increment(packet)
        elif event_type == "measurement_complete":
            if isinstance(payload, TwoWaveMeasurement):
                if distributed is None:
                    raise RuntimeError("two-wave baseline is distributed")
                distributed[
                    payload.listener_id
                ].cooperative.add_two_wave_master_measurement(payload)
            elif isinstance(payload, SharedJointMeasurement):
                if oracle is not None:
                    oracle.add_shared_cycle(payload)
                else:
                    distributed[payload.listener_id].cooperative.add_cycle(
                        payload, CYCLE_SHARED_TIMESTAMP
                    )
            elif isinstance(payload, SharedTimestampMasterMeasurement):
                if oracle is not None:
                    oracle.add_shared_timestamp_master_measurement(payload)
                else:
                    distributed[
                        payload.listener_id
                    ].cooperative.add_shared_timestamp_master_measurement(payload)
            else:
                raise TypeError("unknown acoustic measurement type")
        elif event_type == "evaluate":
            grid_index = int(payload)
            evaluation_time = grid_index * dataset.config.sensor_dt_s
            for follower_id in dataset.config.follower_ids:
                ref = dataset.grid_event_refs[follower_id][grid_index]
                if oracle is not None:
                    estimate, covariance = oracle.state(
                        follower_id, ref.event_id
                    )
                else:
                    estimate, covariance = distributed[
                        follower_id
                    ].cooperative.owner_state(ref.event_id)
                rows.append(
                    metric_row(
                        dataset,
                        method,
                        evaluation_time,
                        follower_id,
                        estimate,
                        covariance,
                    )
                )
        else:
            raise RuntimeError(f"unhandled event type: {event_type}")

    expected_wave_count = 0
    if protocol == PROTOCOL_BASELINE:
        expected_wave_count = dataset.spec.expected_baseline_wave_count
    elif protocol in {PROTOCOL_JOINT, PROTOCOL_JOINT_V2}:
        expected_wave_count = dataset.spec.expected_joint_wave_count
    if len(transmitted_wave_ids) != expected_wave_count:
        raise AssertionError("communication ledger does not match protocol budget")

    if oracle is not None:
        oracle_direct_by_agent = {
            follower_id: sum(
                measurement.listener_id == follower_id
                for measurement in dataset.joint_measurements
            )
            + sum(
                measurement.listener_id == follower_id
                for measurement in dataset.joint_master_measurements
                if measurement.is_relay_row
            )
            for follower_id in dataset.config.follower_ids
        }
        oracle_peer_by_agent = {
            follower_id: 2
            * sum(
                measurement.listener_id == follower_id
                for measurement in dataset.joint_measurements
            )
            for follower_id in dataset.config.follower_ids
        }
        diagnostics = {
            "factor_count": oracle.graph.factor_count,
            "source_factor_count": 0,
            "remote_factor_count": 0,
            "master_prior_count": oracle.master_prior_count,
            "timestamp_prior_count": oracle.timestamp_prior_count,
            "cycle_factor_count": oracle.cycle_factor_count,
            "direct_master_row_count": (
                oracle.joint_cycle_factor_count + oracle.master_row_factor_count
            ),
            "peer_row_count": 2 * oracle.joint_cycle_factor_count,
            "direct_master_rows_by_agent": ";".join(
                f"{agent_id}:{oracle_direct_by_agent[agent_id]}"
                for agent_id in sorted(oracle_direct_by_agent)
            ),
            "peer_rows_by_agent": ";".join(
                f"{agent_id}:{oracle_peer_by_agent[agent_id]}"
                for agent_id in sorted(oracle_peer_by_agent)
            ),
        }
    else:
        direct_rows_by_agent = {
            follower_id: (
                item.cooperative.joint_cycle_factor_count
                + item.cooperative.master_only_factor_count
            )
            for follower_id, item in distributed.items()
        }
        peer_rows_by_agent = {
            follower_id: 2 * item.cooperative.joint_cycle_factor_count
            for follower_id, item in distributed.items()
        }
        diagnostics = {
            "factor_count": sum(
                item.cooperative.factor_count for item in distributed.values()
            ),
            "source_factor_count": sum(
                item.source.factor_count for item in distributed.values()
            ),
            "remote_factor_count": sum(
                item.cooperative.remote_factor_count
                for item in distributed.values()
            ),
            "master_prior_count": sum(
                item.cooperative.master_prior_count
                for item in distributed.values()
            ),
            "timestamp_prior_count": sum(
                item.cooperative.timestamp_prior_count
                for item in distributed.values()
            ),
            "cycle_factor_count": sum(
                item.cooperative.cycle_factor_count
                for item in distributed.values()
            ),
            "direct_master_row_count": sum(direct_rows_by_agent.values()),
            "peer_row_count": sum(peer_rows_by_agent.values()),
            "direct_master_rows_by_agent": ";".join(
                f"{agent_id}:{direct_rows_by_agent[agent_id]}"
                for agent_id in sorted(direct_rows_by_agent)
            ),
            "peer_rows_by_agent": ";".join(
                f"{agent_id}:{peer_rows_by_agent[agent_id]}"
                for agent_id in sorted(peer_rows_by_agent)
            ),
        }
    diagnostics.update(
        {
            "scenario": dataset.spec.name,
            "seed": dataset.seed,
            "method": method,
            "wall_time_s": time.perf_counter() - start_wall,
            "metric_rows": len(rows),
            "physical_wave_count": len(transmitted_wave_ids),
            "unique_physical_wave_count": len(transmitted_wave_ids),
            "source_contamination_count": sum(
                1
                for packet in packets.values()
                if isinstance(packet, SourceOdomIncrement)
                and packet.provenance != SOURCE_ODOMETRY_PROVENANCE
            ),
            "master_contamination_count": sum(
                1
                for packet in packets.values()
                if isinstance(packet, MasterPoseMessage)
                and packet.provenance != MASTER_NAVIGATION_PROVENANCE
            ),
            "shadow_created_before_arrival": shadow_created_before_arrival,
            "raw_odom_segment_count": sum(
                len(segments) for segments in dataset.odom_segments.values()
            ),
            "active_odom_segment_count": sum(
                len(segments) for segments in active_odom_segments.values()
            ),
        }
    )
    return rows, diagnostics
