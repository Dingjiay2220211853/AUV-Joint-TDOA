"""Strict event-time master/follower factor graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gtsam  # type: ignore
import numpy as np

from estimator.communication import (
    MASTER_NAVIGATION_PROVENANCE,
    SOURCE_ODOMETRY_PROVENANCE,
    FollowerInitialStatePublisher,
    FollowerSourcePublisher,
    MasterNavigationPublisher,
    MasterPoseMessage,
    SourceOdomIncrement,
)
from estimator.distributed_graphs import (
    IncrementalPoseGraph,
    SourceGraph,
    SourcePublisher,
    build_cycle_factor,
    pose_key,
)
from estimator.numerics import regularize_covariance
from factors.cycle_snoop_factor import (
    build_cycle_tdoa_covariance,
    create_shared_timestamp_cycle_factor,
    create_shared_timestamp_master_factor,
    create_scalar_cycle_snoop_factor,
)
from factors.snoop_factor import create_quaternary_snoop_factor
from .model import (
    FOLLOWER_IDS,
    MASTER_IDS,
    FormalExperimentDataset,
    ListenerCycleMeasurement,
    OdomSegment,
)


CYCLE_JOINT = "joint"
CYCLE_SCALAR_INDEPENDENT = "scalar_independent"
CYCLE_MASTER_ONLY = "master_only"
CYCLE_SHARED_TIMESTAMP = "shared_timestamp"


def _strict_cycle_keys(cycle: ListenerCycleMeasurement) -> list[int]:
    keys = [int(ref.key) for ref in cycle.logical_event_refs]
    if len(keys) != 8:
        raise AssertionError("formal cycle requires eight logical Pose2 slots")
    if cycle.waves[0].sender_event.key == cycle.waves[3].sender_event.key:
        raise AssertionError("A0 and A1 must use different event keys")
    listener_times = [float(ref.timestamp) for ref in cycle.listener_events]
    if any(
        right < left - 1e-11
        for left, right in zip(listener_times, listener_times[1:])
    ):
        raise AssertionError("listener event times must be nondecreasing")
    if cycle.listener_events[0].key == cycle.listener_events[-1].key:
        raise AssertionError("cycle endpoints must use different listener states")
    return keys


def build_scalar_cycle_factors(
    graph: IncrementalPoseGraph,
    cycle: ListenerCycleMeasurement,
) -> tuple[Any, Any, Any]:
    """Independent scalar-row ablation with no pose-offset covariance."""
    keys = _strict_cycle_keys(cycle)
    missing = [key for key in keys if key not in graph.known_keys]
    if missing:
        raise KeyError(f"cycle lacks event keys: {sorted(missing)}")
    covariance = build_cycle_tdoa_covariance(
        cycle.timestamp_covariance, cycle.sound_speed_mps
    )
    factors: list[Any] = []
    for row in range(3):
        noise = gtsam.noiseModel.Gaussian.Covariance(
            regularize_covariance(
                np.asarray([[covariance[row, row]]], dtype=float)
            )
        )
        factor = create_scalar_cycle_snoop_factor(
            *keys,
            measured_delta_t=cycle.measured_delta_t,
            residual_row=row,
            noise_model=noise,
            sound_speed=cycle.sound_speed_mps,
        )
        if set(int(key) for key in factor.keys()) != set(keys):
            raise AssertionError("scalar cycle factor changed the Pose2 key set")
        factors.append(factor)
    return tuple(factors)  # type: ignore[return-value]


def build_master_only_factor(
    graph: IncrementalPoseGraph,
    cycle: ListenerCycleMeasurement,
) -> Any:
    """A→B listener row using true A0/B/L0/L1 event nodes."""
    refs = (
        cycle.waves[0].sender_event,
        cycle.waves[1].sender_event,
        cycle.listener_events[0],
        cycle.listener_events[1],
    )
    keys = [int(ref.key) for ref in refs]
    missing = [key for key in keys if key not in graph.known_keys]
    if missing:
        raise KeyError(f"master-only factor lacks event keys: {sorted(missing)}")
    covariance = build_cycle_tdoa_covariance(
        cycle.timestamp_covariance, cycle.sound_speed_mps
    )
    noise = gtsam.noiseModel.Gaussian.Covariance(
        regularize_covariance(np.asarray([[covariance[0, 0]]], dtype=float))
    )
    return create_quaternary_snoop_factor(
        *keys,
        measured_delta_t=float(cycle.measured_delta_t[0]),
        noise_model=noise,
        sound_speed=cycle.sound_speed_mps,
    )


def build_two_wave_master_factor(
    graph: IncrementalPoseGraph,
    measurement: Any,
) -> Any:
    """Build a genuine A->B two-wave TDOA factor."""
    refs = (
        measurement.waves[0].sender_event,
        measurement.waves[1].sender_event,
        measurement.listener_events[0],
        measurement.listener_events[1],
    )
    keys = [int(ref.key) for ref in refs]
    if len(keys) != 4:
        raise AssertionError("two-wave TDOA requires four logical Pose2 slots")
    missing = [key for key in keys if key not in graph.known_keys]
    if missing:
        raise KeyError(f"two-wave TDOA lacks event keys: {sorted(missing)}")
    difference = np.asarray([[-1.0, 1.0]], dtype=float)
    covariance = (
        measurement.sound_speed_mps**2
        * difference
        @ np.asarray(measurement.timestamp_covariance, dtype=float)
        @ difference.T
    )
    noise = gtsam.noiseModel.Gaussian.Covariance(
        regularize_covariance(covariance)
    )
    return create_quaternary_snoop_factor(
        *keys,
        measured_delta_t=measurement.measured_delta_t,
        noise_model=noise,
        sound_speed=measurement.sound_speed_mps,
    )


class MasterFollowerCooperativeGraph:
    """One follower's local event-time multi-agent graph."""

    def __init__(
        self,
        dataset: FormalExperimentDataset,
        owner_id: str,
        *,
        include_remote_initial_states: bool = True,
    ) -> None:
        if owner_id not in dataset.config.follower_ids:
            raise ValueError("a cooperative graph owner must be a follower")
        self.dataset = dataset
        self.owner_id = owner_id
        self.graph = IncrementalPoseGraph()
        self.source_epochs: dict[str, int] = {}
        self.initial_times: dict[str, float] = {}
        initial_followers = (
            dataset.config.follower_ids
            if include_remote_initial_states
            else (owner_id,)
        )
        for follower_id in initial_followers:
            message = FollowerInitialStatePublisher.publish(
                dataset.initial_state_messages[follower_id]
            )
            self.graph.add_prior(
                pose_key(follower_id, 0),
                message.pose,
                message.covariance,
                kind=(
                    "owner_initial_state"
                    if follower_id == owner_id
                    else "preshared_remote_initial_state"
                ),
            )
            self.source_epochs[follower_id] = message.source_epoch
            self.initial_times[follower_id] = message.timestamp
        self.graph.update()
        self.owner_latest_event_id = 0
        self.owner_latest_time = self.initial_times[owner_id]
        self.remote_latest_event_id = {
            follower_id: 0
            for follower_id in initial_followers
            if follower_id != owner_id
        }
        self.remote_latest_time = {
            follower_id: self.initial_times[follower_id]
            for follower_id in initial_followers
            if follower_id != owner_id
        }
        self.received_summary_ids: set[tuple[str, int, int, int]] = set()
        self.received_segment_ids: dict[str, set[str]] = {
            follower_id: set() for follower_id in dataset.config.follower_ids
        }
        self.received_master_message_ids: set[tuple[str, int, int]] = set()
        self.master_last_timestamp: dict[str, float] = {}
        self.master_timestamp_by_key: dict[int, float] = {}
        self.remote_factor_count = 0
        self.master_prior_count = 0
        self.joint_cycle_factor_count = 0
        self.scalar_cycle_factor_count = 0
        self.master_only_factor_count = 0
        self.timestamp_prior_count = 0

    @property
    def factor_count(self) -> int:
        return self.graph.factor_count

    @property
    def cycle_factor_count(self) -> int:
        return (
            self.joint_cycle_factor_count
            + self.scalar_cycle_factor_count
            + self.master_only_factor_count
        )

    def add_local_segment(self, segment: OdomSegment) -> None:
        if segment.agent_id != self.owner_id:
            raise ValueError("local segment must belong to graph owner")
        if segment.start_event_id != self.owner_latest_event_id:
            raise ValueError("owner segment event IDs must be contiguous")
        if abs(segment.start_time - self.owner_latest_time) > 1e-10:
            raise ValueError("owner segment timestamps must be contiguous")
        self.graph.add_between(
            pose_key(self.owner_id, segment.start_event_id),
            pose_key(self.owner_id, segment.end_event_id),
            segment.delta_pose,
            segment.covariance,
            kind="owner_odometry_segment",
        )
        self.owner_latest_event_id = segment.end_event_id
        self.owner_latest_time = segment.end_time

    def ingest_follower_increment(
        self,
        increment: SourceOdomIncrement,
    ) -> bool:
        FollowerSourcePublisher.publish(increment)
        source_id = increment.source_id
        if source_id not in self.dataset.config.follower_ids:
            raise ValueError("only followers may send source increments")
        if source_id == self.owner_id:
            return False
        if increment.summary_id in self.received_summary_ids:
            return False
        if increment.source_epoch != self.source_epochs[source_id]:
            raise ValueError("remote source epoch does not match deployment prior")
        expected_event = self.remote_latest_event_id[source_id]
        expected_time = self.remote_latest_time[source_id]
        if increment.start_event_id != expected_event:
            raise ValueError(
                f"remote event gap for {source_id}: expected {expected_event}, "
                f"received {increment.start_event_id}"
            )
        if abs(increment.start_time - expected_time) > 1e-10:
            raise ValueError(
                f"remote timestamp gap for {source_id}: expected "
                f"{expected_time}, received {increment.start_time}"
            )
        seen = self.received_segment_ids[source_id]
        overlap = seen.intersection(increment.segment_ids)
        if overlap:
            raise RuntimeError(f"remote segment overlap: {sorted(overlap)}")
        start_key = pose_key(source_id, increment.start_event_id)
        end_key = pose_key(source_id, increment.end_event_id)
        if end_key in self.graph.known_keys:
            raise ValueError("remote end event already exists without packet ID")
        self.graph.add_between(
            start_key,
            end_key,
            increment.delta_pose,
            increment.covariance,
            kind="remote_source_odometry_increment",
        )
        seen.update(increment.segment_ids)
        self.received_summary_ids.add(increment.summary_id)
        self.remote_latest_event_id[source_id] = increment.end_event_id
        self.remote_latest_time[source_id] = increment.end_time
        self.remote_factor_count += 1
        return True

    def ingest_master_pose(self, message: MasterPoseMessage) -> bool:
        MasterNavigationPublisher.publish(message)
        if message.master_id not in self.dataset.config.master_ids:
            raise ValueError("absolute pose message is not from a master")
        if message.source_epoch != self.dataset.seed:
            raise ValueError("master source epoch does not match dataset epoch")
        if message.provenance != MASTER_NAVIGATION_PROVENANCE:
            raise ValueError("master message provenance is contaminated")
        if message.message_id in self.received_master_message_ids:
            return False
        last_timestamp = self.master_last_timestamp.get(message.master_id)
        if last_timestamp is not None and message.timestamp <= last_timestamp:
            raise ValueError("master message timestamps must strictly increase")
        key = pose_key(message.master_id, message.state_index)
        if key in self.graph.known_keys:
            raise ValueError("master event key already exists")
        self.graph.add_prior(
            key,
            message.pose,
            message.covariance,
            kind="master_absolute_event_prior",
        )
        self.received_master_message_ids.add(message.message_id)
        self.master_last_timestamp[message.master_id] = message.timestamp
        self.master_timestamp_by_key[key] = message.timestamp
        self.master_prior_count += 1
        return True

    def add_cycle(
        self,
        cycle: ListenerCycleMeasurement,
        mode: str,
    ) -> None:
        if cycle.listener_id != self.owner_id:
            raise ValueError("only the listener may add its cycle")
        keys = _strict_cycle_keys(cycle)
        for master_ref in (
            cycle.waves[0].sender_event,
            cycle.waves[1].sender_event,
            cycle.waves[3].sender_event,
        ):
            recorded = self.master_timestamp_by_key.get(master_ref.key)
            if recorded is None or abs(recorded - master_ref.timestamp) > 1e-10:
                raise ValueError("master event prior missing or timestamp-mismatched")
        if mode == CYCLE_JOINT:
            factor = build_cycle_factor(self.graph, cycle)
            if set(int(key) for key in factor.keys()) != set(keys):
                raise AssertionError("joint factor key set changed")
            self.graph.add_custom_factor(factor, kind="joint_cycle_snoop")
            self.joint_cycle_factor_count += 1
        elif mode == CYCLE_SCALAR_INDEPENDENT:
            for factor in build_scalar_cycle_factors(self.graph, cycle):
                self.graph.add_custom_factor(
                    factor, kind="scalar_independent_cycle_snoop"
                )
                self.scalar_cycle_factor_count += 1
        elif mode == CYCLE_MASTER_ONLY:
            self.graph.add_custom_factor(
                build_master_only_factor(self.graph, cycle),
                kind="master_only_snoop",
            )
            self.master_only_factor_count += 1
        elif mode == CYCLE_SHARED_TIMESTAMP:
            timestamp_keys = tuple(int(key) for key in cycle.timestamp_error_keys)
            for key in timestamp_keys:
                if key not in self.graph.known_keys:
                    self.graph.add_scalar_prior(
                        key,
                        0.0,
                        self.dataset.config.timestamp_sigma_s,
                        kind="timestamp_error_prior",
                    )
                    self.timestamp_prior_count += 1
            factor = create_shared_timestamp_cycle_factor(
                *keys,
                timestamp_error_keys=timestamp_keys,
                measured_delta_t=cycle.measured_delta_t,
                sound_speed=cycle.sound_speed_mps,
            )
            self.graph.add_custom_factor(
                factor, kind="shared_timestamp_joint_cycle"
            )
            self.joint_cycle_factor_count += 1
        else:
            raise ValueError(f"unknown cycle mode: {mode}")
        self.graph.update()

    def add_two_wave_master_measurement(self, measurement: Any) -> None:
        if measurement.listener_id != self.owner_id:
            raise ValueError("only the listener may add its two-wave factor")
        for wave in measurement.waves:
            recorded = self.master_timestamp_by_key.get(wave.sender_event.key)
            if recorded is None or abs(recorded - wave.send_time) > 1e-10:
                raise ValueError("two-wave master prior is missing")
        self.graph.add_custom_factor(
            build_two_wave_master_factor(self.graph, measurement),
            kind="true_two_wave_master_tdoa",
        )
        self.master_only_factor_count += 1
        self.graph.update()

    def add_shared_timestamp_master_measurement(self, measurement: Any) -> None:
        """Add one master row tied to the shared physical timestamp states."""
        if measurement.listener_id != self.owner_id:
            raise ValueError("only the listener may add its shared master row")
        refs = measurement.logical_event_refs
        pose_keys = [int(ref.key) for ref in refs]
        if len(pose_keys) != 4:
            raise AssertionError("shared master row requires four logical Pose2 slots")
        for wave in measurement.waves:
            recorded = self.master_timestamp_by_key.get(wave.sender_event.key)
            if recorded is None or abs(recorded - wave.send_time) > 1e-10:
                raise ValueError("shared master-row prior is missing")
        missing = [key for key in pose_keys if key not in self.graph.known_keys]
        if missing:
            raise KeyError(f"shared master row lacks event keys: {sorted(missing)}")
        timestamp_keys = tuple(int(key) for key in measurement.timestamp_error_keys)
        for key in timestamp_keys:
            if key not in self.graph.known_keys:
                self.graph.add_scalar_prior(
                    key,
                    0.0,
                    self.dataset.config.timestamp_sigma_s,
                    kind="timestamp_error_prior",
                )
                self.timestamp_prior_count += 1
        self.graph.add_custom_factor(
            create_shared_timestamp_master_factor(
                *pose_keys,
                timestamp_error_keys=timestamp_keys,
                measured_delta_t=measurement.measured_delta_t,
                sound_speed=measurement.sound_speed_mps,
            ),
            kind="shared_timestamp_master_tdoa",
        )
        self.master_only_factor_count += 1
        self.graph.update()

    def owner_state(self, event_id: int) -> tuple[gtsam.Pose2, np.ndarray]:
        key = pose_key(self.owner_id, event_id)
        return self.graph.pose(key), self.graph.covariance(key)

    def remote_follower_state(
        self,
        follower_id: str,
        event_id: int,
    ) -> tuple[gtsam.Pose2, np.ndarray]:
        if (
            follower_id == self.owner_id
            or follower_id not in self.dataset.config.follower_ids
        ):
            raise ValueError("requested state must be a remote follower")
        key = pose_key(follower_id, event_id)
        return self.graph.pose(key), self.graph.covariance(key)


@dataclass
class MasterFollowerDualGraphEstimator:
    dataset: FormalExperimentDataset
    owner_id: str
    include_remote_initial_states: bool = True

    def __post_init__(self) -> None:
        self.source = SourceGraph(
            self.dataset.initial_state_messages[self.owner_id]
        )
        self.cooperative = MasterFollowerCooperativeGraph(
            self.dataset,
            self.owner_id,
            include_remote_initial_states=self.include_remote_initial_states,
        )

    def add_local_segment(self, segment: OdomSegment) -> None:
        self.source.add_segment(segment)
        self.cooperative.add_local_segment(segment)

    def make_source_increment(
        self,
        send_event_id: int,
    ) -> SourceOdomIncrement | None:
        return SourcePublisher.publish(
            self.source.close_increment(send_event_id)
        )


class FormalCentralizedOracle:
    """All follower segments, all independent master priors and all cycles."""

    def __init__(self, dataset: FormalExperimentDataset) -> None:
        self.dataset = dataset
        self.graph = IncrementalPoseGraph()
        self.latest_event_id: dict[str, int] = {}
        self.latest_time: dict[str, float] = {}
        for follower_id in dataset.config.follower_ids:
            message = FollowerInitialStatePublisher.publish(
                dataset.initial_state_messages[follower_id]
            )
            self.graph.add_prior(
                pose_key(follower_id, 0),
                message.pose,
                message.covariance,
                kind="oracle_follower_initial_state",
            )
            self.latest_event_id[follower_id] = 0
            self.latest_time[follower_id] = message.timestamp
        self.graph.update()
        self.received_master_message_ids: set[tuple[str, int, int]] = set()
        self.master_last_timestamp: dict[str, float] = {}
        self.master_prior_count = 0
        self.cycle_factor_count = 0
        self.joint_cycle_factor_count = 0
        self.master_row_factor_count = 0
        self.timestamp_prior_count = 0

    def add_segment(self, segment: OdomSegment) -> None:
        if segment.start_event_id != self.latest_event_id[segment.agent_id]:
            raise ValueError("oracle segment event IDs must be contiguous")
        if abs(segment.start_time - self.latest_time[segment.agent_id]) > 1e-10:
            raise ValueError("oracle segment timestamps must be contiguous")
        self.graph.add_between(
            pose_key(segment.agent_id, segment.start_event_id),
            pose_key(segment.agent_id, segment.end_event_id),
            segment.delta_pose,
            segment.covariance,
            kind="oracle_odometry_segment",
        )
        self.latest_event_id[segment.agent_id] = segment.end_event_id
        self.latest_time[segment.agent_id] = segment.end_time

    def ingest_master_pose(self, message: MasterPoseMessage) -> bool:
        MasterNavigationPublisher.publish(message)
        if message.source_epoch != self.dataset.seed:
            raise ValueError("master epoch mismatch")
        if message.message_id in self.received_master_message_ids:
            return False
        last_timestamp = self.master_last_timestamp.get(message.master_id)
        if last_timestamp is not None and message.timestamp <= last_timestamp:
            raise ValueError("master timestamps must strictly increase")
        self.graph.add_prior(
            pose_key(message.master_id, message.state_index),
            message.pose,
            message.covariance,
            kind="oracle_master_absolute_event_prior",
        )
        self.received_master_message_ids.add(message.message_id)
        self.master_last_timestamp[message.master_id] = message.timestamp
        self.master_prior_count += 1
        return True

    def add_cycle(self, cycle: ListenerCycleMeasurement) -> None:
        self.graph.add_custom_factor(
            build_cycle_factor(self.graph, cycle),
            kind="oracle_joint_cycle_snoop",
        )
        self.graph.update()
        self.cycle_factor_count += 1
        self.joint_cycle_factor_count += 1

    def add_shared_cycle(self, cycle: Any) -> None:
        keys = _strict_cycle_keys(cycle)
        for timestamp_key in cycle.timestamp_error_keys:
            key = int(timestamp_key)
            if key not in self.graph.known_keys:
                self.graph.add_scalar_prior(
                    key,
                    0.0,
                    self.dataset.config.timestamp_sigma_s,
                    kind="oracle_timestamp_error_prior",
                )
                self.timestamp_prior_count += 1
        self.graph.add_custom_factor(
            create_shared_timestamp_cycle_factor(
                *keys,
                timestamp_error_keys=cycle.timestamp_error_keys,
                measured_delta_t=cycle.measured_delta_t,
                sound_speed=cycle.sound_speed_mps,
            ),
            kind="oracle_shared_timestamp_joint_cycle",
        )
        self.graph.update()
        self.cycle_factor_count += 1
        self.joint_cycle_factor_count += 1

    def add_shared_timestamp_master_measurement(self, measurement: Any) -> None:
        pose_keys = [int(ref.key) for ref in measurement.logical_event_refs]
        if len(pose_keys) != 4:
            raise AssertionError("oracle master row requires four logical Pose2 slots")
        timestamp_keys = tuple(int(key) for key in measurement.timestamp_error_keys)
        for key in timestamp_keys:
            if key not in self.graph.known_keys:
                self.graph.add_scalar_prior(
                    key,
                    0.0,
                    self.dataset.config.timestamp_sigma_s,
                    kind="oracle_timestamp_error_prior",
                )
                self.timestamp_prior_count += 1
        self.graph.add_custom_factor(
            create_shared_timestamp_master_factor(
                *pose_keys,
                timestamp_error_keys=timestamp_keys,
                measured_delta_t=measurement.measured_delta_t,
                sound_speed=measurement.sound_speed_mps,
            ),
            kind="oracle_shared_timestamp_master_tdoa",
        )
        self.graph.update()
        self.cycle_factor_count += 1
        self.master_row_factor_count += 1

    def state(self, follower_id: str, event_id: int) -> tuple[gtsam.Pose2, np.ndarray]:
        key = pose_key(follower_id, event_id)
        return self.graph.pose(key), self.graph.covariance(key)
