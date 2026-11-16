"""Reusable source-only graph primitives for decentralized Pose2 estimation."""

from __future__ import annotations

import math
from typing import Any

import gtsam  # type: ignore
import numpy as np

from estimator.communication import (
    FollowerInitialStateMessage,
    FollowerInitialStatePublisher,
    FollowerSourcePublisher,
    GridFollowerSourcePublisher,
    GridSourceOdomIncrement,
    OdomMeasurement,
    SourceOdomIncrement,
)
from estimator.numerics import regularize_covariance, small_matmul
from estimator.odometry_preintegration import compose_pose_covariance
from factors.cycle_snoop_factor import (
    LOGICAL_POSE_ORDER,
    build_cycle_tdoa_covariance,
    create_cycle_snoop_factor,
    evaluate_cycle_snoop_geometry,
)


def pose_key(agent_id: str, index: int) -> int:
    return int(gtsam.symbol(agent_id, int(index)))


def relative_pose_jacobian(
    first: gtsam.Pose2,
    second: gtsam.Pose2,
) -> np.ndarray:
    """Jacobian of ``first.between(second)`` in endpoint local coordinates."""
    nominal = first.between(second)
    jacobian = np.zeros((3, 6), dtype=float)
    epsilon = 1e-6
    for endpoint in range(2):
        for dimension in range(3):
            perturbation = np.zeros(3, dtype=float)
            perturbation[dimension] = epsilon
            plus_first, minus_first = first, first
            plus_second, minus_second = second, second
            if endpoint == 0:
                plus_first = first.retract(perturbation)
                minus_first = first.retract(-perturbation)
            else:
                plus_second = second.retract(perturbation)
                minus_second = second.retract(-perturbation)
            plus_error = np.asarray(
                nominal.localCoordinates(plus_first.between(plus_second)),
                dtype=float,
            )
            minus_error = np.asarray(
                nominal.localCoordinates(minus_first.between(minus_second)),
                dtype=float,
            )
            jacobian[:, endpoint * 3 + dimension] = (
                plus_error - minus_error
            ) / (2.0 * epsilon)
    return jacobian


class IncrementalPoseGraph:
    """Small full-history iSAM2 wrapper retaining factors for diagnostics."""

    def __init__(self) -> None:
        params = gtsam.ISAM2Params()
        params.setRelinearizeThreshold(0.01)
        params.relinearizeSkip = 1
        self.isam = gtsam.ISAM2(params)
        self.pending_graph = gtsam.NonlinearFactorGraph()
        self.pending_values = gtsam.Values()
        self.full_graph = gtsam.NonlinearFactorGraph()
        self.known_keys: set[int] = set()
        self.factor_kinds: dict[str, int] = {}

    @property
    def factor_count(self) -> int:
        return int(self.full_graph.size())

    def _record_factor(self, factor: Any, kind: str) -> None:
        self.pending_graph.add(factor)
        self.full_graph.add(factor)
        self.factor_kinds[kind] = self.factor_kinds.get(kind, 0) + 1

    def _insert_value(self, key: int, value: Any) -> None:
        if key in self.known_keys:
            return
        self.pending_values.insert(key, value)
        self.known_keys.add(key)

    def add_scalar_prior(
        self,
        key: int,
        value: float,
        sigma: float,
        *,
        kind: str = "scalar_prior",
    ) -> None:
        """Insert one scalar state and its Gaussian prior exactly once."""
        if key in self.known_keys:
            raise ValueError(f"key {key} already exists")
        if not math.isfinite(float(sigma)) or float(sigma) <= 0.0:
            raise ValueError("scalar prior sigma must be finite and positive")
        noise = gtsam.noiseModel.Isotropic.Sigma(1, float(sigma))
        self._record_factor(
            gtsam.PriorFactorDouble(key, float(value), noise), kind
        )
        self._insert_value(key, float(value))

    def add_prior(
        self,
        key: int,
        pose: gtsam.Pose2,
        covariance: np.ndarray,
        *,
        kind: str = "prior",
    ) -> None:
        if key in self.known_keys:
            raise ValueError(f"key {key} already exists")
        noise = gtsam.noiseModel.Gaussian.Covariance(
            regularize_covariance(covariance)
        )
        self._record_factor(gtsam.PriorFactorPose2(key, pose, noise), kind)
        self._insert_value(key, pose)

    def add_between(
        self,
        old_key: int,
        new_key: int,
        delta_pose: gtsam.Pose2,
        covariance: np.ndarray,
        *,
        kind: str,
    ) -> None:
        if old_key not in self.known_keys:
            raise KeyError(f"old key {old_key} is not present")
        if new_key in self.known_keys:
            raise ValueError(f"new key {new_key} already exists")
        old_pose = self.pose(old_key)
        self._insert_value(new_key, old_pose.compose(delta_pose))
        noise = gtsam.noiseModel.Gaussian.Covariance(
            regularize_covariance(covariance)
        )
        self._record_factor(
            gtsam.BetweenFactorPose2(old_key, new_key, delta_pose, noise),
            kind,
        )

    def add_custom_factor(self, factor: Any, *, kind: str) -> None:
        missing = [
            int(key) for key in factor.keys() if int(key) not in self.known_keys
        ]
        if missing:
            raise KeyError(f"factor references missing keys: {missing}")
        self._record_factor(factor, kind)

    def update(self) -> None:
        if self.pending_graph.size() == 0 and self.pending_values.size() == 0:
            return
        self.isam.update(self.pending_graph, self.pending_values)
        self.pending_graph.resize(0)
        self.pending_values.clear()

    def pose(self, key: int) -> gtsam.Pose2:
        if key not in self.known_keys:
            raise KeyError(f"key {key} is not present")
        if self.isam.valueExists(key):
            return self.isam.calculateEstimatePose2(key)
        return self.pending_values.atPose2(key)

    def scalar(self, key: int) -> float:
        if key not in self.known_keys:
            raise KeyError(f"key {key} is not present")
        if self.isam.valueExists(key):
            return float(self.isam.calculateEstimate().atDouble(key))
        return float(self.pending_values.atDouble(key))

    def covariance(self, key: int) -> np.ndarray:
        self.update()
        covariance = np.asarray(self.isam.marginalCovariance(key), dtype=float)
        return regularize_covariance(covariance)

    def values(self) -> gtsam.Values:
        self.update()
        return self.isam.calculateEstimate()


class GridSourceGraph:
    """Legacy owner-only 10 Hz source graph for the frozen experiments."""

    def __init__(self, dataset: Any, owner_id: str) -> None:
        self.dataset = dataset
        self.owner_id = owner_id
        self.source_epoch = int(dataset.seed)
        self.graph = IncrementalPoseGraph()
        self.graph.add_prior(
            pose_key(owner_id, 0),
            dataset.initial_prior_means[owner_id],
            dataset.initial_prior_covariance,
            kind="source_prior",
        )
        self.graph.update()
        self.records: dict[int, OdomMeasurement] = {}
        self.latest_index = 0
        self.last_summary_index = 0
        self.emitted_measurement_ids: set[str] = set()

    @property
    def factor_count(self) -> int:
        return self.graph.factor_count

    def add_odometry(self, record: OdomMeasurement) -> None:
        if record.agent_id != self.owner_id:
            raise ValueError("SourceGraph accepts only owner odometry")
        if record.start_index != self.latest_index:
            raise ValueError("source odometry must be contiguous")
        self.graph.add_between(
            pose_key(self.owner_id, record.start_index),
            pose_key(self.owner_id, record.end_index),
            record.delta_pose,
            record.covariance,
            kind="source_odometry",
        )
        self.graph.update()
        self.records[record.start_index] = record
        self.latest_index = record.end_index

    def close_increment(self, end_index: int) -> GridSourceOdomIncrement | None:
        end_index = int(end_index)
        if end_index > self.latest_index:
            raise ValueError("cannot summarize future source odometry")
        if end_index < self.last_summary_index:
            raise ValueError("source summary indices must be monotonic")
        if end_index == self.last_summary_index:
            return None

        delta = gtsam.Pose2()
        covariance = np.zeros((3, 3), dtype=float)
        measurement_ids: list[str] = []
        for index in range(self.last_summary_index, end_index):
            record = self.records[index]
            covariance = compose_pose_covariance(
                covariance, record.delta_pose, record.covariance
            )
            delta = delta.compose(record.delta_pose)
            measurement_ids.append(record.measurement_id)

        overlap = self.emitted_measurement_ids.intersection(measurement_ids)
        if overlap:
            raise RuntimeError(f"source summary overlap detected: {sorted(overlap)}")
        start_index = self.last_summary_index
        summary = GridSourceOdomIncrement(
            source_id=self.owner_id,
            source_epoch=self.source_epoch,
            start_index=start_index,
            end_index=end_index,
            start_time=start_index * self.dataset.config.sensor_dt_s,
            end_time=end_index * self.dataset.config.sensor_dt_s,
            delta_pose=delta,
            covariance=regularize_covariance(covariance),
            measurement_ids=tuple(measurement_ids),
        )
        self.emitted_measurement_ids.update(measurement_ids)
        self.last_summary_index = end_index
        return summary

    def current_signature(
        self,
    ) -> tuple[int, int, tuple[float, ...], tuple[float, ...]]:
        key = pose_key(self.owner_id, self.latest_index)
        pose = self.graph.pose(key)
        covariance = self.graph.covariance(key)
        return (
            self.factor_count,
            self.last_summary_index,
            (float(pose.x()), float(pose.y()), float(pose.theta())),
            tuple(float(value) for value in covariance.ravel()),
        )


GridSourcePublisher = GridFollowerSourcePublisher


class SourceGraph:
    """Event-time source graph built only from ordered raw odometry segments."""

    def __init__(self, initial_state: FollowerInitialStateMessage) -> None:
        initial_state = FollowerInitialStatePublisher.publish(initial_state)
        self.owner_id = initial_state.follower_id
        self.source_epoch = int(initial_state.source_epoch)
        self.graph = IncrementalPoseGraph()
        self.graph.add_prior(
            pose_key(self.owner_id, 0),
            initial_state.pose,
            initial_state.covariance,
            kind="source_initial_state",
        )
        self.graph.update()
        self.latest_event_id = 0
        self.latest_time = float(initial_state.timestamp)
        self.last_summary_event_id = 0
        self.last_summary_time = float(initial_state.timestamp)
        self.segments_by_start_event: dict[int, Any] = {}
        self.segment_ids: set[str] = set()
        self.emitted_segment_ids: set[str] = set()

    @property
    def factor_count(self) -> int:
        return self.graph.factor_count

    def add_segment(self, segment: Any) -> None:
        if segment.agent_id != self.owner_id:
            raise ValueError("SourceGraph accepts only owner odometry segments")
        if int(segment.start_event_id) != self.latest_event_id:
            raise ValueError("source odometry event IDs must be contiguous")
        if abs(float(segment.start_time) - self.latest_time) > 1e-10:
            raise ValueError("source odometry timestamps must be contiguous")
        if int(segment.end_event_id) <= int(segment.start_event_id):
            raise ValueError("odometry segment event IDs must advance")
        if float(segment.end_time) <= float(segment.start_time):
            raise ValueError("odometry segment time must advance")
        if segment.segment_id in self.segment_ids:
            raise ValueError(f"duplicate odometry segment: {segment.segment_id}")
        self.graph.add_between(
            pose_key(self.owner_id, segment.start_event_id),
            pose_key(self.owner_id, segment.end_event_id),
            segment.delta_pose,
            segment.covariance,
            kind="source_odometry_segment",
        )
        self.segments_by_start_event[int(segment.start_event_id)] = segment
        self.segment_ids.add(segment.segment_id)
        self.latest_event_id = int(segment.end_event_id)
        self.latest_time = float(segment.end_time)

    def close_increment(
        self,
        send_event_id: int,
    ) -> SourceOdomIncrement | None:
        send_event_id = int(send_event_id)
        if send_event_id > self.latest_event_id:
            raise ValueError("cannot summarize a future source event")
        if send_event_id < self.last_summary_event_id:
            raise ValueError("source summary event IDs must be monotonic")
        if send_event_id == self.last_summary_event_id:
            return None

        delta = gtsam.Pose2()
        covariance = np.zeros((3, 3), dtype=float)
        segment_ids: list[str] = []
        cursor = self.last_summary_event_id
        end_time = self.last_summary_time
        while cursor < send_event_id:
            if cursor not in self.segments_by_start_event:
                raise ValueError(
                    f"source segment gap before event {send_event_id}: {cursor}"
                )
            segment = self.segments_by_start_event[cursor]
            if int(segment.end_event_id) > send_event_id:
                raise ValueError("send event lies inside an unsplit segment")
            covariance = compose_pose_covariance(
                covariance, segment.delta_pose, segment.covariance
            )
            delta = delta.compose(segment.delta_pose)
            segment_ids.append(segment.segment_id)
            cursor = int(segment.end_event_id)
            end_time = float(segment.end_time)
        if cursor != send_event_id:
            raise ValueError("source segment chain did not close at send event")
        overlap = self.emitted_segment_ids.intersection(segment_ids)
        if overlap:
            raise RuntimeError(f"source segment overlap: {sorted(overlap)}")
        summary = SourceOdomIncrement(
            source_id=self.owner_id,
            source_epoch=self.source_epoch,
            start_event_id=self.last_summary_event_id,
            end_event_id=send_event_id,
            start_time=self.last_summary_time,
            end_time=end_time,
            delta_pose=delta,
            covariance=covariance,
            segment_ids=tuple(segment_ids),
        )
        FollowerSourcePublisher.publish(summary)
        self.emitted_segment_ids.update(segment_ids)
        self.last_summary_event_id = send_event_id
        self.last_summary_time = end_time
        return summary

    def current_signature(
        self,
    ) -> tuple[int, int, float, tuple[float, ...], tuple[float, ...]]:
        self.graph.update()
        key = pose_key(self.owner_id, self.latest_event_id)
        pose = self.graph.pose(key)
        covariance = self.graph.covariance(key)
        return (
            self.factor_count,
            self.latest_event_id,
            self.latest_time,
            (float(pose.x()), float(pose.y()), float(pose.theta())),
            tuple(float(value) for value in covariance.ravel()),
        )


SourcePublisher = FollowerSourcePublisher


def build_offset_cycle_factor(graph: IncrementalPoseGraph, cycle: Any) -> Any:
    """Legacy grid/offset factor builder retained for frozen experiments."""
    sender_events = [wave.sender_event for wave in cycle.waves]
    listener_events = list(cycle.listener_events)
    event_refs = sender_events + listener_events
    logical_keys = [pose_key(ref.agent_id, ref.base_index) for ref in event_refs]
    missing = [key for key in logical_keys if key not in graph.known_keys]
    if missing:
        raise KeyError(
            f"cycle {cycle.cycle_id} lacks event states {sorted(set(missing))}"
        )
    offsets = {
        name: ref.offset for name, ref in zip(LOGICAL_POSE_ORDER, event_refs)
    }
    event_poses = [
        graph.pose(key).compose(ref.offset)
        for key, ref in zip(logical_keys, event_refs)
    ]
    _, jacobians = evaluate_cycle_snoop_geometry(
        *event_poses,
        measured_delta_t=cycle.measured_delta_t,
        sound_speed=cycle.sound_speed_mps,
    )
    covariance = build_cycle_tdoa_covariance(
        cycle.timestamp_covariance, cycle.sound_speed_mps
    )
    for jacobian, event_ref in zip(jacobians, event_refs):
        covariance += small_matmul(
            small_matmul(
                jacobian,
                np.asarray(event_ref.offset_covariance, dtype=float),
            ),
            jacobian.T,
        )
    covariance = regularize_covariance(covariance)
    noise = gtsam.noiseModel.Gaussian.Covariance(covariance)
    return create_cycle_snoop_factor(
        *logical_keys,
        measured_delta_t=cycle.measured_delta_t,
        noise_model=noise,
        sound_speed=cycle.sound_speed_mps,
        pose_offsets=offsets,
    )


def build_cycle_factor(graph: IncrementalPoseGraph, cycle: Any) -> Any:
    """Build the event-time factor over eight logical pose slots.

    Logical arrival slots may map to one physical Pose2 key when two coded
    waves arrive simultaneously.  ``create_cycle_snoop_factor`` deduplicates
    those graph keys and sums their logical Jacobian contributions.
    """
    event_refs = [wave.sender_event for wave in cycle.waves] + list(
        cycle.listener_events
    )
    if len(event_refs) != 8:
        raise AssertionError("a cycle must contain exactly eight event poses")
    logical_keys = [int(ref.key) for ref in event_refs]
    missing = [key for key in logical_keys if key not in graph.known_keys]
    if missing:
        raise KeyError(
            f"cycle {cycle.cycle_id} lacks event states {sorted(set(missing))}"
        )
    factor = create_cycle_snoop_factor(
        *logical_keys,
        measured_delta_t=cycle.measured_delta_t,
        timestamp_covariance=cycle.timestamp_covariance,
        sound_speed=cycle.sound_speed_mps,
    )
    if set(int(key) for key in factor.keys()) != set(logical_keys):
        raise AssertionError("cycle factor changed the logical Pose2 key set")
    return factor
