"""Data and protocol model for the two paper-validation experiments.

The world, follower priors, and continuous-time odometry segments are sampled
once per seed.  The true two-wave baseline and the shared-A-boundary protocol
then reference that same world data while maintaining separate physical-wave
ledgers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import gtsam  # type: ignore
import numpy as np

from estimator.communication import FollowerInitialStateMessage, MasterPoseMessage
from estimator.distributed_graphs import pose_key
from estimator.numerics import small_matmul
from estimator.odometry_preintegration import (
    compose_pose_covariance,
    exact_unicycle_increment,
    preintegrate_continuous_segment,
    preintegrate_continuous_trajectory,
)
from factors.cycle_snoop_factor import (
    build_cycle_tdoa_covariance,
    evaluate_cycle_snoop_geometry,
)
from .model import (
    AUVSpec,
    EventPoseRef,
    FormalExperimentConfig,
    OdomSegment,
    _sample_tangent_gaussian,
    _solve_arrival_time,
)


PROTOCOL_BASELINE = "dual_master_equal_wave_budget"
PROTOCOL_JOINT = "proposed_joint_shared_boundary"
PROTOCOL_JOINT_V2 = "proposed_joint_bidirectional_v2"
RELAY_SCHEDULE_LEGACY = "legacy_seed_phase"
RELAY_SCHEDULE_ROUND_ROBIN = "round_robin"
RELAY_SCHEDULE_WEAK_DIRECTION = "weak_direction"
PROTOCOL_VARIANT_BIDIRECTIONAL = "bidirectional"
PROTOCOL_VARIANT_A_ONLY = "a_only"
PROTOCOL_VARIANT_B_ONLY = "b_only"
FORMAL_SEEDS = tuple(range(1, 61))
_V2_RELAY_PLAN_CACHE: dict[tuple[object, ...], tuple[str, ...]] = {}


@dataclass(frozen=True)
class JointProtocolBlockSpec:
    block_id: int
    anchor_master_id: str
    second_master_id: str
    anchor_start_s: float
    cycle_count: int
    period_s: float = 3.0

    @property
    def anchor_end_s(self) -> float:
        return self.anchor_start_s + self.cycle_count * self.period_s


@dataclass(frozen=True)
class FormalScenarioSpec:
    name: str
    config: FormalExperimentConfig
    baseline_cycle_count: int
    joint_cycle_count: int
    baseline_period_s: float = 2.0
    joint_period_s: float = 3.0
    relay_phase: int = 0
    joint_protocol: str = PROTOCOL_JOINT
    joint_blocks: tuple[JointProtocolBlockSpec, ...] = ()
    relay_schedule: str = RELAY_SCHEDULE_LEGACY
    radial_rotation_rad: float = 0.0
    protocol_variant: str = PROTOCOL_VARIANT_A_ONLY
    relay_processing_delay_s: float = 0.0

    def validate(self) -> None:
        self.config.validate()
        if self.baseline_cycle_count <= 0:
            raise ValueError("baseline_cycle_count must be positive")
        if self.joint_cycle_count < 0:
            raise ValueError("joint_cycle_count must be non-negative")
        if self.baseline_period_s <= 0.0 or self.joint_period_s <= 0.0:
            raise ValueError("protocol periods must be positive")
        if self.relay_processing_delay_s < 0.0:
            raise ValueError("relay processing delay must be non-negative")
        if self.joint_protocol not in {PROTOCOL_JOINT, PROTOCOL_JOINT_V2}:
            raise ValueError("unknown joint protocol")
        if self.relay_schedule not in {
            RELAY_SCHEDULE_LEGACY,
            RELAY_SCHEDULE_ROUND_ROBIN,
            RELAY_SCHEDULE_WEAK_DIRECTION,
        }:
            raise ValueError("unknown relay schedule")
        if self.joint_blocks:
            if self.joint_protocol != PROTOCOL_JOINT_V2:
                raise ValueError("joint blocks are only valid for protocol v2")
            if sum(block.cycle_count for block in self.joint_blocks) != (
                self.joint_cycle_count
            ):
                raise ValueError("joint block cycle counts do not match total")
            block_ids = [block.block_id for block in self.joint_blocks]
            if len(block_ids) != len(set(block_ids)):
                raise ValueError("joint block IDs must be unique")
            for block in self.joint_blocks:
                if block.cycle_count <= 0 or block.period_s <= 0.0:
                    raise ValueError("joint block count and period must be positive")
                if block.anchor_master_id == block.second_master_id:
                    raise ValueError("block masters must be distinct")
                if {
                    block.anchor_master_id,
                    block.second_master_id,
                } != set(self.config.master_ids):
                    raise ValueError("each block must use masters A and B")
            proposed_waves = sum(
                3 * block.cycle_count + 1 for block in self.joint_blocks
            )
        elif self.joint_cycle_count:
            proposed_waves = 3 * self.joint_cycle_count + 1
        else:
            proposed_waves = 0
        if self.joint_cycle_count:
            baseline_waves = 2 * self.baseline_cycle_count
            if proposed_waves != baseline_waves:
                raise ValueError(
                    "equal-wave radial protocols require 3*K+1 == 2*M"
                )
        latest_anchor = (
            max(block.anchor_end_s for block in self.joint_blocks)
            if self.joint_blocks
            else self.joint_cycle_count * self.joint_period_s
        )
        latest_baseline = (
            (self.baseline_cycle_count - 1) * self.baseline_period_s
        )
        if latest_anchor >= self.config.duration_s:
            raise ValueError("last shared boundary must precede experiment end")
        if latest_baseline >= self.config.duration_s:
            raise ValueError("last baseline cycle must start before experiment end")

    @property
    def expected_baseline_wave_count(self) -> int:
        return 2 * self.baseline_cycle_count

    @property
    def expected_joint_wave_count(self) -> int:
        if not self.joint_cycle_count:
            return 0
        if self.joint_blocks:
            return sum(3 * block.cycle_count + 1 for block in self.joint_blocks)
        return 3 * self.joint_cycle_count + 1


@dataclass(frozen=True)
class PhysicalWave:
    wave_id: int
    protocol: str
    sender_id: str
    send_time: float
    relay_id: str | None
    receive_times: tuple[tuple[str, float], ...]
    sender_event: EventPoseRef
    follower_receive_events: tuple[tuple[str, EventPoseRef], ...]

    def receive_time(self, receiver_id: str) -> float:
        return dict(self.receive_times)[receiver_id]

    def receive_event(self, receiver_id: str) -> EventPoseRef:
        return dict(self.follower_receive_events)[receiver_id]


@dataclass(frozen=True)
class TwoWaveMeasurement:
    measurement_id: int
    protocol_cycle_id: int
    listener_id: str
    waves: tuple[PhysicalWave, PhysicalWave]
    listener_events: tuple[EventPoseRef, EventPoseRef]
    noisy_arrival_times: np.ndarray
    timestamp_covariance: np.ndarray
    sound_speed_mps: float

    @property
    def measured_delta_t(self) -> float:
        return float(self.noisy_arrival_times[1] - self.noisy_arrival_times[0])

    @property
    def completion_time(self) -> float:
        return self.listener_events[-1].timestamp


@dataclass(frozen=True)
class SharedJointMeasurement:
    cycle_id: int
    protocol_cycle_id: int
    initiator_id: str
    second_id: str
    relay_id: str
    listener_id: str
    waves: tuple[PhysicalWave, PhysicalWave, PhysicalWave, PhysicalWave]
    listener_events: tuple[
        EventPoseRef,
        EventPoseRef,
        EventPoseRef,
        EventPoseRef,
    ]
    noisy_arrival_times: np.ndarray
    known_turnaround_corrections: np.ndarray
    timestamp_covariance: np.ndarray
    timestamp_error_keys: tuple[int, int, int, int]
    sound_speed_mps: float

    @property
    def measured_delta_t(self) -> np.ndarray:
        return np.diff(self.noisy_arrival_times) - self.known_turnaround_corrections

    @property
    def completion_time(self) -> float:
        return self.listener_events[-1].timestamp

    @property
    def logical_event_refs(self) -> tuple[EventPoseRef, ...]:
        return tuple(wave.sender_event for wave in self.waves) + self.listener_events


@dataclass(frozen=True)
class SharedTimestampMasterMeasurement:
    measurement_id: int
    protocol_cycle_id: int
    block_id: int
    anchor_master_id: str
    second_master_id: str
    listener_id: str
    is_relay_row: bool
    waves: tuple[PhysicalWave, PhysicalWave]
    listener_events: tuple[EventPoseRef, EventPoseRef]
    noisy_arrival_times: np.ndarray
    known_turnaround_delay_s: float
    timestamp_error_keys: tuple[int, int]
    sound_speed_mps: float

    @property
    def measured_delta_t(self) -> float:
        return float(
            self.noisy_arrival_times[1]
            - self.noisy_arrival_times[0]
            - self.known_turnaround_delay_s
        )

    @property
    def completion_time(self) -> float:
        return self.listener_events[-1].timestamp

    @property
    def logical_event_refs(self) -> tuple[EventPoseRef, ...]:
        return (
            self.waves[0].sender_event,
            self.waves[1].sender_event,
            self.listener_events[0],
            self.listener_events[1],
        )


@dataclass(frozen=True)
class JointProtocolCycle:
    protocol_cycle_id: int
    relay_id: str
    waves: tuple[PhysicalWave, PhysicalWave, PhysicalWave, PhysicalWave]
    listener_measurements: tuple[SharedJointMeasurement, ...]
    anchor_wait_s: float
    block_id: int = 0
    anchor_master_id: str = "A"
    second_master_id: str = "B"
    relay_master_measurement: SharedTimestampMasterMeasurement | None = None
    predicted_cross_gain_by_listener: tuple[tuple[str, float], ...] = ()
    relay_processing_delay_s: float = 0.0

    @property
    def a_wait_s(self) -> float:
        """Backward-compatible name for the old all-A-anchor protocol."""
        return self.anchor_wait_s


@dataclass(frozen=True)
class FormalWorldDataset:
    seed: int
    realization_seed: int
    spec: FormalScenarioSpec
    config: FormalExperimentConfig
    motion_profiles: Mapping[str, object]
    truth_grid: Mapping[str, np.ndarray]
    follower_timelines: Mapping[str, tuple[EventPoseRef, ...]]
    grid_event_refs: Mapping[str, tuple[EventPoseRef, ...]]
    odom_segments: Mapping[str, tuple[OdomSegment, ...]]
    initial_state_messages: Mapping[str, FollowerInitialStateMessage]
    waves: Mapping[int, PhysicalWave]
    baseline_measurements: tuple[TwoWaveMeasurement, ...]
    joint_cycles: tuple[JointProtocolCycle, ...]
    joint_master_measurements: tuple[SharedTimestampMasterMeasurement, ...]
    master_messages: Mapping[int, MasterPoseMessage]
    timestamp_error_keys: Mapping[tuple[str, int], int]
    timestamp_errors: Mapping[tuple[str, int], float]

    @property
    def baseline_waves(self) -> tuple[PhysicalWave, ...]:
        return tuple(
            wave for wave in self.waves.values() if wave.protocol == PROTOCOL_BASELINE
        )

    @property
    def joint_waves(self) -> tuple[PhysicalWave, ...]:
        return tuple(
            wave
            for wave in self.waves.values()
            if wave.protocol in {PROTOCOL_JOINT, PROTOCOL_JOINT_V2}
        )

    @property
    def joint_measurements(self) -> tuple[SharedJointMeasurement, ...]:
        return tuple(
            measurement
            for cycle in self.joint_cycles
            for measurement in cycle.listener_measurements
        )

    def truth_pose(self, agent_id: str, time_s: float) -> gtsam.Pose2:
        profile = self.motion_profiles.get(agent_id)
        if profile is not None:
            return profile.pose_at(float(time_s))
        spec = next(item for item in self.config.agents if item.agent_id == agent_id)
        return spec.initial_pose.compose(
            exact_unicycle_increment(spec.velocity, spec.angular_rate, time_s)
        )


@dataclass(frozen=True)
class _RawWave:
    wave_id: int
    protocol: str
    sender_id: str
    send_time: float
    relay_id: str | None
    receive_times: tuple[tuple[str, float], ...]

    def receive_time(self, receiver_id: str) -> float:
        return dict(self.receive_times)[receiver_id]


@dataclass(frozen=True)
class _RawJointCycle:
    cycle_id: int
    relay_id: str
    wave_ids: tuple[int, int, int, int]
    anchor_wait_s: float
    block_id: int = 0
    anchor_master_id: str = "A"
    second_master_id: str = "B"
    predicted_cross_gain_by_listener: tuple[tuple[str, float], ...] = ()
    relay_processing_delay_s: float = 0.0

    @property
    def a_wait_s(self) -> float:
        return self.anchor_wait_s


def _common_config(
    agents: tuple[AUVSpec, ...],
    *,
    duration_s: float,
    seeds: tuple[int, ...],
    minimum_spacing_m: float,
) -> FormalExperimentConfig:
    return FormalExperimentConfig(
        duration_s=float(duration_s),
        sensor_dt_s=0.1,
        cycle_period_s=3.0,
        turnaround_delay_s=0.0,
        sound_speed_mps=1500.0,
        dvl_sigma_mps=0.05,
        gyro_sigma_radps=math.radians(2.0),
        lateral_sigma_mps=0.05,
        timestamp_sigma_s=0.002,
        follower_initial_xy_sigma_m=1.0,
        follower_initial_heading_sigma_rad=math.radians(5.0),
        master_xy_sigma_m=1e-3,
        master_heading_sigma_rad=math.radians(0.01),
        evaluation_period_s=1.0,
        minimum_initial_spacing_m=minimum_spacing_m,
        agents=agents,
        seeds=seeds,
    )


def build_controlled_spec(
    direction: str,
    *,
    duration_s: float = 200.0,
    seeds: tuple[int, ...] = FORMAL_SEEDS,
    baseline_cycle_count: int = 100,
) -> FormalScenarioSpec:
    direction = direction.lower()
    if direction not in {"x", "y"}:
        raise ValueError("controlled direction must be x or y")
    heading = 0.0 if direction == "x" else math.pi / 2.0
    geometry = "side" if direction == "x" else "front_back"
    config = _common_config(
        (
            AUVSpec("A", "master", 0.0, -500.0, 0.0, 0.0, 0.0),
            AUVSpec("B", "master", 0.0, +500.0, 0.0, 0.0, 0.0),
            AUVSpec("C", "follower", 0.0, 0.0, heading, 1.5, 0.0),
        ),
        duration_s=duration_s,
        seeds=seeds,
        minimum_spacing_m=100.0,
    )
    spec = FormalScenarioSpec(
        name=f"controlled_dual_master_{geometry}",
        config=config,
        baseline_cycle_count=baseline_cycle_count,
        joint_cycle_count=0,
    )
    spec.validate()
    return spec


def build_radial_spec(
    *,
    duration_s: float = 200.0,
    seeds: tuple[int, ...] = FORMAL_SEEDS,
    joint_cycle_count: int = 65,
    baseline_cycle_count: int = 98,
    relay_phase: int = 0,
) -> FormalScenarioSpec:
    followers = tuple(
        AUVSpec(
            chr(ord("C") + index),
            "follower",
            100.0 * math.cos(index * math.pi / 3.0),
            100.0 * math.sin(index * math.pi / 3.0),
            index * math.pi / 3.0,
            1.5,
            0.0,
        )
        for index in range(6)
    )
    config = _common_config(
        (
            AUVSpec("A", "master", 0.0, -500.0, 0.0, 0.0, 0.0),
            AUVSpec("B", "master", 0.0, +500.0, 0.0, 0.0, 0.0),
        )
        + followers,
        duration_s=duration_s,
        seeds=seeds,
        minimum_spacing_m=95.0,
    )
    spec = FormalScenarioSpec(
        name="six_follower_radial_dispersion",
        config=config,
        baseline_cycle_count=baseline_cycle_count,
        joint_cycle_count=joint_cycle_count,
        relay_phase=int(relay_phase) % 6,
    )
    spec.validate()
    return spec


def build_radial_v2_spec(
    *,
    rotation_deg: float = 0.0,
    duration_s: float = 200.0,
    seeds: tuple[int, ...] = FORMAL_SEEDS,
    baseline_cycle_count: int = 100,
    block_cycle_count: int = 33,
    second_block_start_s: float = 100.0,
    relay_schedule: str = RELAY_SCHEDULE_ROUND_ROBIN,
    protocol_variant: str = PROTOCOL_VARIANT_BIDIRECTIONAL,
    relay_processing_delay_s: float = 0.0,
) -> FormalScenarioSpec:
    """Build the frozen 200-wave bidirectional radial experiment.

    The two independent timestamp chains each contain ``3*K+1`` physical
    waves.  With ``K=33`` this is 100 waves per block and 200 waves total,
    exactly matching 100 genuine two-wave dual-master cycles.
    """
    rotation_rad = math.radians(float(rotation_deg))
    if protocol_variant not in {
        PROTOCOL_VARIANT_BIDIRECTIONAL,
        PROTOCOL_VARIANT_A_ONLY,
        PROTOCOL_VARIANT_B_ONLY,
    }:
        raise ValueError("unknown v2 protocol variant")
    if relay_schedule not in {
        RELAY_SCHEDULE_ROUND_ROBIN,
        RELAY_SCHEDULE_WEAK_DIRECTION,
    }:
        raise ValueError("v2 requires round_robin or weak_direction scheduling")
    followers = tuple(
        AUVSpec(
            chr(ord("C") + index),
            "follower",
            100.0 * math.cos(rotation_rad + index * math.pi / 3.0),
            100.0 * math.sin(rotation_rad + index * math.pi / 3.0),
            rotation_rad + index * math.pi / 3.0,
            1.5,
            0.0,
        )
        for index in range(6)
    )
    config = _common_config(
        (
            AUVSpec("A", "master", 0.0, -500.0, 0.0, 0.0, 0.0),
            AUVSpec("B", "master", 0.0, +500.0, 0.0, 0.0, 0.0),
        )
        + followers,
        duration_s=duration_s,
        seeds=seeds,
        minimum_spacing_m=95.0,
    )
    if protocol_variant == PROTOCOL_VARIANT_BIDIRECTIONAL:
        roles = (("A", "B"), ("B", "A"))
    elif protocol_variant == PROTOCOL_VARIANT_A_ONLY:
        roles = (("A", "B"), ("A", "B"))
    else:
        roles = (("B", "A"), ("B", "A"))
    blocks = (
        JointProtocolBlockSpec(
            block_id=0,
            anchor_master_id=roles[0][0],
            second_master_id=roles[0][1],
            anchor_start_s=0.0,
            cycle_count=block_cycle_count,
            period_s=3.0,
        ),
        JointProtocolBlockSpec(
            block_id=1,
            anchor_master_id=roles[1][0],
            second_master_id=roles[1][1],
            anchor_start_s=float(second_block_start_s),
            cycle_count=block_cycle_count,
            period_s=3.0,
        ),
    )
    geometry_name = "rotated_30deg" if abs(rotation_deg - 30.0) < 1e-12 else (
        f"rotated_{rotation_deg:g}deg" if abs(rotation_deg) > 1e-12 else "original"
    )
    spec = FormalScenarioSpec(
        name=f"six_follower_radial_v2_{geometry_name}",
        config=config,
        baseline_cycle_count=baseline_cycle_count,
        joint_cycle_count=2 * int(block_cycle_count),
        joint_period_s=3.0,
        relay_phase=0,
        joint_protocol=PROTOCOL_JOINT_V2,
        joint_blocks=blocks,
        relay_schedule=relay_schedule,
        radial_rotation_rad=rotation_rad,
        protocol_variant=protocol_variant,
        relay_processing_delay_s=float(relay_processing_delay_s),
    )
    spec.validate()
    return spec


def _raw_wave(
    wave_id: int,
    protocol: str,
    sender_id: str,
    send_time: float,
    relay_id: str | None,
    config: FormalExperimentConfig,
    pose_at,
) -> _RawWave:
    receive_times = tuple(
        (
            receiver_id,
            _solve_arrival_time(
                pose_at,
                sender_id,
                receiver_id,
                send_time,
                config.sound_speed_mps,
            ),
        )
        for receiver_id in config.agent_ids
        if receiver_id != sender_id
    )
    return _RawWave(
        wave_id=wave_id,
        protocol=protocol,
        sender_id=sender_id,
        send_time=float(send_time),
        relay_id=relay_id,
        receive_times=receive_times,
    )


def _predicted_peer_cross_gains(
    config: FormalExperimentConfig,
    pose_at,
    waves: tuple[_RawWave, _RawWave, _RawWave, _RawWave],
    relay_id: str,
) -> dict[str, float]:
    """Nominal Schur-complement gain beyond the retained master row.

    The planning score uses only the deterministic mission trajectory and the
    source-only covariance predicted from the deployment prior and continuous
    odometry model.  It never sees sampled truth error, acoustic noise, or a
    cooperative posterior.
    """
    relay_send_time = waves[2].send_time
    nominal_delta, source_process_covariance = preintegrate_continuous_trajectory(
        0.0,
        relay_send_time,
        config.continuous_noise,
        pose_at=lambda time_s: pose_at(relay_id, time_s),
    )
    source_covariance = compose_pose_covariance(
        config.follower_initial_covariance,
        nominal_delta,
        source_process_covariance,
    )
    timing_covariance = build_cycle_tdoa_covariance(
        np.eye(4, dtype=float) * config.timestamp_sigma_s**2,
        config.sound_speed_mps,
    )
    gains = {follower_id: 0.0 for follower_id in config.follower_ids}
    for listener_id in config.follower_ids:
        if listener_id == relay_id:
            continue
        listener_times = [wave.receive_time(listener_id) for wave in waves]
        poses = [
            pose_at(waves[0].sender_id, waves[0].send_time),
            pose_at(waves[1].sender_id, waves[1].send_time),
            pose_at(relay_id, relay_send_time),
            pose_at(waves[3].sender_id, waves[3].send_time),
        ] + [pose_at(listener_id, time_s) for time_s in listener_times]
        _, jacobians = evaluate_cycle_snoop_geometry(
            *poses,
            measured_delta_t=np.zeros(3, dtype=float),
            sound_speed=config.sound_speed_mps,
        )
        listener_jacobian = sum(jacobians[4:], np.zeros((3, 3), dtype=float))
        relay_jacobian = jacobians[2]
        effective_covariance = (
            timing_covariance
            + small_matmul(
                small_matmul(relay_jacobian, source_covariance),
                relay_jacobian.T,
            )
        )
        effective_covariance = 0.5 * (
            effective_covariance + effective_covariance.T
        )
        inverse_effective = _inverse_symmetric_3x3(effective_covariance)
        joint_information = (
            small_matmul(
                small_matmul(listener_jacobian.T, inverse_effective),
                listener_jacobian,
            )
        )
        master_jacobian = listener_jacobian[[0], :]
        master_information = (
            small_matmul(master_jacobian.T, master_jacobian)
            / float(effective_covariance[0, 0])
        )
        incremental_information = 0.5 * (
            joint_information
            - master_information
            + (joint_information - master_information).T
        )
        gains[listener_id] = max(
            0.0,
            float(incremental_information[1, 1]),
        )
    return gains


def _inverse_symmetric_3x3(matrix: np.ndarray) -> np.ndarray:
    """Small explicit inverse, avoiding platform BLAS dispatch in planning."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("matrix must be 3x3")
    a, b, c = (float(matrix[0, index]) for index in range(3))
    d, e = float(matrix[1, 1]), float(matrix[1, 2])
    f = float(matrix[2, 2])
    cofactor00 = d * f - e * e
    cofactor01 = c * e - b * f
    cofactor02 = b * e - c * d
    cofactor11 = a * f - c * c
    cofactor12 = b * c - a * e
    cofactor22 = a * d - b * b
    determinant = a * cofactor00 + b * cofactor01 + c * cofactor02
    if not math.isfinite(determinant) or determinant <= 1e-18:
        raise ValueError("effective timing covariance is not positive definite")
    inverse = np.asarray(
        [
            [cofactor00, cofactor01, cofactor02],
            [cofactor01, cofactor11, cofactor12],
            [cofactor02, cofactor12, cofactor22],
        ],
        dtype=float,
    ) / determinant
    return inverse


def _choose_weak_direction_relay(
    followers: tuple[str, ...],
    relay_counts: dict[str, int],
    relay_quota: int,
    cumulative_gains: dict[str, float],
    candidate_gains: Mapping[str, Mapping[str, float]],
) -> str:
    candidates = [
        follower_id
        for follower_id in followers
        if relay_counts[follower_id] < relay_quota
        and follower_id in candidate_gains
    ]
    if not candidates:
        raise AssertionError("relay scheduler exhausted its quota early")
    best_id = candidates[0]
    best_score: tuple[float, float] | None = None
    for candidate in candidates:
        projected = {
            follower_id: cumulative_gains[follower_id]
            + float(candidate_gains[candidate][follower_id])
            for follower_id in followers
        }
        score = (min(projected.values()), sum(projected.values()))
        if best_score is None or score > best_score:
            best_id = candidate
            best_score = score
    return best_id


def _has_distinct_listener_events(
    config: FormalExperimentConfig,
    waves: tuple[_RawWave, _RawWave, _RawWave, _RawWave],
    relay_id: str,
) -> bool:
    """Reject exact collinear co-arrivals that would collapse two Pose2 events."""
    for listener_id in config.follower_ids:
        if listener_id == relay_id:
            continue
        arrivals = [wave.receive_time(listener_id) for wave in waves]
        if any(
            abs(second - first) <= 1e-10
            for first, second in zip(arrivals, arrivals[1:])
        ):
            return False
    return True


def _v2_plan_cache_key(spec: FormalScenarioSpec) -> tuple[object, ...]:
    return (
        spec.name,
        spec.relay_schedule,
        spec.protocol_variant,
        tuple(spec.joint_blocks),
        spec.relay_processing_delay_s,
        spec.config.timestamp_sigma_s,
        spec.config.sound_speed_mps,
    )


def _plan_v2_relays(
    spec: FormalScenarioSpec, pose_at, *, use_cache: bool = True
) -> tuple[str, ...]:
    cache_key = _v2_plan_cache_key(spec)
    cached = _V2_RELAY_PLAN_CACHE.get(cache_key) if use_cache else None
    if cached is not None:
        return cached
    config = spec.config
    followers = config.follower_ids
    base_quota, remainder = divmod(spec.joint_cycle_count, len(followers))
    target_quota = {
        follower_id: base_quota + int(index < remainder)
        for index, follower_id in enumerate(followers)
    }
    relay_quota = max(target_quota.values())
    safe_by_cycle: list[tuple[str, ...]] = []
    gains_by_cycle: list[dict[str, dict[str, float]]] = []
    for block in spec.joint_blocks:
        for local_cycle_id in range(block.cycle_count):
            anchor_time = block.anchor_start_s + local_cycle_id * block.period_s
            wave_anchor0 = _raw_wave(
                -1,
                PROTOCOL_JOINT_V2,
                block.anchor_master_id,
                anchor_time,
                None,
                config,
                pose_at,
            )
            wave_anchor1 = _raw_wave(
                -4,
                PROTOCOL_JOINT_V2,
                block.anchor_master_id,
                anchor_time + block.period_s,
                None,
                config,
                pose_at,
            )
            safe: list[str] = []
            gains: dict[str, dict[str, float]] = {}
            for candidate in followers:
                wave_second = _raw_wave(
                    -2,
                    PROTOCOL_JOINT_V2,
                    block.second_master_id,
                    wave_anchor0.receive_time(block.second_master_id),
                    candidate,
                    config,
                    pose_at,
                )
                wave_relay = _raw_wave(
                    -3,
                    PROTOCOL_JOINT_V2,
                    candidate,
                    wave_second.receive_time(candidate)
                    + spec.relay_processing_delay_s,
                    candidate,
                    config,
                    pose_at,
                )
                candidate_waves = (
                    wave_anchor0,
                    wave_second,
                    wave_relay,
                    wave_anchor1,
                )
                if not _has_distinct_listener_events(
                    config, candidate_waves, candidate
                ):
                    continue
                safe.append(candidate)
                if spec.relay_schedule == RELAY_SCHEDULE_WEAK_DIRECTION:
                    gains[candidate] = _predicted_peer_cross_gains(
                        config, pose_at, candidate_waves, candidate
                    )
            if not safe:
                raise ValueError("v2 cycle has no relay preserving eight events")
            safe_by_cycle.append(tuple(safe))
            gains_by_cycle.append(gains)

    counts = {follower_id: 0 for follower_id in followers}
    cumulative = {follower_id: 0.0 for follower_id in followers}
    plan: list[str] = []
    for cycle_index, safe in enumerate(safe_by_cycle):
        feasible: list[str] = []
        for candidate in safe:
            if counts[candidate] >= target_quota[candidate]:
                continue
            possible = True
            for follower_id in followers:
                used = counts[follower_id] + int(follower_id == candidate)
                remaining_need = target_quota[follower_id] - used
                future_safe = sum(
                    follower_id in future
                    for future in safe_by_cycle[cycle_index + 1 :]
                )
                if future_safe < remaining_need:
                    possible = False
                    break
            if possible:
                feasible.append(candidate)
        if not feasible:
            raise ValueError(
                f"relay quota is infeasible at v2 cycle {cycle_index}"
            )
        if spec.relay_schedule == RELAY_SCHEDULE_ROUND_ROBIN:
            relay_id = next(
                follower_id
                for offset in range(len(followers))
                for follower_id in (
                    followers[(cycle_index + offset) % len(followers)],
                )
                if follower_id in feasible
            )
            selected_gain = _predicted_peer_cross_gains_for_plan(
                spec, pose_at, cycle_index, relay_id
            )
        else:
            feasible_gains = {
                candidate: gains_by_cycle[cycle_index][candidate]
                for candidate in feasible
            }
            relay_id = _choose_weak_direction_relay(
                followers,
                counts,
                relay_quota,
                cumulative,
                feasible_gains,
            )
            selected_gain = feasible_gains[relay_id]
        plan.append(relay_id)
        counts[relay_id] += 1
        for follower_id in followers:
            cumulative[follower_id] += selected_gain[follower_id]
    if counts != target_quota:
        raise AssertionError("v2 relay planner did not meet its balanced quotas")
    result = tuple(plan)
    if use_cache:
        _V2_RELAY_PLAN_CACHE[cache_key] = result
    return result


def _predicted_peer_cross_gains_for_plan(
    spec: FormalScenarioSpec,
    pose_at,
    cycle_index: int,
    relay_id: str,
) -> dict[str, float]:
    """Rebuild one nominal candidate for round-robin audit scoring."""
    cursor = 0
    for block in spec.joint_blocks:
        if cycle_index < cursor + block.cycle_count:
            local = cycle_index - cursor
            anchor_time = block.anchor_start_s + local * block.period_s
            w0 = _raw_wave(-1, PROTOCOL_JOINT_V2, block.anchor_master_id, anchor_time, None, spec.config, pose_at)
            w1 = _raw_wave(-2, PROTOCOL_JOINT_V2, block.second_master_id, w0.receive_time(block.second_master_id), relay_id, spec.config, pose_at)
            w2 = _raw_wave(-3, PROTOCOL_JOINT_V2, relay_id, w1.receive_time(relay_id) + spec.relay_processing_delay_s, relay_id, spec.config, pose_at)
            w3 = _raw_wave(-4, PROTOCOL_JOINT_V2, block.anchor_master_id, anchor_time + block.period_s, None, spec.config, pose_at)
            return _predicted_peer_cross_gains(spec.config, pose_at, (w0, w1, w2, w3), relay_id)
        cursor += block.cycle_count
    raise IndexError("v2 cycle index is outside all blocks")


def _build_raw_protocols(
    spec: FormalScenarioSpec, pose_at, *, use_relay_plan_cache: bool = True
):
    config = spec.config
    waves: dict[int, _RawWave] = {}
    baseline_cycles: list[tuple[int, int]] = []
    next_wave_id = 0
    for cycle_id in range(spec.baseline_cycle_count):
        start = cycle_id * spec.baseline_period_s
        wave_a = _raw_wave(
            next_wave_id,
            PROTOCOL_BASELINE,
            "A",
            start,
            None,
            config,
            pose_at,
        )
        waves[wave_a.wave_id] = wave_a
        next_wave_id += 1
        send_b = wave_a.receive_time("B")
        wave_b = _raw_wave(
            next_wave_id,
            PROTOCOL_BASELINE,
            "B",
            send_b,
            None,
            config,
            pose_at,
        )
        waves[wave_b.wave_id] = wave_b
        next_wave_id += 1
        baseline_cycles.append((wave_a.wave_id, wave_b.wave_id))

    raw_joint_cycles: list[_RawJointCycle] = []
    if spec.joint_cycle_count and spec.joint_protocol == PROTOCOL_JOINT:
        anchors: list[int] = []
        for anchor_index in range(spec.joint_cycle_count + 1):
            wave = _raw_wave(
                next_wave_id,
                PROTOCOL_JOINT,
                "A",
                anchor_index * spec.joint_period_s,
                None,
                config,
                pose_at,
            )
            waves[wave.wave_id] = wave
            anchors.append(wave.wave_id)
            next_wave_id += 1
        followers = config.follower_ids
        for cycle_id in range(spec.joint_cycle_count):
            relay_id = followers[(spec.relay_phase + cycle_id) % len(followers)]
            wave_a0 = waves[anchors[cycle_id]]
            wave_b = _raw_wave(
                next_wave_id,
                PROTOCOL_JOINT,
                "B",
                wave_a0.receive_time("B"),
                relay_id,
                config,
                pose_at,
            )
            waves[wave_b.wave_id] = wave_b
            next_wave_id += 1
            wave_s = _raw_wave(
                next_wave_id,
                PROTOCOL_JOINT,
                relay_id,
                wave_b.receive_time(relay_id),
                relay_id,
                config,
                pose_at,
            )
            waves[wave_s.wave_id] = wave_s
            next_wave_id += 1
            wave_a1 = waves[anchors[cycle_id + 1]]
            a_receive = wave_s.receive_time("A")
            a_wait = wave_a1.send_time - a_receive
            if a_wait < -1e-10:
                raise ValueError(
                    f"cycle {cycle_id} cannot close before the next 3 s anchor"
                )
            raw_joint_cycles.append(
                _RawJointCycle(
                    cycle_id=cycle_id,
                    relay_id=relay_id,
                    wave_ids=(
                        wave_a0.wave_id,
                        wave_b.wave_id,
                        wave_s.wave_id,
                        wave_a1.wave_id,
                    ),
                    anchor_wait_s=max(0.0, a_wait),
                )
            )
    elif spec.joint_cycle_count:
        followers = config.follower_ids
        relay_plan = _plan_v2_relays(
            spec, pose_at, use_cache=use_relay_plan_cache
        )
        global_cycle_id = 0
        for block in spec.joint_blocks:
            anchors: list[int] = []
            for anchor_index in range(block.cycle_count + 1):
                anchor = _raw_wave(
                    next_wave_id,
                    PROTOCOL_JOINT_V2,
                    block.anchor_master_id,
                    block.anchor_start_s + anchor_index * block.period_s,
                    None,
                    config,
                    pose_at,
                )
                waves[anchor.wave_id] = anchor
                anchors.append(anchor.wave_id)
                next_wave_id += 1
            for local_cycle_id in range(block.cycle_count):
                wave_anchor0 = waves[anchors[local_cycle_id]]
                relay_id = relay_plan[global_cycle_id]
                wave_second = _raw_wave(
                    next_wave_id,
                    PROTOCOL_JOINT_V2,
                    block.second_master_id,
                    wave_anchor0.receive_time(block.second_master_id),
                    relay_id,
                    config,
                    pose_at,
                )
                wave_relay = _raw_wave(
                    next_wave_id + 1,
                    PROTOCOL_JOINT_V2,
                    relay_id,
                    wave_second.receive_time(relay_id)
                    + spec.relay_processing_delay_s,
                    relay_id,
                    config,
                    pose_at,
                )
                selected_waves = (
                    wave_anchor0,
                    wave_second,
                    wave_relay,
                    waves[anchors[local_cycle_id + 1]],
                )
                if not _has_distinct_listener_events(
                    config, selected_waves, relay_id
                ):
                    raise AssertionError("cached relay plan collapses event nodes")
                selected_gains = _predicted_peer_cross_gains(
                    config, pose_at, selected_waves, relay_id
                )
                waves[wave_second.wave_id] = wave_second
                waves[wave_relay.wave_id] = wave_relay
                next_wave_id += 2
                anchor_receive = wave_relay.receive_time(block.anchor_master_id)
                anchor_wait = selected_waves[3].send_time - anchor_receive
                if anchor_wait < -1e-10:
                    raise ValueError(
                        f"v2 cycle {global_cycle_id} cannot close before anchor"
                    )
                raw_joint_cycles.append(
                    _RawJointCycle(
                        cycle_id=global_cycle_id,
                        relay_id=relay_id,
                        wave_ids=tuple(wave.wave_id for wave in selected_waves),
                        anchor_wait_s=max(0.0, anchor_wait),
                        block_id=block.block_id,
                        anchor_master_id=block.anchor_master_id,
                        second_master_id=block.second_master_id,
                        predicted_cross_gain_by_listener=tuple(
                            (follower_id, float(selected_gains[follower_id]))
                            for follower_id in followers
                        ),
                        relay_processing_delay_s=spec.relay_processing_delay_s,
                    )
                )
                global_cycle_id += 1
    return waves, tuple(baseline_cycles), tuple(raw_joint_cycles)


def _build_event_timelines(
    config: FormalExperimentConfig,
    raw_waves: Mapping[int, _RawWave],
):
    tokens: dict[str, list[tuple[float, tuple[str, int]]]] = {
        follower_id: [] for follower_id in config.follower_ids
    }
    for follower_id in config.follower_ids:
        for grid_index in range(config.step_count + 1):
            tokens[follower_id].append(
                (grid_index * config.sensor_dt_s, ("GRID", grid_index))
            )
    for wave in raw_waves.values():
        if wave.sender_id in config.follower_ids:
            tokens[wave.sender_id].append(
                (wave.send_time, ("SEND", wave.wave_id))
            )
        for receiver_id, receive_time in wave.receive_times:
            if receiver_id in config.follower_ids:
                tokens[receiver_id].append(
                    (receive_time, ("RECEIVE", wave.wave_id))
                )

    timelines: dict[str, tuple[EventPoseRef, ...]] = {}
    grid_refs: dict[str, tuple[EventPoseRef, ...]] = {}
    token_refs: dict[tuple[str, str, int], EventPoseRef] = {}
    for follower_id in config.follower_ids:
        ordered = sorted(tokens[follower_id], key=lambda item: (item[0], item[1]))
        groups: list[list[tuple[float, tuple[str, int]]]] = []
        for item in ordered:
            if groups and abs(item[0] - groups[-1][0][0]) <= 1e-11:
                groups[-1].append(item)
            else:
                groups.append([item])
        refs: list[EventPoseRef] = []
        for event_id, group in enumerate(groups):
            event_types = sorted({token[0] for _, token in group})
            ref = EventPoseRef(
                agent_id=follower_id,
                event_id=event_id,
                timestamp=float(group[0][0]),
                key=pose_key(follower_id, event_id),
                event_type="+".join(event_types),
            )
            refs.append(ref)
            for _, (event_type, token_id) in group:
                token_refs[(follower_id, event_type, token_id)] = ref
        timelines[follower_id] = tuple(refs)
        grid_refs[follower_id] = tuple(
            token_refs[(follower_id, "GRID", index)]
            for index in range(config.step_count + 1)
        )
    return timelines, grid_refs, token_refs


def _master_event_data(
    seed: int,
    realization_seed: int,
    config: FormalExperimentConfig,
    raw_waves: Mapping[int, _RawWave],
    pose_at,
):
    master_refs: dict[int, EventPoseRef] = {}
    messages: dict[int, MasterPoseMessage] = {}
    by_master = {
        master_id: sorted(
            (
                wave for wave in raw_waves.values() if wave.sender_id == master_id
            ),
            key=lambda wave: (wave.send_time, wave.wave_id),
        )
        for master_id in config.master_ids
    }
    for master_offset, master_id in enumerate(config.master_ids):
        rng = np.random.RandomState(
            realization_seed * 1009 + 53 + 17 * master_offset
        )
        for event_id, wave in enumerate(by_master[master_id]):
            ref = EventPoseRef(
                agent_id=master_id,
                event_id=event_id,
                timestamp=wave.send_time,
                key=pose_key(master_id, event_id),
                event_type="MASTER_SEND",
            )
            master_refs[wave.wave_id] = ref
            error = np.asarray(
                [
                    rng.normal(0.0, config.master_xy_sigma_m),
                    rng.normal(0.0, config.master_xy_sigma_m),
                    rng.normal(0.0, config.master_heading_sigma_rad),
                ],
                dtype=float,
            )
            messages[wave.wave_id] = MasterPoseMessage(
                master_id=master_id,
                source_epoch=seed,
                sequence=wave.wave_id,
                timestamp=wave.send_time,
                state_index=event_id,
                pose=pose_at(master_id, wave.send_time).retract(error),
                covariance=config.master_covariance,
            )
    return master_refs, messages


def generate_formal_world(
    seed: int,
    spec: FormalScenarioSpec,
    *,
    motion_profiles: Mapping[str, object] | None = None,
    realization_seed: int | None = None,
) -> FormalWorldDataset:
    spec.validate()
    config = spec.config
    specs = {item.agent_id: item for item in config.agents}
    active_profiles = dict(motion_profiles or {})
    random_seed = int(seed if realization_seed is None else realization_seed)
    if random_seed <= 0 or random_seed > 4_000_000:
        raise ValueError("realization_seed must be in [1, 4000000]")
    unknown_profiles = set(active_profiles) - set(specs)
    if unknown_profiles:
        raise ValueError(f"motion profiles contain unknown agents: {unknown_profiles}")

    def pose_at(agent_id: str, time_s: float) -> gtsam.Pose2:
        profile = active_profiles.get(agent_id)
        if profile is not None:
            return profile.pose_at(float(time_s))
        agent = specs[agent_id]
        return agent.initial_pose.compose(
            exact_unicycle_increment(agent.velocity, agent.angular_rate, time_s)
        )

    raw_waves, raw_baseline, raw_joint = _build_raw_protocols(
        spec, pose_at, use_relay_plan_cache=not bool(active_profiles)
    )
    timelines, grid_refs, token_refs = _build_event_timelines(config, raw_waves)
    master_refs, master_messages = _master_event_data(
        seed, random_seed, config, raw_waves, pose_at
    )

    waves: dict[int, PhysicalWave] = {}
    for wave_id, raw in raw_waves.items():
        sender_ref = (
            master_refs[wave_id]
            if raw.sender_id in config.master_ids
            else token_refs[(raw.sender_id, "SEND", wave_id)]
        )
        receive_refs = tuple(
            (
                receiver_id,
                token_refs[(receiver_id, "RECEIVE", wave_id)],
            )
            for receiver_id, _ in raw.receive_times
            if receiver_id in config.follower_ids
        )
        waves[wave_id] = PhysicalWave(
            wave_id=wave_id,
            protocol=raw.protocol,
            sender_id=raw.sender_id,
            send_time=raw.send_time,
            relay_id=raw.relay_id,
            receive_times=raw.receive_times,
            sender_event=sender_ref,
            follower_receive_events=receive_refs,
        )

    prior_rng = np.random.RandomState(random_seed * 1009 + 11)
    initial_messages: dict[str, FollowerInitialStateMessage] = {}
    for follower_id in config.follower_ids:
        error = np.asarray(
            [
                prior_rng.normal(0.0, config.follower_initial_xy_sigma_m),
                prior_rng.normal(0.0, config.follower_initial_xy_sigma_m),
                prior_rng.normal(0.0, config.follower_initial_heading_sigma_rad),
            ],
            dtype=float,
        )
        initial_messages[follower_id] = FollowerInitialStateMessage(
            follower_id=follower_id,
            pose=pose_at(follower_id, 0.0).retract(error),
            covariance=config.follower_initial_covariance,
            timestamp=0.0,
            source_epoch=seed,
        )

    odom_segments: dict[str, tuple[OdomSegment, ...]] = {}
    for follower_offset, follower_id in enumerate(config.follower_ids):
        rng = np.random.RandomState(
            random_seed * 1009 + 101 + 37 * follower_offset
        )
        agent = specs[follower_id]
        segments: list[OdomSegment] = []
        for first, second in zip(
            timelines[follower_id], timelines[follower_id][1:]
        ):
            if follower_id in active_profiles:
                nominal, covariance = preintegrate_continuous_trajectory(
                    first.timestamp,
                    second.timestamp,
                    config.continuous_noise,
                    pose_at=active_profiles[follower_id].pose_at,
                )
            else:
                nominal, covariance = preintegrate_continuous_segment(
                    first.timestamp,
                    second.timestamp,
                    config.continuous_noise,
                    velocity=agent.velocity,
                    angular_rate=agent.angular_rate,
                )
            delta = nominal.retract(_sample_tangent_gaussian(covariance, rng))
            segments.append(
                OdomSegment(
                    segment_id=f"{follower_id}:raw:{first.event_id}:{second.event_id}",
                    agent_id=follower_id,
                    start_event_id=first.event_id,
                    end_event_id=second.event_id,
                    start_time=first.timestamp,
                    end_time=second.timestamp,
                    delta_pose=delta,
                    covariance=covariance,
                )
            )
        odom_segments[follower_id] = tuple(segments)

    acoustic_rng = np.random.RandomState(random_seed * 1009 + 29)
    timestamp_errors: dict[tuple[str, int], float] = {}
    for listener_id in config.follower_ids:
        for wave_id in sorted(waves):
            wave = waves[wave_id]
            if listener_id == wave.sender_id:
                continue
            timestamp_errors[(listener_id, wave_id)] = float(
                acoustic_rng.normal(0.0, config.timestamp_sigma_s)
            )

    timestamp_keys: dict[tuple[str, int], int] = {}
    joint_pairs = sorted(
        {
            (listener_id, wave_id)
            for raw_cycle in raw_joint
            for listener_id in config.follower_ids
            for wave_id in raw_cycle.wave_ids
            if (
                listener_id != raw_cycle.relay_id
                or (
                    spec.joint_protocol == PROTOCOL_JOINT_V2
                    and wave_id in raw_cycle.wave_ids[:2]
                )
            )
        }
    )
    for index, pair in enumerate(joint_pairs):
        timestamp_keys[pair] = int(gtsam.symbol("T", index))

    baseline_measurements: list[TwoWaveMeasurement] = []
    measurement_id = 0
    for cycle_id, wave_ids in enumerate(raw_baseline):
        cycle_waves = tuple(waves[wave_id] for wave_id in wave_ids)
        for listener_id in config.follower_ids:
            listener_events = tuple(
                wave.receive_event(listener_id) for wave in cycle_waves
            )
            true_arrivals = np.asarray(
                [event.timestamp for event in listener_events], dtype=float
            )
            noise = np.asarray(
                [timestamp_errors[(listener_id, wave.wave_id)] for wave in cycle_waves],
                dtype=float,
            )
            baseline_measurements.append(
                TwoWaveMeasurement(
                    measurement_id=measurement_id,
                    protocol_cycle_id=cycle_id,
                    listener_id=listener_id,
                    waves=cycle_waves,
                    listener_events=listener_events,
                    noisy_arrival_times=true_arrivals + noise,
                    timestamp_covariance=(
                        np.eye(2, dtype=float) * config.timestamp_sigma_s**2
                    ),
                    sound_speed_mps=config.sound_speed_mps,
                )
            )
            measurement_id += 1

    joint_cycles: list[JointProtocolCycle] = []
    joint_master_measurements: list[SharedTimestampMasterMeasurement] = []
    joint_measurement_id = 0
    joint_master_measurement_id = 0
    for raw_cycle in raw_joint:
        cycle_waves = tuple(waves[wave_id] for wave_id in raw_cycle.wave_ids)
        measurements: list[SharedJointMeasurement] = []
        relay_master_measurement: SharedTimestampMasterMeasurement | None = None
        for listener_id in config.follower_ids:
            if spec.joint_protocol == PROTOCOL_JOINT_V2:
                master_listener_events = tuple(
                    wave.receive_event(listener_id) for wave in cycle_waves[:2]
                )
                master_true_arrivals = np.asarray(
                    [event.timestamp for event in master_listener_events], dtype=float
                )
                master_noise = np.asarray(
                    [
                        timestamp_errors[(listener_id, wave.wave_id)]
                        for wave in cycle_waves[:2]
                    ],
                    dtype=float,
                )
                master_measurement = SharedTimestampMasterMeasurement(
                    measurement_id=joint_master_measurement_id,
                    protocol_cycle_id=raw_cycle.cycle_id,
                    block_id=raw_cycle.block_id,
                    anchor_master_id=raw_cycle.anchor_master_id,
                    second_master_id=raw_cycle.second_master_id,
                    listener_id=listener_id,
                    is_relay_row=(listener_id == raw_cycle.relay_id),
                    waves=(cycle_waves[0], cycle_waves[1]),
                    listener_events=master_listener_events,
                    noisy_arrival_times=master_true_arrivals + master_noise,
                    known_turnaround_delay_s=0.0,
                    timestamp_error_keys=tuple(
                        timestamp_keys[(listener_id, wave.wave_id)]
                        for wave in cycle_waves[:2]
                    ),
                    sound_speed_mps=config.sound_speed_mps,
                )
                joint_master_measurements.append(master_measurement)
                joint_master_measurement_id += 1
                if listener_id == raw_cycle.relay_id:
                    relay_master_measurement = master_measurement
            if listener_id == raw_cycle.relay_id:
                continue
            listener_events = tuple(
                wave.receive_event(listener_id) for wave in cycle_waves
            )
            true_arrivals = np.asarray(
                [event.timestamp for event in listener_events], dtype=float
            )
            noise = np.asarray(
                [timestamp_errors[(listener_id, wave.wave_id)] for wave in cycle_waves],
                dtype=float,
            )
            measurements.append(
                SharedJointMeasurement(
                    cycle_id=joint_measurement_id,
                    protocol_cycle_id=raw_cycle.cycle_id,
                    initiator_id=raw_cycle.anchor_master_id,
                    second_id=raw_cycle.second_master_id,
                    relay_id=raw_cycle.relay_id,
                    listener_id=listener_id,
                    waves=cycle_waves,
                    listener_events=listener_events,
                    noisy_arrival_times=true_arrivals + noise,
                    known_turnaround_corrections=np.asarray(
                        [
                            0.0,
                            raw_cycle.relay_processing_delay_s,
                            raw_cycle.anchor_wait_s,
                        ],
                        dtype=float,
                    ),
                    timestamp_covariance=(
                        np.eye(4, dtype=float) * config.timestamp_sigma_s**2
                    ),
                    timestamp_error_keys=tuple(
                        timestamp_keys[(listener_id, wave.wave_id)]
                        for wave in cycle_waves
                    ),
                    sound_speed_mps=config.sound_speed_mps,
                )
            )
            joint_measurement_id += 1
        joint_cycles.append(
            JointProtocolCycle(
                protocol_cycle_id=raw_cycle.cycle_id,
                relay_id=raw_cycle.relay_id,
                waves=cycle_waves,
                listener_measurements=tuple(measurements),
                anchor_wait_s=raw_cycle.anchor_wait_s,
                block_id=raw_cycle.block_id,
                anchor_master_id=raw_cycle.anchor_master_id,
                second_master_id=raw_cycle.second_master_id,
                relay_master_measurement=relay_master_measurement,
                predicted_cross_gain_by_listener=(
                    raw_cycle.predicted_cross_gain_by_listener
                ),
                relay_processing_delay_s=raw_cycle.relay_processing_delay_s,
            )
        )

    truth_grid = {
        agent_id: np.vstack(
            [
                np.asarray(
                    [pose.x(), pose.y(), pose.theta()], dtype=float
                )
                for pose in (
                    pose_at(agent_id, index * config.sensor_dt_s)
                    for index in range(config.step_count + 1)
                )
            ]
        )
        for agent_id in config.agent_ids
    }
    dataset = FormalWorldDataset(
        seed=seed,
        realization_seed=random_seed,
        spec=spec,
        config=config,
        motion_profiles=active_profiles,
        truth_grid=truth_grid,
        follower_timelines=timelines,
        grid_event_refs=grid_refs,
        odom_segments=odom_segments,
        initial_state_messages=initial_messages,
        waves=waves,
        baseline_measurements=tuple(baseline_measurements),
        joint_cycles=tuple(joint_cycles),
        joint_master_measurements=tuple(joint_master_measurements),
        master_messages=master_messages,
        timestamp_error_keys=timestamp_keys,
        timestamp_errors=timestamp_errors,
    )
    _validate_dataset(dataset)
    return dataset


def _validate_dataset(dataset: FormalWorldDataset) -> None:
    spec = dataset.spec
    baseline_ids = [wave.wave_id for wave in dataset.baseline_waves]
    joint_ids = [wave.wave_id for wave in dataset.joint_waves]
    if len(baseline_ids) != len(set(baseline_ids)):
        raise AssertionError("baseline physical wave IDs are not unique")
    if len(joint_ids) != len(set(joint_ids)):
        raise AssertionError("joint physical wave IDs are not unique")
    if len(baseline_ids) != spec.expected_baseline_wave_count:
        raise AssertionError("baseline wave ledger count is incorrect")
    if len(joint_ids) != spec.expected_joint_wave_count:
        raise AssertionError("joint wave ledger count is incorrect")
    for first, second in zip(dataset.joint_cycles, dataset.joint_cycles[1:]):
        if first.block_id != second.block_id:
            if first.waves[3].wave_id == second.waves[0].wave_id:
                raise AssertionError("timestamp blocks must not share a boundary wave")
            continue
        if first.waves[3].wave_id != second.waves[0].wave_id:
            raise AssertionError("adjacent cycles do not share the anchor boundary wave")
        for listener_id in dataset.config.follower_ids:
            if listener_id in {first.relay_id, second.relay_id}:
                continue
            left = next(
                item for item in first.listener_measurements
                if item.listener_id == listener_id
            )
            right = next(
                item for item in second.listener_measurements
                if item.listener_id == listener_id
            )
            if left.listener_events[3].key != right.listener_events[0].key:
                raise AssertionError("listener boundary event was not shared")
            if left.timestamp_error_keys[3] != right.timestamp_error_keys[0]:
                raise AssertionError("listener boundary timestamp was not shared")
    for cycle in dataset.joint_cycles:
        if cycle.anchor_wait_s < 0.0:
            raise AssertionError("anchor wait must be non-negative")
        if cycle.waves[3].send_time - cycle.waves[0].send_time > (
            dataset.spec.joint_period_s + 1e-10
        ):
            raise AssertionError("joint cycle exceeded its 3 s anchor interval")
        for measurement in cycle.listener_measurements:
            pose_keys = [ref.key for ref in measurement.logical_event_refs]
            if len(pose_keys) != 8:
                raise AssertionError(
                    "each joint cycle needs eight logical Pose2 slots: "
                    f"cycle={cycle.protocol_cycle_id}, "
                    f"listener={measurement.listener_id}, keys={pose_keys}"
                )
        if spec.joint_protocol == PROTOCOL_JOINT_V2:
            if cycle.relay_master_measurement is None:
                raise AssertionError("v2 relay is missing its master row")
            relay_keys = [
                ref.key for ref in cycle.relay_master_measurement.logical_event_refs
            ]
            if len(relay_keys) != 4:
                raise AssertionError("relay master row needs four logical Pose2 slots")
    if spec.joint_protocol == PROTOCOL_JOINT_V2:
        relay_counts = {
            follower_id: sum(
                cycle.relay_id == follower_id for cycle in dataset.joint_cycles
            )
            for follower_id in dataset.config.follower_ids
        }
        if spec.joint_cycle_count % len(relay_counts) == 0:
            expected_relay_count = spec.joint_cycle_count // len(relay_counts)
            if set(relay_counts.values()) != {expected_relay_count}:
                raise AssertionError("v2 relay quotas are not equal")
        elif max(relay_counts.values()) - min(relay_counts.values()) > 1:
            raise AssertionError("short v2 relay schedule is not balanced")
        if len(dataset.joint_master_measurements) != (
            spec.joint_cycle_count * len(dataset.config.follower_ids)
        ):
            raise AssertionError("v2 master-row count is incorrect")


def aggregate_odom_segments_for_protocol(
    dataset: FormalWorldDataset,
    protocol: str | None,
) -> dict[str, tuple[OdomSegment, ...]]:
    """Collapse the common atomic noise chain to states used by one method.

    All methods are still derived from the same sampled atomic increments.
    Composition therefore preserves the realized odometry and its right-local
    covariance, while unused 10 Hz and other-protocol event variables never
    enter that method's factor graph.
    """
    if protocol not in {
        None,
        PROTOCOL_BASELINE,
        PROTOCOL_JOINT,
        PROTOCOL_JOINT_V2,
    }:
        raise ValueError(f"unknown protocol for odometry aggregation: {protocol}")
    required: dict[str, set[int]] = {
        follower_id: {0} for follower_id in dataset.config.follower_ids
    }
    evaluation_stride = int(
        round(
            dataset.config.evaluation_period_s
            / dataset.config.sensor_dt_s
        )
    )
    if evaluation_stride <= 0:
        raise ValueError("evaluation period must not be shorter than sensor dt")
    for follower_id in dataset.config.follower_ids:
        for grid_index, ref in enumerate(
            dataset.grid_event_refs[follower_id]
        ):
            if grid_index % evaluation_stride == 0:
                required[follower_id].add(int(ref.event_id))
    if protocol == PROTOCOL_BASELINE:
        for measurement in dataset.baseline_measurements:
            required[measurement.listener_id].update(
                int(ref.event_id) for ref in measurement.listener_events
            )
    elif protocol in {PROTOCOL_JOINT, PROTOCOL_JOINT_V2}:
        for cycle in dataset.joint_cycles:
            relay_wave = cycle.waves[2]
            required[cycle.relay_id].add(
                int(relay_wave.sender_event.event_id)
            )
            for measurement in cycle.listener_measurements:
                required[measurement.listener_id].update(
                    int(ref.event_id) for ref in measurement.listener_events
                )
            if cycle.relay_master_measurement is not None:
                required[cycle.relay_id].update(
                    int(ref.event_id)
                    for ref in cycle.relay_master_measurement.listener_events
                )

    result: dict[str, tuple[OdomSegment, ...]] = {}
    for follower_id in dataset.config.follower_ids:
        atomic = dataset.odom_segments[follower_id]
        by_start = {int(segment.start_event_id): segment for segment in atomic}
        selected = sorted(required[follower_id])
        if selected[-1] != dataset.grid_event_refs[follower_id][-1].event_id:
            raise AssertionError("final evaluation state is missing")
        aggregates: list[OdomSegment] = []
        for start_event, end_event in zip(selected, selected[1:]):
            cursor = int(start_event)
            delta = gtsam.Pose2()
            covariance = np.zeros((3, 3), dtype=float)
            raw_ids: list[str] = []
            start_time = float(by_start[cursor].start_time)
            end_time = start_time
            while cursor < end_event:
                segment = by_start.get(cursor)
                if segment is None or int(segment.end_event_id) > end_event:
                    raise AssertionError(
                        f"cannot aggregate {follower_id} {start_event}->{end_event}"
                    )
                covariance = compose_pose_covariance(
                    covariance,
                    segment.delta_pose,
                    segment.covariance,
                )
                delta = delta.compose(segment.delta_pose)
                raw_ids.append(segment.segment_id)
                cursor = int(segment.end_event_id)
                end_time = float(segment.end_time)
            if cursor != end_event:
                raise AssertionError("aggregated odometry did not close")
            aggregates.append(
                OdomSegment(
                    segment_id=(
                        f"{follower_id}:aggregate:{start_event}:{end_event}:"
                        f"raw={len(raw_ids)}"
                    ),
                    agent_id=follower_id,
                    start_event_id=start_event,
                    end_event_id=end_event,
                    start_time=start_time,
                    end_time=end_time,
                    delta_pose=delta,
                    covariance=covariance,
                )
            )
        result[follower_id] = tuple(aggregates)
    return result
