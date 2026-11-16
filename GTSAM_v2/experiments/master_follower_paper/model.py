"""Unified event-time data model for the formal master/follower experiment."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import gtsam  # type: ignore
import numpy as np

from estimator.communication import (
    FollowerInitialStateMessage,
    MasterPoseMessage,
)
from estimator.distributed_graphs import pose_key
from estimator.odometry_preintegration import (
    ContinuousNoiseDensity,
    exact_unicycle_increment,
    preintegrate_continuous_segment,
)


MASTER_IDS = ("A", "B")
FOLLOWER_IDS = ("C", "D", "E")
AGENT_IDS = MASTER_IDS + FOLLOWER_IDS


@dataclass(frozen=True)
class EventPoseRef:
    agent_id: str
    event_id: int
    timestamp: float
    key: int
    event_type: str


@dataclass(frozen=True)
class OdomSegment:
    segment_id: str
    agent_id: str
    start_event_id: int
    end_event_id: int
    start_time: float
    end_time: float
    delta_pose: gtsam.Pose2
    covariance: np.ndarray


@dataclass(frozen=True)
class AUVSpec:
    agent_id: str
    role: str
    initial_x: float
    initial_y: float
    initial_heading_rad: float
    velocity: float
    angular_rate: float

    @property
    def initial_pose(self) -> gtsam.Pose2:
        return gtsam.Pose2(
            self.initial_x,
            self.initial_y,
            self.initial_heading_rad,
        )


@dataclass(frozen=True)
class FormalExperimentConfig:
    duration_s: float
    sensor_dt_s: float
    cycle_period_s: float
    turnaround_delay_s: float
    sound_speed_mps: float
    dvl_sigma_mps: float
    gyro_sigma_radps: float
    lateral_sigma_mps: float
    timestamp_sigma_s: float
    follower_initial_xy_sigma_m: float
    follower_initial_heading_sigma_rad: float
    master_xy_sigma_m: float
    master_heading_sigma_rad: float
    evaluation_period_s: float
    minimum_initial_spacing_m: float
    agents: tuple[AUVSpec, ...]
    seeds: tuple[int, ...]

    @property
    def step_count(self) -> int:
        return int(round(self.duration_s / self.sensor_dt_s))

    @property
    def master_ids(self) -> tuple[str, ...]:
        return tuple(
            spec.agent_id for spec in self.agents if spec.role == "master"
        )

    @property
    def follower_ids(self) -> tuple[str, ...]:
        return tuple(
            spec.agent_id for spec in self.agents if spec.role == "follower"
        )

    @property
    def agent_ids(self) -> tuple[str, ...]:
        return tuple(spec.agent_id for spec in self.agents)

    @property
    def continuous_noise(self) -> ContinuousNoiseDensity:
        return ContinuousNoiseDensity.from_sample_sigmas(
            self.dvl_sigma_mps,
            self.gyro_sigma_radps,
            self.lateral_sigma_mps,
            self.sensor_dt_s,
        )

    @property
    def master_covariance(self) -> np.ndarray:
        return np.diag(
            [
                self.master_xy_sigma_m**2,
                self.master_xy_sigma_m**2,
                self.master_heading_sigma_rad**2,
            ]
        )

    @property
    def follower_initial_covariance(self) -> np.ndarray:
        return np.diag(
            [
                self.follower_initial_xy_sigma_m**2,
                self.follower_initial_xy_sigma_m**2,
                self.follower_initial_heading_sigma_rad**2,
            ]
        )

    def validate(self) -> None:
        if self.duration_s <= 0.0 or self.sensor_dt_s <= 0.0:
            raise ValueError("duration and sensor period must be positive")
        if self.cycle_period_s <= 0.0:
            raise ValueError("cycle period must be positive")
        if self.turnaround_delay_s < 0.0:
            raise ValueError("turnaround delay must be non-negative")
        if self.sound_speed_mps <= 0.0:
            raise ValueError("sound speed must be positive")
        ids = self.agent_ids
        if len(ids) != len(set(ids)):
            raise ValueError("agent IDs must be unique")
        if any(len(agent_id) != 1 for agent_id in ids):
            raise ValueError("GTSAM symbol agent IDs must be one character")
        if self.master_ids != MASTER_IDS:
            raise ValueError("formal experiment requires masters A and B")
        if not self.follower_ids:
            raise ValueError("formal experiment requires at least one follower")
        if any(spec.role not in {"master", "follower"} for spec in self.agents):
            raise ValueError("agent role must be master or follower")
        for first_index, first in enumerate(self.agents):
            for second in self.agents[first_index + 1 :]:
                distance = math.hypot(
                    first.initial_x - second.initial_x,
                    first.initial_y - second.initial_y,
                )
                if distance < self.minimum_initial_spacing_m:
                    raise ValueError(
                        f"initial spacing {first.agent_id}-{second.agent_id} "
                        f"is only {distance:.3f} m"
                    )
        self.continuous_noise.matrix()


def build_default_config() -> FormalExperimentConfig:
    """300 s event-time validation with twice the ten-seed pretest count."""
    config = FormalExperimentConfig(
        duration_s=300.0,
        sensor_dt_s=0.1,
        cycle_period_s=2.0,
        turnaround_delay_s=0.00,
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
        minimum_initial_spacing_m=150.0,
        agents=(
            AUVSpec("A", "master", -150.0, 0.0, 0.0, 0.0, 0.0),
            AUVSpec("B", "master", +150.0, 0.0, math.pi, 0.0, 0.0),
            AUVSpec(
                "C", "follower", -220.0, +220.0, 3.0 * math.pi / 4.0,
                0.55, +0.001,
            ),
            AUVSpec(
                "D", "follower", 0.0, -240.0, -math.pi / 2.0,
                0.50, 0.0,
            ),
            AUVSpec(
                "E", "follower", +220.0, +220.0, math.pi / 4.0,
                0.52, -0.001,
            ),
        ),
        seeds=tuple(range(42, 62)),
    )
    config.validate()
    return config


@dataclass(frozen=True)
class WaveRecord:
    wave_id: int
    protocol_cycle_id: int
    wave_index: int
    sender_id: str
    relay_id: str
    send_time: float
    sender_event: EventPoseRef
    receive_times: tuple[tuple[str, float], ...]
    follower_receive_events: tuple[tuple[str, EventPoseRef], ...]

    def receive_time(self, receiver_id: str) -> float:
        return dict(self.receive_times)[receiver_id]

    def receive_event(self, follower_id: str) -> EventPoseRef:
        return dict(self.follower_receive_events)[follower_id]


@dataclass(frozen=True)
class ListenerCycleMeasurement:
    cycle_id: int
    protocol_cycle_id: int
    initiator_id: str
    second_id: str
    relay_id: str
    listener_id: str
    waves: tuple[WaveRecord, WaveRecord, WaveRecord, WaveRecord]
    listener_events: tuple[
        EventPoseRef,
        EventPoseRef,
        EventPoseRef,
        EventPoseRef,
    ]
    noisy_arrival_times: np.ndarray
    known_turnaround_corrections: np.ndarray
    timestamp_covariance: np.ndarray
    sound_speed_mps: float

    @property
    def measured_delta_t(self) -> np.ndarray:
        # Known reception-to-transmission delays are removed before applying
        # the unchanged three-row geometric measurement equation.
        return (
            np.diff(self.noisy_arrival_times)
            - self.known_turnaround_corrections
        )

    @property
    def completion_time(self) -> float:
        return self.listener_events[-1].timestamp

    @property
    def logical_event_refs(self) -> tuple[EventPoseRef, ...]:
        return tuple(wave.sender_event for wave in self.waves) + tuple(
            self.listener_events
        )


@dataclass(frozen=True)
class ProtocolCycle:
    protocol_cycle_id: int
    relay_id: str
    waves: tuple[WaveRecord, WaveRecord, WaveRecord, WaveRecord]
    listener_measurements: tuple[ListenerCycleMeasurement, ...]


@dataclass(frozen=True)
class FormalExperimentDataset:
    seed: int
    config: FormalExperimentConfig
    truth_grid: Mapping[str, np.ndarray]
    follower_timelines: Mapping[str, tuple[EventPoseRef, ...]]
    grid_event_refs: Mapping[str, tuple[EventPoseRef, ...]]
    odom_segments: Mapping[str, tuple[OdomSegment, ...]]
    initial_state_messages: Mapping[str, FollowerInitialStateMessage]
    protocol_cycles: tuple[ProtocolCycle, ...]
    master_messages: Mapping[int, MasterPoseMessage]

    def truth_pose(self, agent_id: str, time_s: float) -> gtsam.Pose2:
        spec = next(a for a in self.config.agents if a.agent_id == agent_id)
        return spec.initial_pose.compose(
            exact_unicycle_increment(
                spec.velocity,
                spec.angular_rate,
                float(time_s),
            )
        )

    @property
    def listener_measurements(self) -> tuple[ListenerCycleMeasurement, ...]:
        return tuple(
            measurement
            for cycle in self.protocol_cycles
            for measurement in cycle.listener_measurements
        )


@dataclass(frozen=True)
class _PhysicalWave:
    wave_id: int
    protocol_cycle_id: int
    wave_index: int
    sender_id: str
    relay_id: str
    send_time: float
    receive_times: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class _PhysicalCycle:
    protocol_cycle_id: int
    relay_id: str
    waves: tuple[_PhysicalWave, _PhysicalWave, _PhysicalWave, _PhysicalWave]


def _pose_array(pose: gtsam.Pose2) -> np.ndarray:
    return np.asarray([pose.x(), pose.y(), pose.theta()], dtype=float)


def _sample_tangent_gaussian(
    covariance: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    lower = np.zeros((3, 3), dtype=float)
    for row in range(3):
        for column in range(row + 1):
            subtotal = sum(
                float(lower[row, inner]) * float(lower[column, inner])
                for inner in range(column)
            )
            if row == column:
                pivot = float(covariance[row, row]) - subtotal
                if pivot <= 0.0:
                    raise ValueError("segment covariance must be positive definite")
                lower[row, column] = math.sqrt(pivot)
            else:
                lower[row, column] = (
                    float(covariance[row, column]) - subtotal
                ) / lower[column, column]
    standard = rng.normal(0.0, 1.0, size=3)
    sample = np.zeros(3, dtype=float)
    for row in range(3):
        sample[row] = sum(
            float(lower[row, column]) * float(standard[column])
            for column in range(row + 1)
        )
    return sample


def _solve_arrival_time(
    pose_at,
    sender_id: str,
    receiver_id: str,
    send_time: float,
    sound_speed: float,
) -> float:
    source = pose_at(sender_id, send_time).translation()

    def gap(candidate_time: float) -> float:
        target = pose_at(receiver_id, candidate_time).translation()
        dx = float(target[0]) - float(source[0])
        dy = float(target[1]) - float(source[1])
        return sound_speed * (candidate_time - send_time) - math.hypot(dx, dy)

    low = float(send_time)
    high = low + 1.0
    while gap(high) < 0.0:
        high += 1.0
    for _ in range(60):
        middle = 0.5 * (low + high)
        if gap(middle) >= 0.0:
            high = middle
        else:
            low = middle
    return 0.5 * (low + high)


def _build_physical_cycles(
    config: FormalExperimentConfig,
    pose_at,
) -> tuple[_PhysicalCycle, ...]:
    cycles: list[_PhysicalCycle] = []
    wave_id = 0
    cycle_id = 0
    scheduled_time = config.cycle_period_s
    while scheduled_time < config.duration_s - 1e-9:
        relay_id = config.follower_ids[cycle_id % len(config.follower_ids)]
        senders = ("A", "B", relay_id, "A")
        send_times = [float(scheduled_time)]
        for next_sender in senders[1:]:
            previous_sender = senders[len(send_times) - 1]
            arrival = _solve_arrival_time(
                pose_at,
                previous_sender,
                next_sender,
                send_times[-1],
                config.sound_speed_mps,
            )
            send_times.append(arrival + config.turnaround_delay_s)
        waves: list[_PhysicalWave] = []
        for wave_index, (sender_id, send_time) in enumerate(
            zip(senders, send_times)
        ):
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
            waves.append(
                _PhysicalWave(
                    wave_id=wave_id,
                    protocol_cycle_id=cycle_id,
                    wave_index=wave_index,
                    sender_id=sender_id,
                    relay_id=relay_id,
                    send_time=send_time,
                    receive_times=receive_times,
                )
            )
            wave_id += 1
        last_follower_arrival = max(
            receive_time
            for receiver_id, receive_time in waves[-1].receive_times
            if receiver_id in config.follower_ids
        )
        if last_follower_arrival >= config.duration_s:
            break
        cycles.append(
            _PhysicalCycle(
                protocol_cycle_id=cycle_id,
                relay_id=relay_id,
                waves=tuple(waves),
            )
        )
        cycle_id += 1
        scheduled_time += config.cycle_period_s
    return tuple(cycles)


def _build_follower_timelines(
    config: FormalExperimentConfig,
    physical_cycles: tuple[_PhysicalCycle, ...],
) -> tuple[
    dict[str, tuple[EventPoseRef, ...]],
    dict[str, tuple[EventPoseRef, ...]],
    dict[tuple[str, str, int], EventPoseRef],
]:
    raw_events: dict[str, list[tuple[float, str, int]]] = {
        follower_id: [] for follower_id in config.follower_ids
    }
    for follower_id in config.follower_ids:
        for grid_index in range(config.step_count + 1):
            raw_events[follower_id].append(
                (grid_index * config.sensor_dt_s, "GRID", grid_index)
            )
    for cycle in physical_cycles:
        for wave in cycle.waves:
            if wave.sender_id in config.follower_ids:
                raw_events[wave.sender_id].append(
                    (wave.send_time, "SEND", wave.wave_id)
                )
            for receiver_id, receive_time in wave.receive_times:
                if receiver_id in config.follower_ids:
                    raw_events[receiver_id].append(
                        (receive_time, "RECEIVE", wave.wave_id)
                    )

    timelines: dict[str, tuple[EventPoseRef, ...]] = {}
    grid_refs: dict[str, tuple[EventPoseRef, ...]] = {}
    token_refs: dict[tuple[str, str, int], EventPoseRef] = {}
    event_priority = {"GRID": 0, "RECEIVE": 1, "SEND": 2}
    for follower_id in config.follower_ids:
        ordered = sorted(
            raw_events[follower_id],
            key=lambda value: (value[0], event_priority[value[1]], value[2]),
        )
        groups: list[list[tuple[float, str, int]]] = []
        for item in ordered:
            if groups and abs(item[0] - groups[-1][0][0]) <= 1e-12:
                groups[-1].append(item)
            else:
                groups.append([item])
        refs: list[EventPoseRef] = []
        for event_id, group in enumerate(groups):
            timestamp = group[0][0]
            event_type = max(
                (item[1] for item in group), key=event_priority.__getitem__
            )
            ref = EventPoseRef(
                agent_id=follower_id,
                event_id=event_id,
                timestamp=float(timestamp),
                key=pose_key(follower_id, event_id),
                event_type=event_type,
            )
            refs.append(ref)
            for _, token_event_type, token_id in group:
                token_refs[(follower_id, token_event_type, token_id)] = ref
        timelines[follower_id] = tuple(refs)
        grid_refs[follower_id] = tuple(
            token_refs[(follower_id, "GRID", grid_index)]
            for grid_index in range(config.step_count + 1)
        )
    return timelines, grid_refs, token_refs


def generate_dataset(
    seed: int,
    config: FormalExperimentConfig | None = None,
) -> FormalExperimentDataset:
    config = config or build_default_config()
    config.validate()
    specs = {spec.agent_id: spec for spec in config.agents}

    def pose_at(agent_id: str, time_s: float) -> gtsam.Pose2:
        spec = specs[agent_id]
        return spec.initial_pose.compose(
            exact_unicycle_increment(spec.velocity, spec.angular_rate, time_s)
        )

    physical_cycles = _build_physical_cycles(config, pose_at)
    timelines, grid_refs, token_refs = _build_follower_timelines(
        config, physical_cycles
    )

    prior_rng = np.random.RandomState(seed * 1009 + 11)
    acoustic_rng = np.random.RandomState(seed * 1009 + 29)
    master_rngs = {
        master_id: np.random.RandomState(seed * 1009 + 53 + 17 * offset)
        for offset, master_id in enumerate(config.master_ids)
    }
    initial_messages: dict[str, FollowerInitialStateMessage] = {}
    for follower_id in config.follower_ids:
        initial_error = np.asarray(
            [
                prior_rng.normal(0.0, config.follower_initial_xy_sigma_m),
                prior_rng.normal(0.0, config.follower_initial_xy_sigma_m),
                prior_rng.normal(0.0, config.follower_initial_heading_sigma_rad),
            ],
            dtype=float,
        )
        initial_messages[follower_id] = FollowerInitialStateMessage(
            follower_id=follower_id,
            pose=specs[follower_id].initial_pose.retract(initial_error),
            covariance=config.follower_initial_covariance,
            timestamp=0.0,
            source_epoch=int(seed),
        )

    odom_segments: dict[str, tuple[OdomSegment, ...]] = {}
    for follower_offset, follower_id in enumerate(config.follower_ids):
        spec = specs[follower_id]
        rng = np.random.RandomState(seed * 1009 + 101 + 37 * follower_offset)
        segments: list[OdomSegment] = []
        for first, second in zip(
            timelines[follower_id], timelines[follower_id][1:]
        ):
            nominal, covariance = preintegrate_continuous_segment(
                first.timestamp,
                second.timestamp,
                config.continuous_noise,
                velocity=spec.velocity,
                angular_rate=spec.angular_rate,
            )
            error = _sample_tangent_gaussian(covariance, rng)
            segments.append(
                OdomSegment(
                    segment_id=(
                        f"{follower_id}:seg:{first.event_id}:{second.event_id}"
                    ),
                    agent_id=follower_id,
                    start_event_id=first.event_id,
                    end_event_id=second.event_id,
                    start_time=first.timestamp,
                    end_time=second.timestamp,
                    delta_pose=nominal.retract(error),
                    covariance=covariance,
                )
            )
        odom_segments[follower_id] = tuple(segments)

    master_event_counters = {master_id: 0 for master_id in config.master_ids}
    master_refs: dict[int, EventPoseRef] = {}
    master_messages: dict[int, MasterPoseMessage] = {}
    for cycle in physical_cycles:
        for wave in cycle.waves:
            if wave.sender_id not in config.master_ids:
                continue
            event_id = master_event_counters[wave.sender_id]
            master_event_counters[wave.sender_id] += 1
            ref = EventPoseRef(
                agent_id=wave.sender_id,
                event_id=event_id,
                timestamp=wave.send_time,
                key=pose_key(wave.sender_id, event_id),
                event_type="MASTER_SEND",
            )
            master_refs[wave.wave_id] = ref
            nav_error = np.asarray(
                [
                    master_rngs[wave.sender_id].normal(
                        0.0, config.master_xy_sigma_m
                    ),
                    master_rngs[wave.sender_id].normal(
                        0.0, config.master_xy_sigma_m
                    ),
                    master_rngs[wave.sender_id].normal(
                        0.0, config.master_heading_sigma_rad
                    ),
                ],
                dtype=float,
            )
            master_messages[wave.wave_id] = MasterPoseMessage(
                master_id=wave.sender_id,
                source_epoch=int(seed),
                sequence=wave.wave_id,
                timestamp=wave.send_time,
                state_index=event_id,
                pose=pose_at(wave.sender_id, wave.send_time).retract(nav_error),
                covariance=config.master_covariance,
            )

    protocol_cycles: list[ProtocolCycle] = []
    measurement_id = 0
    for physical_cycle in physical_cycles:
        waves: list[WaveRecord] = []
        for physical_wave in physical_cycle.waves:
            sender_event = (
                master_refs[physical_wave.wave_id]
                if physical_wave.sender_id in config.master_ids
                else token_refs[
                    (physical_wave.sender_id, "SEND", physical_wave.wave_id)
                ]
            )
            follower_receive_events = tuple(
                (
                    receiver_id,
                    token_refs[(receiver_id, "RECEIVE", physical_wave.wave_id)],
                )
                for receiver_id, _ in physical_wave.receive_times
                if receiver_id in config.follower_ids
            )
            waves.append(
                WaveRecord(
                    wave_id=physical_wave.wave_id,
                    protocol_cycle_id=physical_wave.protocol_cycle_id,
                    wave_index=physical_wave.wave_index,
                    sender_id=physical_wave.sender_id,
                    relay_id=physical_wave.relay_id,
                    send_time=physical_wave.send_time,
                    sender_event=sender_event,
                    receive_times=physical_wave.receive_times,
                    follower_receive_events=follower_receive_events,
                )
            )
        listener_measurements: list[ListenerCycleMeasurement] = []
        known_corrections: list[float] = []
        for first_wave, second_wave in zip(waves, waves[1:]):
            next_sender = second_wave.sender_id
            receive_time = first_wave.receive_time(next_sender)
            first_position = pose_at(
                first_wave.sender_id, first_wave.send_time
            ).translation()
            receive_position = pose_at(
                next_sender, receive_time
            ).translation()
            send_position = pose_at(
                next_sender, second_wave.send_time
            ).translation()
            receive_range = math.hypot(
                float(first_position[0] - receive_position[0]),
                float(first_position[1] - receive_position[1]),
            )
            send_range = math.hypot(
                float(first_position[0] - send_position[0]),
                float(first_position[1] - send_position[1]),
            )
            known_corrections.append(
                second_wave.send_time
                - receive_time
                + (receive_range - send_range) / config.sound_speed_mps
            )
        for listener_id in config.follower_ids:
            if listener_id == physical_cycle.relay_id:
                continue
            listener_events = tuple(
                wave.receive_event(listener_id) for wave in waves
            )
            true_arrivals = np.asarray(
                [ref.timestamp for ref in listener_events], dtype=float
            )
            listener_measurements.append(
                ListenerCycleMeasurement(
                    cycle_id=measurement_id,
                    protocol_cycle_id=physical_cycle.protocol_cycle_id,
                    initiator_id="A",
                    second_id="B",
                    relay_id=physical_cycle.relay_id,
                    listener_id=listener_id,
                    waves=tuple(waves),
                    listener_events=listener_events,
                    noisy_arrival_times=(
                        true_arrivals
                        + acoustic_rng.normal(
                            0.0, config.timestamp_sigma_s, size=4
                        )
                    ),
                    known_turnaround_corrections=np.asarray(
                        known_corrections, dtype=float
                    ),
                    timestamp_covariance=(
                        np.eye(4, dtype=float) * config.timestamp_sigma_s**2
                    ),
                    sound_speed_mps=config.sound_speed_mps,
                )
            )
            measurement_id += 1
        protocol_cycles.append(
            ProtocolCycle(
                protocol_cycle_id=physical_cycle.protocol_cycle_id,
                relay_id=physical_cycle.relay_id,
                waves=tuple(waves),
                listener_measurements=tuple(listener_measurements),
            )
        )

    truth_grid = {
        agent_id: np.vstack(
            [
                _pose_array(pose_at(agent_id, index * config.sensor_dt_s))
                for index in range(config.step_count + 1)
            ]
        )
        for agent_id in config.agent_ids
    }
    return FormalExperimentDataset(
        seed=int(seed),
        config=config,
        truth_grid=truth_grid,
        follower_timelines=timelines,
        grid_event_refs=grid_refs,
        odom_segments=odom_segments,
        initial_state_messages=initial_messages,
        protocol_cycles=tuple(protocol_cycles),
        master_messages=master_messages,
    )
