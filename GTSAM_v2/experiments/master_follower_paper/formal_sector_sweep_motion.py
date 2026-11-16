"""Structured constant-speed sector S-sweep trajectories.

The path is specified in polar position coordinates rather than by perturbing
vehicle heading.  Each follower starts and ends on its original radial ray,
visits one side of its survey sector and then the other, and travels exactly
``speed * duration`` metres.  The terminal radius is solved from that fixed
arc-length budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Mapping

import gtsam  # type: ignore
import numpy as np

from .formal_motion import ConstantUnicycleMotion
from .formal_protocol import (
    PROTOCOL_VARIANT_BIDIRECTIONAL,
    RELAY_SCHEDULE_ROUND_ROBIN,
    FormalScenarioSpec,
    build_radial_v2_spec,
)


SECTOR_SWEEP_ORIENTATIONS_DEG = (0.0, 15.0, 30.0)
SECTOR_SWEEP_FOLLOWER_IDS = ("C", "D", "E", "F", "G", "H")
SECTOR_SWEEP_DURATION_S = 200.0
SECTOR_SWEEP_SPEED_MPS = 1.5
SECTOR_SWEEP_HALF_ANGLE_DEG = 10.0
SECTOR_SWEEP_MIN_AMPLITUDE_FRACTION = 0.7
SECTOR_SWEEP_FIRST_WAYPOINT_RANGE = (0.25, 0.35)
SECTOR_SWEEP_SECOND_WAYPOINT_RANGE = (0.65, 0.75)
_ARC_GRID_SIZE = 2049
_ROOT_ITERATIONS = 56


def _smoothstep5(value: np.ndarray | float):
    return value**3 * (10.0 + value * (-15.0 + 6.0 * value))


def _smoothstep5_first(value: np.ndarray | float):
    return 30.0 * value**2 * (1.0 - value) ** 2


def _smoothstep5_second(value: np.ndarray | float):
    return 60.0 * value * (1.0 - value) * (1.0 - 2.0 * value)


@dataclass(frozen=True)
class SectorSweepParameter:
    agent_id: str
    order_sign: int
    first_amplitude_rad: float
    second_amplitude_rad: float
    first_radial_fraction: float
    second_radial_fraction: float

    def validate(self, half_angle_rad: float) -> None:
        if self.order_sign not in (-1, 1):
            raise ValueError("sector sweep order_sign must be -1 or +1")
        lower = SECTOR_SWEEP_MIN_AMPLITUDE_FRACTION * float(half_angle_rad)
        for value in (self.first_amplitude_rad, self.second_amplitude_rad):
            if not lower <= float(value) <= float(half_angle_rad):
                raise ValueError("sector waypoint amplitude is outside its bounds")
        if not (
            SECTOR_SWEEP_FIRST_WAYPOINT_RANGE[0]
            <= self.first_radial_fraction
            <= SECTOR_SWEEP_FIRST_WAYPOINT_RANGE[1]
            < SECTOR_SWEEP_SECOND_WAYPOINT_RANGE[0]
            <= self.second_radial_fraction
            <= SECTOR_SWEEP_SECOND_WAYPOINT_RANGE[1]
        ):
            raise ValueError("sector waypoint radial fractions are invalid")


@dataclass(frozen=True)
class StructuredSectorSweepMotion:
    initial_radius_m: float
    center_bearing_rad: float
    speed_mps: float
    duration_s: float
    half_angle_rad: float
    parameter: SectorSweepParameter
    endpoint_radius_m: float = field(init=False)
    _rho_grid: np.ndarray = field(init=False, repr=False, compare=False)
    _arc_grid_m: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.initial_radius_m)
            or self.initial_radius_m <= 0.0
            or not math.isfinite(self.speed_mps)
            or self.speed_mps <= 0.0
            or not math.isfinite(self.duration_s)
            or self.duration_s <= 0.0
            or not math.isfinite(self.half_angle_rad)
            or self.half_angle_rad <= 0.0
        ):
            raise ValueError("sector sweep geometry must be finite and positive")
        self.parameter.validate(self.half_angle_rad)
        rho = np.linspace(0.0, 1.0, _ARC_GRID_SIZE)
        _, q_first, _ = self._offset_derivatives(rho)
        path_length = self.speed_mps * self.duration_s

        def length_for(radial_gain: float) -> float:
            radius = self.initial_radius_m + radial_gain * rho
            density = np.sqrt(radial_gain**2 + (radius * q_first) ** 2)
            return float(
                np.sum(0.5 * (density[:-1] + density[1:]))
                / (_ARC_GRID_SIZE - 1)
            )

        low = 0.0
        high = path_length
        if length_for(low) >= path_length:
            raise ValueError(
                "sector sweep consumes its entire path budget before radial advance"
            )
        for _ in range(_ROOT_ITERATIONS):
            midpoint = 0.5 * (low + high)
            if length_for(midpoint) < path_length:
                low = midpoint
            else:
                high = midpoint
        radial_gain = 0.5 * (low + high)
        radius = self.initial_radius_m + radial_gain * rho
        density = np.sqrt(radial_gain**2 + (radius * q_first) ** 2)
        increments = (
            0.5 * (density[:-1] + density[1:]) / (_ARC_GRID_SIZE - 1)
        )
        cumulative = np.concatenate(([0.0], np.cumsum(increments)))
        # Eliminate sub-micrometre quadrature drift at the fixed endpoint.
        cumulative *= path_length / float(cumulative[-1])
        object.__setattr__(self, "endpoint_radius_m", self.initial_radius_m + radial_gain)
        object.__setattr__(self, "_rho_grid", rho)
        object.__setattr__(self, "_arc_grid_m", cumulative)

    @property
    def path_length_m(self) -> float:
        return self.speed_mps * self.duration_s

    @property
    def initial_x_m(self) -> float:
        return self.initial_radius_m * math.cos(self.center_bearing_rad)

    @property
    def initial_y_m(self) -> float:
        return self.initial_radius_m * math.sin(self.center_bearing_rad)

    @property
    def initial_heading_rad(self) -> float:
        return self.center_bearing_rad

    @property
    def terminal_heading_rad(self) -> float:
        return self.center_bearing_rad

    @property
    def first_waypoint_radius_m(self) -> float:
        return self.initial_radius_m + (
            self.endpoint_radius_m - self.initial_radius_m
        ) * self.parameter.first_radial_fraction

    @property
    def second_waypoint_radius_m(self) -> float:
        return self.initial_radius_m + (
            self.endpoint_radius_m - self.initial_radius_m
        ) * self.parameter.second_radial_fraction

    def _offset_derivatives(
        self, rho: np.ndarray | float
    ) -> tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float]:
        values = np.asarray(rho, dtype=float)
        scalar = values.ndim == 0
        values = np.atleast_1d(values)
        knots = (
            0.0,
            float(self.parameter.first_radial_fraction),
            float(self.parameter.second_radial_fraction),
            1.0,
        )
        signed_first = (
            self.parameter.order_sign * self.parameter.first_amplitude_rad
        )
        signed_second = (
            -self.parameter.order_sign * self.parameter.second_amplitude_rad
        )
        offsets = (0.0, signed_first, signed_second, 0.0)
        result = np.zeros_like(values)
        first = np.zeros_like(values)
        second = np.zeros_like(values)
        for segment in range(3):
            left, right = knots[segment], knots[segment + 1]
            mask = (
                (values >= left) & (values < right)
                if segment < 2
                else (values >= left) & (values <= right)
            )
            width = right - left
            local = (values[mask] - left) / width
            delta = offsets[segment + 1] - offsets[segment]
            result[mask] = offsets[segment] + delta * _smoothstep5(local)
            first[mask] = delta * _smoothstep5_first(local) / width
            second[mask] = delta * _smoothstep5_second(local) / (width * width)
        if scalar:
            return float(result[0]), float(first[0]), float(second[0])
        return result, first, second

    def _rho_at(self, time_s: float) -> float:
        time_s = float(time_s)
        tolerance = 1e-9
        if (
            not math.isfinite(time_s)
            or time_s < -tolerance
            or time_s > self.duration_s + tolerance
        ):
            raise ValueError("sector sweep time lies outside the mission")
        time_s = min(max(time_s, 0.0), self.duration_s)
        return float(
            np.interp(
                self.speed_mps * time_s,
                self._arc_grid_m,
                self._rho_grid,
            )
        )

    def bearing_offset_at(self, time_s: float) -> float:
        value, _, _ = self._offset_derivatives(self._rho_at(time_s))
        return float(value)

    def radial_distance_at(self, time_s: float) -> float:
        rho = self._rho_at(time_s)
        return self.initial_radius_m + (
            self.endpoint_radius_m - self.initial_radius_m
        ) * rho

    def time_at_radial_fraction(self, radial_fraction: float) -> float:
        radial_fraction = float(radial_fraction)
        if not 0.0 <= radial_fraction <= 1.0:
            raise ValueError("radial_fraction must lie in [0, 1]")
        arc_length = float(
            np.interp(radial_fraction, self._rho_grid, self._arc_grid_m)
        )
        return arc_length / self.speed_mps

    def _state_at(self, time_s: float) -> tuple[float, float, float, float]:
        rho = self._rho_at(time_s)
        offset, q_first, q_second = self._offset_derivatives(rho)
        radial_gain = self.endpoint_radius_m - self.initial_radius_m
        radius = self.initial_radius_m + radial_gain * rho
        bearing = self.center_bearing_rad + float(offset)
        tangential_component = radius * float(q_first)
        heading = bearing + math.atan2(tangential_component, radial_gain)
        first_norm_squared = radial_gain**2 + tangential_component**2
        cross_first_second = (
            radial_gain
            * (2.0 * radial_gain * float(q_first) + radius * float(q_second))
            + radius**2 * float(q_first) ** 3
        )
        curvature = cross_first_second / (first_norm_squared**1.5)
        return radius, bearing, heading, curvature

    def heading_at(self, time_s: float) -> float:
        return self._state_at(time_s)[2]

    def control_at(self, time_s: float) -> tuple[float, float]:
        curvature = self._state_at(time_s)[3]
        return self.speed_mps, self.speed_mps * curvature

    def pose_at(self, time_s: float) -> gtsam.Pose2:
        radius, bearing, heading, _ = self._state_at(time_s)
        return gtsam.Pose2(
            radius * math.cos(bearing),
            radius * math.sin(bearing),
            heading,
        )


def sample_sector_sweep_parameters(
    seed: int,
    follower_ids: tuple[str, ...] = SECTOR_SWEEP_FOLLOWER_IDS,
    *,
    half_angle_rad: float = math.radians(SECTOR_SWEEP_HALF_ANGLE_DEG),
) -> tuple[SectorSweepParameter, ...]:
    """Draw structured route parameters shared across orientations."""
    seed = int(seed)
    if seed <= 0:
        raise ValueError("seed must be positive")
    half_angle_rad = float(half_angle_rad)
    if not math.isfinite(half_angle_rad) or half_angle_rad <= 0.0:
        raise ValueError("half_angle_rad must be finite and positive")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x53535750]))
    lower = SECTOR_SWEEP_MIN_AMPLITUDE_FRACTION * half_angle_rad
    parameters = []
    for agent_id in follower_ids:
        parameter = SectorSweepParameter(
            agent_id=agent_id,
            order_sign=1 if int(rng.integers(0, 2)) else -1,
            first_amplitude_rad=float(rng.uniform(lower, half_angle_rad)),
            second_amplitude_rad=float(rng.uniform(lower, half_angle_rad)),
            first_radial_fraction=float(
                rng.uniform(*SECTOR_SWEEP_FIRST_WAYPOINT_RANGE)
            ),
            second_radial_fraction=float(
                rng.uniform(*SECTOR_SWEEP_SECOND_WAYPOINT_RANGE)
            ),
        )
        parameter.validate(half_angle_rad)
        parameters.append(parameter)
    return tuple(parameters)


def sector_sweep_realization_seed(seed: int, orientation_deg: float) -> int:
    seed = int(seed)
    if seed <= 0:
        raise ValueError("seed must be positive")
    matches = [
        index
        for index, value in enumerate(SECTOR_SWEEP_ORIENTATIONS_DEG)
        if abs(float(orientation_deg) - value) <= 1e-12
    ]
    if len(matches) != 1:
        raise ValueError("sector-sweep orientation must be 0, 15, or 30 degrees")
    value = 1_000_000 + (seed - 1) * len(SECTOR_SWEEP_ORIENTATIONS_DEG) + matches[0] + 1
    if value > 4_000_000:
        raise ValueError("seed is too large for the legacy RandomState streams")
    return value


def build_structured_sector_sweep_case(
    *,
    seed: int,
    orientation_deg: float,
    duration_s: float = SECTOR_SWEEP_DURATION_S,
    half_angle_deg: float = SECTOR_SWEEP_HALF_ANGLE_DEG,
    baseline_cycle_count: int = 100,
    block_cycle_count: int = 33,
    second_block_start_s: float = 100.0,
) -> tuple[
    FormalScenarioSpec,
    Mapping[str, ConstantUnicycleMotion | StructuredSectorSweepMotion],
    tuple[SectorSweepParameter, ...],
    int,
]:
    seed = int(seed)
    orientation_deg = float(orientation_deg)
    half_angle_rad = math.radians(float(half_angle_deg))
    spec = build_radial_v2_spec(
        rotation_deg=orientation_deg,
        duration_s=float(duration_s),
        seeds=(seed,),
        baseline_cycle_count=int(baseline_cycle_count),
        block_cycle_count=int(block_cycle_count),
        second_block_start_s=float(second_block_start_s),
        relay_schedule=RELAY_SCHEDULE_ROUND_ROBIN,
        protocol_variant=PROTOCOL_VARIANT_BIDIRECTIONAL,
    )
    parameters = sample_sector_sweep_parameters(
        seed, spec.config.follower_ids, half_angle_rad=half_angle_rad
    )
    parameter_by_agent = {item.agent_id: item for item in parameters}
    profiles: dict[str, ConstantUnicycleMotion | StructuredSectorSweepMotion] = {}
    agents = []
    for agent in spec.config.agents:
        if agent.role == "master":
            profile: ConstantUnicycleMotion | StructuredSectorSweepMotion = (
                ConstantUnicycleMotion(
                    agent.initial_x,
                    agent.initial_y,
                    agent.initial_heading_rad,
                    agent.velocity,
                    agent.angular_rate,
                )
            )
        else:
            center_bearing = math.atan2(agent.initial_y, agent.initial_x)
            profile = StructuredSectorSweepMotion(
                initial_radius_m=math.hypot(agent.initial_x, agent.initial_y),
                center_bearing_rad=center_bearing,
                speed_mps=SECTOR_SWEEP_SPEED_MPS,
                duration_s=float(duration_s),
                half_angle_rad=half_angle_rad,
                parameter=parameter_by_agent[agent.agent_id],
            )
            agent = replace(
                agent,
                initial_heading_rad=profile.initial_heading_rad,
                velocity=profile.speed_mps,
                angular_rate=profile.control_at(0.0)[1],
            )
        profiles[agent.agent_id] = profile
        agents.append(agent)
    orientation_label = f"{orientation_deg:g}".replace("-", "m").replace(".", "p")
    spec = replace(
        spec,
        name=f"six_follower_structured_sector_sweep_v2_{orientation_label}deg",
        config=replace(spec.config, agents=tuple(agents), seeds=(seed,)),
    )
    spec.validate()
    return (
        spec,
        profiles,
        parameters,
        sector_sweep_realization_seed(seed, orientation_deg),
    )


def sector_sweep_parameter_rows(dataset) -> list[dict[str, float | int | str]]:
    result: list[dict[str, float | int | str]] = []
    orientation_deg = math.degrees(float(dataset.spec.radial_rotation_rad))
    follower_truth = {
        agent_id: np.asarray(dataset.truth_grid[agent_id], dtype=float)
        for agent_id in dataset.config.follower_ids
    }
    minimum_separation = {agent_id: float("inf") for agent_id in dataset.config.follower_ids}
    for first_index, first_id in enumerate(dataset.config.follower_ids):
        for second_id in dataset.config.follower_ids[first_index + 1 :]:
            delta = follower_truth[first_id][:, :2] - follower_truth[second_id][:, :2]
            distance = np.sqrt(np.sum(delta * delta, axis=1))
            pair_minimum = float(np.min(distance))
            minimum_separation[first_id] = min(minimum_separation[first_id], pair_minimum)
            minimum_separation[second_id] = min(minimum_separation[second_id], pair_minimum)
    for agent_id in dataset.config.follower_ids:
        profile = dataset.motion_profiles[agent_id]
        if not isinstance(profile, StructuredSectorSweepMotion):
            continue
        truth = follower_truth[agent_id]
        initial_xy = truth[0, :2]
        displacement = truth[:, :2] - initial_xy
        radial = np.asarray(
            [math.cos(profile.center_bearing_rad), math.sin(profile.center_bearing_rad)],
            dtype=float,
        )
        tangential = np.asarray([-radial[1], radial[0]], dtype=float)
        offsets = np.asarray(
            [profile.bearing_offset_at(index * dataset.config.sensor_dt_s)
             for index in range(dataset.config.step_count + 1)],
            dtype=float,
        )
        angular_rates = np.asarray(
            [abs(profile.control_at(index * dataset.config.sensor_dt_s)[1])
             for index in range(dataset.config.step_count + 1)],
            dtype=float,
        )
        radii = np.sqrt(truth[:, 0] ** 2 + truth[:, 1] ** 2)
        parameter = profile.parameter
        result.append(
            {
                "scenario": dataset.spec.name,
                "seed": int(dataset.seed),
                "realization_seed": int(dataset.realization_seed),
                "orientation_deg": orientation_deg,
                "agent_id": agent_id,
                "motion_family": "structured_sector_s_sweep",
                "sector_half_angle_deg": math.degrees(profile.half_angle_rad),
                "order_sign": int(parameter.order_sign),
                "first_signed_offset_deg": math.degrees(
                    parameter.order_sign * parameter.first_amplitude_rad
                ),
                "second_signed_offset_deg": math.degrees(
                    -parameter.order_sign * parameter.second_amplitude_rad
                ),
                "first_radial_fraction": parameter.first_radial_fraction,
                "second_radial_fraction": parameter.second_radial_fraction,
                "first_waypoint_radius_m": profile.first_waypoint_radius_m,
                "second_waypoint_radius_m": profile.second_waypoint_radius_m,
                "first_waypoint_time_s": profile.time_at_radial_fraction(
                    parameter.first_radial_fraction
                ),
                "second_waypoint_time_s": profile.time_at_radial_fraction(
                    parameter.second_radial_fraction
                ),
                "center_bearing_rad": profile.center_bearing_rad,
                "actual_initial_heading_rad": profile.initial_heading_rad,
                "actual_terminal_heading_rad": profile.terminal_heading_rad,
                "path_length_m": profile.path_length_m,
                "final_radius_m": profile.endpoint_radius_m,
                "radial_net_progress_m": float(displacement[-1] @ radial),
                "max_abs_tangential_displacement_m": float(
                    np.max(np.abs(displacement @ tangential))
                ),
                "minimum_radius_m": float(np.min(radii)),
                "maximum_position_bearing_offset_deg": math.degrees(
                    float(np.max(np.abs(offsets)))
                ),
                "position_bearing_span_deg": math.degrees(
                    float(np.max(offsets) - np.min(offsets))
                ),
                "sector_utilization_fraction": (
                    float(np.max(offsets) - np.min(offsets))
                    / (2.0 * profile.half_angle_rad)
                ),
                "max_abs_angular_rate_radps": float(np.max(angular_rates)),
                "max_abs_curvature_per_m": float(np.max(angular_rates))
                / profile.speed_mps,
                "minimum_intervehicle_distance_m": minimum_separation[agent_id],
                "radial_monotonic_violation_count": int(
                    np.count_nonzero(np.diff(radii) < -1e-9)
                ),
                "terminal_position_bearing_offset_deg": math.degrees(offsets[-1]),
            }
        )
    return result
