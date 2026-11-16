"""Continuous motion laws for the random non-straight radial experiment."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import math
from typing import Mapping

import gtsam  # type: ignore
import numpy as np

from estimator.odometry_preintegration import exact_unicycle_increment
from .formal_protocol import (
    PROTOCOL_VARIANT_BIDIRECTIONAL,
    RELAY_SCHEDULE_ROUND_ROBIN,
    FormalScenarioSpec,
    build_radial_v2_spec,
)


RANDOM_PATH_ORIENTATIONS_DEG = (0.0, 15.0, 30.0)
RANDOM_PATH_FOLLOWER_IDS = ("C", "D", "E", "F", "G", "H")
RANDOM_PATH_MISSION_DURATION_S = 200.0
RANDOM_PATH_SPEED_MPS = 1.5
RANDOM_PATH_HEADING_CYCLES = 2.0
RANDOM_PATH_AMPLITUDE_MEAN_DEG = 15.0
RANDOM_PATH_AMPLITUDE_STD_DEG = 5.0
RANDOM_PATH_AMPLITUDE_MIN_DEG = 5.0
RANDOM_PATH_AMPLITUDE_MAX_DEG = 25.0
_TRUTH_QUADRATURE_ORDER = 32
_TRUTH_QUADRATURE = (
    (-0.9972638618494816, 0.007018610009469298),
    (-0.9856115115452684, 0.016274394730905965),
    (-0.9647622555875064, 0.025392065309262427),
    (-0.9349060759377397, 0.034273862913021626),
    (-0.8963211557660522, 0.042835898022226426),
    (-0.84936761373257, 0.050998059262376244),
    (-0.7944837959679424, 0.058684093478535704),
    (-0.7321821187402897, 0.06582222277636175),
    (-0.6630442669302152, 0.07234579410884845),
    (-0.5877157572407623, 0.07819389578707031),
    (-0.5068999089322294, 0.08331192422694685),
    (-0.42135127613063533, 0.08765209300440391),
    (-0.33186860228212767, 0.09117387869576386),
    (-0.23928736225213706, 0.09384439908080457),
    (-0.1444719615827965, 0.09563872007927483),
    (-0.04830766568773831, 0.09654008851472781),
    (+0.04830766568773831, 0.09654008851472781),
    (+0.1444719615827965, 0.09563872007927483),
    (+0.23928736225213706, 0.09384439908080457),
    (+0.33186860228212767, 0.09117387869576386),
    (+0.42135127613063533, 0.08765209300440391),
    (+0.5068999089322294, 0.08331192422694685),
    (+0.5877157572407623, 0.07819389578707031),
    (+0.6630442669302152, 0.07234579410884845),
    (+0.7321821187402897, 0.06582222277636175),
    (+0.7944837959679424, 0.058684093478535704),
    (+0.84936761373257, 0.050998059262376244),
    (+0.8963211557660522, 0.042835898022226426),
    (+0.9349060759377397, 0.034273862913021626),
    (+0.9647622555875064, 0.025392065309262427),
    (+0.9856115115452684, 0.016274394730905965),
    (+0.9972638618494816, 0.007018610009469298),
)
_TRUTH_NODES = np.asarray([item[0] for item in _TRUTH_QUADRATURE], dtype=float)
_TRUTH_WEIGHTS = np.asarray([item[1] for item in _TRUTH_QUADRATURE], dtype=float)


@dataclass(frozen=True)
class ConstantUnicycleMotion:
    initial_x_m: float
    initial_y_m: float
    initial_heading_rad: float
    speed_mps: float
    angular_rate_radps: float

    def heading_at(self, time_s: float) -> float:
        return self.initial_heading_rad + self.angular_rate_radps * float(time_s)

    def control_at(self, time_s: float) -> tuple[float, float]:
        del time_s
        return self.speed_mps, self.angular_rate_radps

    def pose_at(self, time_s: float) -> gtsam.Pose2:
        initial = gtsam.Pose2(
            self.initial_x_m, self.initial_y_m, self.initial_heading_rad
        )
        return initial.compose(
            exact_unicycle_increment(
                self.speed_mps, self.angular_rate_radps, float(time_s)
            )
        )


@lru_cache(maxsize=131_072)
def _sinusoidal_displacement(
    speed_mps: float,
    nominal_heading_rad: float,
    amplitude_rad: float,
    phase_rad: float,
    angular_frequency_radps: float,
    time_s: float,
) -> tuple[float, float]:
    """Integrate world velocity using fixed 32-point Gauss--Legendre."""
    time_s = float(time_s)
    if time_s < 0.0 or not math.isfinite(time_s):
        raise ValueError("trajectory time must be finite and non-negative")
    if time_s == 0.0:
        return 0.0, 0.0
    half = 0.5 * time_s
    sample_times = half * (_TRUTH_NODES + 1.0)
    headings = nominal_heading_rad + amplitude_rad * np.sin(
        angular_frequency_radps * sample_times + phase_rad
    )
    dx = half * speed_mps * float(np.dot(_TRUTH_WEIGHTS, np.cos(headings)))
    dy = half * speed_mps * float(np.dot(_TRUTH_WEIGHTS, np.sin(headings)))
    return dx, dy


@dataclass(frozen=True)
class SinusoidalHeadingMotion:
    initial_x_m: float
    initial_y_m: float
    nominal_heading_rad: float
    speed_mps: float
    amplitude_rad: float
    phase_rad: float
    angular_frequency_radps: float = (
        2.0 * math.pi * RANDOM_PATH_HEADING_CYCLES
        / RANDOM_PATH_MISSION_DURATION_S
    )

    @property
    def initial_heading_rad(self) -> float:
        return self.heading_at(0.0)

    def heading_at(self, time_s: float) -> float:
        return self.nominal_heading_rad + self.amplitude_rad * math.sin(
            self.angular_frequency_radps * float(time_s) + self.phase_rad
        )

    def control_at(self, time_s: float) -> tuple[float, float]:
        angular_rate = (
            self.amplitude_rad
            * self.angular_frequency_radps
            * math.cos(
                self.angular_frequency_radps * float(time_s) + self.phase_rad
            )
        )
        return self.speed_mps, angular_rate

    def pose_at(self, time_s: float) -> gtsam.Pose2:
        dx, dy = _sinusoidal_displacement(
            self.speed_mps,
            self.nominal_heading_rad,
            self.amplitude_rad,
            self.phase_rad,
            self.angular_frequency_radps,
            float(time_s),
        )
        return gtsam.Pose2(
            self.initial_x_m + dx,
            self.initial_y_m + dy,
            self.heading_at(time_s),
        )


@dataclass(frozen=True)
class RandomPathParameter:
    agent_id: str
    amplitude_rad: float
    phase_rad: float


def sample_random_path_parameters(
    seed: int,
    follower_ids: tuple[str, ...] = RANDOM_PATH_FOLLOWER_IDS,
) -> tuple[RandomPathParameter, ...]:
    """Draw seed-only path parameters shared across all orientations."""
    if int(seed) <= 0:
        raise ValueError("seed must be positive")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x52505448]))
    result: list[RandomPathParameter] = []
    for agent_id in follower_ids:
        while True:
            amplitude_deg = float(
                rng.normal(
                    RANDOM_PATH_AMPLITUDE_MEAN_DEG,
                    RANDOM_PATH_AMPLITUDE_STD_DEG,
                )
            )
            if (
                RANDOM_PATH_AMPLITUDE_MIN_DEG
                <= amplitude_deg
                <= RANDOM_PATH_AMPLITUDE_MAX_DEG
            ):
                break
        result.append(
            RandomPathParameter(
                agent_id=agent_id,
                amplitude_rad=math.radians(amplitude_deg),
                phase_rad=float(rng.uniform(0.0, 2.0 * math.pi)),
            )
        )
    return tuple(result)


def realization_seed(seed: int, orientation_deg: float) -> int:
    """Return an injective seed for the frozen three-orientation panel."""
    seed = int(seed)
    if seed <= 0:
        raise ValueError("seed must be positive")
    matches = [
        index
        for index, value in enumerate(RANDOM_PATH_ORIENTATIONS_DEG)
        if abs(float(orientation_deg) - value) <= 1e-12
    ]
    if len(matches) != 1:
        raise ValueError(
            "random-path orientation must be one of 0, 15, or 30 degrees"
        )
    value = (seed - 1) * len(RANDOM_PATH_ORIENTATIONS_DEG) + matches[0] + 1
    if value > 4_000_000:
        raise ValueError("seed is too large for the legacy RandomState streams")
    return value


def build_random_nonstraight_case(
    *,
    seed: int,
    orientation_deg: float,
    duration_s: float = RANDOM_PATH_MISSION_DURATION_S,
    baseline_cycle_count: int = 100,
    block_cycle_count: int = 33,
    second_block_start_s: float = 100.0,
) -> tuple[
    FormalScenarioSpec,
    Mapping[str, ConstantUnicycleMotion | SinusoidalHeadingMotion],
    tuple[RandomPathParameter, ...],
    int,
]:
    """Build one random-path case without changing the frozen v2 protocol."""
    seed = int(seed)
    orientation_deg = float(orientation_deg)
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
    parameters = sample_random_path_parameters(seed, spec.config.follower_ids)
    parameter_by_agent = {item.agent_id: item for item in parameters}
    profiles: dict[str, ConstantUnicycleMotion | SinusoidalHeadingMotion] = {}
    agents = []
    for agent in spec.config.agents:
        if agent.role == "master":
            profile: ConstantUnicycleMotion | SinusoidalHeadingMotion = (
                ConstantUnicycleMotion(
                    agent.initial_x,
                    agent.initial_y,
                    agent.initial_heading_rad,
                    agent.velocity,
                    agent.angular_rate,
                )
            )
        else:
            parameter = parameter_by_agent[agent.agent_id]
            nominal_heading = math.atan2(agent.initial_y, agent.initial_x)
            profile = SinusoidalHeadingMotion(
                initial_x_m=agent.initial_x,
                initial_y_m=agent.initial_y,
                nominal_heading_rad=nominal_heading,
                speed_mps=RANDOM_PATH_SPEED_MPS,
                amplitude_rad=parameter.amplitude_rad,
                phase_rad=parameter.phase_rad,
            )
            agent = replace(
                agent,
                initial_heading_rad=profile.initial_heading_rad,
                velocity=RANDOM_PATH_SPEED_MPS,
                angular_rate=profile.control_at(0.0)[1],
            )
        profiles[agent.agent_id] = profile
        agents.append(agent)
    orientation_label = f"{orientation_deg:g}".replace("-", "m").replace(".", "p")
    spec = replace(
        spec,
        name=f"six_follower_random_nonstraight_v2_{orientation_label}deg",
        config=replace(spec.config, agents=tuple(agents), seeds=(seed,)),
    )
    spec.validate()
    return spec, profiles, parameters, realization_seed(seed, orientation_deg)


def random_path_parameter_rows(dataset) -> list[dict[str, float | int | str]]:
    """Build per-agent trajectory audit rows from a generated dataset."""
    result: list[dict[str, float | int | str]] = []
    orientation_deg = math.degrees(float(dataset.spec.radial_rotation_rad))
    for agent_id in dataset.config.follower_ids:
        profile = dataset.motion_profiles[agent_id]
        if not isinstance(profile, SinusoidalHeadingMotion):
            continue
        truth = np.asarray(dataset.truth_grid[agent_id], dtype=float)
        initial_xy = truth[0, :2]
        displacement = truth[:, :2] - initial_xy
        radial = np.asarray(
            [math.cos(profile.nominal_heading_rad), math.sin(profile.nominal_heading_rad)],
            dtype=float,
        )
        tangential = np.asarray([-radial[1], radial[0]], dtype=float)
        final_xy = truth[-1, :2]
        result.append(
            {
                "scenario": dataset.spec.name,
                "seed": int(dataset.seed),
                "realization_seed": int(dataset.realization_seed),
                "orientation_deg": orientation_deg,
                "agent_id": agent_id,
                "amplitude_deg": math.degrees(profile.amplitude_rad),
                "phase_rad": profile.phase_rad,
                "nominal_heading_rad": profile.nominal_heading_rad,
                "actual_initial_heading_rad": profile.initial_heading_rad,
                "max_abs_angular_rate_radps": (
                    profile.amplitude_rad * profile.angular_frequency_radps
                ),
                "path_length_m": profile.speed_mps * dataset.config.duration_s,
                "final_radius_m": float(np.linalg.norm(final_xy)),
                "radial_net_progress_m": float(displacement[-1] @ radial),
                "max_abs_tangential_displacement_m": float(
                    np.max(np.abs(displacement @ tangential))
                ),
            }
        )
    return result
