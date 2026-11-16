"""Deterministic nominal covariance prediction for fixed acoustic schedules.

The module compiles the small, preregistered candidate set into physical
waves, builds the same event-time Pose2/scalar factor types used online, and
propagates covariance without sampling measurement or process noise.  It is a
ranking simulation, not a Monte Carlo estimator launcher.

For each planned mission the all-ordinary ``D100`` schedule is evaluated
first.  Its predicted cross-track covariance supplies the weak-window scores
needed by the ``*-DW`` templates and is also the fixed normalization in the
ranking criterion.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import cmp_to_key
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import gtsam  # type: ignore
import numpy as np
import pandas as pd
from tqdm import tqdm  # type: ignore

from estimator.distributed_graphs import IncrementalPoseGraph
from estimator.numerics import small_matmul
from estimator.odometry_preintegration import (
    exact_unicycle_increment,
    preintegrate_continuous_trajectory,
)
from factors.cycle_snoop_factor import (
    create_shared_timestamp_cycle_factor,
    create_shared_timestamp_master_factor,
)

from .formal_candidate_schedules import (
    ACTION_DUAL_MASTER,
    ACTION_JOINT,
    CANDIDATE_D100,
    FIXED_CANDIDATE_IDS,
    FixedCandidateSchedule,
    ScheduledAcousticAction,
    build_fixed_candidate_schedule,
)
from .formal_protocol import FormalExperimentConfig, _RawWave, _raw_wave
from .formal_protocol import build_radial_v2_spec
from .formal_motion import ConstantUnicycleMotion
from .formal_schedule_compiler import CompiledNominalAction, CompiledNominalSchedule, compile_nominal_schedule
from .formal_sector_sweep_motion import (
    SECTOR_SWEEP_ORIENTATIONS_DEG,
    build_structured_sector_sweep_case,
)


DEFAULT_START_SEED = 1
DEFAULT_SEED_COUNT = 1
DEFAULT_WORKERS = 12
DEFAULT_LAMBDA_C = 0.5
DEFAULT_LAMBDA_W = 0.5
DNCP_AUC_START_S = 20.0
DEFAULT_OUTPUT = Path(
    "experiments/master_follower_paper/results/"
    "dncp_structured_sector_s_sweep_seed_1_1000"
)
DEFAULT_FIXED_PROXY_OUTPUT = Path(
    "experiments/master_follower_paper/results/"
    "dncp_fixed_radial_sector_proxy_lambda_1_0p5_0"
)
DNCP_SCHEMA = "formal-dncp-ranking-v1"
FIXED_PROXY_SCHEMA = "formal-dncp-fixed-radial-sector-proxy-v1"
_TIME_SCALE = 10**10




@dataclass(frozen=True)
class ListenerPrediction:
    agent_id: str
    evaluation_times_s: tuple[float, ...]
    perpendicular_variances_m2: tuple[float, ...]
    auc_perpendicular_m2: float
    factor_count: int
    pose_variable_count: int
    timestamp_variable_count: int


class _KeyAllocator:
    def __init__(self) -> None:
        self._pose_index = 0
        self._timestamp_index = 0

    def pose(self) -> int:
        key = int(gtsam.symbol("X", self._pose_index))
        self._pose_index += 1
        return key

    def timestamp(self) -> int:
        key = int(gtsam.symbol("T", self._timestamp_index))
        self._timestamp_index += 1
        return key

    @property
    def pose_count(self) -> int:
        return self._pose_index

    @property
    def timestamp_count(self) -> int:
        return self._timestamp_index


def _time_token(time_s: float) -> int:
    return int(round(float(time_s) * _TIME_SCALE))


def _pose_function(
    config: FormalExperimentConfig,
    motion_profiles: Mapping[str, object],
) -> Callable[[str, float], gtsam.Pose2]:
    specs = {item.agent_id: item for item in config.agents}

    def pose_at(agent_id: str, time_s: float) -> gtsam.Pose2:
        profile = motion_profiles.get(agent_id)
        if profile is not None:
            return profile.pose_at(float(time_s))  # type: ignore[attr-defined]
        spec = specs[agent_id]
        return spec.initial_pose.compose(
            exact_unicycle_increment(
                spec.velocity, spec.angular_rate, float(time_s)
            )
        )

    return pose_at




def _world_position_covariance(
    pose: gtsam.Pose2, covariance: np.ndarray
) -> np.ndarray:
    cosine = math.cos(float(pose.theta()))
    sine = math.sin(float(pose.theta()))
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=float)
    result = small_matmul(
        small_matmul(
            rotation, np.asarray(covariance[:2, :2], dtype=float)
        ),
        rotation.T,
    )
    return 0.5 * (result + result.T)


def _normalized_auc(times: Sequence[float], values: Sequence[float]) -> float:
    x = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or x.size < 2:
        raise ValueError("AUC needs at least two aligned samples")
    duration = float(x[-1] - x[0])
    if duration <= 0.0:
        raise ValueError("AUC time span must be positive")
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)) / duration)


def _predict_listener(
    compiled: CompiledNominalSchedule,
    config: FormalExperimentConfig,
    motion_profiles: Mapping[str, object],
    listener_id: str,
) -> ListenerPrediction:
    """Run causal nominal covariance propagation for one local graph."""

    pose_at = _pose_function(config, motion_profiles)
    allocator = _KeyAllocator()
    graph = IncrementalPoseGraph()
    evaluation_times = tuple(
        float(index) for index in range(int(round(config.duration_s)) + 1)
    )

    local_times: dict[int, float] = {
        _time_token(time_s): time_s for time_s in evaluation_times
    }
    for wave in compiled.waves:
        if wave.sender_id == listener_id:
            local_times.setdefault(_time_token(wave.send_time), wave.send_time)
        else:
            receive_time = wave.receive_time(listener_id)
            local_times.setdefault(_time_token(receive_time), receive_time)
    ordered_local_times = tuple(sorted(local_times.values()))
    local_keys = {
        _time_token(time_s): allocator.pose() for time_s in ordered_local_times
    }

    initial_key = local_keys[_time_token(0.0)]
    graph.add_prior(
        initial_key,
        pose_at(listener_id, 0.0),
        config.follower_initial_covariance,
        kind="owner_initial_state",
    )

    relay_sources = tuple(
        sorted(
            {
                action.intent.relay_id
                for action in compiled.actions
                if action.intent.action_type == ACTION_JOINT
                and action.intent.relay_id != listener_id
                and action.intent.relay_id is not None
            }
        )
    )
    remote_initial_keys: dict[str, int] = {}
    for source_id in relay_sources:
        key = allocator.pose()
        remote_initial_keys[source_id] = key
        graph.add_prior(
            key,
            pose_at(source_id, 0.0),
            config.follower_initial_covariance,
            kind="preshared_remote_initial_state",
        )
    graph.update()

    serial = 0
    events: list[tuple[float, int, int, str, Any]] = []
    process_cache: dict[tuple[str, int, int], tuple[gtsam.Pose2, np.ndarray]] = {}

    def process_segment(
        agent_id: str, start_time: float, end_time: float
    ) -> tuple[gtsam.Pose2, np.ndarray]:
        cache_key = (
            agent_id,
            _time_token(start_time),
            _time_token(end_time),
        )
        cached = process_cache.get(cache_key)
        if cached is None:
            profile = motion_profiles[agent_id]
            cached = preintegrate_continuous_trajectory(
                start_time,
                end_time,
                config.continuous_noise,
                pose_at=profile.pose_at,  # type: ignore[attr-defined]
            )
            process_cache[cache_key] = cached
        return cached

    for start_time, end_time in zip(ordered_local_times, ordered_local_times[1:]):
        events.append(
            (
                end_time,
                0,
                serial,
                "local_segment",
                (
                    local_keys[_time_token(start_time)],
                    local_keys[_time_token(end_time)],
                    start_time,
                    end_time,
                ),
            )
        )
        serial += 1

    master_keys: dict[int, int] = {}
    for wave in compiled.waves:
        if wave.sender_id not in config.master_ids:
            continue
        key = allocator.pose()
        master_keys[wave.wave_id] = key
        events.append(
            (
                wave.receive_time(listener_id),
                1,
                serial,
                "master_prior",
                (key, wave),
            )
        )
        serial += 1

    remote_keys: dict[tuple[str, int], int] = {}
    remote_previous_key = dict(remote_initial_keys)
    remote_previous_time = {source_id: 0.0 for source_id in relay_sources}
    for action in compiled.actions:
        if (
            action.intent.action_type != ACTION_JOINT
            or action.intent.relay_id == listener_id
        ):
            continue
        source_id = action.intent.relay_id
        assert source_id is not None
        relay_wave = action.waves[2]
        new_key = allocator.pose()
        remote_keys[(source_id, relay_wave.wave_id)] = new_key
        events.append(
            (
                relay_wave.receive_time(listener_id),
                1,
                serial,
                "remote_increment",
                (
                    source_id,
                    remote_previous_key[source_id],
                    new_key,
                    remote_previous_time[source_id],
                    relay_wave.send_time,
                ),
            )
        )
        serial += 1
        remote_previous_key[source_id] = new_key
        remote_previous_time[source_id] = relay_wave.send_time

    for action in compiled.actions:
        if action.intent.action_type == ACTION_DUAL_MASTER:
            completion_time = action.waves[1].receive_time(listener_id)
        elif action.intent.relay_id == listener_id:
            completion_time = action.waves[1].receive_time(listener_id)
        else:
            completion_time = action.waves[3].receive_time(listener_id)
        events.append((completion_time, 2, serial, "measurement", action))
        serial += 1

    for time_s in evaluation_times[1:]:
        events.append((time_s, 3, serial, "evaluate", time_s))
        serial += 1

    timestamp_keys: dict[int, int] = {}

    def timestamp_key(wave_id: int) -> int:
        key = timestamp_keys.get(wave_id)
        if key is None:
            key = allocator.timestamp()
            graph.add_scalar_prior(
                key,
                0.0,
                config.timestamp_sigma_s,
                kind="timestamp_error_prior",
            )
            timestamp_keys[wave_id] = key
        return key

    variances: list[float] = []

    def record_variance(time_s: float) -> None:
        pose = pose_at(listener_id, time_s)
        covariance = graph.covariance(local_keys[_time_token(time_s)])
        covariance_xy = _world_position_covariance(pose, covariance)
        perpendicular = np.asarray(
            [-math.sin(pose.theta()), math.cos(pose.theta())], dtype=float
        )
        projected = small_matmul(covariance_xy, perpendicular.reshape(2, 1))
        variances.append(
            float(
                perpendicular[0] * projected[0, 0]
                + perpendicular[1] * projected[1, 0]
            )
        )

    record_variance(0.0)
    for _, _, _, event_type, payload in sorted(events):
        if event_type == "local_segment":
            old_key, new_key, start_time, end_time = payload
            delta, covariance = process_segment(
                listener_id, start_time, end_time
            )
            graph.add_between(
                old_key,
                new_key,
                delta,
                covariance,
                kind="owner_odometry_segment",
            )
        elif event_type == "master_prior":
            key, wave = payload
            graph.add_prior(
                key,
                pose_at(wave.sender_id, wave.send_time),
                config.master_covariance,
                kind="master_absolute_event_prior",
            )
        elif event_type == "remote_increment":
            source_id, old_key, new_key, start_time, end_time = payload
            delta, covariance = process_segment(source_id, start_time, end_time)
            graph.add_between(
                old_key,
                new_key,
                delta,
                covariance,
                kind="remote_source_odometry_increment",
            )
        elif event_type == "measurement":
            action = payload
            wave_0, wave_1 = action.waves[:2]
            local_0 = local_keys[
                _time_token(wave_0.receive_time(listener_id))
            ]
            local_1 = local_keys[
                _time_token(wave_1.receive_time(listener_id))
            ]
            first_timestamp = timestamp_key(wave_0.wave_id)
            second_timestamp = timestamp_key(wave_1.wave_id)
            if (
                action.intent.action_type == ACTION_DUAL_MASTER
                or action.intent.relay_id == listener_id
            ):
                measured_delta = (
                    wave_1.receive_time(listener_id)
                    - wave_0.receive_time(listener_id)
                )
                factor = create_shared_timestamp_master_factor(
                    master_keys[wave_0.wave_id],
                    master_keys[wave_1.wave_id],
                    local_0,
                    local_1,
                    timestamp_error_keys=(first_timestamp, second_timestamp),
                    measured_delta_t=measured_delta,
                    sound_speed=config.sound_speed_mps,
                )
                graph.add_custom_factor(
                    factor, kind="shared_timestamp_master_tdoa"
                )
                continue

            relay_id = action.intent.relay_id
            assert relay_id is not None
            wave_2, wave_3 = action.waves[2:]
            local_2 = local_keys[
                _time_token(wave_2.receive_time(listener_id))
            ]
            local_3 = local_keys[
                _time_token(wave_3.receive_time(listener_id))
            ]
            pose_keys = (
                master_keys[wave_0.wave_id],
                master_keys[wave_1.wave_id],
                remote_keys[(relay_id, wave_2.wave_id)],
                master_keys[wave_3.wave_id],
                local_0,
                local_1,
                local_2,
                local_3,
            )
            arrivals = np.asarray(
                [wave.receive_time(listener_id) for wave in action.waves],
                dtype=float,
            )
            corrections = np.asarray(
                [0.0, 0.0, action.anchor_wait_s], dtype=float
            )
            factor = create_shared_timestamp_cycle_factor(
                *pose_keys,
                timestamp_error_keys=tuple(
                    timestamp_key(wave.wave_id) for wave in action.waves
                ),
                measured_delta_t=np.diff(arrivals) - corrections,
                sound_speed=config.sound_speed_mps,
            )
            graph.add_custom_factor(factor, kind="shared_timestamp_joint_cycle")
        elif event_type == "evaluate":
            record_variance(float(payload))
        else:
            raise RuntimeError(f"unknown DNCP event type: {event_type}")

    if len(variances) != len(evaluation_times):
        raise AssertionError("DNCP evaluation grid is incomplete")
    if not np.all(np.isfinite(variances)) or min(variances) <= 0.0:
        raise RuntimeError("DNCP produced a non-finite covariance")
    auc_start_index = evaluation_times.index(DNCP_AUC_START_S)
    return ListenerPrediction(
        agent_id=listener_id,
        evaluation_times_s=evaluation_times,
        perpendicular_variances_m2=tuple(variances),
        auc_perpendicular_m2=_normalized_auc(
            evaluation_times[auc_start_index:], variances[auc_start_index:]
        ),
        factor_count=graph.factor_count,
        pose_variable_count=allocator.pose_count,
        timestamp_variable_count=allocator.timestamp_count,
    )


def predict_schedule(
    schedule: FixedCandidateSchedule,
    config: FormalExperimentConfig,
    motion_profiles: Mapping[str, object],
) -> tuple[CompiledNominalSchedule, tuple[ListenerPrediction, ...]]:
    pose_at = _pose_function(config, motion_profiles)
    compiled = compile_nominal_schedule(schedule, config, pose_at)
    predictions = tuple(
        _predict_listener(compiled, config, motion_profiles, listener_id)
        for listener_id in config.follower_ids
    )
    return compiled, predictions


def weak_window_scores_from_d100(
    predictions: Sequence[ListenerPrediction],
    *,
    lambda_w: float = DEFAULT_LAMBDA_W,
) -> dict[tuple[int, int], float]:
    """Create the 32 preregistered weak-window scores from D100."""

    lambda_w = float(lambda_w)
    if not 0.0 <= lambda_w <= 1.0:
        raise ValueError("lambda_w must lie in [0, 1]")
    by_agent = {item.agent_id: item for item in predictions}
    if len(by_agent) != 6:
        raise ValueError("weak-window scoring requires six followers")
    scores: dict[tuple[int, int], float] = {}
    for block_id in range(2):
        for macro_index in range(16):
            center_time = block_id * 100.0 + macro_index * 6.0 + 3.0
            values = []
            for prediction in by_agent.values():
                time_index = prediction.evaluation_times_s.index(center_time)
                values.append(prediction.perpendicular_variances_m2[time_index])
            scores[(block_id, macro_index)] = float(
                lambda_w * max(values)
                + (1.0 - lambda_w) * sum(values) / len(values)
            )
    return scores


def criterion_components(
    candidate_auc_by_agent: Mapping[str, float],
    baseline_auc_by_agent: Mapping[str, float],
    *,
    lambda_c: float,
) -> tuple[dict[str, float], float, float, float]:
    """Return per-agent ratios, fleet ratio, worst ratio, and C(lambda_c)."""

    lambda_c = float(lambda_c)
    if not 0.0 <= lambda_c <= 1.0:
        raise ValueError("lambda_c must lie in [0, 1]")
    if set(candidate_auc_by_agent) != set(baseline_auc_by_agent):
        raise ValueError("candidate and D100 must contain the same followers")
    ratios = {
        agent_id: float(candidate_auc_by_agent[agent_id])
        / float(baseline_auc_by_agent[agent_id])
        for agent_id in sorted(baseline_auc_by_agent)
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in ratios.values()):
        raise ValueError("DNCP AUC ratios must be finite and positive")
    candidate_values = tuple(float(value) for value in candidate_auc_by_agent.values())
    baseline_values = tuple(float(value) for value in baseline_auc_by_agent.values())
    fleet_ratio = float(
        (sum(candidate_values) / len(candidate_values))
        / (sum(baseline_values) / len(baseline_values))
    )
    worst_ratio = float(max(candidate_values) / max(baseline_values))
    criterion = lambda_c * worst_ratio + (1.0 - lambda_c) * fleet_ratio
    return ratios, fleet_ratio, worst_ratio, float(criterion)


def _rank_score_rows(rows: list[dict[str, Any]]) -> None:
    tolerance_scale = 1e-10

    def compare(left: dict[str, Any], right: dict[str, Any]) -> int:
        left_c = float(left["criterion"])
        right_c = float(right["criterion"])
        tolerance = tolerance_scale * max(1.0, abs(left_c), abs(right_c))
        if abs(left_c - right_c) > tolerance:
            return -1 if left_c < right_c else 1
        tie_left = (
            float(left["worst_member_ratio"]),
            float(left["fleet_mean_ratio"]),
            int(left["joint_action_count"]),
            str(left["candidate_id"]),
        )
        tie_right = (
            float(right["worst_member_ratio"]),
            float(right["fleet_mean_ratio"]),
            int(right["joint_action_count"]),
            str(right["candidate_id"]),
        )
        return -1 if tie_left < tie_right else (1 if tie_left > tie_right else 0)

    rows.sort(key=cmp_to_key(compare))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank


def run_dncp_case(
    *,
    seed: int,
    orientation_deg: float,
    lambda_c: float = DEFAULT_LAMBDA_C,
    lambda_w: float = DEFAULT_LAMBDA_W,
    candidate_ids: Sequence[str] = FIXED_CANDIDATE_IDS,
) -> dict[str, Any]:
    """Evaluate and rank every requested template for one planned mission."""

    candidate_ids = tuple(dict.fromkeys((CANDIDATE_D100, *candidate_ids)))
    unknown = sorted(set(candidate_ids) - set(FIXED_CANDIDATE_IDS))
    if unknown:
        raise ValueError(f"unknown DNCP candidates: {unknown}")
    spec, profiles, parameters, _ = build_structured_sector_sweep_case(
        seed=int(seed), orientation_deg=float(orientation_deg)
    )
    sector_angles = {
        agent_id: float(profiles[agent_id].center_bearing_rad)  # type: ignore[attr-defined]
        for agent_id in spec.config.follower_ids
    }
    d100 = build_fixed_candidate_schedule(
        CANDIDATE_D100,
        follower_ids=spec.config.follower_ids,
        sector_angles_rad=sector_angles,
    )
    d100_compiled, d100_predictions = predict_schedule(
        d100, spec.config, profiles
    )
    weak_scores = weak_window_scores_from_d100(
        d100_predictions, lambda_w=lambda_w
    )
    prediction_cache = {
        CANDIDATE_D100: (d100_compiled, d100_predictions)
    }
    baseline_auc = {
        item.agent_id: item.auc_perpendicular_m2 for item in d100_predictions
    }

    agent_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        if candidate_id not in prediction_cache:
            schedule = build_fixed_candidate_schedule(
                candidate_id,
                follower_ids=spec.config.follower_ids,
                sector_angles_rad=sector_angles,
                weak_window_scores=weak_scores,
            )
            prediction_cache[candidate_id] = predict_schedule(
                schedule, spec.config, profiles
            )
        compiled, predictions = prediction_cache[candidate_id]
        candidate_auc = {
            item.agent_id: item.auc_perpendicular_m2 for item in predictions
        }
        ratios, fleet_ratio, worst_ratio, criterion = criterion_components(
            candidate_auc, baseline_auc, lambda_c=lambda_c
        )
        for prediction in predictions:
            agent_rows.append(
                {
                    "seed": int(seed),
                    "orientation_deg": float(orientation_deg),
                    "candidate_id": candidate_id,
                    "agent_id": prediction.agent_id,
                    "auc_perpendicular_m2": prediction.auc_perpendicular_m2,
                    "auc_ratio_to_d100": ratios[prediction.agent_id],
                    "factor_count": prediction.factor_count,
                    "pose_variable_count": prediction.pose_variable_count,
                    "timestamp_variable_count": prediction.timestamp_variable_count,
                    "ordinary_action_count": compiled.schedule.ordinary_action_count,
                    "joint_action_count": compiled.schedule.joint_action_count,
                }
            )
        score_rows.append(
            {
                "seed": int(seed),
                "orientation_deg": float(orientation_deg),
                "candidate_id": candidate_id,
                "fleet_mean_ratio": fleet_ratio,
                "worst_member_ratio": worst_ratio,
                "criterion": criterion,
                "lambda_c": float(lambda_c),
                "ordinary_action_count": compiled.schedule.ordinary_action_count,
                "joint_action_count": compiled.schedule.joint_action_count,
                "physical_wave_count": compiled.physical_wave_count,
                "schedule_fingerprint": compiled.schedule.fingerprint,
                "selected_joint_macros_block_0": ";".join(
                    str(item)
                    for item in compiled.schedule.selected_joint_macros_by_block[0]
                ),
                "selected_joint_macros_block_1": ";".join(
                    str(item)
                    for item in compiled.schedule.selected_joint_macros_by_block[1]
                ),
                "relay_quotas": ";".join(
                    f"{agent}:{count}"
                    for agent, count in compiled.schedule.relay_quotas
                ),
            }
        )
    _rank_score_rows(score_rows)
    return {
        "seed": int(seed),
        "orientation_deg": float(orientation_deg),
        "lambda_c": float(lambda_c),
        "lambda_w": float(lambda_w),
        "candidate_ids": list(candidate_ids),
        "route_parameters": [asdict(item) for item in parameters],
        "agent_rows": agent_rows,
        "score_rows": score_rows,
    }


def _case_task(payload: tuple[int, float, float, float, tuple[str, ...]]) -> dict[str, Any]:
    seed, orientation_deg, lambda_c, lambda_w, candidate_ids = payload
    started = time.perf_counter()
    result = run_dncp_case(
        seed=seed,
        orientation_deg=orientation_deg,
        lambda_c=lambda_c,
        lambda_w=lambda_w,
        candidate_ids=candidate_ids,
    )
    result["elapsed_wall_s"] = time.perf_counter() - started
    return result


def _parse_orientations(text: str | Iterable[float]) -> tuple[float, ...]:
    if isinstance(text, str):
        values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    else:
        values = tuple(float(item) for item in text)
    if not values or len(values) != len(set(values)):
        raise ValueError("orientations must be a non-empty unique list")
    unsupported = sorted(set(values) - set(SECTOR_SWEEP_ORIENTATIONS_DEG))
    if unsupported:
        raise ValueError(f"unsupported orientations: {unsupported}")
    return values


def _case_path(output: Path, orientation_deg: float, seed: int) -> Path:
    label = f"{orientation_deg:g}".replace("-", "m").replace(".", "p")
    return output / "cases" / f"orientation_{label}" / f"seed_{seed:04d}.json"


def _configuration(
    *,
    orientations: Sequence[float],
    start_seed: int,
    seed_count: int,
    lambda_c: float,
    lambda_w: float,
    candidate_ids: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema": DNCP_SCHEMA,
        "mission_family": "structured_sector_s_sweep_beta10",
        "orientations_deg": list(orientations),
        "start_seed": int(start_seed),
        "end_seed": int(start_seed + seed_count - 1),
        "seed_count": int(seed_count),
        "candidate_ids": list(candidate_ids),
        "lambda_c": float(lambda_c),
        "lambda_w": float(lambda_w),
        "duration_s": 200.0,
        "wave_budget": 200,
        "evaluation_period_s": 1.0,
        "auc_start_s": DNCP_AUC_START_S,
        "noise_sampling": False,
        "nonzero_covariances_retained": True,
        "causal_filtering_covariance": True,
    }


def _fingerprint(configuration: Mapping[str, Any]) -> str:
    payload = json.dumps(
        configuration, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_case(path: Path, result: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_cases(
    output: Path,
    orientations: Sequence[float],
    seeds: Sequence[int],
    config_fingerprint: str,
) -> list[dict[str, Any]]:
    results = []
    for orientation in orientations:
        for seed in seeds:
            path = _case_path(output, orientation, seed)
            if not path.exists():
                raise RuntimeError(f"missing completed DNCP case: {path}")
            result = json.loads(path.read_text(encoding="utf-8"))
            if int(result["seed"]) != seed or abs(
                float(result["orientation_deg"]) - orientation
            ) > 1e-12:
                raise RuntimeError(f"DNCP checkpoint key mismatch: {path}")
            if result.get("config_fingerprint") != config_fingerprint:
                raise RuntimeError(f"DNCP checkpoint configuration mismatch: {path}")
            results.append(result)
    return results


def _aggregate_rankings(
    agent_frame: pd.DataFrame, lambda_c: float
) -> pd.DataFrame:
    result_rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, pd.DataFrame]] = [
        (f"orientation_{orientation:g}", group)
        for orientation, group in agent_frame.groupby("orientation_deg", sort=True)
    ]
    scopes.append(("all_orientations", agent_frame))
    for scope, frame in scopes:
        mean_auc = (
            frame.groupby(["candidate_id", "agent_id"], as_index=False)[
                "auc_perpendicular_m2"
            ]
            .mean()
        )
        baseline = {
            str(row.agent_id): float(row.auc_perpendicular_m2)
            for row in mean_auc.loc[
                mean_auc["candidate_id"] == CANDIDATE_D100
            ].itertuples()
        }
        rows: list[dict[str, Any]] = []
        for candidate_id, group in mean_auc.groupby("candidate_id", sort=False):
            candidate = {
                str(row.agent_id): float(row.auc_perpendicular_m2)
                for row in group.itertuples()
            }
            _, fleet_ratio, worst_ratio, criterion = criterion_components(
                candidate, baseline, lambda_c=lambda_c
            )
            score_row = frame.loc[frame["candidate_id"] == candidate_id].iloc[0]
            rows.append(
                {
                    "scope": scope,
                    "candidate_id": candidate_id,
                    "fleet_mean_ratio": fleet_ratio,
                    "worst_member_ratio": worst_ratio,
                    "criterion": criterion,
                    "lambda_c": lambda_c,
                    "ordinary_action_count": int(score_row.get("ordinary_action_count", 0)),
                    "joint_action_count": int(score_row.get("joint_action_count", 0)),
                }
            )
        _rank_score_rows(rows)
        result_rows.extend(rows)
    return pd.DataFrame(result_rows)


def _write_report(
    output: Path,
    configuration: Mapping[str, Any],
    aggregate: pd.DataFrame,
    score_frame: pd.DataFrame,
    elapsed_s: float,
) -> None:
    lines = [
        "# DNCP fixed-candidate ranking",
        "",
        "This report uses deterministic nominal covariance prediction: no noise",
        "realization is sampled, while every configured non-zero covariance is retained.",
        "D100 is both a ranked candidate and the fixed normalization baseline.",
        "",
        f"- seeds: {configuration['start_seed']}–{configuration['end_seed']}",
        f"- orientations: {configuration['orientations_deg']}",
        f"- lambda_c: {configuration['lambda_c']}",
        f"- candidates: {configuration['candidate_ids']}",
        f"- elapsed wall time: {elapsed_s:.1f} s",
        "",
        "The ranking is conditional on the planned mission, covariance model, fixed",
        "candidate set, 200-wave budget, and lambda_c. It is not a claim of universal",
        "or globally optimal scheduling.",
        "",
    ]
    for scope, frame in aggregate.groupby("scope", sort=False):
        lines.extend(
            [
                f"## {scope}",
                "",
                "| Rank | Candidate | Criterion | Worst ratio | Fleet ratio | D/J |",
                "|---:|---|---:|---:|---:|---:|",
            ]
        )
        for row in frame.sort_values("rank").itertuples():
            lines.append(
                f"| {int(row.rank)} | {row.candidate_id} | {row.criterion:.6f} | "
                f"{row.worst_member_ratio:.6f} | {row.fleet_mean_ratio:.6f} | "
                f"{int(row.ordinary_action_count)}/{int(row.joint_action_count)} |"
            )
        lines.append("")
    winners = (
        score_frame.loc[score_frame["rank"] == 1, "candidate_id"]
        .value_counts()
        .sort_index()
    )
    lines.extend(["## Per-mission first-place frequency", ""])
    for candidate_id, count in winners.items():
        lines.append(f"- {candidate_id}: {int(count)}")
    lines.append("")
    (output / "dncp_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_dncp_batch(
    output_dir: str | Path,
    *,
    orientations: Iterable[float] = SECTOR_SWEEP_ORIENTATIONS_DEG,
    start_seed: int = DEFAULT_START_SEED,
    seed_count: int = DEFAULT_SEED_COUNT,
    workers: int = DEFAULT_WORKERS,
    lambda_c: float = DEFAULT_LAMBDA_C,
    lambda_w: float = DEFAULT_LAMBDA_W,
    candidate_ids: Sequence[str] = FIXED_CANDIDATE_IDS,
) -> dict[str, Any]:
    orientations = _parse_orientations(orientations)
    start_seed = int(start_seed)
    seed_count = int(seed_count)
    workers = int(workers)
    lambda_c = float(lambda_c)
    lambda_w = float(lambda_w)
    if start_seed <= 0 or seed_count <= 0 or workers <= 0:
        raise ValueError("start_seed, seed_count, and workers must be positive")
    if not 0.0 <= lambda_c <= 1.0 or not 0.0 <= lambda_w <= 1.0:
        raise ValueError("lambda_c and lambda_w must lie in [0, 1]")
    candidate_ids = tuple(dict.fromkeys((CANDIDATE_D100, *candidate_ids)))
    unknown = sorted(set(candidate_ids) - set(FIXED_CANDIDATE_IDS))
    if unknown:
        raise ValueError(f"unknown DNCP candidates: {unknown}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    configuration = _configuration(
        orientations=orientations,
        start_seed=start_seed,
        seed_count=seed_count,
        lambda_c=lambda_c,
        lambda_w=lambda_w,
        candidate_ids=candidate_ids,
    )
    configuration["config_fingerprint"] = _fingerprint(configuration)
    config_path = output / "dncp_run_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("config_fingerprint") != configuration["config_fingerprint"]:
            raise RuntimeError("DNCP output contains an incompatible configuration")
    else:
        config_path.write_text(
            json.dumps(configuration, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    seeds = tuple(range(start_seed, start_seed + seed_count))
    pending_payloads = [
        (seed, orientation, lambda_c, lambda_w, candidate_ids)
        for orientation in orientations
        for seed in seeds
        if not _case_path(output, orientation, seed).exists()
    ]
    total_cases = len(orientations) * len(seeds)
    completed_before = total_cases - len(pending_payloads)
    effective_workers = min(workers, max(1, len(pending_payloads)), os.cpu_count() or workers)
    print(
        "[dncp] "
        f"orientations={','.join(f'{item:g}' for item in orientations)} "
        f"seeds={seeds[0]}..{seeds[-1]} candidates={len(candidate_ids)} "
        f"workers={effective_workers} cases={total_cases} "
        f"resume_skipped={completed_before} lambda_c={lambda_c:g}",
        flush=True,
    )
    wall_started = time.perf_counter()
    if pending_payloads:
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(_case_task, payload): payload
                for payload in pending_payloads[:effective_workers]
            }
            next_index = effective_workers
            with tqdm(
                total=total_cases,
                initial=completed_before,
                desc="DNCP missions",
                unit="mission",
                dynamic_ncols=True,
            ) as progress:
                while futures:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        payload = futures.pop(future)
                        try:
                            result = future.result()
                        except Exception as exc:
                            raise RuntimeError(
                                "DNCP case failed: "
                                f"seed={payload[0]}, orientation={payload[1]:g}"
                            ) from exc
                        result["config_fingerprint"] = configuration[
                            "config_fingerprint"
                        ]
                        _write_case(
                            _case_path(
                                output,
                                float(result["orientation_deg"]),
                                int(result["seed"]),
                            ),
                            result,
                        )
                        progress.update(1)
                        progress.set_postfix_str(
                            f"elapsed={time.perf_counter()-wall_started:.0f}s"
                        )
                        if next_index < len(pending_payloads):
                            next_payload = pending_payloads[next_index]
                            futures[executor.submit(_case_task, next_payload)] = next_payload
                            next_index += 1

    cases = _load_cases(
        output,
        orientations,
        seeds,
        str(configuration["config_fingerprint"]),
    )
    agent_rows = [row for case in cases for row in case["agent_rows"]]
    score_rows = [row for case in cases for row in case["score_rows"]]
    agent_frame = pd.DataFrame(agent_rows)
    score_frame = pd.DataFrame(score_rows)
    agent_frame.to_csv(output / "dncp_agent_auc.csv", index=False)
    score_frame.to_csv(output / "dncp_case_scores.csv", index=False)
    aggregate = _aggregate_rankings(agent_frame, lambda_c)
    aggregate.to_csv(output / "dncp_aggregate_ranking.csv", index=False)
    frequency = (
        score_frame.loc[score_frame["rank"] == 1]
        .groupby(["orientation_deg", "candidate_id"], as_index=False)
        .size()
        .rename(columns={"size": "first_place_count"})
    )
    frequency["case_count"] = seed_count
    frequency["first_place_fraction"] = frequency["first_place_count"] / seed_count
    frequency.to_csv(output / "dncp_selection_frequency.csv", index=False)
    elapsed_s = time.perf_counter() - wall_started
    _write_report(output, configuration, aggregate, score_frame, elapsed_s)
    manifest = {
        "schema": DNCP_SCHEMA,
        "completed_at": datetime.now().astimezone().isoformat(),
        "elapsed_wall_s_this_invocation": elapsed_s,
        "case_count": total_cases,
        "candidate_predictions": total_cases * len(candidate_ids),
        "complete": True,
    }
    (output / "dncp_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def build_fixed_radial_sector_proxy(
    *, proxy_orientation_deg: float = 0.0
) -> tuple[FormalExperimentConfig, dict[str, ConstantUnicycleMotion]]:
    """Return the single deterministic six-sector planning surrogate.

    The internal scenario constructor requires an epoch field, but no random
    generator is called here.  Every follower follows its sector centreline
    radially at 1.5 m/s for 200 s and both masters remain stationary.
    """

    proxy_orientation_deg = float(proxy_orientation_deg)
    if not math.isfinite(proxy_orientation_deg):
        raise ValueError("proxy_orientation_deg must be finite")
    spec = build_radial_v2_spec(
        rotation_deg=proxy_orientation_deg,
        duration_s=200.0,
        seeds=(1,),
        baseline_cycle_count=100,
        block_cycle_count=33,
        second_block_start_s=100.0,
    )
    profiles = {
        agent.agent_id: ConstantUnicycleMotion(
            agent.initial_x,
            agent.initial_y,
            agent.initial_heading_rad,
            agent.velocity,
            agent.angular_rate,
        )
        for agent in spec.config.agents
    }
    return spec.config, profiles


def _parse_float_list(text: str | Iterable[float]) -> tuple[float, ...]:
    values = (
        tuple(float(item.strip()) for item in text.split(",") if item.strip())
        if isinstance(text, str)
        else tuple(float(item) for item in text)
    )
    if not values or len(values) != len(set(values)):
        raise ValueError("weight list must contain unique values")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("every lambda_c value must lie in [0, 1]")
    return values


def _fixed_proxy_predictions(
    *,
    proxy_orientation_deg: float,
    lambda_w: float,
    candidate_ids: Sequence[str],
) -> tuple[
    FormalExperimentConfig,
    dict[str, ConstantUnicycleMotion],
    dict[tuple[int, int], float],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Evaluate all candidates once; no route or noise seed is sampled."""

    config, profiles = build_fixed_radial_sector_proxy(
        proxy_orientation_deg=proxy_orientation_deg
    )
    candidate_ids = tuple(dict.fromkeys((CANDIDATE_D100, *candidate_ids)))
    unknown = sorted(set(candidate_ids) - set(FIXED_CANDIDATE_IDS))
    if unknown:
        raise ValueError(f"unknown DNCP candidates: {unknown}")
    sector_angles = {
        agent_id: math.atan2(
            profiles[agent_id].initial_y_m, profiles[agent_id].initial_x_m
        )
        for agent_id in config.follower_ids
    }
    d100 = build_fixed_candidate_schedule(
        CANDIDATE_D100,
        follower_ids=config.follower_ids,
        sector_angles_rad=sector_angles,
    )
    d100_compiled, d100_predictions = predict_schedule(d100, config, profiles)
    weak_scores = weak_window_scores_from_d100(
        d100_predictions, lambda_w=lambda_w
    )
    prediction_cache = {
        CANDIDATE_D100: (d100_compiled, d100_predictions)
    }
    agent_rows: list[dict[str, Any]] = []
    schedule_rows: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        if candidate_id not in prediction_cache:
            schedule = build_fixed_candidate_schedule(
                candidate_id,
                follower_ids=config.follower_ids,
                sector_angles_rad=sector_angles,
                weak_window_scores=weak_scores,
            )
            prediction_cache[candidate_id] = predict_schedule(
                schedule, config, profiles
            )
        compiled, predictions = prediction_cache[candidate_id]
        for prediction in predictions:
            agent_rows.append(
                {
                    "planning_scenario": "fixed_radial_sector_proxy",
                    "proxy_orientation_deg": float(proxy_orientation_deg),
                    "candidate_id": candidate_id,
                    "agent_id": prediction.agent_id,
                    "auc_perpendicular_m2": prediction.auc_perpendicular_m2,
                    "factor_count": prediction.factor_count,
                    "pose_variable_count": prediction.pose_variable_count,
                    "timestamp_variable_count": prediction.timestamp_variable_count,
                }
            )
        schedule_rows.append(
            {
                "candidate_id": candidate_id,
                "ordinary_action_count": compiled.schedule.ordinary_action_count,
                "joint_action_count": compiled.schedule.joint_action_count,
                "physical_wave_count": compiled.physical_wave_count,
                "schedule_fingerprint": compiled.schedule.fingerprint,
                "relay_quotas": ";".join(
                    f"{agent}:{count}"
                    for agent, count in compiled.schedule.relay_quotas
                ),
                "selected_joint_macros_block_0": ";".join(
                    str(item)
                    for item in compiled.schedule.selected_joint_macros_by_block[0]
                ),
                "selected_joint_macros_block_1": ";".join(
                    str(item)
                    for item in compiled.schedule.selected_joint_macros_by_block[1]
                ),
                "actions": [asdict(action) for action in compiled.schedule.actions],
            }
        )
    return config, profiles, weak_scores, agent_rows, schedule_rows


def _rank_fixed_proxy(
    agent_rows: Sequence[Mapping[str, Any]],
    schedule_rows: Sequence[Mapping[str, Any]],
    lambda_c_values: Sequence[float],
) -> list[dict[str, Any]]:
    auc_by_candidate: dict[str, dict[str, float]] = {}
    for row in agent_rows:
        auc_by_candidate.setdefault(str(row["candidate_id"]), {})[
            str(row["agent_id"])
        ] = float(row["auc_perpendicular_m2"])
    baseline = auc_by_candidate[CANDIDATE_D100]
    schedule_by_id = {str(row["candidate_id"]): row for row in schedule_rows}
    result: list[dict[str, Any]] = []
    for lambda_c in lambda_c_values:
        rows: list[dict[str, Any]] = []
        for candidate_id, candidate_auc in auc_by_candidate.items():
            ratios, fleet_ratio, worst_ratio, criterion = criterion_components(
                candidate_auc, baseline, lambda_c=float(lambda_c)
            )
            schedule = schedule_by_id[candidate_id]
            rows.append(
                {
                    "lambda_c": float(lambda_c),
                    "candidate_id": candidate_id,
                    "criterion": criterion,
                    "worst_member_ratio": worst_ratio,
                    "fleet_mean_ratio": fleet_ratio,
                    "worst_agent_id": max(
                        candidate_auc, key=lambda item: candidate_auc[item]
                    ),
                    "worst_auc_perpendicular_m2": max(candidate_auc.values()),
                    "fleet_mean_auc_perpendicular_m2": sum(candidate_auc.values())
                    / len(candidate_auc),
                    "max_agent_relative_ratio": max(ratios.values()),
                    "ordinary_action_count": int(schedule["ordinary_action_count"]),
                    "joint_action_count": int(schedule["joint_action_count"]),
                    "physical_wave_count": int(schedule["physical_wave_count"]),
                    "schedule_fingerprint": str(schedule["schedule_fingerprint"]),
                }
            )
        _rank_score_rows(rows)
        result.extend(rows)
    return result


def _write_fixed_proxy_report(
    output: Path,
    configuration: Mapping[str, Any],
    ranking_rows: Sequence[Mapping[str, Any]],
    elapsed_s: float,
) -> None:
    lines = [
        "# Fixed radial-sector proxy DNCP ranking",
        "",
        "The planning geometry is a single deterministic six-follower radial",
        "centreline mission. No trajectory parameter, process error, initial error,",
        "master error, or timestamp error is sampled. All configured non-zero",
        "covariances remain in the causal event-time factor graphs.",
        "",
        f"- proxy orientation: {configuration['proxy_orientation_deg']:g} deg",
        f"- horizon / speed: 200 s / 1.5 m/s",
        f"- AUC window: {DNCP_AUC_START_S:g}--200 s",
        f"- physical-wave budget: 200 for every candidate",
        f"- lambda_c values: {configuration['lambda_c_values']}",
        f"- elapsed wall time: {elapsed_s:.1f} s",
        "",
        "D100 is the ordinary dual-master baseline, J66-RR is the sequential",
        "round-robin Joint protocol, and H* entries are adjusted fixed offline",
        "schedules. The result is conditional on this proxy geometry and the fixed",
        "candidate set; it is not a universal or global optimum.",
        "",
    ]
    frame = pd.DataFrame(ranking_rows)
    for lambda_c in configuration["lambda_c_values"]:
        subset = frame.loc[np.isclose(frame["lambda_c"], float(lambda_c))]
        lines.extend(
            [
                f"## lambda_c = {float(lambda_c):g}",
                "",
                "| Rank | Candidate | C | Worst ratio | Fleet ratio | Worst AUC | Fleet AUC | D/J |",
                "|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in subset.sort_values("rank").itertuples():
            lines.append(
                f"| {int(row.rank)} | {row.candidate_id} | {row.criterion:.6f} | "
                f"{row.worst_member_ratio:.6f} | {row.fleet_mean_ratio:.6f} | "
                f"{row.worst_auc_perpendicular_m2:.6f} | "
                f"{row.fleet_mean_auc_perpendicular_m2:.6f} | "
                f"{int(row.ordinary_action_count)}/{int(row.joint_action_count)} |"
            )
        lines.append("")
    (output / "dncp_fixed_proxy_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def run_fixed_proxy_dncp(
    output_dir: str | Path,
    *,
    proxy_orientation_deg: float = 0.0,
    lambda_c_values: Sequence[float] = (1.0, 0.5, 0.0),
    lambda_w: float = DEFAULT_LAMBDA_W,
    candidate_ids: Sequence[str] = FIXED_CANDIDATE_IDS,
) -> dict[str, Any]:
    """Run the seed-free proxy prediction once and emit all weight rankings."""

    lambda_c_values = _parse_float_list(lambda_c_values)
    lambda_w = float(lambda_w)
    if not 0.0 <= lambda_w <= 1.0:
        raise ValueError("lambda_w must lie in [0, 1]")
    candidate_ids = tuple(dict.fromkeys((CANDIDATE_D100, *candidate_ids)))
    unknown = sorted(set(candidate_ids) - set(FIXED_CANDIDATE_IDS))
    if unknown:
        raise ValueError(f"unknown DNCP candidates: {unknown}")
    configuration = {
        "schema": FIXED_PROXY_SCHEMA,
        "planning_scenario": "fixed_radial_sector_proxy",
        "proxy_orientation_deg": float(proxy_orientation_deg),
        "candidate_ids": list(candidate_ids),
        "lambda_c_values": list(lambda_c_values),
        "lambda_w": lambda_w,
        "duration_s": 200.0,
        "speed_mps": 1.5,
        "initial_radius_m": 100.0,
        "master_positions_m": {"A": [0.0, -500.0], "B": [0.0, 500.0]},
        "auc_start_s": DNCP_AUC_START_S,
        "wave_budget": 200,
        "trajectory_sampling": False,
        "noise_sampling": False,
        "nonzero_covariances_retained": True,
        "causal_filtering_covariance": True,
    }
    configuration["config_fingerprint"] = _fingerprint(configuration)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print(
        "[dncp-fixed-proxy] "
        f"orientation={proxy_orientation_deg:g} candidates={len(candidate_ids)} "
        f"lambda_c={','.join(f'{item:g}' for item in lambda_c_values)}",
        flush=True,
    )
    _, _, weak_scores, agent_rows, schedule_rows = _fixed_proxy_predictions(
        proxy_orientation_deg=float(proxy_orientation_deg),
        lambda_w=lambda_w,
        candidate_ids=candidate_ids,
    )
    ranking_rows = _rank_fixed_proxy(
        agent_rows, schedule_rows, lambda_c_values
    )
    elapsed_s = time.perf_counter() - started
    pd.DataFrame(agent_rows).to_csv(
        output / "dncp_fixed_proxy_agent_auc.csv", index=False
    )
    pd.DataFrame(ranking_rows).to_csv(
        output / "dncp_fixed_proxy_rankings.csv", index=False
    )
    (output / "dncp_fixed_proxy_schedules.json").write_text(
        json.dumps(
            {
                "schema": FIXED_PROXY_SCHEMA,
                "weak_window_scores": {
                    f"{block}:{macro}": score
                    for (block, macro), score in weak_scores.items()
                },
                "schedules": schedule_rows,
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    (output / "dncp_fixed_proxy_config.json").write_text(
        json.dumps(configuration, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_fixed_proxy_report(output, configuration, ranking_rows, elapsed_s)
    manifest = {
        "schema": FIXED_PROXY_SCHEMA,
        "completed_at": datetime.now().astimezone().isoformat(),
        "elapsed_wall_s": elapsed_s,
        "candidate_count": len(candidate_ids),
        "lambda_c_count": len(lambda_c_values),
        "covariance_predictions_reused_across_lambda": True,
        "complete": True,
    }
    (output / "dncp_fixed_proxy_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[dncp-fixed-proxy] complete elapsed={elapsed_s:.1f}s "
        f"report={output / 'dncp_fixed_proxy_report.md'}",
        flush=True,
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank fixed schedules by deterministic nominal covariance prediction."
    )
    parser.add_argument(
        "--planning-mode",
        choices=("fixed_proxy", "route_ensemble"),
        default="fixed_proxy",
        help="fixed_proxy is the seed-free paper planning model",
    )
    parser.add_argument("--proxy-orientation", type=float, default=0.0)
    parser.add_argument(
        "--lambda-c-values",
        type=str,
        default="1,0.5,0",
        help="fixed_proxy rankings computed from one shared covariance prediction",
    )
    parser.add_argument("--start-seed", type=int, default=DEFAULT_START_SEED)
    parser.add_argument("--seed-count", type=int, default=DEFAULT_SEED_COUNT)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--orientations",
        type=str,
        default=",".join(f"{item:g}" for item in SECTOR_SWEEP_ORIENTATIONS_DEG),
    )
    parser.add_argument("--lambda-c", type=float, default=DEFAULT_LAMBDA_C)
    parser.add_argument("--lambda-w", type=float, default=DEFAULT_LAMBDA_W)
    parser.add_argument(
        "--candidates",
        type=str,
        default=",".join(FIXED_CANDIDATE_IDS),
        help="comma-separated fixed candidate IDs; D100 is always included",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    candidates = tuple(
        item.strip() for item in args.candidates.split(",") if item.strip()
    )
    if args.planning_mode == "fixed_proxy":
        run_fixed_proxy_dncp(
            args.output or DEFAULT_FIXED_PROXY_OUTPUT,
            proxy_orientation_deg=args.proxy_orientation,
            lambda_c_values=_parse_float_list(args.lambda_c_values),
            lambda_w=args.lambda_w,
            candidate_ids=candidates,
        )
    else:
        run_dncp_batch(
            args.output or DEFAULT_OUTPUT,
            orientations=_parse_orientations(args.orientations),
            start_seed=args.start_seed,
            seed_count=args.seed_count,
            workers=args.workers,
            lambda_c=args.lambda_c,
            lambda_w=args.lambda_w,
            candidate_ids=candidates,
        )


if __name__ == "__main__":
    main()
