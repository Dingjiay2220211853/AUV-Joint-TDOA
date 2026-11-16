"""Paper section-II Monte Carlo simulation for cyclic relay schedules.

The five methods are D-only, J-RR, J-D1, J-D2 and J-D3.  D-only retains
the original two-wave/one-row master factor.  All four Joint schedules use
the same 62 Joint action times, five-cycle alternating-anchor backbone and
200-wave finite rendering; only their deterministic cyclic relay words differ.

This entry writes simulation checkpoints only.  It deliberately performs no
bootstrap, plotting or post-hoc schedule selection.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
from queue import Empty
import sqlite3
import time
import traceback
from typing import Any, Mapping, Sequence

from tqdm import tqdm

from .formal_candidate_schedules import (
    CANDIDATE_J_RR,
    DEFAULT_FOLLOWER_IDS,
    DIRECTIONAL_JOINT_CANDIDATE_IDS,
    FixedCandidateSchedule,
    build_all_paper_directional_schedules,
)
from .formal_experiment_5_offline_schedules import (
    OfflineScheduleCheckpoint,
    _canonical,
    _write_json,
    scientific_fingerprint,
    validate_case_payload,
)


SCHEMA = "paper-directional-joint-schedules-phi15-anchor5-v2"
DEFAULT_OUTPUT = Path(
    "experiments/master_follower_paper/results/"
    "formal_directional_joint_schedules_phi15_beta10_anchor5_seed_1_1000"
)
PRIMARY_LAMBDA_C = 0.5
_EVENT_QUEUE: Any = None
EVENT_COALESCENCE_MIGRATION_ID = "physical-coarrival-pose-dedup-v1"
_EVENT_COALESCENCE_CHANGED_SOURCES = frozenset(
    {
        "experiments/master_follower_paper/formal_experiment_6_directional_joint_schedules.py",
        "experiments/master_follower_paper/formal_schedule_simulation.py",
        "experiments/master_follower_paper/formal_dncp_ranking.py",
        "experiments/master_follower_paper/formal_protocol.py",
        "experiments/master_follower_paper/graphs.py",
        "estimator/distributed_graphs.py",
        "factors/cycle_snoop_factor.py",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _code_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    files = [
        "experiments/master_follower_paper/formal_experiment_6_directional_joint_schedules.py",
        "experiments/master_follower_paper/formal_candidate_schedules.py",
        "experiments/master_follower_paper/formal_schedule_simulation.py",
        "experiments/master_follower_paper/formal_schedule_compiler.py",
        "experiments/master_follower_paper/formal_dncp_ranking.py",
        "experiments/master_follower_paper/formal_protocol.py",
        "experiments/master_follower_paper/formal_sector_sweep_motion.py",
        "experiments/master_follower_paper/formal_motion.py",
        "experiments/master_follower_paper/model.py",
        "experiments/master_follower_paper/graphs.py",
        "experiments/master_follower_paper/formal_runner.py",
        "estimator/distributed_graphs.py",
        "estimator/odometry_preintegration.py",
        "estimator/numerics.py",
        "factors/cycle_snoop_factor.py",
    ]
    return {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in files
    }


def _configuration() -> dict[str, Any]:
    from .formal_dncp_ranking import build_fixed_radial_sector_proxy

    config, _ = build_fixed_radial_sector_proxy(proxy_orientation_deg=15.0)
    return {
        "schema": SCHEMA,
        "orientation_deg": 15.0,
        "candidate_ids": list(DIRECTIONAL_JOINT_CANDIDATE_IDS),
        "primary_lambda_c": PRIMARY_LAMBDA_C,
        "model": asdict(config),
        "trajectory": {
            "family": "structured_sector_s_sweep",
            "half_angle_deg": 10.0,
            "duration_s": 200.0,
            "speed_mps": 1.5,
            "path_length_m": 300.0,
            "initial_radius_m": 100.0,
            "amplitude_fraction_range": [0.7, 1.0],
            "first_waypoint_radial_fraction_range": [0.25, 0.35],
            "second_waypoint_radial_fraction_range": [0.65, 0.75],
            "interpolation": "piecewise_quintic_smoothstep_c2",
        },
        "measurement": {
            "ordinary_master_factor": "original_two_wave_one_row",
            "cross_action_master_row": False,
            "coincident_event_pose_policy": (
                "merge_physical_pose_key_and_accumulate_logical_jacobians"
            ),
            "simultaneous_timestamp_policy": (
                "distinct_sender_coded_wave_timestamp_states"
            ),
            "joint_action_count": 62,
            "joint_anchor_block_lengths": [5] * 12 + [1, 1],
            "joint_anchor_block_count": 14,
            "joint_anchor_switch_count": 13,
            "physical_wave_count": 200,
            "common_union_atomic_odometry": True,
            "common_identical_physical_event_noise": True,
            "report_times_s": list(range(1, 201)),
        },
        "source_sha256": _code_hashes(),
    }


def _angles() -> dict[str, float]:
    return {
        agent: math.radians(15.0 + 60.0 * index)
        for index, agent in enumerate(DEFAULT_FOLLOWER_IDS)
    }


def schedules_from_frozen(
    design: Mapping[str, Any],
) -> tuple[FixedCandidateSchedule, ...]:
    schedules = build_all_paper_directional_schedules(
        follower_ids=DEFAULT_FOLLOWER_IDS,
        sector_angles_rad=_angles(),
    )
    if _canonical([asdict(item) for item in schedules]) != _canonical(
        design["schedules"]
    ):
        raise ValueError("frozen schedules no longer match the cyclic generator")
    return schedules


def _nominal_predictions(
    schedules: Sequence[FixedCandidateSchedule],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Freeze one deterministic covariance ranking relative to J-RR."""

    from .formal_dncp_ranking import (
        build_fixed_radial_sector_proxy,
        criterion_components,
        predict_schedule,
    )

    config, profiles = build_fixed_radial_sector_proxy(
        proxy_orientation_deg=15.0
    )
    agent_rows: list[dict[str, Any]] = []
    auc: dict[str, dict[str, float]] = {}
    for schedule in schedules:
        compiled, predictions = predict_schedule(schedule, config, profiles)
        auc[schedule.candidate_id] = {
            item.agent_id: float(item.auc_perpendicular_m2)
            for item in predictions
        }
        for item in predictions:
            agent_rows.append(
                {
                    "candidate_id": schedule.candidate_id,
                    "agent_id": item.agent_id,
                    "auc_perpendicular_m2": float(item.auc_perpendicular_m2),
                    "factor_count": int(item.factor_count),
                    "pose_variable_count": int(item.pose_variable_count),
                    "timestamp_variable_count": int(item.timestamp_variable_count),
                    "physical_wave_count": int(compiled.physical_wave_count),
                }
            )
    reference = auc[CANDIDATE_J_RR]
    scores: list[dict[str, Any]] = []
    for schedule in schedules:
        if schedule.candidate_id == DIRECTIONAL_JOINT_CANDIDATE_IDS[0]:
            # D-only remains contextual and is not part of relay allocation.
            continue
        ratios, fleet, worst, criterion = criterion_components(
            auc[schedule.candidate_id], reference, lambda_c=PRIMARY_LAMBDA_C
        )
        scores.append(
            {
                "candidate_id": schedule.candidate_id,
                "criterion": criterion,
                "fleet_mean_ratio_to_j_rr": fleet,
                "worst_member_ratio_to_j_rr": worst,
                "per_agent_ratios_to_j_rr": ratios,
            }
        )
    scores.sort(key=lambda row: (float(row["criterion"]), row["candidate_id"]))
    for rank, row in enumerate(scores, start=1):
        row["rank"] = rank
    return agent_rows, scores


def _configuration_before_event_coalescence_policy(
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    comparable = json.loads(_canonical(configuration))
    comparable.pop("source_sha256", None)
    measurement = comparable.get("measurement", {})
    measurement.pop("coincident_event_pose_policy", None)
    measurement.pop("simultaneous_timestamp_policy", None)
    return comparable


def _migrate_event_coalescence_checkpoint(
    output: Path,
    design: Mapping[str, Any],
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit and migrate checkpoints produced before co-arrival support.

    Every old completed case necessarily passed the former eight-distinct-key
    guard, so the new factor follows the identical one-to-one key path for all
    stored results.  Only the failed and future co-arrival cases use the new
    exact Jacobian accumulation path.
    """

    old_configuration = design.get("configuration")
    if not isinstance(old_configuration, Mapping):
        raise ValueError("existing frozen design has no scientific configuration")
    if _configuration_before_event_coalescence_policy(old_configuration) != (
        _configuration_before_event_coalescence_policy(configuration)
    ):
        raise ValueError(
            "checkpoint differs by more than the physical co-arrival fix"
        )
    schedules = build_all_paper_directional_schedules(
        follower_ids=DEFAULT_FOLLOWER_IDS,
        sector_angles_rad=_angles(),
    )
    encoded_schedules = [asdict(item) for item in schedules]
    if _canonical(encoded_schedules) != _canonical(design.get("schedules")):
        raise ValueError("checkpoint schedules changed; migration is forbidden")

    old_sources = dict(old_configuration.get("source_sha256", {}))
    current_sources = dict(configuration.get("source_sha256", {}))
    changed_sources = {
        source
        for source in set(old_sources) | set(current_sources)
        if old_sources.get(source) != current_sources.get(source)
    }
    if not changed_sources or not changed_sources <= (
        _EVENT_COALESCENCE_CHANGED_SOURCES
    ):
        raise ValueError(
            "source changes are not limited to the reviewed co-arrival fix: "
            f"{sorted(changed_sources)}"
        )

    checkpoint_path = output / "checkpoint.sqlite3"
    if not checkpoint_path.exists():
        raise ValueError("cannot migrate a missing checkpoint database")
    old_fingerprint = str(design.get("fingerprint", ""))
    new_fingerprint = scientific_fingerprint(configuration, encoded_schedules)
    with sqlite3.connect(str(checkpoint_path), timeout=30.0) as connection:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        if metadata.get("schema") != SCHEMA:
            raise ValueError("checkpoint schema is not eligible for migration")
        if metadata.get("candidate_ids") != _canonical(
            list(DIRECTIONAL_JOINT_CANDIDATE_IDS)
        ):
            raise ValueError("checkpoint candidates are not eligible for migration")
        if metadata.get("fingerprint") not in {
            old_fingerprint,
            new_fingerprint,
        }:
            raise ValueError("checkpoint fingerprint has an unknown lineage")
        completed_case_count = int(
            connection.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
        )
        latest_failure = connection.execute(
            "SELECT context FROM failures ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if completed_case_count and (
            latest_failure is None
            or "coincident Joint events" not in str(latest_failure[0])
        ):
            raise ValueError(
                "migration requires the audited coincident-event failure"
            )
        with connection:
            connection.execute(
                "UPDATE metadata SET value=? WHERE key='fingerprint'",
                (new_fingerprint,),
            )

    backup = output / (
        "frozen_design.before_" + EVENT_COALESCENCE_MIGRATION_ID + ".json"
    )
    if not backup.exists():
        _write_json(backup, design)
    migrated = json.loads(_canonical(design))
    migrated["configuration"] = configuration
    migrated["schedules"] = encoded_schedules
    migrated["fingerprint"] = new_fingerprint
    history = list(migrated.get("compatible_migrations", []))
    history.append(
        {
            "migration_id": EVENT_COALESCENCE_MIGRATION_ID,
            "migrated_at": _utc_now(),
            "old_fingerprint": old_fingerprint,
            "new_fingerprint": new_fingerprint,
            "completed_case_count": completed_case_count,
            "changed_sources": sorted(changed_sources),
            "stored_cases_coarrival_count": 0,
            "justification": (
                "the former generator rejected every co-arrival before a case "
                "could be transactionally committed"
            ),
        }
    )
    migrated["compatible_migrations"] = history
    _write_json(output / "frozen_design.json", migrated)
    _write_json(output / "event_coalescence_migration.json", history[-1])
    return migrated


def freeze_design(
    output: Path,
    *,
    allow_event_coalescence_migration: bool = False,
) -> dict[str, Any]:
    configuration = _configuration()
    freeze_path = output / "frozen_design.json"
    if freeze_path.exists():
        design = json.loads(freeze_path.read_text(encoding="utf-8"))
        expected = scientific_fingerprint(configuration, design["schedules"])
        if design.get("fingerprint") != expected:
            if not allow_event_coalescence_migration:
                raise ValueError(
                    "scientific configuration/source changed; use a new output "
                    "directory or pass --allow-event-coalescence-migration for "
                    "the reviewed co-arrival-only checkpoint migration"
                )
            design = _migrate_event_coalescence_checkpoint(
                output, design, configuration
            )
        schedules_from_frozen(design)
        return design
    schedules = build_all_paper_directional_schedules(
        follower_ids=DEFAULT_FOLLOWER_IDS,
        sector_angles_rad=_angles(),
    )
    encoded = [asdict(item) for item in schedules]
    print(
        "[directional-joint] freezing five cyclic schedules and nominal ranking...",
        flush=True,
    )
    agent_rows, ranking = _nominal_predictions(schedules)
    design = {
        "configuration": configuration,
        "schedules": encoded,
        "dncp_agent_rows": agent_rows,
        "dncp_ranking_rows": ranking,
        "fingerprint": scientific_fingerprint(configuration, encoded),
        "created_at": _utc_now(),
    }
    _write_json(freeze_path, design)
    return design


def _worker_init(event_queue: Any) -> None:
    global _EVENT_QUEUE
    _EVENT_QUEUE = event_queue
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")


def _abort_process_pool(
    executor: ProcessPoolExecutor,
    futures: Mapping[Any, int],
) -> None:
    """Stop failed Windows workers without waiting on unfinished seed cases."""

    for future in futures:
        future.cancel()
    processes = list((getattr(executor, "_processes", {}) or {}).values())
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=1.0)
    for process in processes:
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(timeout=1.0)
    executor.shutdown(wait=False, cancel_futures=True)


def _case_task(seed: int, design: Mapping[str, Any]) -> dict[str, Any]:
    from .formal_schedule_simulation import generate_schedule_case, run_schedule_method

    started = time.perf_counter()
    stage = "data_generation"

    def notify(detail: Any = None, **extra: Any) -> None:
        if _EVENT_QUEUE is not None:
            _EVENT_QUEUE.put(
                {
                    "seed": seed,
                    "stage": stage,
                    "detail": "" if detail is None else str(detail),
                    "elapsed_wall_s": time.perf_counter() - started,
                    **extra,
                }
            )

    try:
        schedules = schedules_from_frozen(design)
        notify("start")
        case = generate_schedule_case(seed, schedules, progress_callback=notify)
        raw_rows: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        for schedule in schedules:
            stage = schedule.candidate_id
            notify("start")
            method_started = time.perf_counter()
            rows, diagnostic = run_schedule_method(case, stage)
            for row in rows:
                row.update(
                    seed=int(seed), method=stage, candidate_id=stage,
                    orientation_deg=15.0,
                )
            diagnostic.update(
                seed=int(seed), method=stage, candidate_id=stage,
                elapsed_wall_s=time.perf_counter() - method_started,
            )
            raw_rows.extend(rows)
            diagnostics.append(diagnostic)
            notify("complete", estimator_complete=True)
        result = {
            "seed": int(seed),
            "orientation_deg": 15.0,
            "raw_rows": raw_rows,
            "run_diagnostics": diagnostics,
            "protocol_agent_audit": case.audit_rows,
            "path_parameters": case.path_parameters,
            "generation_wall_s": case.generation_wall_s,
            "shared_data_audit": case.shared_data_audit,
            "elapsed_wall_s": time.perf_counter() - started,
        }
        validate_case_payload(result, DIRECTIONAL_JOINT_CANDIDATE_IDS)
        return {"ok": True, "payload": result}
    except Exception as exc:
        return {
            "ok": False,
            "seed": int(seed),
            "candidate_or_stage": stage,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_wall_s": time.perf_counter() - started,
        }


def run_directional_joint_experiment(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    start_seed: int = 1,
    seed_count: int = 1000,
    workers: int = 12,
    allow_event_coalescence_migration: bool = False,
) -> dict[str, Any]:
    start_seed = int(start_seed)
    seed_count = int(seed_count)
    workers = int(workers)
    if start_seed < 1 or seed_count < 1 or not 1 <= workers <= 12:
        raise ValueError(
            "positive start-seed/seed-count and workers in [1,12] are required"
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_started = time.perf_counter()
    design = freeze_design(
        output,
        allow_event_coalescence_migration=allow_event_coalescence_migration,
    )
    seeds = tuple(range(start_seed, start_seed + seed_count))
    stage_dir = output / "stages" / f"seed_{seeds[0]}_{seeds[-1]}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    candidate_count = len(DIRECTIONAL_JOINT_CANDIDATE_IDS)
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "start_seed": seeds[0],
        "end_seed": seeds[-1],
        "seed_count": seed_count,
        "workers": workers,
        "candidate_ids": list(DIRECTIONAL_JOINT_CANDIDATE_IDS),
        "estimator_runs": seed_count * candidate_count,
        "fingerprint": design["fingerprint"],
        "started_at": _utc_now(),
        "simulation_only": True,
        "complete": False,
    }
    _write_json(stage_dir / "run_manifest.json", manifest)
    print(
        f"[directional-joint] seeds={seeds[0]}..{seeds[-1]}, workers={workers}, "
        f"estimator_runs={seed_count * candidate_count}, five schedules, "
        "full 200 s / 200 waves per schedule",
        flush=True,
    )
    with OfflineScheduleCheckpoint(
        output / "checkpoint.sqlite3",
        design["fingerprint"],
        DIRECTIONAL_JOINT_CANDIDATE_IDS,
        schema=SCHEMA,
    ) as checkpoint:
        completed = checkpoint.completed_seeds()
        reused = sorted(set(seeds) & completed)
        for seed in reused:
            checkpoint.load_case(seed)
        pending = [seed for seed in seeds if seed not in completed]
        manifest["reused_seed_count"] = len(reused)
        case_times: list[float] = []
        task_started = time.perf_counter()
        stages: dict[int, str] = {}
        estimator_complete = len(reused) * candidate_count
        failure: dict[str, Any] | None = None
        if pending:
            context = multiprocessing.get_context("spawn")
            with context.Manager() as manager:
                event_queue = manager.Queue()
                executor = ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=context,
                    initializer=_worker_init,
                    initargs=(event_queue,),
                )
                aborted = False
                futures: dict[Any, int] = {}
                iterator = iter(pending)

                def fill_queue() -> None:
                    while len(futures) < 2 * workers:
                        seed = next(iterator, None)
                        if seed is None:
                            return
                        futures[executor.submit(_case_task, seed, design)] = seed

                fill_queue()
                bar = tqdm(
                    total=seed_count,
                    initial=len(reused),
                    desc="directional-joint-phi15",
                    unit="seed",
                    dynamic_ncols=True,
                )
                try:
                    while futures:
                        done, _ = wait(
                            tuple(futures), timeout=1.0,
                            return_when=FIRST_COMPLETED,
                        )
                        while True:
                            try:
                                event = event_queue.get_nowait()
                            except Empty:
                                break
                            stages[int(event["seed"])] = str(event["stage"])
                            estimator_complete += int(
                                event.get("estimator_complete", False)
                            )
                        for future in done:
                            seed = futures.pop(future)
                            try:
                                result = future.result()
                            except Exception as exc:
                                result = {
                                    "ok": False,
                                    "seed": seed,
                                    "candidate_or_stage": stages.get(seed, "worker"),
                                    "error": str(exc),
                                    "traceback": traceback.format_exc(),
                                }
                            if not result["ok"]:
                                failure = result
                                checkpoint.record_failure(seed, result)
                                _write_json(stage_dir / "last_failure.json", result)
                                break
                            payload = result["payload"]
                            checkpoint.save_case(payload)
                            case_times.append(float(payload["elapsed_wall_s"]))
                            stages.pop(seed, None)
                            bar.update(1)
                        elapsed = time.perf_counter() - task_started
                        active = ";".join(
                            f"{seed}:{name}"
                            for seed, name in sorted(stages.items())[:3]
                        )
                        postfix = (
                            f"runs={estimator_complete}/{seed_count * candidate_count} "
                            f"elapsed={elapsed:.0f}s"
                        )
                        if case_times:
                            postfix += (
                                f" mean_case={sum(case_times) / len(case_times):.1f}s"
                            )
                        if active:
                            postfix += f" {active}"
                        bar.set_postfix_str(postfix)
                        bar.refresh()
                        if failure:
                            break
                        fill_queue()
                except BaseException:
                    _abort_process_pool(executor, futures)
                    aborted = True
                    raise
                finally:
                    if failure and not aborted:
                        _abort_process_pool(executor, futures)
                        aborted = True
                    if not aborted:
                        executor.shutdown(wait=True, cancel_futures=True)
                    bar.close()
        if failure:
            manifest.update(
                failed=True,
                failure=failure,
                elapsed_wall_s=time.perf_counter() - run_started,
            )
            _write_json(stage_dir / "run_manifest.json", manifest)
            raise RuntimeError(
                f"directional schedule seed {failure['seed']} failed at "
                f"{failure.get('candidate_or_stage')}: {failure.get('error')}"
            )
        if not set(seeds) <= checkpoint.completed_seeds():
            raise RuntimeError("requested range is incomplete")
        timings = [
            float(
                checkpoint.connection.execute(
                    "SELECT elapsed_wall_s FROM cases WHERE seed=?", (seed,)
                ).fetchone()[0]
            )
            for seed in seeds
        ]
        manifest.update(
            simulation_complete=True,
            new_seed_count=len(pending),
            average_case_wall_s=sum(timings) / len(timings),
            simulation_wall_s=time.perf_counter() - task_started,
        )
    manifest.update(
        complete=True,
        completed_at=_utc_now(),
        elapsed_wall_s=time.perf_counter() - run_started,
    )
    _write_json(stage_dir / "run_manifest.json", manifest)
    print(
        f"[directional-joint] simulation complete, "
        f"elapsed={manifest['elapsed_wall_s']:.1f}s, "
        f"mean_case={manifest['average_case_wall_s']:.1f}s, "
        f"output={stage_dir}",
        flush=True,
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--seed-count", type=int, default=None)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("smoke", "formal"), default="formal")
    parser.add_argument(
        "--allow-event-coalescence-migration",
        action="store_true",
        help=(
            "audit and migrate a checkpoint created before exact simultaneous-"
            "arrival Pose2 key coalescence was supported"
        ),
    )
    args = parser.parse_args(argv)
    run_directional_joint_experiment(
        args.output,
        start_seed=args.start_seed,
        seed_count=(args.seed_count if args.seed_count is not None
                    else (1 if args.mode == "smoke" else 1000)),
        workers=args.workers,
        allow_event_coalescence_migration=(
            args.allow_event_coalescence_migration
        ),
    )


if __name__ == "__main__":
    main()
