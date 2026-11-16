"""Frozen phi=15 degree, six-schedule Monte Carlo validation.

The first invocation predicts six schedules on one deterministic radial proxy.
Each Monte Carlo seed then generates one random S-sweep and one common atomic
odometry realization for all six frozen schedules.  Lambda only affects the
report, never the generated schedule or the number of estimator runs.

Run from GTSAM_v2, for example::

    python -u -m experiments.master_follower_paper.formal_experiment_5_offline_schedules \
        --start-seed 1 --seed-count 200 --workers 12

Increasing --seed-count reuses complete seeds in the same output directory.
``--mode smoke`` means one full 200-second seed (unless --seed-count is given);
it does not shorten the trajectory or change the communication budget.
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
from typing import Any, Iterable, Mapping, Sequence
import zlib

from tqdm import tqdm

from .formal_candidate_schedules import (
    DEFAULT_FOLLOWER_IDS,
    FIXED_CANDIDATE_IDS,
    FixedCandidateSchedule,
    build_all_fixed_candidate_schedules,
)


SCHEMA = "offline-fixed-schedules-phi15-v1"
DEFAULT_OUTPUT = Path(
    "experiments/master_follower_paper/results/"
    "formal_offline_schedules_phi15_beta10"
)
DEFAULT_LAMBDA_C_VALUES = (1.0, 0.5, 0.0)
_EVENT_QUEUE: Any = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, default=_json_default)


def scientific_fingerprint(configuration: Mapping[str, Any], schedules: Any) -> str:
    """Execution range, worker count and report weights are deliberately absent."""
    return hashlib.sha256(_canonical({
        "configuration": configuration, "schedules": schedules,
    }).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False, default=_json_default),
                         encoding="utf-8")
    temporary.replace(path)


def _code_hashes() -> dict[str, str]:
    """Freeze model, factor, protocol, estimator and analysis implementation."""
    root = Path(__file__).resolve().parents[2]
    relative_files = [
        "experiments/master_follower_paper/" + name for name in (
            "formal_experiment_5_offline_schedules.py",
            "formal_schedule_simulation.py", "formal_schedule_compiler.py",
            "process_offline_schedule_validation.py", "formal_dncp_ranking.py",
            "formal_candidate_schedules.py", "formal_protocol.py",
            "formal_sector_sweep_motion.py", "formal_motion.py", "model.py",
            "graphs.py", "formal_runner.py", "formal_analysis.py",
        )
    ] + [
        "estimator/distributed_graphs.py", "estimator/odometry_preintegration.py",
        "estimator/numerics.py", "factors/cycle_snoop_factor.py",
    ]
    # The compiler can remain in the DNCP module; whichever implementation is
    # present is hashed, while every mandatory source must exist before freeze.
    hashes = {}
    for relative in relative_files:
        source = root / relative
        if not source.exists() and relative.endswith("formal_schedule_compiler.py"):
            continue
        hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    return hashes


def _configuration() -> dict[str, Any]:
    from .formal_dncp_ranking import build_fixed_radial_sector_proxy

    proxy, _ = build_fixed_radial_sector_proxy(proxy_orientation_deg=15.0)
    return {
        "schema": SCHEMA,
        "proxy_orientation_deg": 15.0,
        "orientation_deg": 15.0,
        "lambda_w": 0.5,
        "candidate_ids": list(FIXED_CANDIDATE_IDS),
        "model": asdict(proxy),
        "trajectory": {
            "family": "structured_sector_s_sweep", "half_angle_deg": 10.0,
            "duration_s": 200.0, "speed_mps": 1.5, "path_length_m": 300.0,
            "initial_radius_m": 100.0, "amplitude_fraction_range": [0.7, 1.0],
            "first_waypoint_radial_fraction_range": [0.25, 0.35],
            "second_waypoint_radial_fraction_range": [0.65, 0.75],
            "interpolation": "piecewise_quintic_smoothstep_c2",
            "endpoint_bearing_offset_deg": 0.0, "endpoint_heading_offset_deg": 0.0,
        },
        "measurement": {
            "common_union_atomic_odometry": True,
            "common_identical_physical_event_noise": True,
            "report_times_s": list(range(1, 201)),
            "auc_start_s": 20.0, "slope_start_s": 150.0,
        },
        "source_sha256": _code_hashes(),
    }


def schedules_from_frozen(design: Mapping[str, Any]) -> tuple[FixedCandidateSchedule, ...]:
    weak = {(int(item["block_id"]), int(item["macro_index"])): float(item["score"])
            for item in design["weak_scores"]}
    angles = {agent: math.radians(15.0 + 60.0 * index)
              for index, agent in enumerate(DEFAULT_FOLLOWER_IDS)}
    schedules = build_all_fixed_candidate_schedules(
        follower_ids=DEFAULT_FOLLOWER_IDS, sector_angles_rad=angles,
        weak_window_scores=weak,
    )
    actual = [asdict(schedule) for schedule in schedules]
    if _canonical(actual) != _canonical(design["schedules"]):
        raise ValueError("frozen schedule no longer matches the fixed candidate generator")
    return schedules


def freeze_design(output: Path, lambda_c_values: Sequence[float]) -> dict[str, Any]:
    from .formal_dncp_ranking import _fixed_proxy_predictions, _rank_fixed_proxy

    configuration = _configuration()
    freeze_path = output / "frozen_design.json"
    if freeze_path.exists():
        design = json.loads(freeze_path.read_text(encoding="utf-8"))
        expected = scientific_fingerprint(configuration, design["schedules"])
        if design["fingerprint"] != expected:
            raise ValueError("scientific configuration/source changed; use a new output directory")
        schedules_from_frozen(design)
        # Requested report weights may change without rerunning covariance or MC.
        design["dncp_ranking_rows"] = _rank_fixed_proxy(
            design["dncp_agent_rows"], design["dncp_schedule_rows"], lambda_c_values,
        )
        return design
    started = time.perf_counter()
    print("[offline-schedules] freezing six candidates on deterministic phi=15 proxy...", flush=True)
    _, _, weak, agent_rows, schedule_rows = _fixed_proxy_predictions(
        proxy_orientation_deg=15.0, lambda_w=0.5, candidate_ids=FIXED_CANDIDATE_IDS,
    )
    angles = {agent: math.radians(15.0 + 60.0 * index)
              for index, agent in enumerate(DEFAULT_FOLLOWER_IDS)}
    schedules = build_all_fixed_candidate_schedules(
        sector_angles_rad=angles, weak_window_scores=weak,
    )
    encoded_schedules = [asdict(schedule) for schedule in schedules]
    design = {
        "configuration": configuration, "schedules": encoded_schedules,
        "dncp_agent_rows": agent_rows, "dncp_schedule_rows": schedule_rows,
        "dncp_ranking_rows": _rank_fixed_proxy(agent_rows, schedule_rows, lambda_c_values),
        "weak_scores": [{"block_id": key[0], "macro_index": key[1], "score": value}
                        for key, value in sorted(weak.items())],
        "fingerprint": scientific_fingerprint(configuration, encoded_schedules),
        "created_at": _utc_now(), "dncp_wall_s": time.perf_counter() - started,
    }
    _write_json(freeze_path, design)
    return design


def validate_case_payload(payload: Mapping[str, Any],
                          candidate_ids: Sequence[str] = FIXED_CANDIDATE_IDS) -> None:
    seed = int(payload["seed"])
    if seed <= 0 or float(payload["elapsed_wall_s"]) < 0:
        raise ValueError("invalid seed or elapsed time")
    required = ("raw_rows", "run_diagnostics", "protocol_agent_audit",
                "path_parameters", "shared_data_audit")
    if any(key not in payload for key in required):
        raise ValueError("case is incomplete")
    keys: set[tuple[str, str, float]] = set()
    for row in payload["raw_rows"]:
        if int(row["seed"]) != seed:
            raise ValueError("case contains another seed")
        key = (str(row["method"]), str(row["agent_id"]), float(row["time_s"]))
        if key in keys:
            raise ValueError("duplicate method/agent/time observation")
        keys.add(key)
    expected = {(str(method), agent, float(time_s)) for method in candidate_ids
                for agent in DEFAULT_FOLLOWER_IDS for time_s in range(1, 201)}
    if keys != expected:
        raise ValueError(f"case observation coverage mismatch: missing={len(expected-keys)}, extra={len(keys-expected)}")
    diagnostics = [str(row.get("candidate_id", row.get("method")))
                   for row in payload["run_diagnostics"]]
    if sorted(diagnostics) != sorted(candidate_ids):
        raise ValueError("case must contain exactly one diagnostic per schedule")


class OfflineScheduleCheckpoint:
    """Parent-owned transactional case store, readable without importing GTSAM."""

    def __init__(self, path: str | Path, fingerprint: str,
                 candidate_ids: Sequence[str] = FIXED_CANDIDATE_IDS,
                 schema: str = SCHEMA) -> None:
        self.path = Path(path)
        self.candidate_ids = tuple(candidate_ids)
        self.connection = sqlite3.connect(str(self.path))
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.connection.execute("CREATE TABLE IF NOT EXISTS cases (seed INTEGER PRIMARY KEY, payload BLOB NOT NULL, elapsed_wall_s REAL NOT NULL, completed_at TEXT NOT NULL)")
        self.connection.execute("CREATE TABLE IF NOT EXISTS failures (id INTEGER PRIMARY KEY AUTOINCREMENT, seed INTEGER NOT NULL, context TEXT NOT NULL, created_at TEXT NOT NULL)")
        metadata = dict(self.connection.execute("SELECT key,value FROM metadata"))
        expected = {"schema": str(schema), "fingerprint": str(fingerprint),
                    "candidate_ids": _canonical(list(self.candidate_ids))}
        if metadata and metadata != expected:
            self.connection.close()
            raise ValueError("checkpoint scientific configuration is incompatible")
        if not metadata:
            with self.connection:
                self.connection.executemany("INSERT INTO metadata(key,value) VALUES (?,?)", expected.items())

    def __enter__(self) -> "OfflineScheduleCheckpoint":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def completed_seeds(self) -> set[int]:
        return {int(row[0]) for row in self.connection.execute("SELECT seed FROM cases")}

    def save_case(self, payload: Mapping[str, Any]) -> None:
        validate_case_payload(payload, self.candidate_ids)
        blob = zlib.compress(_canonical(payload).encode("utf-8"), level=6)
        with self.connection:
            self.connection.execute(
                "INSERT INTO cases(seed,payload,elapsed_wall_s,completed_at) VALUES (?,?,?,?)",
                (int(payload["seed"]), blob, float(payload["elapsed_wall_s"]), _utc_now()),
            )

    def load_case(self, seed: int) -> dict[str, Any]:
        result = self.connection.execute("SELECT payload FROM cases WHERE seed=?", (int(seed),)).fetchone()
        if result is None:
            raise ValueError(f"missing complete seed {seed}")
        payload = json.loads(zlib.decompress(result[0]).decode("utf-8"))
        validate_case_payload(payload, self.candidate_ids)
        return payload

    def iter_cases(self, seeds: Iterable[int]) -> Iterable[dict[str, Any]]:
        requested = tuple(int(seed) for seed in seeds)
        if len(requested) != len(set(requested)):
            raise ValueError("duplicate requested seed")
        missing = set(requested) - self.completed_seeds()
        if missing:
            raise ValueError(f"missing complete seeds: {sorted(missing)}")
        for seed in requested:
            yield self.load_case(seed)

    def record_failure(self, seed: int, context: Mapping[str, Any]) -> None:
        with self.connection:
            self.connection.execute("INSERT INTO failures(seed,context,created_at) VALUES (?,?,?)",
                                    (int(seed), _canonical(context), _utc_now()))


def _worker_init(event_queue: Any) -> None:
    global _EVENT_QUEUE
    _EVENT_QUEUE = event_queue
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")


def _case_task(seed: int, design: Mapping[str, Any]) -> dict[str, Any]:
    from .formal_schedule_simulation import generate_schedule_case, run_schedule_method

    started = time.perf_counter()
    stage = "data_generation"

    def notify(message: Any = None, **details: Any) -> None:
        if _EVENT_QUEUE is not None:
            _EVENT_QUEUE.put({"seed": seed, "stage": stage,
                              "detail": str(message) if message is not None else "",
                              "elapsed_wall_s": time.perf_counter()-started, **details})

    try:
        schedules = schedules_from_frozen(design)
        notify("start")
        case = generate_schedule_case(seed, schedules, progress_callback=notify)
        raw_rows, diagnostics = [], []
        for schedule in schedules:
            stage = schedule.candidate_id
            notify("start")
            method_started = time.perf_counter()
            rows, diagnostic = run_schedule_method(case, stage)
            for row in rows:
                row.update(seed=int(seed), method=stage, candidate_id=stage, orientation_deg=15.0)
            diagnostic.update(seed=int(seed), method=stage, candidate_id=stage,
                              elapsed_wall_s=time.perf_counter()-method_started)
            raw_rows.extend(rows)
            diagnostics.append(diagnostic)
            notify("complete", estimator_complete=True)
        result = {
            "seed": int(seed), "orientation_deg": 15.0, "raw_rows": raw_rows,
            "run_diagnostics": diagnostics, "protocol_agent_audit": case.audit_rows,
            "path_parameters": case.path_parameters,
            "generation_wall_s": case.generation_wall_s,
            "shared_data_audit": case.shared_data_audit,
            "elapsed_wall_s": time.perf_counter()-started,
        }
        validate_case_payload(result)
        return {"ok": True, "payload": result}
    except Exception as exc:
        return {"ok": False, "seed": int(seed), "candidate_or_stage": stage,
                "error": str(exc), "traceback": traceback.format_exc(),
                "elapsed_wall_s": time.perf_counter()-started}


def _parse_lambdas(value: str | Iterable[float]) -> tuple[float, ...]:
    values = tuple(dict.fromkeys(float(item) for item in
                   (value.split(",") if isinstance(value, str) else value)))
    if not values or any(not math.isfinite(item) or not 0 <= item <= 1 for item in values):
        raise ValueError("lambda-c-values must be comma-separated numbers in [0,1]")
    return values


def run_offline_schedule_experiment(
    output_dir: str | Path = DEFAULT_OUTPUT, *, start_seed: int = 1,
    seed_count: int = 200, workers: int = 12,
    lambda_c_values: Sequence[float] = DEFAULT_LAMBDA_C_VALUES,
    run_analysis: bool = True,
) -> dict[str, Any]:
    start_seed, seed_count, workers = int(start_seed), int(seed_count), int(workers)
    if start_seed < 1 or seed_count < 1 or not 1 <= workers <= 12:
        raise ValueError("positive start-seed/seed-count and workers in [1,12] are required")
    lambdas = _parse_lambdas(lambda_c_values)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_started = time.perf_counter()
    design = freeze_design(output, lambdas)
    seeds = tuple(range(start_seed, start_seed+seed_count))
    stage_dir = output / "stages" / f"seed_{seeds[0]}_{seeds[-1]}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": SCHEMA, "start_seed": seeds[0], "end_seed": seeds[-1],
        "seed_count": seed_count, "workers": workers,
        "candidate_ids": list(FIXED_CANDIDATE_IDS), "estimator_runs": seed_count*6,
        "lambda_c_values": list(lambdas), "fingerprint": design["fingerprint"],
        "started_at": _utc_now(), "complete": False,
    }
    _write_json(stage_dir / "run_manifest.json", manifest)
    print(f"[offline-schedules] seeds={seeds[0]}..{seeds[-1]}, workers={workers}, "
          f"estimator_runs={6*seed_count}, full 200 s / 200 waves per schedule", flush=True)
    with OfflineScheduleCheckpoint(output / "checkpoint.sqlite3", design["fingerprint"]) as checkpoint:
        completed = checkpoint.completed_seeds()
        reused = sorted(set(seeds) & completed)
        # A completed marker is useful only after the actual payload is checked.
        for seed in reused:
            checkpoint.load_case(seed)
        pending = [seed for seed in seeds if seed not in completed]
        manifest["reused_seed_count"] = len(reused)
        case_times: list[float] = []
        task_started = time.perf_counter()
        stages: dict[int, str] = {}
        estimator_complete = len(reused)*6
        failure: dict[str, Any] | None = None
        if pending:
            context = multiprocessing.get_context("spawn")
            with context.Manager() as manager:
                event_queue = manager.Queue()
                executor = ProcessPoolExecutor(max_workers=workers, mp_context=context,
                                               initializer=_worker_init, initargs=(event_queue,))
                futures: dict[Any, int] = {}
                iterator = iter(pending)

                def fill_queue() -> None:
                    while len(futures) < 2*workers:
                        seed = next(iterator, None)
                        if seed is None:
                            return
                        futures[executor.submit(_case_task, seed, design)] = seed

                fill_queue()
                bar = tqdm(total=seed_count, initial=len(reused), desc="offline-phi15",
                           unit="seed", dynamic_ncols=True,
                           bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")
                try:
                    while futures:
                        done, _ = wait(tuple(futures), timeout=1.0, return_when=FIRST_COMPLETED)
                        while True:
                            try:
                                event = event_queue.get_nowait()
                            except Empty:
                                break
                            stages[int(event["seed"])] = str(event["stage"])
                            estimator_complete += int(event.get("estimator_complete", False))
                        for future in done:
                            seed = futures.pop(future)
                            try:
                                result = future.result()
                            except Exception as exc:
                                result = {"ok": False, "seed": seed,
                                          "candidate_or_stage": stages.get(seed, "worker"),
                                          "error": str(exc), "traceback": traceback.format_exc()}
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
                        elapsed = time.perf_counter()-task_started
                        active = ";".join(f"{seed}:{stage}" for seed,stage in sorted(stages.items())[:3])
                        bar.set_postfix_str(
                            f"runs={estimator_complete}/{6*seed_count} elapsed={elapsed:.0f}s "
                            f"mean_case={sum(case_times)/len(case_times):.1f}s {active}"
                            if case_times else f"runs={estimator_complete}/{6*seed_count} elapsed={elapsed:.0f}s {active}"
                        )
                        bar.refresh()
                        if failure:
                            break
                        fill_queue()
                except BaseException:
                    for future in futures:
                        future.cancel()
                    for process in (getattr(executor, "_processes", {}) or {}).values():
                        if process.is_alive():
                            process.terminate()
                    raise
                finally:
                    if failure:
                        for future in futures:
                            future.cancel()
                        # Stop already-running siblings promptly after the first
                        # failed seed; completed transactions remain recoverable.
                        for process in (getattr(executor, "_processes", {}) or {}).values():
                            if process.is_alive():
                                process.terminate()
                    executor.shutdown(wait=True, cancel_futures=True)
                    bar.close()
        if failure:
            manifest.update(failed=True, failure=failure,
                            elapsed_wall_s=time.perf_counter()-run_started)
            _write_json(stage_dir / "run_manifest.json", manifest)
            raise RuntimeError(f"offline schedule seed {failure['seed']} failed at "
                               f"{failure.get('candidate_or_stage')}: {failure.get('error')}")
        complete_cases = checkpoint.completed_seeds()
        if not set(seeds) <= complete_cases:
            raise RuntimeError("requested range is incomplete; no complete report was emitted")
        timings = [float(checkpoint.connection.execute(
            "SELECT elapsed_wall_s FROM cases WHERE seed=?", (seed,)).fetchone()[0]) for seed in seeds]
        manifest.update(
            simulation_complete=True, new_seed_count=len(pending),
            average_case_wall_s=sum(timings)/len(timings),
            simulation_wall_s=time.perf_counter()-task_started,
            elapsed_wall_s=time.perf_counter()-run_started,
        )
    if run_analysis:
        from .process_offline_schedule_validation import analyze_offline_schedule_results
        print("[offline-schedules] simulation complete; analyzing paired seeds and plots...", flush=True)
        manifest["analysis"] = analyze_offline_schedule_results(
            output, start_seed=start_seed, seed_count=seed_count,
            lambda_c_values=lambdas, bootstrap_samples=10000, make_plots=True,
        )
    manifest.update(complete=True, completed_at=_utc_now(),
                    elapsed_wall_s=time.perf_counter()-run_started)
    _write_json(stage_dir / "run_manifest.json", manifest)
    print(f"[offline-schedules] complete, elapsed={manifest['elapsed_wall_s']:.1f}s, "
          f"mean_case={manifest['average_case_wall_s']:.1f}s, output={stage_dir}", flush=True)
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--seed-count", type=int, default=None)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--lambda-c-values", default="1,0.5,0")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("smoke", "formal"), default="formal",
                        help="smoke defaults to one seed but retains the full 200 s protocol")
    parser.add_argument("--no-analysis", action="store_true")
    args = parser.parse_args(argv)
    run_offline_schedule_experiment(
        args.output, start_seed=args.start_seed,
        seed_count=args.seed_count if args.seed_count is not None else (1 if args.mode == "smoke" else 200),
        workers=args.workers, lambda_c_values=_parse_lambdas(args.lambda_c_values),
        run_analysis=not args.no_analysis,
    )


if __name__ == "__main__":
    main()
