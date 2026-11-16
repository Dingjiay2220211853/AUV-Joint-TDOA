"""Seed-paired analysis of the fixed six-schedule paper experiment.

This module deliberately imports no simulator/GTSAM modules.  It can be run
with the SB3 Python environment against a read-only experiment checkpoint.
Bootstrap resamples whole seeds, shared by all candidates and followers.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
import zlib
from contextlib import closing
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


CANDIDATES = ("D100", "J66-RR", "H42-U", "H42-D", "H42-DW", "H24-DW")
AGENTS = ("C", "D", "E", "F", "G", "H")
TIMES = np.arange(1.0, 201.0)
PAIRS = tuple((candidate, "D100") for candidate in CANDIDATES[1:]) + (
    ("H42-D", "H42-U"), ("H42-DW", "H42-D"), ("H24-DW", "H42-DW"),
)
COVARIANCES = ("p_perpendicular_m2", "p_parallel_m2", "trace_p_xy_m2", "lambda_max_p_xy_m2")
ERRORS = ("perpendicular", "parallel", "xy")
CONSISTENCY = ("nees_perpendicular", "nees_xy", "nees_se2", "covered_perpendicular_95", "covered_xy_95", "covered_se2_95")
RAW_FIELDS = COVARIANCES + tuple(f"squared_error_{x}_m2" for x in ERRORS) + CONSISTENCY


def _auc(values: np.ndarray, times: np.ndarray = TIMES) -> np.ndarray:
    select = (times >= 20.0) & (times <= 200.0)
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return integrate(values[..., select], times[select], axis=-1) / 180.0


def _slope(values: np.ndarray) -> np.ndarray:
    selected = TIMES >= 150.0
    centered = TIMES[selected] - TIMES[selected].mean()
    return np.sum(values[..., selected] * centered, axis=-1) / np.sum(centered**2)


def _quantiles(values: np.ndarray) -> tuple[float, float]:
    return tuple(float(x) for x in np.quantile(values, (0.025, 0.975)))


def _safe_ratio(numerator: np.ndarray | float, denominator: np.ndarray | float) -> np.ndarray:
    return np.divide(numerator, denominator, out=np.full(np.broadcast_shapes(np.shape(numerator), np.shape(denominator)), np.nan), where=np.asarray(denominator) > 0)


def validate_case(payload: dict[str, Any], seed: int) -> pd.DataFrame:
    """Require every candidate/follower/second and reject duplicate/nonfinite data."""
    if int(payload.get("seed", -1)) != seed:
        raise ValueError(f"checkpoint seed {seed} has mismatching payload seed")
    frame = pd.DataFrame(payload["raw_rows"])
    required = {"seed", "method", "agent_id", "time_s", *RAW_FIELDS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"seed {seed}: missing metric fields {missing}")
    if not frame["seed"].eq(seed).all():
        raise ValueError(f"seed {seed}: mixed raw-row seeds")
    if set(frame["method"]) != set(CANDIDATES) or set(frame["agent_id"]) != set(AGENTS):
        raise ValueError(f"seed {seed}: expected the six fixed candidates and six followers")
    # Optional t=0 is permitted; all other rows must lie on the frozen grid.
    frame = frame[frame.time_s != 0.0].copy()
    if frame.duplicated(["method", "agent_id", "time_s"]).any():
        raise ValueError(f"seed {seed}: duplicate candidate/follower/time row")
    actual = pd.MultiIndex.from_frame(frame[["method", "agent_id", "time_s"]])
    expected = pd.MultiIndex.from_product((CANDIDATES, AGENTS, TIMES), names=actual.names)
    if len(actual) != len(expected) or len(expected.difference(actual)) or len(actual.difference(expected)):
        raise ValueError(f"seed {seed}: incomplete 1..200 s six-candidate/six-follower panel")
    if not np.isfinite(frame[list(RAW_FIELDS)].to_numpy(float)).all():
        raise ValueError(f"seed {seed}: nonfinite metrics")
    if (frame[list(COVARIANCES) + [f"squared_error_{x}_m2" for x in ERRORS]] < 0).any().any():
        raise ValueError(f"seed {seed}: negative covariance or squared error")
    return frame.set_index(["method", "agent_id", "time_s"]).reindex(expected).reset_index()


def case_agent_metrics(frame: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return [candidate,agent] summaries and [candidate,agent,time] curves."""
    curves = {name: frame[name].to_numpy(float).reshape(6, 6, 200) for name in RAW_FIELDS}
    result: dict[str, np.ndarray] = {}
    for name in COVARIANCES:
        result[f"auc_{name}"] = _auc(curves[name])
        result[f"final_{name}"] = curves[name][..., -1]
        result[f"slope_{name}_per_s"] = _slope(curves[name])
    for direction in ERRORS:
        errors = curves[f"squared_error_{direction}_m2"]
        result[f"mse_{direction}_m2"] = _auc(errors)
        result[f"final_squared_error_{direction}_m2"] = errors[..., -1]
        result[f"peak_error_{direction}_m"] = np.sqrt(errors.max(axis=-1))
    for name in CONSISTENCY:
        result[name] = curves[name][..., TIMES >= 20.0].mean(axis=-1)
    return result, curves


def bootstrap_agent_means(agent_values: np.ndarray, samples: int, random_seed: int = 1729) -> np.ndarray:
    """Whole-seed means; one resampling draw is reused for every trailing field."""
    if samples < 1 or len(agent_values) < 1:
        raise ValueError("bootstrap needs positive samples and a nonempty seed set")
    rng = np.random.default_rng(random_seed)
    n = len(agent_values)
    flat = agent_values.reshape(n, -1)
    result = np.empty((samples,) + agent_values.shape[1:])
    for start in range(0, samples, 128):
        count = min(128, samples - start)
        indices = rng.integers(0, n, size=(count, n))
        weights = np.zeros((count, n))
        np.add.at(weights, (np.arange(count)[:, None], indices), 1.0 / n)
        # Avoid a platform BLAS dependency: the simulation's KalmanNet runtime
        # has a broken delayed BLAS DLL, while this reduction needs no BLAS.
        result[start:start + count] = np.einsum(
            'bn,nf->bf', weights, flat, optimize=False
        ).reshape((count,) + agent_values.shape[1:])
    return result


def _scoped(values: np.ndarray, scope: str, metric: str) -> np.ndarray:
    if scope == "fleet_mean":
        result = values.mean(axis=-1)
    elif scope == "worst_individual":
        result = values.min(axis=-1) if metric.startswith("covered_") else values.max(axis=-1)
    else:
        raise ValueError(scope)
    return np.sqrt(np.maximum(result, 0)) if metric.startswith("rmse_") else result


def _kendall_many(predicted: np.ndarray, observed: np.ndarray) -> np.ndarray:
    left, right = np.triu_indices(len(predicted), 1)
    p = np.sign(predicted[left] - predicted[right])
    o = np.sign(observed[..., left] - observed[..., right])
    denominator = np.sqrt(np.sum(p != 0) * np.sum(o != 0, axis=-1))
    return np.divide(np.sum(p * o, axis=-1), denominator, out=np.zeros_like(denominator, dtype=float), where=denominator > 0)


def score_aggregates(agent_means: np.ndarray, lambda_c: float) -> np.ndarray:
    """Max is AFTER seed averaging; agent_means ends in [candidate,agent]."""
    fleet, worst = agent_means.mean(axis=-1), agent_means.max(axis=-1)
    return lambda_c * _safe_ratio(worst, worst[..., :1]) + (1 - lambda_c) * _safe_ratio(fleet, fleet[..., :1])


def _table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "No rows."
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame[list(columns)].itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{x:.6g}" if isinstance(x, (float, np.floating)) else str(x) for x in row) + " |")
    return "\n".join(lines)


def _checkpoint_path(root: Path) -> Path:
    for name in ("checkpoint.sqlite3", "checkpoint.sqlite", "results.sqlite"):
        if (root / name).is_file():
            return root / name
    raise FileNotFoundError(f"no supported checkpoint in {root}")


def _load_cases(root: Path, seeds: Sequence[int]):
    database = _checkpoint_path(root)
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
        available = {int(row[0]) for row in connection.execute("SELECT seed FROM cases")}
        missing = sorted(set(seeds) - available)
        if missing:
            raise ValueError(f"incomplete requested seed range; missing {missing[:20]}")
        for seed in seeds:
            row = connection.execute("SELECT payload FROM cases WHERE seed=?", (seed,)).fetchone()
            try:
                payload = json.loads(zlib.decompress(bytes(row[0])).decode("utf-8"))
            except (ValueError, zlib.error, TypeError) as exc:
                raise ValueError(f"corrupt checkpoint payload for seed {seed}") from exc
            yield seed, payload


def analyze_offline_schedule_results(
    output_dir: str | Path, *, start_seed: int = 1, seed_count: int = 200,
    lambda_c_values: Sequence[float] = (1.0, 0.5, 0.0), bootstrap_samples: int = 10000,
    make_plots: bool = True,
) -> dict[str, Any]:
    """Create an auditable stage report without loading all raw rows at once."""
    started = time.perf_counter()
    if start_seed < 1 or seed_count < 1 or not lambda_c_values or any(not 0 <= x <= 1 for x in lambda_c_values):
        raise ValueError("invalid seed range or lambda_c")
    root = Path(output_dir).resolve()
    frozen = json.loads((root / "frozen_design.json").read_text(encoding="utf-8"))
    seeds = list(range(start_seed, start_seed + seed_count))
    metrics_by_seed: list[dict[str, np.ndarray]] = []
    instant_by_seed: list[dict[str, np.ndarray]] = []
    curve_sum: dict[str, np.ndarray] = {}
    case_rows, parameter_rows, audit_rows, diagnostics = [], [], [], []
    representative: pd.DataFrame | None = None
    for seed, payload in _load_cases(root, seeds):
        frame = validate_case(payload, seed)
        metrics, curves = case_agent_metrics(frame)
        metrics_by_seed.append(metrics)
        instant = {}
        for name in COVARIANCES:
            instant[f"auc_{name}"] = _auc(curves[name].max(axis=1))
        for direction in ERRORS:
            instant[f"mse_{direction}_m2"] = _auc(curves[f"squared_error_{direction}_m2"].max(axis=1))
        instant_by_seed.append(instant)
        for name, value in curves.items():
            curve_sum[name] = curve_sum.get(name, np.zeros_like(value)) + value
        for method_index, candidate in enumerate(CANDIDATES):
            for agent_index, agent in enumerate(AGENTS):
                case_rows.append({"seed": seed, "candidate_id": candidate, "agent_id": agent, **{name: float(v[method_index, agent_index]) for name, v in metrics.items()}})
        for key, sink in (("path_parameters", parameter_rows), ("protocol_agent_audit", audit_rows), ("run_diagnostics", diagnostics)):
            sink.extend({"seed": seed, **row} for row in payload.get(key, []))
        if representative is None or seed == 1:
            representative = frame[frame.method == "D100"].copy()
    # Only validated complete panels may create a stage output.
    stage = root / f"analysis_seed_{start_seed:04d}_{seeds[-1]:04d}"
    stage.mkdir(parents=True, exist_ok=True)
    names = list(metrics_by_seed[0])
    tensor = np.stack([np.stack([row[name] for name in names], axis=-1) for row in metrics_by_seed])
    means, boot = tensor.mean(axis=0), bootstrap_agent_means(tensor, bootstrap_samples)
    summary_rows, agent_rows, compare_rows, seed_summary_rows = [], [], [], []
    absolute_boot: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for metric_index, name in enumerate(names):
        display_names = [name]
        if name.startswith("mse_"):
            display_names.append(name.replace("mse_", "rmse_", 1).replace("_m2", "_m"))
        for display in display_names:
            is_rmse = display.startswith("rmse_")
            for agent_index, agent in enumerate(AGENTS):
                vals = means[:, agent_index, metric_index]
                draws = boot[:, :, agent_index, metric_index]
                if is_rmse:
                    vals, draws = np.sqrt(vals), np.sqrt(draws)
                for ci, candidate in enumerate(CANDIDATES):
                    lo, hi = _quantiles(draws[:, ci])
                    agent_rows.append(dict(candidate_id=candidate, agent_id=agent, metric=display, estimate=vals[ci], ci_lower=lo, ci_upper=hi))
            for scope in ("fleet_mean", "worst_individual"):
                points = _scoped(means[..., metric_index], scope, display)
                draws = _scoped(boot[..., metric_index], scope, display)
                seed_vals = _scoped(tensor[..., metric_index], scope, display)
                absolute_boot[(display, scope)] = (points, draws, seed_vals)
            # Deliberately distinct: E_seed[max_i A_i] vs max_i E_seed[A_i].
            seed_max = (tensor[..., metric_index].min(axis=-1) if name.startswith("covered_")
                        else tensor[..., metric_index].max(axis=-1))
            draw_max = bootstrap_agent_means(seed_max, bootstrap_samples)
            points_max, seed_vals = seed_max.mean(axis=0), seed_max
            if is_rmse:
                points_max, draw_max, seed_vals = np.sqrt(points_max), np.sqrt(draw_max), np.sqrt(seed_vals)
            absolute_boot[(display, "mean_seed_worst")] = (points_max, draw_max, seed_vals)
            if name in instant_by_seed[0]:
                instant_vals = np.stack([row[name] for row in instant_by_seed])
                instant_draws = bootstrap_agent_means(instant_vals, bootstrap_samples)
                instant_mean = instant_vals.mean(axis=0)
                if is_rmse:
                    instant_mean, instant_draws, instant_vals = np.sqrt(instant_mean), np.sqrt(instant_draws), np.sqrt(instant_vals)
                absolute_boot[(display, "instant_worst")] = (instant_mean, instant_draws, instant_vals)
    for (metric, scope), (points, draws, seed_values) in absolute_boot.items():
        for ci, candidate in enumerate(CANDIDATES):
            lo, hi = _quantiles(draws[:, ci])
            summary_rows.append(dict(candidate_id=candidate, metric=metric, scope=scope, estimate=float(points[ci]), ci_lower=lo, ci_upper=hi))
            for si, seed in enumerate(seeds):
                seed_summary_rows.append(dict(seed=seed, candidate_id=candidate, metric=metric, scope=scope, value=float(seed_values[si, ci])))
        if metric.startswith(("nees_", "covered_")):
            continue
        signed = metric.startswith("slope_")
        for candidate, reference in PAIRS:
            ci, ri = CANDIDATES.index(candidate), CANDIDATES.index(reference)
            values = draws[:, ci] - draws[:, ri] if signed else _safe_ratio(draws[:, ci], draws[:, ri])
            point = points[ci] - points[ri] if signed else float(_safe_ratio(points[ci], points[ri]))
            lo, hi = _quantiles(values)
            compare_rows.append(dict(candidate_id=candidate, reference_id=reference, metric=metric, scope=scope, contrast="difference" if signed else "ratio", estimate=point, ci_lower=lo, ci_upper=hi, lower_seed_fraction=float(np.mean(seed_values[:, ci] < seed_values[:, ri]))))
    summary, comparisons = pd.DataFrame(summary_rows), pd.DataFrame(compare_rows)
    dncp = pd.DataFrame(frozen["dncp_agent_rows"])
    nominal_index = pd.MultiIndex.from_product((CANDIDATES, AGENTS), names=["candidate_id", "agent_id"])
    if dncp.duplicated(["candidate_id", "agent_id"]).any():
        raise ValueError("duplicate DNCP candidate/agent rows")
    nominal = dncp.set_index(["candidate_id", "agent_id"]).reindex(nominal_index)["auc_perpendicular_m2"].to_numpy(float).reshape(6, 6)
    if not np.isfinite(nominal).all() or np.any(nominal <= 0):
        raise ValueError("frozen design has missing or nonpositive DNCP AUCs")
    score_rows, ranking_rows, score_comparisons = [], [], []
    for weight in lambda_c_values:
        prediction = score_aggregates(nominal, weight)
        for ci, candidate in enumerate(CANDIDATES):
            rank = int(np.where(np.argsort(prediction, kind="stable") == ci)[0][0] + 1)
            score_rows.append(dict(lambda_c=weight, source="dncp", candidate_id=candidate, score=float(prediction[ci]), rank=rank, ci_lower=float(prediction[ci]), ci_upper=float(prediction[ci])))
        for source, metric in (("mc_covariance", "auc_p_perpendicular_m2"), ("mc_squared_error", "mse_perpendicular_m2")):
            mi = names.index(metric)
            scores, score_draws = score_aggregates(means[..., mi], weight), score_aggregates(boot[..., mi], weight)
            choice = int(np.argmin(prediction))
            best = int(np.argmin(scores))
            regret = scores[choice] - scores[best]
            regret_draws = score_draws[:, choice] - score_draws.min(axis=1)
            tau = float(_kendall_many(prediction, scores))
            tau_lo, tau_hi = _quantiles(_kendall_many(prediction, score_draws))
            regret_lo, regret_hi = _quantiles(regret_draws)
            ranking_rows.append(dict(lambda_c=weight, source=source, kendall_tau_b=tau, tau_ci_lower=tau_lo, tau_ci_upper=tau_hi, predicted_choice=CANDIDATES[choice], empirical_choice=CANDIDATES[best], selected_score=float(scores[choice]), best_score=float(scores[best]), selection_regret=float(regret), regret_ci_lower=regret_lo, regret_ci_upper=regret_hi, selected_is_best_bootstrap_fraction=float(np.mean(np.argmin(score_draws, axis=1) == choice))))
            order = np.argsort(scores, kind="stable")
            for ci, candidate in enumerate(CANDIDATES):
                lo, hi = _quantiles(score_draws[:, ci])
                score_rows.append(dict(lambda_c=weight, source=source, candidate_id=candidate, score=float(scores[ci]), rank=int(np.where(order == ci)[0][0] + 1), ci_lower=lo, ci_upper=hi))
            # All candidate pairs make uncertain neighboring ranks reviewable.
            for ci in range(1, 6):
                for ri in range(ci):
                    differences = score_draws[:, ci] - score_draws[:, ri]
                    lo, hi = _quantiles(differences)
                    score_comparisons.append(dict(lambda_c=weight, source=source, candidate_id=CANDIDATES[ci], reference_id=CANDIDATES[ri], score_difference=float(scores[ci] - scores[ri]), ci_lower=lo, ci_upper=hi, lower_bootstrap_fraction=float(np.mean(differences < 0))))
    scores, ranking = pd.DataFrame(score_rows), pd.DataFrame(ranking_rows)
    consistency = summary[(summary.scope == "fleet_mean") & summary.metric.isin(CONSISTENCY)].copy()
    consistency["theoretical_target"] = consistency.metric.map({"nees_perpendicular": 1., "nees_xy": 2., "nees_se2": 3., "covered_perpendicular_95": .95, "covered_xy_95": .95, "covered_se2_95": .95})
    consistency["signed_deviation"] = consistency.estimate - consistency.theoretical_target
    time_rows = []
    for name, summed in curve_sum.items():
        averaged = summed / seed_count
        for ci, candidate in enumerate(CANDIDATES):
            for ai, agent in enumerate(AGENTS):
                time_rows.extend(dict(candidate_id=candidate, agent_id=agent, time_s=float(t), metric=name, ensemble_mean=float(v)) for t, v in zip(TIMES, averaged[ci, ai]))
    tables = {
        "offline_schedule_summary.csv": summary, "offline_schedule_agent_metrics.csv": pd.DataFrame(agent_rows),
        "offline_schedule_case_agent_metrics.csv": pd.DataFrame(case_rows), "offline_schedule_seed_metrics.csv": pd.DataFrame(seed_summary_rows),
        "offline_schedule_paired_bootstrap.csv": comparisons, "offline_schedule_scores.csv": scores,
        "offline_schedule_ranking_validation.csv": ranking, "offline_schedule_score_differences.csv": pd.DataFrame(score_comparisons),
        "offline_schedule_consistency.csv": consistency, "offline_schedule_time_metrics.csv": pd.DataFrame(time_rows),
        "offline_schedule_path_parameters.csv": pd.DataFrame(parameter_rows), "offline_schedule_protocol_audit.csv": pd.DataFrame(audit_rows),
        "offline_schedule_run_diagnostics.csv": pd.DataFrame(diagnostics),
    }
    for filename, table in tables.items():
        table.to_csv(stage / filename, index=False, encoding="utf-8-sig")
    figures = []
    if make_plots:
        figures = _write_figures(stage, frozen, representative, summary, comparisons, pd.DataFrame(agent_rows), scores, consistency, {k: v / seed_count for k, v in curve_sum.items()})
    dr_info = _dr_reference(stage, make_plots)
    headline = comparisons[(comparisons.scope.isin(("fleet_mean", "worst_individual"))) & comparisons.metric.isin(("auc_p_perpendicular_m2", "rmse_perpendicular_m"))]
    report = ["# 固定15°扇区任务：六组离线通信调度验证", "", f"Seeds {start_seed}–{seeds[-1]}；{seed_count}个共同任务，{seed_count * 6}次估计器运行；每组200波。", "", "DNCP使用固定15°径向直线代理，Monte Carlo使用随机S形扇区路径。全部候选在仿真前冻结；三个权重复用同一组估计器结果。", "", "## 统计定义", "", "AUC与MSE在20–200 s以梯形积分除以180 s计算；末期斜率为150–200 s最小二乘拟合。RMSE始终先汇总平方误差，再开方。主评分采用 max_i E_seed[A_i]；mean_seed_worst 单列 E_seed[max_i A_i]；instant_worst 为每时刻取最差艇后再积分。最差RMSE使用平方误差同样的汇总次序。", "", f"置信区间采用{bootstrap_samples:,}次整seed配对bootstrap，六候选共享每次重采样，艇间及时间相关性保留。比值为聚合量之比，不是逐seed比值的平均。NEES/coverage目标分别为1、2、3与0.95。", "", "## 主要对比", "", _table(headline, ("candidate_id", "reference_id", "metric", "scope", "estimate", "ci_lower", "ci_upper")), "", "## DNCP及实验评分排序", "", _table(scores.sort_values(["lambda_c", "source", "rank"], ascending=[False, True, True]), ("lambda_c", "source", "rank", "candidate_id", "score", "ci_lower", "ci_upper")), "", "## 排序与选中方案验证", "", _table(ranking, ("lambda_c", "source", "kendall_tau_b", "predicted_choice", "empirical_choice", "selection_regret", "regret_ci_lower", "regret_ci_upper")), "", "评分相邻候选的全部配对差值见 offline_schedule_score_differences.csv。区间包含零时仅能说明当前精度下无法区分；本报告不强制精确名次、不自动追加seed。N=200之后扩展样本属于精度增加，前200个seed不再称为独立确认数据。", "", "## 一致性", "", _table(consistency, ("candidate_id", "metric", "estimate", "theoretical_target", "ci_lower", "ci_upper")), "", "## 直航DR辅助分析", "", dr_info["description"], "", "## 图表", ""]
    for filename in figures + dr_info.get("figures", []):
        report.extend([f"![{Path(filename).stem}]({filename})", ""])
    (stage / "offline_schedule_validation_report.md").write_text("\n".join(report), encoding="utf-8")
    manifest = dict(complete=True, start_seed=start_seed, end_seed=seeds[-1], seed_count=seed_count, estimator_runs=6 * seed_count, lambda_c_values=list(lambda_c_values), bootstrap_samples=bootstrap_samples, fingerprint=frozen.get("fingerprint"), analysis_directory=str(stage), elapsed_wall_s=time.perf_counter() - started, figures=figures, dr_reference=dr_info)
    (stage / "analysis_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 180})
    return plt


def _write_figures(stage, frozen, representative, summary, comparisons, agents, scores, consistency, curves):
    plt = _plotting()
    colors = dict(zip(CANDIDATES, plt.get_cmap("tab10").colors[:6]))
    figures = []
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(stage / (name + ".png"))
        fig.savefig(stage / (name + ".svg"))
        plt.close(fig)
        figures.append(name + ".png")
    if representative is not None and {"truth_x_m", "truth_y_m"}.issubset(representative):
        from matplotlib.collections import LineCollection
        from matplotlib.colors import Normalize
        fig, ax = plt.subplots(figsize=(8, 8))
        for agent, group in representative.groupby("agent_id"):
            xy = group[["truth_x_m", "truth_y_m"]].to_numpy()
            lines = LineCollection(np.stack((xy[:-1], xy[1:]), axis=1), cmap="viridis", norm=Normalize(0, 200), linewidths=2)
            lines.set_array(group.time_s.to_numpy()[:-1])
            ax.add_collection(lines)
            ax.plot(xy[0, 0], xy[0, 1], "o", mfc="white", mec="black", ms=4)
            ax.annotate(agent, xy[-1])
        ax.scatter([0, 0], [-500, 500], marker="*", color="firebrick", s=120, label="Masters A/B")
        ax.autoscale(); ax.set_aspect("equal"); ax.set(xlabel="x (m)", ylabel="y (m)", title=f"Fixed 15 degree sectors: seed {int(representative.seed.iloc[0])}")
        fig.colorbar(lines, ax=ax, label="Time (s)", fraction=.04); ax.legend()
        save(fig, "offline_schedule_trajectory")
    schedules = frozen.get("schedules", [])
    if schedules:
        fig, axes = plt.subplots(2, 1, figsize=(12, 7))
        quotas = np.zeros((6, 6))
        for ci, schedule in enumerate(schedules):
            candidate = schedule.get("candidate_id", schedule.get("definition", {}).get("candidate_id"))
            if candidate not in CANDIDATES:
                continue
            index = CANDIDATES.index(candidate)
            for action in schedule.get("actions", []):
                joint = action["action_type"] == "joint"
                axes[0].plot(action["start_time_s"], index, "|", color="darkorange" if joint else "steelblue", markersize=12)
            quota = dict(schedule.get("relay_quotas", []))
            quotas[index] = [quota.get(agent, 0) for agent in AGENTS]
        axes[0].set(yticks=np.arange(6), yticklabels=CANDIDATES, xlabel="Action start time (s)", title="Blue: ordinary; orange: joint (shared boundaries counted once)")
        for ai, agent in enumerate(AGENTS):
            axes[1].bar(np.arange(6) + (ai - 2.5) * .12, quotas[:, ai], width=.12, label=agent)
        axes[1].set(xticks=np.arange(6), xticklabels=CANDIDATES, ylabel="Relay count"); axes[1].legend(ncol=6)
        save(fig, "offline_schedule_communications")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ci, candidate in enumerate(CANDIDATES):
        covariance = curves["p_perpendicular_m2"][ci]
        mse = curves["squared_error_perpendicular_m2"][ci]
        for ax, values in zip(axes.flat, (covariance.mean(axis=0), covariance.max(axis=0), np.sqrt(mse.mean(axis=0)), np.sqrt(mse.max(axis=0)))):
            ax.plot(TIMES, values, label=candidate, color=colors[candidate])
    for ax, title in zip(axes.flat, ("Fleet transverse covariance (m2)", "Worst ensemble-member covariance (m2)", "Fleet transverse RMSE (m)", "Worst ensemble-member transverse RMSE (m)")):
        ax.set(xlabel="Time (s)", title=title); ax.grid(alpha=.2)
    axes[0, 0].legend(ncol=2, fontsize=8)
    save(fig, "offline_schedule_time_curves")
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot = agents[agents.metric == "auc_p_perpendicular_m2"].pivot(index="candidate_id", columns="agent_id", values="estimate").reindex(index=CANDIDATES, columns=AGENTS)
    ratio = pivot.to_numpy() / pivot.loc["D100"].to_numpy()[None, :]
    heat = ax.imshow(100 * (ratio - 1), cmap="coolwarm", aspect="auto")
    for ci in range(6):
        for ai in range(6):
            ax.text(ai, ci, f"{100 * (ratio[ci, ai] - 1):+.1f}%", ha="center", va="center")
    ax.set(xticks=np.arange(6), xticklabels=AGENTS, yticks=np.arange(6), yticklabels=CANDIDATES, title="Per-follower transverse covariance AUC change from D100")
    fig.colorbar(heat, ax=ax, label="Change (%)")
    save(fig, "offline_schedule_agent_changes")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, scope in zip(axes, ("fleet_mean", "worst_individual")):
        selection = comparisons[(comparisons.metric == "auc_p_perpendicular_m2") & (comparisons.scope == scope)].reset_index(drop=True)
        ax.errorbar(selection.estimate, np.arange(len(selection)), xerr=np.maximum(np.vstack((selection.estimate - selection.ci_lower, selection.ci_upper - selection.estimate)), 0), fmt="o", capsize=3)
        ax.axvline(1, color="gray", ls="--")
        ax.set(yticks=np.arange(len(selection)), yticklabels=[f"{r.candidate_id}/{r.reference_id}" for r in selection.itertuples()], xlabel="AUC ratio, paired 95% CI", title=scope)
    save(fig, "offline_schedule_paired_comparisons")
    fig, ax = plt.subplots(figsize=(8, 6))
    for candidate in CANDIDATES:
        chosen = summary[(summary.candidate_id == candidate) & (summary.metric == "auc_p_perpendicular_m2")].set_index("scope")
        x, y = chosen.loc["fleet_mean", "estimate"], chosen.loc["worst_individual", "estimate"]
        ax.scatter(x, y, color=colors[candidate]); ax.annotate(candidate, (x, y), xytext=(4, 3), textcoords="offset points")
    ax.set(xlabel="Fleet transverse covariance AUC (m2)", ylabel="Worst-member transverse covariance AUC (m2)", title="Mean / worst-member trade-off")
    save(fig, "offline_schedule_tradeoff")
    weights = list(dict.fromkeys(scores.lambda_c))
    fig, axes = plt.subplots(1, len(weights), figsize=(5 * len(weights), 5), squeeze=False)
    for ax, weight in zip(axes.flat, weights):
        for source, group in scores[scores.lambda_c == weight].groupby("source"):
            group = group.set_index("candidate_id").reindex(CANDIDATES)
            ax.plot(np.arange(6), group["rank"], "o-", label=source)
        ax.set(xticks=np.arange(6), xticklabels=CANDIDATES, yticks=np.arange(1, 7), title=f"lambda_c={weight:g}", ylabel="Rank (1 is lowest loss)")
        ax.tick_params(axis="x", rotation=45); ax.invert_yaxis(); ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    save(fig, "offline_schedule_rankings")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, metric in zip(axes.flat, CONSISTENCY):
        chosen = consistency[consistency.metric == metric].set_index("candidate_id").reindex(CANDIDATES)
        ax.errorbar(np.arange(6), chosen.estimate, yerr=np.maximum(np.vstack((chosen.estimate - chosen.ci_lower, chosen.ci_upper - chosen.estimate)), 0), fmt="o", capsize=3)
        ax.axhline(chosen.theoretical_target.iloc[0], ls="--", color="gray")
        ax.set(xticks=np.arange(6), xticklabels=CANDIDATES, title=metric); ax.tick_params(axis="x", rotation=45)
    save(fig, "offline_schedule_consistency")
    return figures


def _dr_reference(stage: Path, make_plots: bool) -> dict[str, Any]:
    source = Path(__file__).parent / "results" / "formal_experiment_1_seed_1_60"
    if not (source / "config.json").is_file() or not (source / "time_fleet_metrics.csv").is_file():
        return dict(description="未找到既有直航DR记录；未运行新的DR实验。", figures=[])
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))["scenarios"][0]["config"]
    follower = next(a for a in config["agents"] if a["role"] == "follower")
    velocity = follower["velocity"]
    # Existing continuous model maps sample sigmas to PSD by sigma^2 * dt.
    q_lat = config["lateral_sigma_mps"] ** 2 * config["sensor_dt_s"]
    q_omega = config["gyro_sigma_radps"] ** 2 * config["sensor_dt_s"]
    times = np.arange(201.0)
    theory = pd.DataFrame(dict(time_s=times, initial_position_m2=np.full_like(times, config["follower_initial_xy_sigma_m"] ** 2), initial_heading_t2_m2=velocity**2 * config["follower_initial_heading_sigma_rad"] ** 2 * times**2, lateral_t_m2=q_lat * times, gyro_t3_m2=velocity**2 * q_omega * times**3 / 3))
    theory["total_m2"] = theory.iloc[:, 1:].sum(axis=1)
    original = pd.read_csv(source / "time_fleet_metrics.csv")
    dr = original[original.method == "dr_only"].groupby(["scenario", "time_s"])["mean_p_perpendicular_m2"].mean().reset_index()
    theory.to_csv(stage / "dr_covariance_theory_components.csv", index=False)
    dr.to_csv(stage / "dr_existing_covariance_curves.csv", index=False)
    figures = []
    if make_plots:
        plt = _plotting()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for name in theory.columns[1:-1]:
            axes[0].plot(times, theory[name], label=name)
        axes[0].plot(times, theory.total_m2, color="black", ls="--", label="total")
        for scenario, group in dr.groupby("scenario"):
            axes[1].plot(group.time_s, group.mean_p_perpendicular_m2, label=scenario)
        axes[1].plot(times, theory.total_m2, "k--", label="linearized full theory")
        for ax in axes:
            ax.set(xlabel="Time (s)", ylabel="Transverse covariance (m2)"); ax.legend(fontsize=7); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(stage / "dr_covariance_components.png", dpi=180); fig.savefig(stage / "dr_covariance_components.svg"); plt.close(fig)
        figures.append("dr_covariance_components.png")
    description = ("复用既有直航DR实验记录，完整一阶理论为 P_perp(t)=P_perp,0+v²P_heading,0 t²+q_lat t+v²q_omega t³/3。"
        f"这里 q_lat={q_lat:.9g}、q_omega={q_omega:.9g}，由旧模型的 sample_sigma²×sensor_dt 换算。"
        "图中分离初始位置、初始航向二次项、横向过程一次项及角速度噪声三次项，并与原DR协方差曲线比较。"
        "三次项存在不等于总协方差在整个200 s区间均是纯t³；该式是一阶直航传播解释，并非非线性随机估计的逐样本恒等式。")
    return dict(description=description, source_directory=str(source), figures=figures)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--seed-count", type=int, default=200)
    parser.add_argument("--lambda-c-values", default="1,0.5,0")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    result = analyze_offline_schedule_results(args.output, start_seed=args.start_seed, seed_count=args.seed_count, lambda_c_values=tuple(float(x) for x in args.lambda_c_values.split(",")), bootstrap_samples=args.bootstrap_samples, make_plots=not args.no_plots)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
