"""Analyze the five fixed directional Joint-TDOA schedules without GTSAM.

Run this file as a script in an analysis environment without GTSAM.
Bootstrap resamples whole common seeds to preserve the within-seed coupling
between schedules and AUVs.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
import zlib
from contextlib import closing
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


CANDIDATES = ("D-only", "J-RR", "J-D1", "J-D2", "J-D3")
JOINT_CANDIDATES = CANDIDATES[1:]
AGENTS = ("C", "D", "E", "F", "G", "H")
TIMES = np.arange(1.0, 201.0)
PRIMARY_LAMBDA_C = 0.5

COVARIANCES = (
    "p_perpendicular_m2",
    "p_parallel_m2",
    "trace_p_xy_m2",
    "lambda_max_p_xy_m2",
)
ERROR_DIRECTIONS = ("perpendicular", "parallel", "xy")
CONSISTENCY = (
    "nees_perpendicular",
    "nees_xy",
    "nees_se2",
    "covered_perpendicular_95",
    "covered_xy_95",
    "covered_se2_95",
)
RAW_FIELDS = COVARIANCES + tuple(
    f"squared_error_{direction}_m2" for direction in ERROR_DIRECTIONS
) + CONSISTENCY
PRIMARY_PAIRS = (
    ("J-RR", "D-only"),
    ("J-D1", "J-RR"),
    ("J-D2", "J-RR"),
    ("J-D3", "J-RR"),
)
PAIRS = PRIMARY_PAIRS + (
    ("J-D1", "D-only"),
    ("J-D2", "D-only"),
    ("J-D3", "D-only"),
)


def _integrate(values: np.ndarray) -> np.ndarray:
    selected = TIMES >= 20.0
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return integrate(values[..., selected], TIMES[selected], axis=-1) / 180.0


def _slope(values: np.ndarray) -> np.ndarray:
    selected = TIMES >= 150.0
    centered = TIMES[selected] - TIMES[selected].mean()
    return np.sum(values[..., selected] * centered, axis=-1) / np.sum(centered**2)


def _ratio(numerator: np.ndarray | float, denominator: np.ndarray | float) -> np.ndarray:
    numerator_array = np.asarray(numerator, dtype=float)
    denominator_array = np.asarray(denominator, dtype=float)
    shape = np.broadcast_shapes(numerator_array.shape, denominator_array.shape)
    return np.divide(
        numerator_array,
        denominator_array,
        out=np.full(shape, np.nan),
        where=denominator_array > 0,
    )


def _quantiles(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, (0.025, 0.975))
    return float(low), float(high)


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "No rows."
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame[list(columns)].itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _load_cases(root: Path, seeds: Sequence[int]) -> Iterable[tuple[int, dict[str, Any]]]:
    database = root / "checkpoint.sqlite3"
    if not database.is_file():
        raise FileNotFoundError(database)
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
        available = {int(row[0]) for row in connection.execute("SELECT seed FROM cases")}
        missing = sorted(set(seeds) - available)
        if missing:
            raise ValueError(f"checkpoint is incomplete; missing seeds {missing[:20]}")
        for seed in seeds:
            row = connection.execute(
                "SELECT payload FROM cases WHERE seed=?", (seed,)
            ).fetchone()
            try:
                payload = json.loads(zlib.decompress(bytes(row[0])).decode("utf-8"))
            except (ValueError, TypeError, zlib.error) as exc:
                raise ValueError(f"seed {seed}: corrupt payload") from exc
            yield seed, payload


def _validate_case(payload: dict[str, Any], seed: int) -> pd.DataFrame:
    if int(payload.get("seed", -1)) != seed:
        raise ValueError(f"seed {seed}: payload seed mismatch")
    frame = pd.DataFrame(payload["raw_rows"])
    required = {"seed", "method", "agent_id", "time_s", *RAW_FIELDS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"seed {seed}: missing fields {missing}")
    if not frame.seed.eq(seed).all():
        raise ValueError(f"seed {seed}: mixed seeds")
    frame = frame[frame.time_s != 0.0].copy()
    if set(frame.method) != set(CANDIDATES) or set(frame.agent_id) != set(AGENTS):
        raise ValueError(f"seed {seed}: unexpected method or follower set")
    if frame.duplicated(["method", "agent_id", "time_s"]).any():
        raise ValueError(f"seed {seed}: duplicate method/follower/time row")
    expected = pd.MultiIndex.from_product(
        (CANDIDATES, AGENTS, TIMES), names=["method", "agent_id", "time_s"]
    )
    actual = pd.MultiIndex.from_frame(frame[["method", "agent_id", "time_s"]])
    if len(actual) != len(expected) or len(expected.difference(actual)):
        raise ValueError(f"seed {seed}: incomplete five-method panel")
    numeric = frame[list(RAW_FIELDS)].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"seed {seed}: nonfinite metrics")
    nonnegative = list(COVARIANCES) + [
        f"squared_error_{direction}_m2" for direction in ERROR_DIRECTIONS
    ]
    if (frame[nonnegative] < 0).any().any():
        raise ValueError(f"seed {seed}: negative loss or covariance")
    return frame.set_index(["method", "agent_id", "time_s"]).reindex(expected).reset_index()


def _case_metrics(frame: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    shape = (len(CANDIDATES), len(AGENTS), len(TIMES))
    curves = {field: frame[field].to_numpy(float).reshape(shape) for field in RAW_FIELDS}
    metrics: dict[str, np.ndarray] = {}
    for field in COVARIANCES:
        metrics[f"auc_{field}"] = _integrate(curves[field])
        metrics[f"final_{field}"] = curves[field][..., -1]
        metrics[f"slope_{field}_per_s"] = _slope(curves[field])
    for direction in ERROR_DIRECTIONS:
        squared = curves[f"squared_error_{direction}_m2"]
        metrics[f"mse_{direction}_m2"] = _integrate(squared)
        metrics[f"final_squared_error_{direction}_m2"] = squared[..., -1]
        metrics[f"peak_error_{direction}_m"] = np.sqrt(squared.max(axis=-1))
    selected = TIMES >= 20.0
    for field in CONSISTENCY:
        metrics[field] = curves[field][..., selected].mean(axis=-1)
    return metrics, curves


def _display_metric(metric: str, values: np.ndarray) -> tuple[str, np.ndarray]:
    if metric.startswith("mse_"):
        return metric.replace("mse_", "rmse_", 1).replace("_m2", "_m"), np.sqrt(
            np.maximum(values, 0.0)
        )
    return metric, values


def _scope(values: np.ndarray, scope: str, metric: str) -> np.ndarray:
    if scope == "fleet_mean":
        aggregated = values.mean(axis=-1)
    elif scope == "worst_individual":
        aggregated = values.min(axis=-1) if metric.startswith("covered_") else values.max(axis=-1)
    else:
        raise ValueError(scope)
    return aggregated


def _scores(agent_values: np.ndarray, lambda_c: float) -> np.ndarray:
    """Score joint schedules only, with J-RR as the unit reference."""
    fleet = agent_values.mean(axis=-1)
    worst = agent_values.max(axis=-1)
    return lambda_c * _ratio(worst, worst[..., :1]) + (1.0 - lambda_c) * _ratio(
        fleet, fleet[..., :1]
    )


def _kendall(predicted: np.ndarray, observed: np.ndarray) -> np.ndarray:
    left, right = np.triu_indices(len(predicted), 1)
    predicted_sign = np.sign(predicted[left] - predicted[right])
    observed_sign = np.sign(observed[..., left] - observed[..., right])
    denominator = np.sqrt(
        np.sum(predicted_sign != 0) * np.sum(observed_sign != 0, axis=-1)
    )
    return np.divide(
        np.sum(predicted_sign * observed_sign, axis=-1),
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator > 0,
    )


def _plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 180,
        }
    )
    return plt


def _write_figures(
    stage: Path,
    frozen: dict[str, Any],
    representative: pd.DataFrame,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    agent_metrics: pd.DataFrame,
    scores: pd.DataFrame,
    consistency: pd.DataFrame,
    curves: dict[str, np.ndarray],
) -> list[str]:
    plt = _plotting()
    colors = dict(zip(CANDIDATES, plt.get_cmap("tab10").colors[: len(CANDIDATES)]))
    figures: list[str] = []

    def save(fig, name: str, *, tight: bool = True) -> None:
        if tight:
            fig.tight_layout()
        fig.savefig(stage / f"{name}.png")
        fig.savefig(stage / f"{name}.svg")
        plt.close(fig)
        figures.append(f"{name}.png")

    if {"truth_x_m", "truth_y_m"}.issubset(representative.columns):
        from matplotlib.collections import LineCollection
        from matplotlib.colors import Normalize

        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        colored = None
        for agent, group in representative.groupby("agent_id", sort=False):
            xy = group[["truth_x_m", "truth_y_m"]].to_numpy()
            colored = LineCollection(
                np.stack((xy[:-1], xy[1:]), axis=1),
                cmap="viridis",
                norm=Normalize(0, 200),
                linewidths=2.2,
            )
            colored.set_array(group.time_s.to_numpy()[:-1])
            ax.add_collection(colored)
            ax.plot(*xy[0], "o", mfc="white", mec="black", ms=4)
            ax.annotate(agent, xy[-1], xytext=(3, 3), textcoords="offset points")
        ax.scatter([0, 0], [-500, 500], marker="*", color="firebrick", s=130, label="Masters A/B")
        ax.autoscale()
        ax.set_aspect("equal")
        ax.set(xlabel="x (m)", ylabel="y (m)", title="Seed 1 structured S-sector trajectories (15°)")
        if colored is not None:
            fig.colorbar(colored, ax=ax, label="Time (s)", fraction=0.04)
        ax.legend()
        save(fig, "directional_schedule_trajectory")

    schedules = frozen.get("schedules", [])
    quotas = np.zeros((len(CANDIDATES), len(AGENTS)))
    if schedules:
        fig, axes = plt.subplots(2, 1, figsize=(11, 6.8))
        for schedule in schedules:
            candidate = schedule.get("definition", {}).get("candidate_id")
            if candidate not in CANDIDATES:
                continue
            candidate_index = CANDIDATES.index(candidate)
            for action in schedule.get("actions", []):
                color = "darkorange" if action["action_type"] == "joint" else "steelblue"
                axes[0].plot(action["start_time_s"], candidate_index, "|", color=color, markersize=12)
            quota = dict(schedule.get("relay_quotas", []))
            quotas[candidate_index] = [quota.get(agent, 0) for agent in AGENTS]
        axes[0].set(
            yticks=np.arange(len(CANDIDATES)),
            yticklabels=CANDIDATES,
            xlabel="Action start time (s)",
            title="All schedules are time-uniform; blue=ordinary, orange=Joint",
        )
        for agent_index, agent in enumerate(AGENTS):
            axes[1].bar(
                np.arange(len(CANDIDATES)) + (agent_index - 2.5) * 0.12,
                quotas[:, agent_index],
                width=0.12,
                label=agent,
            )
        axes[1].set(
            xticks=np.arange(len(CANDIDATES)),
            xticklabels=CANDIDATES,
            ylabel="Relay count in 62 Joint actions",
        )
        axes[1].legend(ncol=6)
        save(fig, "directional_schedule_protocols")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for candidate_index, candidate in enumerate(CANDIDATES):
        covariance = curves["p_perpendicular_m2"][candidate_index]
        error = curves["squared_error_perpendicular_m2"][candidate_index]
        values = (
            covariance.mean(axis=0),
            covariance.max(axis=0),
            np.sqrt(error.mean(axis=0)),
            np.sqrt(error.max(axis=0)),
        )
        for ax, curve in zip(axes.flat, values):
            ax.plot(TIMES, curve, label=candidate, color=colors[candidate])
    titles = (
        "Fleet-mean transverse covariance",
        "Worst-member transverse covariance",
        "Fleet transverse RMSE",
        "Worst-member transverse RMSE",
    )
    for ax, title in zip(axes.flat, titles):
        ax.set(xlabel="Time (s)", title=title)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(ncol=2, fontsize=8)
    save(fig, "directional_schedule_time_curves")

    chosen = agent_metrics[agent_metrics.metric == "auc_p_perpendicular_m2"]
    pivot = chosen.pivot(index="candidate_id", columns="agent_id", values="estimate").reindex(
        index=CANDIDATES, columns=AGENTS
    )
    ratio = pivot.to_numpy() / pivot.loc["J-RR"].to_numpy()[None, :]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    heat = ax.imshow(100.0 * (ratio - 1.0), cmap="coolwarm", aspect="auto", vmin=-15, vmax=15)
    for candidate_index in range(len(CANDIDATES)):
        for agent_index in range(len(AGENTS)):
            ax.text(
                agent_index,
                candidate_index,
                f"{100 * (ratio[candidate_index, agent_index] - 1):+.1f}%",
                ha="center",
                va="center",
                fontsize=9,
            )
    ax.set(
        xticks=np.arange(len(AGENTS)),
        xticklabels=AGENTS,
        yticks=np.arange(len(CANDIDATES)),
        yticklabels=CANDIDATES,
        title="Per-follower transverse covariance AUC change relative to J-RR",
    )
    fig.colorbar(heat, ax=ax, label="Change (%)")
    save(fig, "directional_schedule_agent_changes")

    primary_pair_mask = np.array(
        [
            (candidate, reference) in PRIMARY_PAIRS
            for candidate, reference in zip(
                comparisons.candidate_id, comparisons.reference_id
            )
        ]
    )
    headline = comparisons[
        primary_pair_mask
        &
        (comparisons.metric.isin(("auc_p_perpendicular_m2", "rmse_perpendicular_m")))
        & (comparisons.scope.isin(("fleet_mean", "worst_individual")))
    ].reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, metric in zip(axes, ("auc_p_perpendicular_m2", "rmse_perpendicular_m")):
        selection = headline[headline.metric == metric].reset_index(drop=True)
        error = np.vstack(
            (selection.estimate - selection.ci_lower, selection.ci_upper - selection.estimate)
        )
        ax.errorbar(selection.estimate, np.arange(len(selection)), xerr=np.maximum(error, 0), fmt="o", capsize=3)
        ax.axvline(1.0, color="gray", linestyle="--")
        ax.set(
            yticks=np.arange(len(selection)),
            yticklabels=[f"{row.candidate_id}/{row.reference_id} · {row.scope}" for row in selection.itertuples()],
            xlabel="Paired ratio with 95% bootstrap CI",
            title=metric,
        )
    save(fig, "directional_schedule_primary_contrasts")

    # Direct localization performance from individual followers to fleet-level
    # aggregates.  RMSE is always obtained after averaging squared error.
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    width = 0.16
    for ax, metric, title in zip(
        axes[:2],
        ("rmse_perpendicular_m", "rmse_xy_m"),
        ("Per-follower transverse RMSE", "Per-follower XY RMSE"),
    ):
        selected = agent_metrics[agent_metrics.metric == metric]
        for candidate_index, candidate in enumerate(CANDIDATES):
            rows = selected[selected.candidate_id == candidate].set_index("agent_id").reindex(AGENTS)
            positions = np.arange(len(AGENTS)) + (candidate_index - 2) * width
            yerr = np.maximum(
                np.vstack((rows.estimate - rows.ci_lower, rows.ci_upper - rows.estimate)),
                0,
            )
            ax.bar(
                positions,
                rows.estimate,
                width=width,
                yerr=yerr,
                capsize=2,
                color=colors[candidate],
                label=candidate,
            )
        ax.set(xticks=np.arange(len(AGENTS)), xticklabels=AGENTS, ylabel="RMSE (m)", title=title)
        ax.grid(axis="y", alpha=0.2)
    aggregate_labels = (
        ("rmse_perpendicular_m", "fleet_mean", "Fleet transverse"),
        ("rmse_perpendicular_m", "worst_individual", "Worst transverse"),
        ("rmse_xy_m", "fleet_mean", "Fleet XY"),
        ("rmse_xy_m", "worst_individual", "Worst XY"),
    )
    for candidate_index, candidate in enumerate(CANDIDATES):
        rows = []
        for metric, scope, _ in aggregate_labels:
            rows.append(
                summary[
                    (summary.candidate_id == candidate)
                    & (summary.metric == metric)
                    & (summary.scope == scope)
                ].iloc[0]
            )
        positions = np.arange(len(aggregate_labels)) + (candidate_index - 2) * width
        estimates = np.array([row.estimate for row in rows])
        yerr = np.maximum(
            np.vstack(
                (
                    estimates - np.array([row.ci_lower for row in rows]),
                    np.array([row.ci_upper for row in rows]) - estimates,
                )
            ),
            0,
        )
        axes[2].bar(
            positions,
            estimates,
            width=width,
            yerr=yerr,
            capsize=2,
            color=colors[candidate],
            label=candidate,
        )
    axes[2].set(
        xticks=np.arange(len(aggregate_labels)),
        xticklabels=[label for _, _, label in aggregate_labels],
        ylabel="RMSE (m)",
        title="Fleet and worst-member RMSE",
    )
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].grid(axis="y", alpha=0.2)
    axes[0].legend(ncol=2, fontsize=8)
    save(fig, "directional_schedule_rmse_absolute")

    # The three directional candidates are compared against both J-RR and
    # D-only at every requested aggregation level.
    comparison_methods = ("J-D1", "J-D2", "J-D3")
    scopes = tuple(f"agent_{agent}" for agent in AGENTS) + (
        "fleet_mean",
        "worst_individual",
    )
    scope_labels = AGENTS + ("Fleet", "Worst")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    for row_index, metric in enumerate(("rmse_perpendicular_m", "rmse_xy_m")):
        for column_index, reference in enumerate(("J-RR", "D-only")):
            values = np.full((len(scopes), len(comparison_methods)), np.nan)
            for scope_index, scope in enumerate(scopes):
                for method_index, method in enumerate(comparison_methods):
                    matched = comparisons[
                        (comparisons.candidate_id == method)
                        & (comparisons.reference_id == reference)
                        & (comparisons.metric == metric)
                        & (comparisons.scope == scope)
                    ]
                    values[scope_index, method_index] = matched.estimate.iloc[0]
            ax = axes[row_index, column_index]
            heat = ax.imshow(
                100.0 * (values - 1.0),
                cmap="coolwarm",
                aspect="auto",
                vmin=-35,
                vmax=35,
            )
            for scope_index in range(len(scopes)):
                for method_index in range(len(comparison_methods)):
                    ax.text(
                        method_index,
                        scope_index,
                        f"{100 * (values[scope_index, method_index] - 1):+.1f}%",
                        ha="center",
                        va="center",
                        fontsize=8.5,
                    )
            ax.set(
                xticks=np.arange(len(comparison_methods)),
                xticklabels=comparison_methods,
                yticks=np.arange(len(scopes)),
                yticklabels=scope_labels,
                title=f"{metric} relative to {reference}",
            )
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.07, top=0.95, hspace=0.10, wspace=0.08)
    colorbar_axis = fig.add_axes((0.91, 0.18, 0.018, 0.64))
    fig.colorbar(heat, cax=colorbar_axis, label="RMSE change (%)")
    save(fig, "directional_schedule_rmse_relative", tight=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    for ax, lambda_c in zip(axes, (1.0, 0.5, 0.0)):
        selected = scores[scores.lambda_c == lambda_c]
        for source, group in selected.groupby("source"):
            group = group.set_index("candidate_id").reindex(JOINT_CANDIDATES)
            ax.plot(np.arange(len(JOINT_CANDIDATES)), group["rank"], "o-", label=source)
        ax.set(
            xticks=np.arange(len(JOINT_CANDIDATES)),
            xticklabels=JOINT_CANDIDATES,
            yticks=np.arange(1, len(JOINT_CANDIDATES) + 1),
            title=f"lambda_c={lambda_c:g}",
            ylabel="Rank (1 is best)",
        )
        ax.tick_params(axis="x", rotation=35)
        ax.invert_yaxis()
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    save(fig, "directional_schedule_rankings")

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    targets = {
        "nees_perpendicular": 1.0,
        "nees_xy": 2.0,
        "nees_se2": 3.0,
        "covered_perpendicular_95": 0.95,
        "covered_xy_95": 0.95,
        "covered_se2_95": 0.95,
    }
    for ax, metric in zip(axes.flat, CONSISTENCY):
        selected = consistency[consistency.metric == metric].set_index("candidate_id").reindex(CANDIDATES)
        errors = np.maximum(
            np.vstack((selected.estimate - selected.ci_lower, selected.ci_upper - selected.estimate)), 0
        )
        ax.errorbar(np.arange(len(CANDIDATES)), selected.estimate, yerr=errors, fmt="o", capsize=3)
        ax.axhline(targets[metric], color="gray", linestyle="--")
        ax.set(xticks=np.arange(len(CANDIDATES)), xticklabels=CANDIDATES, title=metric)
        ax.tick_params(axis="x", rotation=35)
    save(fig, "directional_schedule_consistency")
    return figures


def analyze(
    output_dir: str | Path,
    *,
    start_seed: int = 1,
    seed_count: int = 1000,
    bootstrap_samples: int = 10000,
    make_plots: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    root = Path(output_dir).resolve()
    seeds = list(range(start_seed, start_seed + seed_count))
    frozen = json.loads((root / "frozen_design.json").read_text(encoding="utf-8"))
    configured = tuple(frozen["configuration"]["candidate_ids"])
    if configured != CANDIDATES:
        raise ValueError(f"unexpected frozen candidates {configured}")
    if float(frozen["configuration"]["primary_lambda_c"]) != PRIMARY_LAMBDA_C:
        raise ValueError("unexpected primary lambda_c")

    metric_rows: list[np.ndarray] = []
    metric_names: list[str] | None = None
    curve_sum: dict[str, np.ndarray] = {}
    case_agent_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    representative: pd.DataFrame | None = None

    for offset, (seed, payload) in enumerate(_load_cases(root, seeds), start=1):
        frame = _validate_case(payload, seed)
        metrics, curves = _case_metrics(frame)
        if metric_names is None:
            metric_names = list(metrics)
        metric_rows.append(np.stack([metrics[name] for name in metric_names], axis=-1))
        for field, values in curves.items():
            curve_sum[field] = curve_sum.get(field, np.zeros_like(values)) + values
        for candidate_index, candidate in enumerate(CANDIDATES):
            for agent_index, agent in enumerate(AGENTS):
                case_agent_rows.append(
                    {
                        "seed": seed,
                        "candidate_id": candidate,
                        "agent_id": agent,
                        **{
                            name: float(metrics[name][candidate_index, agent_index])
                            for name in metric_names
                        },
                    }
                )
        parameter_rows.extend({"seed": seed, **row} for row in payload.get("path_parameters", []))
        audit_rows.extend({"seed": seed, **row} for row in payload.get("protocol_agent_audit", []))
        diagnostic_rows.extend({"seed": seed, **row} for row in payload.get("run_diagnostics", []))
        if representative is None:
            representative = frame[frame.method == "D-only"].copy()
        if offset % 100 == 0 or offset == seed_count:
            print(f"[analysis] validated {offset}/{seed_count} seeds", flush=True)

    assert metric_names is not None and representative is not None
    tensor = np.stack(metric_rows)  # seed, candidate, agent, metric
    means = tensor.mean(axis=0)
    rng = np.random.default_rng(1729)
    weights = rng.multinomial(
        seed_count, np.full(seed_count, 1.0 / seed_count), size=bootstrap_samples
    ).astype(np.float64) / seed_count
    boot = (weights @ tensor.reshape(seed_count, -1)).reshape(
        bootstrap_samples, len(CANDIDATES), len(AGENTS), len(metric_names)
    )

    summary_rows: list[dict[str, Any]] = []
    agent_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    scoped_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for metric_index, raw_name in enumerate(metric_names):
        is_rmse = raw_name.startswith("mse_")
        display_name, point_agent = _display_metric(raw_name, means[..., metric_index])
        _, boot_agent = _display_metric(raw_name, boot[..., metric_index])
        for candidate_index, candidate in enumerate(CANDIDATES):
            for agent_index, agent in enumerate(AGENTS):
                low, high = _quantiles(boot_agent[:, candidate_index, agent_index])
                agent_rows.append(
                    {
                        "candidate_id": candidate,
                        "agent_id": agent,
                        "metric": display_name,
                        "estimate": float(point_agent[candidate_index, agent_index]),
                        "ci_lower": low,
                        "ci_upper": high,
                    }
                )
        if not display_name.startswith(("nees_", "covered_")):
            signed_agent = display_name.startswith("slope_")
            raw_seed_agent = tensor[..., metric_index]
            if is_rmse:
                raw_seed_agent = np.sqrt(np.maximum(raw_seed_agent, 0.0))
            for agent_index, agent in enumerate(AGENTS):
                for candidate, reference in PAIRS:
                    candidate_index = CANDIDATES.index(candidate)
                    reference_index = CANDIDATES.index(reference)
                    contrast_draws = (
                        boot_agent[:, candidate_index, agent_index]
                        - boot_agent[:, reference_index, agent_index]
                        if signed_agent
                        else _ratio(
                            boot_agent[:, candidate_index, agent_index],
                            boot_agent[:, reference_index, agent_index],
                        )
                    )
                    estimate = (
                        point_agent[candidate_index, agent_index]
                        - point_agent[reference_index, agent_index]
                        if signed_agent
                        else float(
                            _ratio(
                                point_agent[candidate_index, agent_index],
                                point_agent[reference_index, agent_index],
                            )
                        )
                    )
                    low, high = _quantiles(contrast_draws)
                    comparison_rows.append(
                        {
                            "candidate_id": candidate,
                            "reference_id": reference,
                            "metric": display_name,
                            "scope": f"agent_{agent}",
                            "contrast": "difference" if signed_agent else "ratio",
                            "estimate": float(estimate),
                            "ci_lower": low,
                            "ci_upper": high,
                            "lower_seed_fraction": float(
                                np.mean(
                                    raw_seed_agent[:, candidate_index, agent_index]
                                    < raw_seed_agent[:, reference_index, agent_index]
                                )
                            ),
                        }
                    )
        for scope in ("fleet_mean", "worst_individual"):
            # RMSE is sqrt(E[e^2]), not E[sqrt(E[e^2])].  Aggregate the
            # squared-error quantity over followers before applying sqrt.
            point = _scope(means[..., metric_index], scope, raw_name)
            draws = _scope(boot[..., metric_index], scope, raw_name)
            if is_rmse:
                point = np.sqrt(np.maximum(point, 0.0))
                draws = np.sqrt(np.maximum(draws, 0.0))
            scoped_cache[(display_name, scope)] = (point, draws)
            for candidate_index, candidate in enumerate(CANDIDATES):
                low, high = _quantiles(draws[:, candidate_index])
                summary_rows.append(
                    {
                        "candidate_id": candidate,
                        "metric": display_name,
                        "scope": scope,
                        "estimate": float(point[candidate_index]),
                        "ci_lower": low,
                        "ci_upper": high,
                    }
                )
            if display_name.startswith(("nees_", "covered_")):
                continue
            signed = display_name.startswith("slope_")
            raw_seed_agent = tensor[..., metric_index]
            seed_scope = _scope(raw_seed_agent, scope, raw_name)
            if is_rmse:
                seed_scope = np.sqrt(np.maximum(seed_scope, 0.0))
            for candidate, reference in PAIRS:
                candidate_index = CANDIDATES.index(candidate)
                reference_index = CANDIDATES.index(reference)
                contrast_draws = (
                    draws[:, candidate_index] - draws[:, reference_index]
                    if signed
                    else _ratio(draws[:, candidate_index], draws[:, reference_index])
                )
                estimate = (
                    point[candidate_index] - point[reference_index]
                    if signed
                    else float(_ratio(point[candidate_index], point[reference_index]))
                )
                low, high = _quantiles(contrast_draws)
                comparison_rows.append(
                    {
                        "candidate_id": candidate,
                        "reference_id": reference,
                        "metric": display_name,
                        "scope": scope,
                        "contrast": "difference" if signed else "ratio",
                        "estimate": float(estimate),
                        "ci_lower": low,
                        "ci_upper": high,
                        "lower_seed_fraction": float(
                            np.mean(seed_scope[:, candidate_index] < seed_scope[:, reference_index])
                        ),
                    }
                )

    summary = pd.DataFrame(summary_rows)
    agent_metrics = pd.DataFrame(agent_rows)
    comparisons = pd.DataFrame(comparison_rows)

    nominal_frame = pd.DataFrame(frozen["dncp_agent_rows"])
    nominal_index = pd.MultiIndex.from_product(
        (JOINT_CANDIDATES, AGENTS), names=["candidate_id", "agent_id"]
    )
    nominal = (
        nominal_frame.set_index(["candidate_id", "agent_id"])
        .reindex(nominal_index)["auc_perpendicular_m2"]
        .to_numpy(float)
        .reshape(len(JOINT_CANDIDATES), len(AGENTS))
    )
    if not np.isfinite(nominal).all() or np.any(nominal <= 0):
        raise ValueError("frozen DNCP agent AUC panel is invalid")

    score_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    score_difference_rows: list[dict[str, Any]] = []
    for lambda_c in (1.0, PRIMARY_LAMBDA_C, 0.0):
        predicted = _scores(nominal, lambda_c)
        predicted_order = np.argsort(predicted, kind="stable")
        for index, candidate in enumerate(JOINT_CANDIDATES):
            score_rows.append(
                {
                    "lambda_c": lambda_c,
                    "source": "dncp",
                    "candidate_id": candidate,
                    "score": float(predicted[index]),
                    "rank": int(np.where(predicted_order == index)[0][0] + 1),
                    "ci_lower": float(predicted[index]),
                    "ci_upper": float(predicted[index]),
                }
            )
        for source, metric in (
            ("mc_covariance", "auc_p_perpendicular_m2"),
            ("mc_squared_error", "mse_perpendicular_m2"),
        ):
            metric_index = metric_names.index(metric)
            joint_means = means[1:, :, metric_index]
            joint_boot = boot[:, 1:, :, metric_index]
            observed = _scores(joint_means, lambda_c)
            observed_boot = _scores(joint_boot, lambda_c)
            observed_order = np.argsort(observed, kind="stable")
            predicted_choice = int(np.argmin(predicted))
            empirical_choice = int(np.argmin(observed))
            tau_draws = _kendall(predicted, observed_boot)
            tau_low, tau_high = _quantiles(tau_draws)
            regret_draws = observed_boot[:, predicted_choice] - observed_boot.min(axis=1)
            regret_low, regret_high = _quantiles(regret_draws)
            ranking_rows.append(
                {
                    "lambda_c": lambda_c,
                    "source": source,
                    "kendall_tau_b": float(_kendall(predicted, observed)),
                    "tau_ci_lower": tau_low,
                    "tau_ci_upper": tau_high,
                    "predicted_choice": JOINT_CANDIDATES[predicted_choice],
                    "empirical_choice": JOINT_CANDIDATES[empirical_choice],
                    "selection_regret": float(observed[predicted_choice] - observed[empirical_choice]),
                    "regret_ci_lower": regret_low,
                    "regret_ci_upper": regret_high,
                    "selected_is_best_bootstrap_fraction": float(
                        np.mean(np.argmin(observed_boot, axis=1) == predicted_choice)
                    ),
                }
            )
            for index, candidate in enumerate(JOINT_CANDIDATES):
                low, high = _quantiles(observed_boot[:, index])
                score_rows.append(
                    {
                        "lambda_c": lambda_c,
                        "source": source,
                        "candidate_id": candidate,
                        "score": float(observed[index]),
                        "rank": int(np.where(observed_order == index)[0][0] + 1),
                        "ci_lower": low,
                        "ci_upper": high,
                    }
                )
            for candidate_index in range(1, len(JOINT_CANDIDATES)):
                for reference_index in range(candidate_index):
                    differences = (
                        observed_boot[:, candidate_index]
                        - observed_boot[:, reference_index]
                    )
                    low, high = _quantiles(differences)
                    score_difference_rows.append(
                        {
                            "lambda_c": lambda_c,
                            "source": source,
                            "candidate_id": JOINT_CANDIDATES[candidate_index],
                            "reference_id": JOINT_CANDIDATES[reference_index],
                            "score_difference": float(
                                observed[candidate_index] - observed[reference_index]
                            ),
                            "ci_lower": low,
                            "ci_upper": high,
                            "lower_bootstrap_fraction": float(np.mean(differences < 0)),
                        }
                    )

    scores = pd.DataFrame(score_rows)
    ranking = pd.DataFrame(ranking_rows)
    rmse_comparisons = comparisons[
        comparisons.metric.isin(
            ("rmse_perpendicular_m", "rmse_parallel_m", "rmse_xy_m")
        )
    ].copy()
    consistency = summary[
        (summary.scope == "fleet_mean") & summary.metric.isin(CONSISTENCY)
    ].copy()
    targets = {
        "nees_perpendicular": 1.0,
        "nees_xy": 2.0,
        "nees_se2": 3.0,
        "covered_perpendicular_95": 0.95,
        "covered_xy_95": 0.95,
        "covered_se2_95": 0.95,
    }
    consistency["theoretical_target"] = consistency.metric.map(targets)
    consistency["signed_deviation"] = consistency.estimate - consistency.theoretical_target

    stage = root / f"analysis_seed_{start_seed:04d}_{seeds[-1]:04d}"
    stage.mkdir(parents=True, exist_ok=True)
    averaged_curves = {field: values / seed_count for field, values in curve_sum.items()}
    time_rows = []
    for field, values in averaged_curves.items():
        for candidate_index, candidate in enumerate(CANDIDATES):
            for agent_index, agent in enumerate(AGENTS):
                time_rows.extend(
                    {
                        "candidate_id": candidate,
                        "agent_id": agent,
                        "time_s": float(time_s),
                        "metric": field,
                        "ensemble_mean": float(value),
                    }
                    for time_s, value in zip(TIMES, values[candidate_index, agent_index])
                )

    tables = {
        "directional_schedule_summary.csv": summary,
        "directional_schedule_agent_metrics.csv": agent_metrics,
        "directional_schedule_paired_bootstrap.csv": comparisons,
        "directional_schedule_rmse_comparisons.csv": rmse_comparisons,
        "directional_schedule_scores.csv": scores,
        "directional_schedule_ranking_validation.csv": ranking,
        "directional_schedule_score_differences.csv": pd.DataFrame(score_difference_rows),
        "directional_schedule_consistency.csv": consistency,
        "directional_schedule_case_agent_metrics.csv": pd.DataFrame(case_agent_rows),
        "directional_schedule_time_metrics.csv": pd.DataFrame(time_rows),
        "directional_schedule_path_parameters.csv": pd.DataFrame(parameter_rows),
        "directional_schedule_protocol_audit.csv": pd.DataFrame(audit_rows),
        "directional_schedule_run_diagnostics.csv": pd.DataFrame(diagnostic_rows),
    }
    for filename, table in tables.items():
        table.to_csv(stage / filename, index=False, encoding="utf-8-sig")

    diagnostics = tables["directional_schedule_run_diagnostics.csv"]
    required_zero = (
        "duplicate_master_prior_count",
        "duplicate_timestamp_prior_count",
        "master_contamination_count",
        "source_contamination_count",
        "source_gap_count",
        "source_overlap_count",
        "shadow_created_before_arrival",
    )
    audit_summary = {
        "case_count": seed_count,
        "estimator_run_count": seed_count * len(CANDIDATES),
        "physical_wave_counts": sorted(diagnostics.physical_wave_count.unique().tolist()),
        "unique_physical_wave_counts": sorted(diagnostics.unique_physical_wave_count.unique().tolist()),
        "joint_cycle_counts": sorted(diagnostics.joint_cycle_count.unique().tolist()),
        "ordinary_action_counts": sorted(diagnostics.ordinary_action_count.unique().tolist()),
        "max_structural_violation_count": int(diagnostics[list(required_zero)].to_numpy().max()),
    }
    failure_count = 0
    with closing(sqlite3.connect(root / "checkpoint.sqlite3")) as connection:
        failure_count = int(connection.execute("SELECT COUNT(*) FROM failures").fetchone()[0])
    audit_summary["historical_failure_record_count"] = failure_count
    (stage / "directional_schedule_audit_summary.json").write_text(
        json.dumps(audit_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    figures: list[str] = []
    if make_plots:
        figures = _write_figures(
            stage,
            frozen,
            representative,
            summary,
            comparisons,
            agent_metrics,
            scores,
            consistency,
            averaged_curves,
        )

    headline = comparisons[
        comparisons.metric.isin(("auc_p_perpendicular_m2", "rmse_perpendicular_m"))
        & comparisons.scope.isin(("fleet_mean", "worst_individual"))
    ]
    primary_ranking = ranking[ranking.lambda_c == PRIMARY_LAMBDA_C]
    primary_scores = scores[scores.lambda_c == PRIMARY_LAMBDA_C].sort_values(
        ["source", "rank"]
    )
    primary_score_differences = pd.DataFrame(score_difference_rows)
    primary_score_differences = primary_score_differences[
        primary_score_differences.lambda_c == PRIMARY_LAMBDA_C
    ]
    directional_rmse_headline = rmse_comparisons[
        rmse_comparisons.candidate_id.isin(("J-D1", "J-D2", "J-D3"))
        & rmse_comparisons.reference_id.isin(("J-RR", "D-only"))
        & rmse_comparisons.metric.isin(("rmse_perpendicular_m", "rmse_xy_m"))
        & rmse_comparisons.scope.isin(("fleet_mean", "worst_individual"))
    ]
    report = [
        "# 固定15°随机S形扇区：方向性联合TDOA调度结果",
        "",
        f"完整分析 seeds {start_seed}–{seeds[-1]}，共{seed_count}个公共Monte Carlo任务、{seed_count * len(CANDIDATES)}次估计器运行。每组均为200次真实发波。",
        "",
        "D-only是普通双主TDOA背景基线；DNCP排序只比较J-RR、J-D1、J-D2、J-D3四个联合schedule。主评分固定lambda_c=0.5；lambda_c=1和0仅作权重敏感性展示，不增加仿真。",
        "",
        "## 数据与结构审计",
        "",
        f"最终 checkpoint 包含全部 {seed_count} 个 seed。结构异常计数最大值为"
        f"{audit_summary['max_structural_violation_count']}，实际发波计数为"
        f"{audit_summary['physical_wave_counts']}。该 seed 范围内保留 {failure_count} 条历史失败记录。",
        "",
        "## 主要对比",
        "",
        _markdown_table(
            headline,
            ("candidate_id", "reference_id", "metric", "scope", "estimate", "ci_lower", "ci_upper", "lower_seed_fraction"),
        ),
        "",
        "比值小于1表示候选优于其reference。J-RR/D-only回答联合测量在该任务中的总体效果；J-D1/2/3相对J-RR回答方向性relay分配是否改善顺序循环。",
        "",
        "## 三种方向调度的直接定位RMSE",
        "",
        _markdown_table(
            directional_rmse_headline,
            ("candidate_id", "reference_id", "metric", "scope", "estimate", "ci_lower", "ci_upper", "lower_seed_fraction"),
        ),
        "",
        "逐艇绝对RMSE与逐艇配对比值见directional_schedule_agent_metrics.csv和directional_schedule_rmse_comparisons.csv；图中同时展示C–H、fleet mean与worst individual。",
        "",
        "## DNCP与Monte Carlo排序（主权重lambda_c=0.5）",
        "",
        _markdown_table(
            primary_scores,
            ("source", "rank", "candidate_id", "score", "ci_lower", "ci_upper"),
        ),
        "",
        _markdown_table(
            primary_ranking,
            ("source", "kendall_tau_b", "tau_ci_lower", "tau_ci_upper", "predicted_choice", "empirical_choice", "selection_regret", "regret_ci_lower", "regret_ci_upper", "selected_is_best_bootstrap_fraction"),
        ),
        "",
        "主权重下候选之间的直接配对评分差：",
        "",
        _markdown_table(
            primary_score_differences,
            ("source", "candidate_id", "reference_id", "score_difference", "ci_lower", "ci_upper", "lower_bootstrap_fraction"),
        ),
        "",
        "DNCP与Monte Carlo covariance同属同一噪声模型，但DNCP只使用固定径向直线代理和确定性名义协方差；Monte Carlo使用随机S形轨迹与随机测量误差。因此排序一致性验证的是代理规划跨路径扰动的可迁移性，不是独立物理模型之间的一致性。",
        "",
        "## 一致性",
        "",
        _markdown_table(
            consistency,
            ("candidate_id", "metric", "estimate", "theoretical_target", "ci_lower", "ci_upper"),
        ),
        "",
        "## 输出图表",
        "",
    ]
    for figure in figures:
        report.extend((f"![{Path(figure).stem}]({figure})", ""))
    (stage / "directional_schedule_validation_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    manifest = {
        "complete": True,
        "start_seed": start_seed,
        "end_seed": seeds[-1],
        "seed_count": seed_count,
        "estimator_runs": seed_count * len(CANDIDATES),
        "bootstrap_samples": bootstrap_samples,
        "primary_lambda_c": PRIMARY_LAMBDA_C,
        "fingerprint": frozen.get("fingerprint"),
        "analysis_directory": str(stage),
        "elapsed_wall_s": time.perf_counter() - started,
        "figures": figures,
        "audit": audit_summary,
    }
    (stage / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--seed-count", type=int, default=1000)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    result = analyze(
        args.output,
        start_seed=args.start_seed,
        seed_count=args.seed_count,
        bootstrap_samples=args.bootstrap_samples,
        make_plots=not args.no_plots,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
