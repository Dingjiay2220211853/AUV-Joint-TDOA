"""Statistical-consistency diagnostics for the Pose2 factor graph.

The diagnostics in this module are evaluation-only.  They may use simulation
ground truth to compute NEES, but none of the returned quantities are fed back
to any controller or the estimator.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy.stats import chi2


_EPS = 1e-12


def _solve_small(
    matrix: np.ndarray,
    vector: np.ndarray,
) -> np.ndarray:
    """Solve a 1x1--3x3 system without platform BLAS/LAPACK dispatch."""
    matrix = np.asarray(matrix, dtype=float)
    vector = np.asarray(vector, dtype=float)
    size = int(vector.size)
    augmented = [
        [
            float(matrix[row, column])
            + (float(_EPS) if row == column else 0.0)
            for column in range(size)
        ] + [float(vector[row])]
        for row in range(size)
    ]
    for pivot_column in range(size):
        pivot_row = max(
            range(pivot_column, size),
            key=lambda row: abs(augmented[row][pivot_column]),
        )
        if abs(augmented[pivot_row][pivot_column]) <= _EPS:
            raise ValueError("covariance is singular")
        if pivot_row != pivot_column:
            augmented[pivot_column], augmented[pivot_row] = (
                augmented[pivot_row],
                augmented[pivot_column],
            )
        pivot = augmented[pivot_column][pivot_column]
        for column in range(pivot_column, size + 1):
            augmented[pivot_column][column] /= pivot
        for row in range(size):
            if row == pivot_column:
                continue
            scale = augmented[row][pivot_column]
            for column in range(pivot_column, size + 1):
                augmented[row][column] -= (
                    scale * augmented[pivot_column][column]
                )
    return np.asarray([augmented[row][-1] for row in range(size)])


def chi_square_interval(
    degrees_of_freedom: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return the central chi-square consistency interval."""
    dof = int(degrees_of_freedom)
    if dof <= 0:
        raise ValueError("degrees_of_freedom must be positive")
    confidence = float(confidence)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    tail = 0.5 * (1.0 - confidence)
    return (
        float(chi2.ppf(tail, dof)),
        float(chi2.ppf(1.0 - tail, dof)),
    )


def normalized_squared_error(
    error: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """Compute ``error.T @ covariance^-1 @ error`` robustly."""
    error = np.asarray(error, dtype=float).reshape(-1)
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (error.size, error.size):
        raise ValueError("error and covariance dimensions do not match")
    covariance = 0.5 * (covariance + covariance.T)
    if not np.all(np.isfinite(error)) or not np.all(np.isfinite(covariance)):
        return float("nan")
    try:
        solved = _solve_small(covariance, error)
    except ValueError:
        return float("nan")
    value = sum(
        float(error[index]) * float(solved[index])
        for index in range(error.size)
    )
    return max(value, 0.0) if np.isfinite(value) else float("nan")


def scalar_nis(
    residual: float,
    innovation_variance: float,
) -> tuple[float, float]:
    """Return scalar NIS and the signed normalized innovation."""
    variance = float(innovation_variance)
    if not np.isfinite(variance) or variance <= 0.0:
        return float("nan"), float("nan")
    normalized = float(residual) / float(np.sqrt(variance))
    return normalized * normalized, normalized


def pose2_local_error(estimate_pose: Any, true_pose: Any) -> np.ndarray:
    """Return the Pose2 tangent error from estimate to truth."""
    return np.asarray(
        estimate_pose.localCoordinates(true_pose),
        dtype=float,
    ).reshape(3)


def pose_consistency(
    estimate_pose: Any,
    true_pose: Any,
    covariance: np.ndarray,
    confidence: float = 0.95,
) -> dict[str, float | bool]:
    """Compute position and full-Pose2 NEES for one state."""
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (3, 3):
        raise ValueError("Pose2 covariance must be 3x3")
    covariance = 0.5 * (covariance + covariance.T)
    error = pose2_local_error(estimate_pose, true_pose)
    nees_xy = normalized_squared_error(error[:2], covariance[:2, :2])
    nees_pose = normalized_squared_error(error, covariance)
    xy_lower, xy_upper = chi_square_interval(2, confidence)
    pose_lower, pose_upper = chi_square_interval(3, confidence)
    return {
        "position_error_m": float(np.linalg.norm(error[:2])),
        "heading_error_rad": float(error[2]),
        "nees_xy": nees_xy,
        "nees_pose": nees_pose,
        "nees_xy_lower": xy_lower,
        "nees_xy_upper": xy_upper,
        "nees_pose_lower": pose_lower,
        "nees_pose_upper": pose_upper,
        "nees_xy_in_interval": bool(xy_lower <= nees_xy <= xy_upper),
        "nees_pose_in_interval": bool(
            pose_lower <= nees_pose <= pose_upper
        ),
    }


def consistency_summary(
    values: Iterable[float],
    degrees_of_freedom: int,
    confidence: float = 0.95,
) -> dict[str, float | int | bool]:
    """Summarize a sequence of NIS/NEES samples.

    For ``N`` independent samples with per-sample dimension ``d``,
    ``sum(statistic)`` follows ``chi2(N*d)`` under a consistent model.  The
    returned mean interval is therefore the rigorous aggregate consistency
    interval, while ``sample_coverage`` reports the fraction of individual
    values inside their central interval.
    """
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    count = int(array.size)
    dof = int(degrees_of_freedom)
    lower_single, upper_single = chi_square_interval(dof, confidence)
    if count == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "expected_mean": float(dof),
            "mean_lower": float("nan"),
            "mean_upper": float("nan"),
            "sample_coverage": float("nan"),
            "mean_consistent": False,
        }
    lower_total, upper_total = chi_square_interval(count * dof, confidence)
    mean_value = float(np.mean(array))
    mean_lower = lower_total / count
    mean_upper = upper_total / count
    coverage = float(
        np.mean((array >= lower_single) & (array <= upper_single))
    )
    return {
        "count": count,
        "mean": mean_value,
        "expected_mean": float(dof),
        "mean_lower": float(mean_lower),
        "mean_upper": float(mean_upper),
        "sample_coverage": coverage,
        "mean_consistent": bool(mean_lower <= mean_value <= mean_upper),
    }
