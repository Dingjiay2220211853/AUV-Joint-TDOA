"""BLAS-free helpers for the small covariance matrices used by experiments."""

from __future__ import annotations

import math

import numpy as np


def small_matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
        raise ValueError("matrix dimensions do not match")
    result = np.zeros((left.shape[0], right.shape[1]), dtype=float)
    for row in range(left.shape[0]):
        for column in range(right.shape[1]):
            result[row, column] = sum(
                float(left[row, inner]) * float(right[inner, column])
                for inner in range(left.shape[1])
            )
    return result


def _cholesky(matrix: np.ndarray) -> list[list[float]]:
    size = matrix.shape[0]
    lower = [[0.0 for _ in range(size)] for _ in range(size)]
    for row in range(size):
        for column in range(row + 1):
            subtotal = sum(
                lower[row][inner] * lower[column][inner]
                for inner in range(column)
            )
            if row == column:
                pivot = float(matrix[row, row]) - subtotal
                if not math.isfinite(pivot) or pivot <= 0.0:
                    raise ValueError("matrix is not positive definite")
                lower[row][column] = math.sqrt(pivot)
            else:
                lower[row][column] = (
                    float(matrix[row, column]) - subtotal
                ) / lower[column][column]
    return lower


def regularize_covariance(
    covariance: np.ndarray,
    minimum_jitter: float = 1e-10,
) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("covariance must be finite")
    result = 0.5 * (covariance + covariance.T)
    try:
        _cholesky(result)
        return result
    except ValueError:
        pass
    scale = max(1.0, max(abs(float(value)) for value in result.ravel()))
    jitter = max(float(minimum_jitter), scale * 1e-12)
    for _ in range(10):
        candidate = result.copy()
        for index in range(candidate.shape[0]):
            candidate[index, index] += jitter
        try:
            _cholesky(candidate)
            return candidate
        except ValueError:
            jitter *= 10.0
    raise ValueError("covariance could not be regularized")


def solve_spd(covariance: np.ndarray, vector: np.ndarray) -> np.ndarray:
    covariance = regularize_covariance(covariance)
    vector = np.asarray(vector, dtype=float)
    lower = _cholesky(covariance)
    size = covariance.shape[0]
    forward = [0.0 for _ in range(size)]
    for row in range(size):
        subtotal = sum(lower[row][column] * forward[column] for column in range(row))
        forward[row] = (float(vector[row]) - subtotal) / lower[row][row]
    solution = [0.0 for _ in range(size)]
    for row in range(size - 1, -1, -1):
        subtotal = sum(
            lower[column][row] * solution[column]
            for column in range(row + 1, size)
        )
        solution[row] = (forward[row] - subtotal) / lower[row][row]
    return np.asarray(solution, dtype=float)


def quadratic_form(error: np.ndarray, covariance: np.ndarray) -> float:
    error = np.asarray(error, dtype=float)
    solution = solve_spd(covariance, error)
    return sum(
        float(error[index]) * float(solution[index])
        for index in range(error.size)
    )
