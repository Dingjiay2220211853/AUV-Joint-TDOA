"""Correlated triple-TDOA factor for one relay-snoop micro-cycle.

The logical event-pose order follows the paper derivation exactly:

    A0, B, relay, A1, listener0, listener1, listener2, listener3

Four arrival timestamps generate three consecutive time differences.  Their
distance-domain covariance is ``c^2 D Sigma_tau D.T`` and must be represented
by one three-dimensional Gaussian factor rather than three independent scalar
factors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import gtsam  # type: ignore
import numpy as np


LOGICAL_POSE_ORDER = (
    "a0",
    "b",
    "relay",
    "a1",
    "listener0",
    "listener1",
    "listener2",
    "listener3",
)

TIMESTAMP_DIFFERENCE_MATRIX = np.asarray(
    [
        [-1.0, 1.0, 0.0, 0.0],
        [0.0, -1.0, 1.0, 0.0],
        [0.0, 0.0, -1.0, 1.0],
    ],
    dtype=float,
)

# GTSAM's exact Constrained noise becomes numerically rank-deficient after a
# long timestamp chain is incrementally re-eliminated by iSAM2.  This distance-
# domain variance is 1e-6 m^2 versus about 9 m^2 timestamp variance in the
# paper configuration, so it is a numerical realization of the otherwise
# deterministic relation without adding a material second noise source.
_DETERMINISTIC_RELATION_SIGMA_M = 1e-3

_PAIR_TERMS = (
    # residual row, first logical pose, second logical pose, sign
    (0, 0, 1, 1.0),  # d(A0, B)
    (0, 1, 5, 1.0),  # d(B, C1)
    (0, 0, 4, -1.0),  # -d(A0, C0)
    (1, 1, 2, 1.0),  # d(B, relay)
    (1, 2, 6, 1.0),  # d(relay, C2)
    (1, 1, 5, -1.0),  # -d(B, C1)
    (2, 2, 3, 1.0),  # d(relay, A1)
    (2, 3, 7, 1.0),  # d(A1, C3)
    (2, 2, 6, -1.0),  # -d(relay, C2)
)


def _deduplicate_graph_keys(
    logical_keys: Sequence[Any],
) -> tuple[list[Any], list[int]]:
    """Return unique GTSAM keys and one index for each logical event slot.

    Two separately identified acoustic arrivals can be simultaneous at one
    moving receiver (for example when a relay lies exactly on the direct
    propagation ray).  They then share one physical Pose2 state even though
    their timestamp errors remain separate measurement variables.  GTSAM
    factors cannot list the same key twice, so the callback must accumulate
    both logical Jacobians into the shared graph-key block.
    """

    unique_keys: list[Any] = []
    logical_to_unique: list[int] = []
    unique_index_by_key: dict[int, int] = {}
    for key in logical_keys:
        key_integer = int(key)
        if key_integer not in unique_index_by_key:
            unique_index_by_key[key_integer] = len(unique_keys)
            unique_keys.append(key)
        logical_to_unique.append(unique_index_by_key[key_integer])
    return unique_keys, logical_to_unique


def _small_matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply the small fixed-size matrices without BLAS dispatch."""
    if left.shape[1] != right.shape[0]:
        raise ValueError("matrix dimensions do not match")
    result = np.zeros((left.shape[0], right.shape[1]), dtype=float)
    for row in range(left.shape[0]):
        for column in range(right.shape[1]):
            result[row, column] = sum(
                float(left[row, inner]) * float(right[inner, column])
                for inner in range(left.shape[1])
            )
    return result


def _require_symmetric_positive_definite(
    matrix: np.ndarray,
    name: str,
) -> None:
    """Validate a covariance using a small explicit Cholesky factorization."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")

    maximum = max(
        abs(float(matrix[row, column]))
        for row in range(matrix.shape[0])
        for column in range(matrix.shape[1])
    )
    if maximum <= 0.0:
        raise ValueError(f"{name} must be positive definite")
    symmetry_tolerance = maximum * 1e-10
    for row in range(matrix.shape[0]):
        for column in range(row):
            if abs(float(matrix[row, column] - matrix[column, row])) > (
                symmetry_tolerance
            ):
                raise ValueError(f"{name} must be symmetric")

    lower = [
        [0.0 for _ in range(matrix.shape[0])]
        for _ in range(matrix.shape[0])
    ]
    pivot_tolerance = maximum * 1e-12
    for row in range(matrix.shape[0]):
        for column in range(row + 1):
            subtotal = sum(
                lower[row][inner] * lower[column][inner]
                for inner in range(column)
            )
            if row == column:
                pivot = float(matrix[row, row]) - subtotal
                if pivot <= pivot_tolerance:
                    raise ValueError(f"{name} must be positive definite")
                lower[row][column] = math.sqrt(pivot)
            else:
                lower[row][column] = (
                    float(matrix[row, column]) - subtotal
                ) / lower[column][column]


def build_cycle_tdoa_covariance(
    timestamp_covariance: np.ndarray,
    sound_speed: float = 1500.0,
) -> np.ndarray:
    """Map a 4x4 timestamp covariance [s^2] to 3x3 residual covariance [m^2]."""
    timestamp_covariance = np.asarray(timestamp_covariance, dtype=float)
    if timestamp_covariance.shape != (4, 4):
        raise ValueError("timestamp_covariance must have shape (4, 4)")
    _require_symmetric_positive_definite(
        timestamp_covariance,
        "timestamp_covariance",
    )
    sound_speed = float(sound_speed)
    if not math.isfinite(sound_speed) or sound_speed <= 0.0:
        raise ValueError("sound_speed must be finite and positive")

    time_difference_covariance = _small_matmul(
        _small_matmul(
            TIMESTAMP_DIFFERENCE_MATRIX,
            timestamp_covariance,
        ),
        TIMESTAMP_DIFFERENCE_MATRIX.T,
    )
    covariance = sound_speed * sound_speed * time_difference_covariance
    covariance = 0.5 * (covariance + covariance.T)
    _require_symmetric_positive_definite(covariance, "cycle covariance")
    return covariance


def create_cycle_tdoa_noise_model(
    timestamp_covariance: np.ndarray,
    sound_speed: float = 1500.0,
) -> Any:
    """Create the full correlated GTSAM Gaussian noise model for a cycle."""
    return gtsam.noiseModel.Gaussian.Covariance(
        build_cycle_tdoa_covariance(timestamp_covariance, sound_speed)
    )


def _coerce_measurement(measured_delta_t: Sequence[float]) -> np.ndarray:
    measurement = np.asarray(measured_delta_t, dtype=float)
    if measurement.shape != (3,):
        raise ValueError("measured_delta_t must contain exactly three values")
    if not np.all(np.isfinite(measurement)):
        raise ValueError("measured_delta_t must contain only finite values")
    return measurement.copy()


def _range_with_local_jacobians(
    first_pose: Any,
    second_pose: Any,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return a planar range and its right-local Pose2 Jacobians."""
    dx = float(first_pose.x()) - float(second_pose.x())
    dy = float(first_pose.y()) - float(second_pose.y())
    distance = math.hypot(dx, dy)
    if distance <= 1e-9:
        raise ValueError("cycle factor range is not differentiable at zero")

    unit_x = dx / distance
    unit_y = dy / distance

    first_theta = float(first_pose.theta())
    first_cos = math.cos(first_theta)
    first_sin = math.sin(first_theta)
    first_jacobian = np.asarray(
        [
            unit_x * first_cos + unit_y * first_sin,
            -unit_x * first_sin + unit_y * first_cos,
            0.0,
        ],
        dtype=float,
    )

    second_theta = float(second_pose.theta())
    second_cos = math.cos(second_theta)
    second_sin = math.sin(second_theta)
    second_unit_x = -unit_x
    second_unit_y = -unit_y
    second_jacobian = np.asarray(
        [
            second_unit_x * second_cos + second_unit_y * second_sin,
            -second_unit_x * second_sin + second_unit_y * second_cos,
            0.0,
        ],
        dtype=float,
    )
    return distance, first_jacobian, second_jacobian


def evaluate_cycle_snoop_geometry(
    pose_a0: Any,
    pose_b: Any,
    pose_relay: Any,
    pose_a1: Any,
    pose_listener0: Any,
    pose_listener1: Any,
    pose_listener2: Any,
    pose_listener3: Any,
    measured_delta_t: Sequence[float],
    sound_speed: float = 1500.0,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Evaluate the 3D cycle residual and eight Pose2-local Jacobian blocks."""
    measurement = _coerce_measurement(measured_delta_t)
    sound_speed = float(sound_speed)
    if not math.isfinite(sound_speed) or sound_speed <= 0.0:
        raise ValueError("sound_speed must be finite and positive")

    poses = (
        pose_a0,
        pose_b,
        pose_relay,
        pose_a1,
        pose_listener0,
        pose_listener1,
        pose_listener2,
        pose_listener3,
    )
    residual = np.zeros(3, dtype=float)
    jacobians = [np.zeros((3, 3), dtype=float) for _ in poses]

    for row, first_index, second_index, sign in _PAIR_TERMS:
        distance, first_jacobian, second_jacobian = (
            _range_with_local_jacobians(
                poses[first_index],
                poses[second_index],
            )
        )
        residual[row] += sign * distance
        jacobians[first_index][row, :] += sign * first_jacobian
        jacobians[second_index][row, :] += sign * second_jacobian

    residual -= sound_speed * measurement
    return residual, jacobians


def create_cycle_snoop_factor(
    idx_a0: Any,
    idx_b: Any,
    idx_relay: Any,
    idx_a1: Any,
    idx_listener0: Any,
    idx_listener1: Any,
    idx_listener2: Any,
    idx_listener3: Any,
    measured_delta_t: Sequence[float],
    *,
    timestamp_covariance: np.ndarray | None = None,
    noise_model: Any = None,
    sound_speed: float = 1500.0,
    pose_offsets: Mapping[str, Any] | None = None,
) -> Any:
    """Create one correlated triple-TDOA ``gtsam.CustomFactor``.

    Exactly one of ``timestamp_covariance`` or ``noise_model`` must be given.
    ``timestamp_covariance`` is the 4x4 covariance of the listener's arrival
    timestamps in seconds squared; it is converted internally to the full
    3x3 distance covariance.

    ``pose_offsets`` optionally maps names in ``LOGICAL_POSE_ORDER`` to a
    deterministic Pose2 offset from the graph state to the event pose.  This
    permits several logical events to reuse one graph key without collapsing
    their event-time geometry.  Jacobians for duplicate keys are accumulated
    before they are returned to GTSAM.
    """
    if (timestamp_covariance is None) == (noise_model is None):
        raise ValueError(
            "provide exactly one of timestamp_covariance or noise_model"
        )
    if timestamp_covariance is not None:
        noise_model = create_cycle_tdoa_noise_model(
            timestamp_covariance,
            sound_speed,
        )

    measurement = _coerce_measurement(measured_delta_t)
    logical_keys = [
        idx_a0,
        idx_b,
        idx_relay,
        idx_a1,
        idx_listener0,
        idx_listener1,
        idx_listener2,
        idx_listener3,
    ]

    supplied_offsets = dict(pose_offsets or {})
    unknown_offset_names = sorted(
        set(supplied_offsets) - set(LOGICAL_POSE_ORDER)
    )
    if unknown_offset_names:
        raise ValueError(
            f"unknown pose offset names: {unknown_offset_names}"
        )
    logical_offsets = [
        supplied_offsets.get(name, gtsam.Pose2())
        for name in LOGICAL_POSE_ORDER
    ]

    unique_keys, logical_to_unique = _deduplicate_graph_keys(logical_keys)

    def error_func(this, values, H):
        event_poses = [
            values.atPose2(key).compose(offset)
            for key, offset in zip(logical_keys, logical_offsets)
        ]
        error, logical_jacobians = evaluate_cycle_snoop_geometry(
            *event_poses,
            measured_delta_t=measurement,
            sound_speed=sound_speed,
        )

        if H is not None:
            accumulated = [
                np.zeros((3, 3), dtype=float) for _ in unique_keys
            ]
            for logical_index, logical_jacobian in enumerate(
                logical_jacobians
            ):
                offset_adjoint = np.asarray(
                    logical_offsets[logical_index]
                    .inverse()
                    .AdjointMap(),
                    dtype=float,
                )
                mapped_jacobian = _small_matmul(
                    logical_jacobian,
                    offset_adjoint,
                )
                accumulated[
                    logical_to_unique[logical_index]
                ] += mapped_jacobian
            for unique_index, jacobian in enumerate(accumulated):
                H[unique_index] = jacobian

        return error

    return gtsam.CustomFactor(noise_model, unique_keys, error_func)  # type: ignore


def create_shared_timestamp_cycle_factor(
    idx_a0: Any,
    idx_b: Any,
    idx_relay: Any,
    idx_a1: Any,
    idx_listener0: Any,
    idx_listener1: Any,
    idx_listener2: Any,
    idx_listener3: Any,
    timestamp_error_keys: Sequence[Any],
    measured_delta_t: Sequence[float],
    *,
    sound_speed: float = 1500.0,
) -> Any:
    """Create the online shared-timestamp form of the 8-Pose2 cycle.

    ``measured_delta_t`` is ``D * noisy_arrivals - known_delays``.  The four
    scalar states are the timestamp errors for the listener/wave pairs and
    each must have one external ``N(0, sigma_t**2)`` prior.  Consequently this
    factor is an effectively deterministic relation.  A 1e-3 m numerical
    sigma is used instead of GTSAM's exact constrained model because long
    shared chains otherwise become rank-indeterminate under incremental
    re-elimination.  Attaching ``R_cycle`` here would still count the
    timestamp noise twice.
    """
    timestamp_error_keys = tuple(timestamp_error_keys)
    if len(timestamp_error_keys) != 4:
        raise ValueError("timestamp_error_keys must contain four keys")
    if len({int(key) for key in timestamp_error_keys}) != 4:
        raise ValueError("one cycle must reference four distinct timestamps")
    measurement = _coerce_measurement(measured_delta_t)
    sound_speed = float(sound_speed)
    if not math.isfinite(sound_speed) or sound_speed <= 0.0:
        raise ValueError("sound_speed must be finite and positive")

    logical_pose_keys = (
        idx_a0,
        idx_b,
        idx_relay,
        idx_a1,
        idx_listener0,
        idx_listener1,
        idx_listener2,
        idx_listener3,
    )
    pose_keys, logical_to_unique = _deduplicate_graph_keys(logical_pose_keys)
    all_keys = list(pose_keys) + list(timestamp_error_keys)
    if len({int(key) for key in all_keys}) != len(all_keys):
        raise ValueError("Pose2 and timestamp keys must be globally distinct")

    def error_func(this, values, H):
        poses = [values.atPose2(key) for key in logical_pose_keys]
        timestamp_errors = np.asarray(
            [values.atDouble(key) for key in timestamp_error_keys],
            dtype=float,
        )
        error, pose_jacobians, timestamp_jacobian = (
            evaluate_shared_timestamp_cycle_geometry(
                *poses,
                measured_delta_t=measurement,
                timestamp_errors=timestamp_errors,
                sound_speed=sound_speed,
            )
        )
        if H is not None:
            accumulated = [
                np.zeros((3, 3), dtype=float) for _ in pose_keys
            ]
            for logical_index, jacobian in enumerate(pose_jacobians):
                accumulated[logical_to_unique[logical_index]] += jacobian
            for index, jacobian in enumerate(accumulated):
                H[index] = jacobian
            for index in range(4):
                H[len(pose_keys) + index] = timestamp_jacobian[:, [index]]
        return error

    deterministic_noise = gtsam.noiseModel.Isotropic.Sigma(
        3, _DETERMINISTIC_RELATION_SIGMA_M
    )
    return gtsam.CustomFactor(deterministic_noise, all_keys, error_func)  # type: ignore


def evaluate_shared_timestamp_cycle_geometry(
    pose_a0: Any,
    pose_b: Any,
    pose_relay: Any,
    pose_a1: Any,
    pose_listener0: Any,
    pose_listener1: Any,
    pose_listener2: Any,
    pose_listener3: Any,
    measured_delta_t: Sequence[float],
    timestamp_errors: Sequence[float],
    *,
    sound_speed: float = 1500.0,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Evaluate the augmented residual and all Pose2/scalar Jacobians."""
    timestamp_errors = np.asarray(timestamp_errors, dtype=float)
    if timestamp_errors.shape != (4,) or not np.all(
        np.isfinite(timestamp_errors)
    ):
        raise ValueError("timestamp_errors must contain four finite values")
    error, pose_jacobians = evaluate_cycle_snoop_geometry(
        pose_a0,
        pose_b,
        pose_relay,
        pose_a1,
        pose_listener0,
        pose_listener1,
        pose_listener2,
        pose_listener3,
        measured_delta_t=measured_delta_t,
        sound_speed=sound_speed,
    )
    timestamp_jacobian = float(sound_speed) * TIMESTAMP_DIFFERENCE_MATRIX
    error += timestamp_jacobian @ timestamp_errors
    return error, pose_jacobians, timestamp_jacobian


def evaluate_shared_timestamp_master_geometry(
    pose_anchor: Any,
    pose_second: Any,
    pose_listener0: Any,
    pose_listener1: Any,
    measured_delta_t: float,
    timestamp_errors: Sequence[float],
    *,
    sound_speed: float = 1500.0,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Evaluate one latent-timestamp dual-master TDOA row.

    This is exactly the first row of the joint-cycle model.  It is exposed as
    a separate factor so the current relay can use the two master waves that
    it already received, and so the master-backbone mechanism control can use
    the identical physical wave/timestamp ledger without any peer state.
    """
    sound_speed = float(sound_speed)
    if not math.isfinite(sound_speed) or sound_speed <= 0.0:
        raise ValueError("sound_speed must be finite and positive")
    measured_delta_t = float(measured_delta_t)
    if not math.isfinite(measured_delta_t):
        raise ValueError("measured_delta_t must be finite")
    timestamp_errors = np.asarray(timestamp_errors, dtype=float)
    if timestamp_errors.shape != (2,) or not np.all(
        np.isfinite(timestamp_errors)
    ):
        raise ValueError("timestamp_errors must contain two finite values")

    poses = (pose_anchor, pose_second, pose_listener0, pose_listener1)
    jacobians = [np.zeros((1, 3), dtype=float) for _ in poses]
    anchor_second, h_anchor_ab, h_second_ab = _range_with_local_jacobians(
        pose_anchor, pose_second
    )
    second_listener, h_second_l1, h_listener1 = _range_with_local_jacobians(
        pose_second, pose_listener1
    )
    anchor_listener, h_anchor_l0, h_listener0 = _range_with_local_jacobians(
        pose_anchor, pose_listener0
    )
    residual = np.asarray(
        [
            anchor_second
            + second_listener
            - anchor_listener
            - sound_speed * measured_delta_t
            + sound_speed * (timestamp_errors[1] - timestamp_errors[0])
        ],
        dtype=float,
    )
    jacobians[0][0, :] = h_anchor_ab - h_anchor_l0
    jacobians[1][0, :] = h_second_ab + h_second_l1
    jacobians[2][0, :] = -h_listener0
    jacobians[3][0, :] = h_listener1
    timestamp_jacobian = sound_speed * np.asarray([[-1.0, 1.0]], dtype=float)
    return residual, jacobians, timestamp_jacobian


def create_shared_timestamp_master_factor(
    idx_anchor: Any,
    idx_second: Any,
    idx_listener0: Any,
    idx_listener1: Any,
    timestamp_error_keys: Sequence[Any],
    measured_delta_t: float,
    *,
    sound_speed: float = 1500.0,
) -> Any:
    """Create the 4-Pose2/2-scalar latent-timestamp master row."""
    logical_pose_keys = (
        idx_anchor,
        idx_second,
        idx_listener0,
        idx_listener1,
    )
    pose_keys, logical_to_unique = _deduplicate_graph_keys(logical_pose_keys)
    timestamp_error_keys = tuple(timestamp_error_keys)
    if len(timestamp_error_keys) != 2 or len(
        {int(key) for key in timestamp_error_keys}
    ) != 2:
        raise ValueError("shared master row requires two distinct timestamp keys")
    all_keys = list(pose_keys) + list(timestamp_error_keys)
    if len({int(key) for key in all_keys}) != len(all_keys):
        raise ValueError("Pose2 and timestamp keys must be globally distinct")

    def error_func(this, values, H):
        poses = [values.atPose2(key) for key in logical_pose_keys]
        timestamp_errors = np.asarray(
            [values.atDouble(key) for key in timestamp_error_keys], dtype=float
        )
        error, pose_jacobians, timestamp_jacobian = (
            evaluate_shared_timestamp_master_geometry(
                *poses,
                measured_delta_t=measured_delta_t,
                timestamp_errors=timestamp_errors,
                sound_speed=sound_speed,
            )
        )
        if H is not None:
            accumulated = [
                np.zeros((1, 3), dtype=float) for _ in pose_keys
            ]
            for logical_index, jacobian in enumerate(pose_jacobians):
                accumulated[logical_to_unique[logical_index]] += jacobian
            for index, jacobian in enumerate(accumulated):
                H[index] = jacobian
            for index in range(2):
                H[len(pose_keys) + index] = timestamp_jacobian[:, [index]]
        return error

    deterministic_noise = gtsam.noiseModel.Isotropic.Sigma(
        1, _DETERMINISTIC_RELATION_SIGMA_M
    )
    return gtsam.CustomFactor(deterministic_noise, all_keys, error_func)  # type: ignore


def build_timestamp_chain_incidence(
    cycle_wave_ids: Sequence[Sequence[int]],
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Build the stacked physical-timestamp incidence matrix for a chain."""
    normalized = tuple(tuple(int(wave) for wave in cycle) for cycle in cycle_wave_ids)
    if any(len(cycle) != 4 for cycle in normalized):
        raise ValueError("each cycle must reference four physical wave IDs")
    unique_wave_ids = tuple(dict.fromkeys(wave for cycle in normalized for wave in cycle))
    column = {wave_id: index for index, wave_id in enumerate(unique_wave_ids)}
    incidence = np.zeros((3 * len(normalized), len(unique_wave_ids)), dtype=float)
    for cycle_index, wave_ids in enumerate(normalized):
        for row in range(3):
            incidence[3 * cycle_index + row, column[wave_ids[row]]] = -1.0
            incidence[3 * cycle_index + row, column[wave_ids[row + 1]]] = 1.0
    row_gram = _small_matmul(incidence, incidence.T)
    _require_symmetric_positive_definite(
        row_gram, "timestamp difference row Gramian"
    )
    return incidence, unique_wave_ids


def build_timestamp_chain_covariance(
    cycle_wave_ids: Sequence[Sequence[int]],
    timestamp_covariance: np.ndarray,
    sound_speed: float = 1500.0,
) -> np.ndarray:
    """Return ``c^2 D_chain Sigma_timestamp D_chain.T``."""
    incidence, unique_wave_ids = build_timestamp_chain_incidence(cycle_wave_ids)
    timestamp_covariance = np.asarray(timestamp_covariance, dtype=float)
    expected_shape = (len(unique_wave_ids), len(unique_wave_ids))
    if timestamp_covariance.shape != expected_shape:
        raise ValueError(
            f"timestamp_covariance must have shape {expected_shape}"
        )
    _require_symmetric_positive_definite(
        timestamp_covariance, "timestamp_covariance"
    )
    sound_speed = float(sound_speed)
    if not math.isfinite(sound_speed) or sound_speed <= 0.0:
        raise ValueError("sound_speed must be finite and positive")
    covariance = sound_speed * sound_speed * _small_matmul(
        _small_matmul(incidence, timestamp_covariance), incidence.T
    )
    covariance = 0.5 * (covariance + covariance.T)
    _require_symmetric_positive_definite(covariance, "chain covariance")
    return covariance


def create_scalar_cycle_snoop_factor(
    idx_a0: Any,
    idx_b: Any,
    idx_relay: Any,
    idx_a1: Any,
    idx_listener0: Any,
    idx_listener1: Any,
    idx_listener2: Any,
    idx_listener3: Any,
    measured_delta_t: Sequence[float],
    *,
    residual_row: int,
    noise_model: Any,
    sound_speed: float = 1500.0,
    pose_offsets: Mapping[str, Any] | None = None,
) -> Any:
    """Create one scalar row for the independent-TDOA paper ablation.

    The proposed estimator must use :func:`create_cycle_snoop_factor`.  This
    helper intentionally discards cross-row timestamp covariance and exists
    only to implement the scalar-independent comparison requested by the
    experiment plan.
    """
    residual_row = int(residual_row)
    if residual_row not in (0, 1, 2):
        raise ValueError("residual_row must be 0, 1, or 2")
    measurement = _coerce_measurement(measured_delta_t)
    logical_keys = [
        idx_a0,
        idx_b,
        idx_relay,
        idx_a1,
        idx_listener0,
        idx_listener1,
        idx_listener2,
        idx_listener3,
    ]
    supplied_offsets = dict(pose_offsets or {})
    unknown_offset_names = sorted(
        set(supplied_offsets) - set(LOGICAL_POSE_ORDER)
    )
    if unknown_offset_names:
        raise ValueError(f"unknown pose offset names: {unknown_offset_names}")
    logical_offsets = [
        supplied_offsets.get(name, gtsam.Pose2())
        for name in LOGICAL_POSE_ORDER
    ]

    unique_keys: list[Any] = []
    logical_to_unique: list[int] = []
    unique_index_by_key: dict[int, int] = {}
    for key in logical_keys:
        key_integer = int(key)
        if key_integer not in unique_index_by_key:
            unique_index_by_key[key_integer] = len(unique_keys)
            unique_keys.append(key)
        logical_to_unique.append(unique_index_by_key[key_integer])

    def error_func(this, values, H):
        event_poses = [
            values.atPose2(key).compose(offset)
            for key, offset in zip(logical_keys, logical_offsets)
        ]
        error, logical_jacobians = evaluate_cycle_snoop_geometry(
            *event_poses,
            measured_delta_t=measurement,
            sound_speed=sound_speed,
        )
        if H is not None:
            accumulated = [
                np.zeros((1, 3), dtype=float) for _ in unique_keys
            ]
            for logical_index, logical_jacobian in enumerate(
                logical_jacobians
            ):
                offset_adjoint = np.asarray(
                    logical_offsets[logical_index]
                    .inverse()
                    .AdjointMap(),
                    dtype=float,
                )
                mapped = _small_matmul(
                    logical_jacobian[[residual_row], :],
                    offset_adjoint,
                )
                accumulated[logical_to_unique[logical_index]] += mapped
            for unique_index, jacobian in enumerate(accumulated):
                H[unique_index] = jacobian
        return np.asarray([error[residual_row]], dtype=float)

    return gtsam.CustomFactor(noise_model, unique_keys, error_func)  # type: ignore
