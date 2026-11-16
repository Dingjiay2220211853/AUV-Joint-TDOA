"""DVL/gyro preintegration for the planar unicycle motion model."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable

import gtsam  # type: ignore
import numpy as np


_EPS = 1e-12


@dataclass(frozen=True)
class ContinuousNoiseDensity:
    """Body-frame white-noise spectral densities for ``[v, v_lat, omega]``."""

    q_v: float
    q_lateral: float
    q_omega: float

    @classmethod
    def from_sample_sigmas(
        cls,
        velocity_sigma: float,
        angular_rate_sigma: float,
        lateral_velocity_sigma: float,
        sample_period: float,
    ) -> "ContinuousNoiseDensity":
        sample_period = float(sample_period)
        if sample_period <= 0.0:
            raise ValueError("sample_period must be positive")
        return cls(
            q_v=float(velocity_sigma) ** 2 * sample_period,
            q_lateral=float(lateral_velocity_sigma) ** 2 * sample_period,
            q_omega=float(angular_rate_sigma) ** 2 * sample_period,
        )

    def matrix(self) -> np.ndarray:
        values = (self.q_v, self.q_lateral, self.q_omega)
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("continuous noise densities must be finite and positive")
        return np.diag(values)


_GAUSS_LEGENDRE_16 = (
    (-0.9894009349916499, 0.027152459411754095),
    (-0.9445750230732326, 0.06225352393864789),
    (-0.8656312023878318, 0.09515851168249278),
    (-0.7554044083550030, 0.12462897125553387),
    (-0.6178762444026438, 0.14959598881657673),
    (-0.4580167776572274, 0.16915651939500254),
    (-0.2816035507792589, 0.18260341504492360),
    (-0.09501250983763744, 0.18945061045506850),
    (+0.09501250983763744, 0.18945061045506850),
    (+0.2816035507792589, 0.18260341504492360),
    (+0.4580167776572274, 0.16915651939500254),
    (+0.6178762444026438, 0.14959598881657673),
    (+0.7554044083550030, 0.12462897125553387),
    (+0.8656312023878318, 0.09515851168249278),
    (+0.9445750230732326, 0.06225352393864789),
    (+0.9894009349916499, 0.027152459411754095),
)


def _sinc(angle: float) -> float:
    if abs(angle) < 1e-6:
        angle2 = angle * angle
        return 1.0 - angle2 / 6.0 + angle2 * angle2 / 120.0
    return math.sin(angle) / angle


def _cosc(angle: float) -> float:
    if abs(angle) < 1e-6:
        angle2 = angle * angle
        return angle * (
            0.5 - angle2 / 24.0 + angle2 * angle2 / 720.0
        )
    return (1.0 - math.cos(angle)) / angle


def exact_unicycle_increment(
    velocity: float,
    angular_rate: float,
    dt: float,
) -> gtsam.Pose2:
    """Integrate a constant unicycle control exactly over ``dt``."""
    dt = max(float(dt), 0.0)
    velocity = float(velocity)
    angle = float(angular_rate) * dt
    distance = velocity * dt
    return gtsam.Pose2(
        distance * _sinc(angle),
        distance * _cosc(angle),
        angle,
    )


def preintegrate_continuous_segment(
    t0: float,
    t1: float,
    continuous_covariance: ContinuousNoiseDensity | np.ndarray,
    *,
    velocity: float,
    angular_rate: float,
) -> tuple[gtsam.Pose2, np.ndarray]:
    """Integrate one event-to-event interval under continuous white noise.

    ``continuous_covariance`` is the body-twist spectral-density matrix in
    ``[forward, lateral, yaw-rate]`` order.  The covariance integral is
    evaluated in the right-local tangent frame at ``t1``.  Consequently its
    semigroup is exactly the same Pose2 composition rule used by
    :func:`compose_pose_covariance`; changing event boundaries does not invent
    another sensor observation.
    """
    t0 = float(t0)
    t1 = float(t1)
    if not math.isfinite(t0) or not math.isfinite(t1) or t1 < t0:
        raise ValueError("segment times must be finite and ordered")
    duration = t1 - t0
    increment = exact_unicycle_increment(velocity, angular_rate, duration)
    if duration == 0.0:
        return increment, np.zeros((3, 3), dtype=float)

    if isinstance(continuous_covariance, ContinuousNoiseDensity):
        density = continuous_covariance.matrix()
    else:
        density = np.asarray(continuous_covariance, dtype=float)
        if density.shape != (3, 3) or not np.all(np.isfinite(density)):
            raise ValueError("continuous covariance must be finite 3x3")
        density = 0.5 * (density + density.T)
    # The experiment uses strictly positive diagonal densities.  This direct
    # check avoids adding per-segment numerical jitter, which would break the
    # required split/composition identity.
    for index in range(3):
        if float(density[index, index]) <= 0.0:
            raise ValueError("continuous covariance must be positive definite")

    covariance = np.zeros((3, 3), dtype=float)
    half_duration = 0.5 * duration
    for node, weight in _GAUSS_LEGENDRE_16:
        elapsed = half_duration * (node + 1.0)
        remaining = duration - elapsed
        future_increment = exact_unicycle_increment(
            velocity, angular_rate, remaining
        )
        transport = np.asarray(
            future_increment.inverse().AdjointMap(), dtype=float
        )
        integrand = _small_matmul(
            _small_matmul(transport, density), transport.T
        )
        covariance += half_duration * weight * integrand
    covariance = 0.5 * (covariance + covariance.T)
    return increment, covariance


def preintegrate_continuous_trajectory(
    t0: float,
    t1: float,
    continuous_covariance: ContinuousNoiseDensity | np.ndarray,
    *,
    pose_at: Callable[[float], gtsam.Pose2],
) -> tuple[gtsam.Pose2, np.ndarray]:
    """Preintegrate a deterministic time-varying trajectory and white noise.

    ``pose_at`` supplies the noise-free world pose at an arbitrary physical
    time.  The returned mean and covariance use the same right-local tangent
    convention as :func:`preintegrate_continuous_segment`, so independently
    sampled atomic event intervals can be composed without inventing another
    measurement at an event boundary.
    """
    t0 = float(t0)
    t1 = float(t1)
    if not math.isfinite(t0) or not math.isfinite(t1) or t1 < t0:
        raise ValueError("segment times must be finite and ordered")
    duration = t1 - t0
    start_pose = pose_at(t0)
    end_pose = pose_at(t1)
    increment = start_pose.between(end_pose)
    if duration == 0.0:
        return increment, np.zeros((3, 3), dtype=float)

    if isinstance(continuous_covariance, ContinuousNoiseDensity):
        density = continuous_covariance.matrix()
    else:
        density = np.asarray(continuous_covariance, dtype=float)
        if density.shape != (3, 3) or not np.all(np.isfinite(density)):
            raise ValueError("continuous covariance must be finite 3x3")
        density = 0.5 * (density + density.T)
    # Match the constant-control path and avoid invoking a platform BLAS just
    # to validate the experiment's strictly positive diagonal density.
    for index in range(3):
        if float(density[index, index]) <= 0.0:
            raise ValueError("continuous covariance must be positive definite")

    covariance = np.zeros((3, 3), dtype=float)
    half_duration = 0.5 * duration
    for node, weight in _GAUSS_LEGENDRE_16:
        sample_time = t0 + half_duration * (node + 1.0)
        future_increment = pose_at(sample_time).between(end_pose)
        transport = np.asarray(
            future_increment.inverse().AdjointMap(), dtype=float
        )
        integrand = _small_matmul(
            _small_matmul(transport, density), transport.T
        )
        covariance += half_duration * weight * integrand
    return increment, 0.5 * (covariance + covariance.T)


def _control_jacobian(
    velocity: float,
    angular_rate: float,
    dt: float,
) -> np.ndarray:
    """Numerically map sampled control errors to Pose2 tangent errors."""
    nominal = exact_unicycle_increment(velocity, angular_rate, dt)
    jacobian = np.zeros((3, 2), dtype=float)
    perturbations = [
        max(1e-7, abs(float(velocity)) * 1e-6),
        max(1e-8, abs(float(angular_rate)) * 1e-6),
    ]
    controls = [float(velocity), float(angular_rate)]
    for column, epsilon in enumerate(perturbations):
        plus = controls.copy()
        minus = controls.copy()
        plus[column] += epsilon
        minus[column] -= epsilon
        plus_pose = exact_unicycle_increment(plus[0], plus[1], dt)
        minus_pose = exact_unicycle_increment(minus[0], minus[1], dt)
        plus_error = np.asarray(
            nominal.localCoordinates(plus_pose), dtype=float
        )
        minus_error = np.asarray(
            nominal.localCoordinates(minus_pose), dtype=float
        )
        jacobian[:, column] = (
            plus_error - minus_error
        ) / (2.0 * epsilon)
    return jacobian


def preintegrate_constant_measurement(
    velocity: float,
    angular_rate: float,
    dt: float,
    velocity_sigma: float,
    angular_rate_sigma: float,
    lateral_velocity_sigma: float,
) -> tuple[gtsam.Pose2, np.ndarray]:
    """Return exact mean increment and full local tangent covariance."""
    dt = max(float(dt), 0.0)
    increment = exact_unicycle_increment(velocity, angular_rate, dt)
    control_jacobian = _control_jacobian(
        velocity, angular_rate, dt
    )
    sigmas = [
        max(float(velocity_sigma), 0.0),
        max(float(angular_rate_sigma), 0.0),
    ]
    covariance = np.zeros((3, 3), dtype=float)
    for noise_index, sigma in enumerate(sigmas):
        column = control_jacobian[:, noise_index]
        for row in range(3):
            for column_index in range(3):
                covariance[row, column_index] += (
                    float(column[row])
                    * float(column[column_index])
                    * sigma
                    * sigma
                )
    lateral_std = max(float(lateral_velocity_sigma), 0.0) * dt
    covariance[1, 1] += lateral_std * lateral_std
    covariance = 0.5 * (covariance + covariance.T)
    covariance[np.diag_indices(3)] += _EPS
    return increment, covariance


def _small_matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.zeros((left.shape[0], right.shape[1]), dtype=float)
    for row in range(left.shape[0]):
        for column in range(right.shape[1]):
            result[row, column] = sum(
                float(left[row, inner]) * float(right[inner, column])
                for inner in range(left.shape[1])
            )
    return result


def compose_pose_covariance(
    base_covariance: np.ndarray,
    offset: gtsam.Pose2,
    offset_covariance: np.ndarray,
) -> np.ndarray:
    """Propagate right-local covariance through ``base * offset``.

    This is also used when a timestamped belief is reported at an acoustic
    event lying between two 10 Hz odometry nodes.
    """
    adjoint = np.asarray(offset.inverse().AdjointMap(), dtype=float)
    propagated = _small_matmul(
        _small_matmul(
            adjoint,
            np.asarray(base_covariance, dtype=float),
        ),
        adjoint.T,
    )
    result = propagated + np.asarray(offset_covariance, dtype=float)
    result = 0.5 * (result + result.T)
    return result


@dataclass
class PreintegratedOdometry:
    """Accumulate independent sampled DVL/gyro intervals."""

    delta: gtsam.Pose2 = field(default_factory=gtsam.Pose2)
    covariance: np.ndarray = field(
        default_factory=lambda: np.zeros((3, 3), dtype=float)
    )
    duration: float = 0.0
    sample_count: int = 0

    def __post_init__(self):
        self.covariance = np.asarray(self.covariance, dtype=float).copy()

    def integrate(
        self,
        velocity: float,
        angular_rate: float,
        dt: float,
        velocity_sigma: float,
        angular_rate_sigma: float,
        lateral_velocity_sigma: float,
    ) -> None:
        increment, increment_covariance = (
            preintegrate_constant_measurement(
                velocity,
                angular_rate,
                dt,
                velocity_sigma,
                angular_rate_sigma,
                lateral_velocity_sigma,
            )
        )
        # Right-local tangent covariance for T = T_previous * increment.
        self.covariance = compose_pose_covariance(
            self.covariance,
            increment,
            increment_covariance,
        )
        self.delta = self.delta.compose(increment)
        self.duration += max(float(dt), 0.0)
        self.sample_count += 1

    def reset(self) -> None:
        self.delta = gtsam.Pose2()
        self.covariance = np.zeros((3, 3), dtype=float)
        self.duration = 0.0
        self.sample_count = 0
