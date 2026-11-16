"""Typed messages shared by distributed cooperative factor graphs.

The two public message classes deliberately encode different trust boundaries:
followers may publish only raw-measurement odometry increments, whereas masters
publish independent globally referenced navigation states.
"""

from __future__ import annotations

from dataclasses import dataclass

import gtsam  # type: ignore
import numpy as np


SOURCE_ODOMETRY_PROVENANCE = frozenset({"SOURCE", "ODOMETRY"})
MASTER_NAVIGATION_PROVENANCE = frozenset(
    {"MASTER", "ABSOLUTE", "SOURCE_NAVIGATION"}
)


@dataclass(frozen=True)
class OdomMeasurement:
    measurement_id: str
    agent_id: str
    start_index: int
    end_index: int
    start_time: float
    end_time: float
    measured_velocity: float
    measured_angular_rate: float
    delta_pose: gtsam.Pose2
    covariance: np.ndarray


@dataclass(frozen=True)
class EventOdomRef:
    """Graph state and deterministic offset representing one event pose."""

    agent_id: str
    event_time: float
    base_index: int
    base_time: float
    offset: gtsam.Pose2
    offset_covariance: np.ndarray
    measurement_id: str | None


@dataclass(frozen=True)
class GridSourceOdomIncrement:
    """Legacy 10 Hz summary retained by the frozen earlier experiments."""

    source_id: str
    source_epoch: int
    start_index: int
    end_index: int
    start_time: float
    end_time: float
    delta_pose: gtsam.Pose2
    covariance: np.ndarray
    measurement_ids: tuple[str, ...]
    provenance: frozenset[str] = SOURCE_ODOMETRY_PROVENANCE

    @property
    def summary_id(self) -> tuple[str, int, int, int]:
        return (
            self.source_id,
            self.source_epoch,
            self.start_index,
            self.end_index,
        )


@dataclass(frozen=True)
class MasterPoseMessage:
    """Independent master navigation state ``(X_bar(t), Sigma(t), t)``."""

    master_id: str
    source_epoch: int
    sequence: int
    timestamp: float
    state_index: int
    pose: gtsam.Pose2
    covariance: np.ndarray
    provenance: frozenset[str] = MASTER_NAVIGATION_PROVENANCE

    @property
    def message_id(self) -> tuple[str, int, int]:
        return self.master_id, self.source_epoch, self.sequence


@dataclass(frozen=True)
class SourceOdomIncrement:
    """Event-to-event follower summary made from non-overlapping segments."""

    source_id: str
    source_epoch: int
    start_event_id: int
    end_event_id: int
    start_time: float
    end_time: float
    delta_pose: gtsam.Pose2
    covariance: np.ndarray
    segment_ids: tuple[str, ...]
    provenance: frozenset[str] = SOURCE_ODOMETRY_PROVENANCE

    @property
    def summary_id(self) -> tuple[str, int, int, int]:
        return (
            self.source_id,
            self.source_epoch,
            self.start_event_id,
            self.end_event_id,
        )


@dataclass(frozen=True)
class FollowerInitialStateMessage:
    """Deployment-time follower prior shared once before the mission."""

    follower_id: str
    pose: gtsam.Pose2
    covariance: np.ndarray
    timestamp: float
    source_epoch: int

    @property
    def message_id(self) -> tuple[str, int, str]:
        return self.follower_id, self.source_epoch, "INITIAL"


def _require_spd(covariance: np.ndarray, name: str) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
        raise ValueError(f"{name} must be finite 3x3")
    symmetric = 0.5 * (covariance + covariance.T)
    if not np.allclose(covariance, symmetric, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    lower = np.zeros((3, 3), dtype=float)
    for row in range(3):
        for column in range(row + 1):
            subtotal = sum(
                float(lower[row, inner]) * float(lower[column, inner])
                for inner in range(column)
            )
            if row == column:
                pivot = float(symmetric[row, row]) - subtotal
                if not np.isfinite(pivot) or pivot <= 0.0:
                    raise ValueError(f"{name} must be positive definite")
                lower[row, column] = np.sqrt(pivot)
            else:
                lower[row, column] = (
                    float(symmetric[row, column]) - subtotal
                ) / lower[column, column]
    return symmetric


class FollowerSourcePublisher:
    """Strict capability boundary for event-time follower output."""

    @staticmethod
    def publish(
        increment: SourceOdomIncrement | None,
    ) -> SourceOdomIncrement | None:
        if increment is None:
            return None
        if not isinstance(increment, SourceOdomIncrement):
            raise TypeError("follower publisher accepts only SourceOdomIncrement")
        if increment.provenance != SOURCE_ODOMETRY_PROVENANCE:
            raise ValueError("source increment has contaminated provenance")
        if increment.source_epoch < 0:
            raise ValueError("source epoch must be nonnegative")
        if increment.end_event_id <= increment.start_event_id:
            raise ValueError("source event IDs must advance")
        if not np.isfinite(increment.start_time) or not np.isfinite(
            increment.end_time
        ):
            raise ValueError("source summary timestamps must be finite")
        if increment.end_time <= increment.start_time:
            raise ValueError("source summary time must advance")
        if not increment.segment_ids:
            raise ValueError("source summary must contain at least one segment")
        if len(increment.segment_ids) != len(set(increment.segment_ids)):
            raise ValueError("source summary contains duplicate segment IDs")
        _require_spd(increment.covariance, "source covariance")
        return increment


class GridFollowerSourcePublisher:
    """Compatibility boundary for the frozen grid/offset experiments."""

    @staticmethod
    def publish(
        increment: GridSourceOdomIncrement | None,
    ) -> GridSourceOdomIncrement | None:
        if increment is None:
            return None
        if not isinstance(increment, GridSourceOdomIncrement):
            raise TypeError(
                "grid publisher accepts only GridSourceOdomIncrement"
            )
        if increment.provenance != SOURCE_ODOMETRY_PROVENANCE:
            raise ValueError("source increment has contaminated provenance")
        return increment


class FollowerInitialStatePublisher:
    """Validate the one-time deployment prior shared by each follower."""

    @staticmethod
    def publish(
        message: FollowerInitialStateMessage,
    ) -> FollowerInitialStateMessage:
        if not isinstance(message, FollowerInitialStateMessage):
            raise TypeError(
                "initial-state publisher accepts only FollowerInitialStateMessage"
            )
        if message.source_epoch < 0:
            raise ValueError("source epoch must be nonnegative")
        if not np.isfinite(message.timestamp):
            raise ValueError("initial timestamp must be finite")
        _require_spd(message.covariance, "follower initial covariance")
        return message


class MasterNavigationPublisher:
    """Capability boundary for master source-navigation output."""

    @staticmethod
    def publish(message: MasterPoseMessage) -> MasterPoseMessage:
        if not isinstance(message, MasterPoseMessage):
            raise TypeError("master publisher accepts only MasterPoseMessage")
        if message.provenance != MASTER_NAVIGATION_PROVENANCE:
            raise ValueError("master navigation message has contaminated provenance")
        if message.source_epoch < 0:
            raise ValueError("master source epoch must be nonnegative")
        if not np.isfinite(message.timestamp):
            raise ValueError("master timestamp must be finite")
        _require_spd(message.covariance, "master covariance")
        return message
