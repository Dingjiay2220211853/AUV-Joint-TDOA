"""Shared physical compiler for fixed schedule prediction and simulation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import gtsam
from .formal_candidate_schedules import ACTION_DUAL_MASTER, ACTION_JOINT, FixedCandidateSchedule, ScheduledAcousticAction
from .formal_protocol import FormalExperimentConfig, _RawWave, _raw_wave

@dataclass(frozen=True)
class CompiledNominalAction:
    """One schedule action after exact physical-wave compilation."""

    intent: ScheduledAcousticAction
    waves: tuple[_RawWave, ...]
    anchor_wait_s: float = 0.0


@dataclass(frozen=True)
class CompiledNominalSchedule:
    """Physical ledger used by one DNCP candidate."""

    schedule: FixedCandidateSchedule
    actions: tuple[CompiledNominalAction, ...]
    waves: tuple[_RawWave, ...]

    @property
    def physical_wave_count(self) -> int:
        return len(self.waves)

    def validate(self) -> None:
        self.schedule.validate()
        if self.physical_wave_count != self.schedule.wave_budget:
            raise AssertionError(
                f"{self.schedule.candidate_id} compiled "
                f"{self.physical_wave_count} waves, expected "
                f"{self.schedule.wave_budget}"
            )
        ids = [int(item.wave_id) for item in self.waves]
        if ids != list(range(len(ids))):
            raise AssertionError("compiled wave IDs are not contiguous")
        if len(ids) != len(set(ids)):
            raise AssertionError("a physical wave was created more than once")
        if len(self.actions) != len(self.schedule.actions):
            raise AssertionError("compiled action count changed")
        for compiled in self.actions:
            intent = compiled.intent
            expected = 2 if intent.action_type == ACTION_DUAL_MASTER else 4
            if len(compiled.waves) != expected:
                raise AssertionError("compiled action has the wrong wave count")
            first = compiled.waves[0]
            if first.sender_id != intent.anchor_master_id:
                raise AssertionError("action starts from the wrong anchor master")
            if abs(first.send_time - intent.start_time_s) > 1e-10:
                raise AssertionError("compiled action start time changed")
            if intent.action_type == ACTION_JOINT:
                if compiled.waves[2].sender_id != intent.relay_id:
                    raise AssertionError("Joint action uses the wrong relay")
                if compiled.anchor_wait_s < -1e-10:
                    raise AssertionError("Joint action has a negative anchor wait")


def compile_nominal_schedule(
    schedule: FixedCandidateSchedule,
    config: FormalExperimentConfig,
    pose_at: Callable[[str, float], gtsam.Pose2],
) -> CompiledNominalSchedule:
    """Turn a fixed intent into one exact 200-wave physical ledger.

    A terminal anchor wave of a Joint action is reused as the starting wave
    of the immediately following action when anchor identity and time match.
    This is the finite-horizon accounting that makes the hybrid candidates
    exactly, rather than asymptotically, equal to D100's 200 transmissions.
    """

    waves: list[_RawWave] = []
    actions: list[CompiledNominalAction] = []
    pending_boundary: _RawWave | None = None

    def new_wave(
        sender_id: str,
        send_time_s: float,
        relay_id: str | None,
    ) -> _RawWave:
        wave = _raw_wave(
            len(waves),
            f"dncp:{schedule.candidate_id}",
            sender_id,
            float(send_time_s),
            relay_id,
            config,
            pose_at,
        )
        waves.append(wave)
        return wave

    for intent in schedule.actions:
        if (
            pending_boundary is not None
            and pending_boundary.sender_id == intent.anchor_master_id
            and abs(pending_boundary.send_time - intent.start_time_s) <= 1e-10
        ):
            wave_0 = pending_boundary
            pending_boundary = None
        else:
            pending_boundary = None
            wave_0 = new_wave(
                intent.anchor_master_id, intent.start_time_s, intent.relay_id
            )

        wave_1 = new_wave(
            intent.second_master_id,
            wave_0.receive_time(intent.second_master_id),
            intent.relay_id,
        )
        if intent.action_type == ACTION_DUAL_MASTER:
            actions.append(CompiledNominalAction(intent, (wave_0, wave_1)))
            continue
        if intent.action_type != ACTION_JOINT or intent.relay_id is None:
            raise ValueError("candidate contains an unknown acoustic action")
        wave_2 = new_wave(
            intent.relay_id,
            wave_1.receive_time(intent.relay_id),
            intent.relay_id,
        )
        terminal_send_time = intent.start_time_s + 3.0
        wave_3 = new_wave(
            intent.anchor_master_id,
            terminal_send_time,
            intent.relay_id,
        )
        anchor_receive_time = wave_2.receive_time(intent.anchor_master_id)
        anchor_wait_s = terminal_send_time - anchor_receive_time
        if anchor_wait_s < -1e-10:
            raise ValueError(
                f"Joint action at {intent.start_time_s:g}s cannot close by "
                f"{terminal_send_time:g}s"
            )
        pending_boundary = wave_3
        actions.append(
            CompiledNominalAction(
                intent,
                (wave_0, wave_1, wave_2, wave_3),
                max(0.0, float(anchor_wait_s)),
            )
        )

    compiled = CompiledNominalSchedule(
        schedule=schedule,
        actions=tuple(actions),
        waves=tuple(waves),
    )
    compiled.validate()
    return compiled



