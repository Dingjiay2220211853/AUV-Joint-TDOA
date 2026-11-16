"""Deterministic fixed candidate schedules for the offline Joint-TDOA study.

This module only constructs preregistered communication intents.  It does not
sample measurements, propagate a cooperative posterior, build an experiment
dataset, or rank candidates.  A later protocol compiler must still turn each
intent into physical waves and audit event-time feasibility.

The historical hybrid candidates use two 100 s anchor blocks.  The paper's
directional-only candidates instead reverse the anchor after five consecutive
Joint cycles and use fourteen timestamp-chain blocks while retaining the exact
200 s / 200-wave budget.  Both families remain available through separate
builders in this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import math
from typing import Mapping, Sequence


ACTION_DUAL_MASTER = "dual_master"
ACTION_JOINT = "joint"

TIMING_UNIFORM = "uniform"
TIMING_WEAK_WINDOWS = "weak_windows"

RELAY_NONE = "none"
RELAY_UNIFORM = "uniform"
RELAY_DIRECTIONAL = "directional"

CANDIDATE_D100 = "D100"
CANDIDATE_J66_RR = "J66-RR"
CANDIDATE_H42_U = "H42-U"
CANDIDATE_H42_D = "H42-D"
CANDIDATE_H42_DW = "H42-DW"
CANDIDATE_H24_DW = "H24-DW"

# Paper-section-II schedules.  These are deliberately separate from the
# historical H42/H24 development candidates: all four Joint schedules use the
# same acoustic action times and differ only in the cyclic relay word.
CANDIDATE_D_ONLY = "D-only"
CANDIDATE_J_RR = "J-RR"
CANDIDATE_J_D1 = "J-D1"
CANDIDATE_J_D2 = "J-D2"
CANDIDATE_J_D3 = "J-D3"

DIRECTIONAL_JOINT_CANDIDATE_IDS = (
    CANDIDATE_D_ONLY,
    CANDIDATE_J_RR,
    CANDIDATE_J_D1,
    CANDIDATE_J_D2,
    CANDIDATE_J_D3,
)

# One 62-action relay period.  C/F are directionally complementary to the
# north-south master baseline, D/G are nearly redundant, and E/H are
# intermediate at the fixed 15 degree mission orientation.  The integer
# quotas preserve opposite-agent symmetry exactly over the finite 200 s run.
# Sixty-two Joint actions and fourteen timestamp-chain blocks consume exactly
# 3 * 62 + 14 = 200 physical waves.
DIRECTIONAL_JOINT_RELAY_QUOTAS = {
    CANDIDATE_J_RR: (10, 10, 11, 10, 10, 11),
    CANDIDATE_J_D1: (14, 7, 10, 14, 7, 10),
    CANDIDATE_J_D2: (17, 4, 10, 17, 4, 10),
    CANDIDATE_J_D3: (21, 4, 6, 21, 4, 6),
}

# The first twelve blocks contain five Joint cycles each.  The two one-cycle
# terminal blocks are the exact finite-horizon truncation needed to retain
# equal A/B exposure and the 200-wave budget.  Every block reverses the anchor
# order, so no vehicle sees one ordered geometry for longer than 15 s.
PAPER_JOINT_BLOCK_LENGTHS = (5,) * 12 + (1, 1)
PAPER_JOINT_ACTION_COUNT = sum(PAPER_JOINT_BLOCK_LENGTHS)
PAPER_JOINT_BLOCK_COUNT = len(PAPER_JOINT_BLOCK_LENGTHS)

FIXED_CANDIDATE_IDS = (
    CANDIDATE_D100,
    CANDIDATE_J66_RR,
    CANDIDATE_H42_U,
    CANDIDATE_H42_D,
    CANDIDATE_H42_DW,
    CANDIDATE_H24_DW,
)

DEFAULT_FOLLOWER_IDS = ("C", "D", "E", "F", "G", "H")
MACROS_PER_ANCHOR_BLOCK = 16
MACRO_DURATION_S = 6.0
ANCHOR_BLOCK_DURATION_S = 100.0
FIXED_DURATION_S = 200.0
FIXED_WAVE_BUDGET = 200


@dataclass(frozen=True)
class FixedCandidateDefinition:
    """Frozen structure of one physically motivated schedule candidate."""

    candidate_id: str
    ordinary_action_count: int
    joint_action_count: int
    joint_macros_by_block: tuple[int, int] | None
    relay_policy: str
    timing_policy: str
    directional_epsilon: float | None = None
    boundary_waves_by_block: tuple[int, ...] = (0, 0)

    @property
    def boundary_wave_count(self) -> int:
        return sum(self.boundary_waves_by_block)

    @property
    def expected_wave_count(self) -> int:
        return (
            2 * self.ordinary_action_count
            + 3 * self.joint_action_count
            + self.boundary_wave_count
        )


@dataclass(frozen=True)
class ScheduledAcousticAction:
    """One deterministic measurement action in a fixed candidate plan."""

    sequence_index: int
    action_type: str
    start_time_s: float
    block_id: int
    anchor_master_id: str
    second_master_id: str
    relay_id: str | None
    macro_index: int | None
    nominal_wave_cost: int


@dataclass(frozen=True)
class FixedCandidateSchedule:
    """Auditable fixed schedule returned by :func:`build_fixed_candidate_schedule`."""

    definition: FixedCandidateDefinition
    follower_ids: tuple[str, ...]
    sector_angles_rad: tuple[float, ...]
    master_baseline_angle_rad: float
    actions: tuple[ScheduledAcousticAction, ...]
    relay_quotas: tuple[tuple[str, int], ...]
    relay_quotas_by_block: tuple[tuple[str, int, int], ...]
    selected_joint_macros_by_block: tuple[tuple[int, ...], tuple[int, ...]]
    duration_s: float
    wave_budget: int
    fingerprint: str

    @property
    def candidate_id(self) -> str:
        return self.definition.candidate_id

    @property
    def ordinary_action_count(self) -> int:
        return sum(action.action_type == ACTION_DUAL_MASTER for action in self.actions)

    @property
    def joint_action_count(self) -> int:
        return sum(action.action_type == ACTION_JOINT for action in self.actions)

    @property
    def physical_wave_count(self) -> int:
        return (
            sum(action.nominal_wave_cost for action in self.actions)
            + self.definition.boundary_wave_count
        )

    @property
    def relay_plan(self) -> tuple[str, ...]:
        return tuple(
            action.relay_id
            for action in self.actions
            if action.action_type == ACTION_JOINT and action.relay_id is not None
        )

    def validate(self) -> None:
        if self.duration_s != FIXED_DURATION_S:
            raise AssertionError("fixed candidates require a 200 s horizon")
        if self.wave_budget != FIXED_WAVE_BUDGET:
            raise AssertionError("fixed candidates require a 200-wave budget")
        if self.definition.expected_wave_count != self.wave_budget:
            raise AssertionError("candidate definition does not consume 200 waves")
        if self.physical_wave_count != self.wave_budget:
            raise AssertionError("generated schedule does not consume 200 waves")
        if self.ordinary_action_count != self.definition.ordinary_action_count:
            raise AssertionError("ordinary action count differs from definition")
        if self.joint_action_count != self.definition.joint_action_count:
            raise AssertionError("Joint action count differs from definition")
        if tuple(action.sequence_index for action in self.actions) != tuple(
            range(len(self.actions))
        ):
            raise AssertionError("action sequence indices are not contiguous")
        action_times = tuple(action.start_time_s for action in self.actions)
        if action_times != tuple(sorted(action_times)):
            raise AssertionError("action times are not monotonic")
        if len(action_times) != len(set(action_times)):
            raise AssertionError("two actions occupy the same nominal start time")
        if any(time < 0.0 or time >= self.duration_s for time in action_times):
            raise AssertionError("action time lies outside the fixed horizon")
        if any(
            action.action_type == ACTION_DUAL_MASTER and action.relay_id is not None
            for action in self.actions
        ):
            raise AssertionError("ordinary actions cannot have a relay")
        if any(
            action.action_type == ACTION_JOINT
            and action.relay_id not in self.follower_ids
            for action in self.actions
        ):
            raise AssertionError("Joint action is missing a valid relay")
        quota = dict(self.relay_quotas)
        actual = {follower_id: 0 for follower_id in self.follower_ids}
        actual_by_anchor = {
            follower_id: [0, 0] for follower_id in self.follower_ids
        }
        for action in self.actions:
            if action.action_type != ACTION_JOINT:
                continue
            assert action.relay_id is not None
            actual[action.relay_id] += 1
            if (action.anchor_master_id, action.second_master_id) == ("A", "B"):
                anchor_index = 0
            elif (action.anchor_master_id, action.second_master_id) == ("B", "A"):
                anchor_index = 1
            else:
                raise AssertionError("Joint action has an unknown master order")
            actual_by_anchor[action.relay_id][anchor_index] += 1
        if actual != quota:
            raise AssertionError("relay plan does not meet total quotas")
        expected_by_block = {
            follower_id: (block_0, block_1)
            for follower_id, block_0, block_1 in self.relay_quotas_by_block
        }
        if {
            follower_id: tuple(counts)
            for follower_id, counts in actual_by_anchor.items()
        } != expected_by_block:
            raise AssertionError("relay plan does not meet A/B-anchor quotas")
        if any(
            abs(block_counts[0] - block_counts[1]) > 1
            for block_counts in expected_by_block.values()
        ):
            raise AssertionError("per-relay A/B anchor counts differ by more than one")


_DEFINITIONS = {
    CANDIDATE_D100: FixedCandidateDefinition(
        candidate_id=CANDIDATE_D100,
        ordinary_action_count=100,
        joint_action_count=0,
        joint_macros_by_block=(0, 0),
        relay_policy=RELAY_NONE,
        timing_policy=TIMING_UNIFORM,
    ),
    CANDIDATE_J66_RR: FixedCandidateDefinition(
        candidate_id=CANDIDATE_J66_RR,
        ordinary_action_count=0,
        joint_action_count=66,
        joint_macros_by_block=None,
        relay_policy=RELAY_UNIFORM,
        timing_policy=TIMING_UNIFORM,
        boundary_waves_by_block=(1, 1),
    ),
    CANDIDATE_H42_U: FixedCandidateDefinition(
        candidate_id=CANDIDATE_H42_U,
        ordinary_action_count=37,
        joint_action_count=42,
        joint_macros_by_block=(10, 11),
        relay_policy=RELAY_UNIFORM,
        timing_policy=TIMING_UNIFORM,
        directional_epsilon=1.0,
    ),
    CANDIDATE_H42_D: FixedCandidateDefinition(
        candidate_id=CANDIDATE_H42_D,
        ordinary_action_count=37,
        joint_action_count=42,
        joint_macros_by_block=(10, 11),
        relay_policy=RELAY_DIRECTIONAL,
        timing_policy=TIMING_UNIFORM,
        directional_epsilon=0.25,
    ),
    CANDIDATE_H42_DW: FixedCandidateDefinition(
        candidate_id=CANDIDATE_H42_DW,
        ordinary_action_count=37,
        joint_action_count=42,
        joint_macros_by_block=(10, 11),
        relay_policy=RELAY_DIRECTIONAL,
        timing_policy=TIMING_WEAK_WINDOWS,
        directional_epsilon=0.25,
    ),
    CANDIDATE_H24_DW: FixedCandidateDefinition(
        candidate_id=CANDIDATE_H24_DW,
        ordinary_action_count=64,
        joint_action_count=24,
        joint_macros_by_block=(6, 6),
        relay_policy=RELAY_DIRECTIONAL,
        timing_policy=TIMING_WEAK_WINDOWS,
        directional_epsilon=0.05,
    ),
}


def fixed_candidate_definitions() -> tuple[FixedCandidateDefinition, ...]:
    """Return the preregistered candidates in their stable report order."""

    return tuple(_DEFINITIONS[candidate_id] for candidate_id in FIXED_CANDIDATE_IDS)


def _validate_followers(follower_ids: Sequence[str]) -> tuple[str, ...]:
    result = tuple(str(follower_id) for follower_id in follower_ids)
    if len(result) != 6 or len(set(result)) != 6:
        raise ValueError("fixed six-sector candidates require six unique followers")
    if any(not follower_id for follower_id in result):
        raise ValueError("follower IDs must be non-empty")
    return result


def _sector_angles(
    follower_ids: tuple[str, ...],
    sector_angles_rad: Mapping[str, float] | Sequence[float] | None,
) -> tuple[float, ...]:
    if sector_angles_rad is None:
        values = tuple(index * math.pi / 3.0 for index in range(6))
    elif isinstance(sector_angles_rad, Mapping):
        if set(sector_angles_rad) != set(follower_ids):
            raise ValueError("sector-angle mapping must contain every follower exactly once")
        values = tuple(float(sector_angles_rad[item]) for item in follower_ids)
    else:
        values = tuple(float(value) for value in sector_angles_rad)
        if len(values) != len(follower_ids):
            raise ValueError("sector-angle sequence length differs from follower count")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("sector angles must be finite")
    return values


def _largest_remainder_quotas(
    total: int,
    weights: Mapping[str, float],
    follower_ids: tuple[str, ...],
) -> dict[str, int]:
    if total < 0:
        raise ValueError("quota total must be non-negative")
    if set(weights) != set(follower_ids):
        raise ValueError("weights must contain every follower exactly once")
    if any(
        not math.isfinite(float(weights[item])) or float(weights[item]) < 0.0
        for item in follower_ids
    ):
        raise ValueError("quota weights must be finite and non-negative")
    denominator = sum(float(weights[item]) for item in follower_ids)
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise ValueError("quota weights must have a finite positive sum")
    raw = {
        item: total * float(weights[item]) / denominator for item in follower_ids
    }
    quota = {item: math.floor(raw[item]) for item in follower_ids}
    remaining = total - sum(quota.values())
    order = sorted(
        follower_ids,
        key=lambda item: (-(raw[item] - quota[item]), follower_ids.index(item)),
    )
    for item in order[:remaining]:
        quota[item] += 1
    return quota


def _relay_quotas(
    definition: FixedCandidateDefinition,
    follower_ids: tuple[str, ...],
    angles: tuple[float, ...],
    master_baseline_angle_rad: float,
) -> dict[str, int]:
    if definition.joint_action_count == 0:
        return {item: 0 for item in follower_ids}
    if definition.relay_policy == RELAY_UNIFORM:
        weights = {item: 1.0 for item in follower_ids}
    elif definition.relay_policy == RELAY_DIRECTIONAL:
        epsilon = definition.directional_epsilon
        if epsilon is None or not 0.0 < epsilon <= 1.0:
            raise ValueError("directional candidates need epsilon in (0, 1]")
        weights = {}
        for item, angle in zip(follower_ids, angles):
            projection = math.cos(angle - master_baseline_angle_rad)
            lateral_complementarity = max(0.0, 1.0 - projection * projection)
            weights[item] = epsilon + (1.0 - epsilon) * lateral_complementarity
    else:
        raise ValueError("unknown relay policy")
    return _largest_remainder_quotas(
        definition.joint_action_count, weights, follower_ids
    )


def _split_quotas_by_block(
    quotas: Mapping[str, int],
    block_joint_counts: tuple[int, int],
    follower_ids: tuple[str, ...],
) -> tuple[dict[str, int], dict[str, int]]:
    total = sum(quotas.values())
    if sum(block_joint_counts) != total:
        raise ValueError("anchor-block Joint counts do not match total relay quota")
    if total == 0:
        empty = {item: 0 for item in follower_ids}
        return empty.copy(), empty.copy()
    first_total = block_joint_counts[0]
    raw = {item: quotas[item] * first_total / total for item in follower_ids}
    first = {item: math.floor(raw[item]) for item in follower_ids}
    remaining = first_total - sum(first.values())
    order = sorted(
        follower_ids,
        key=lambda item: (-(raw[item] - first[item]), follower_ids.index(item)),
    )
    for item in order[:remaining]:
        first[item] += 1
    second = {item: quotas[item] - first[item] for item in follower_ids}
    if any(abs(first[item] - second[item]) > 1 for item in follower_ids):
        raise ValueError("relay quota cannot be balanced across the two anchor blocks")
    return first, second


def _weighted_round_robin(
    quotas: Mapping[str, int], follower_ids: tuple[str, ...]
) -> tuple[str, ...]:
    total = sum(quotas.values())
    used = {item: 0 for item in follower_ids}
    plan: list[str] = []
    previous: str | None = None
    for step in range(1, total + 1):
        candidates = [item for item in follower_ids if used[item] < quotas[item]]
        if not candidates:
            raise AssertionError("weighted round-robin exhausted quotas early")
        deficit = {
            item: step * quotas[item] - total * used[item] for item in candidates
        }
        best = max(deficit.values())
        tied = [item for item in candidates if deficit[item] == best]
        if previous in tied and len(tied) > 1:
            tied.remove(previous)
        selected = min(tied, key=follower_ids.index)
        plan.append(selected)
        used[selected] += 1
        previous = selected
    if used != dict(quotas):
        raise AssertionError("weighted round-robin failed to meet quotas")
    return tuple(plan)


def _repair_anchor_balanced_relay_word(
    relay_word: Sequence[str],
    anchor_indices: Sequence[int],
) -> tuple[str, ...]:
    """Remove adjacent duplicates without changing per-anchor relay counts."""

    plan = list(relay_word)
    anchors = tuple(int(value) for value in anchor_indices)
    if len(plan) != len(anchors):
        raise ValueError("relay word and anchor sequence have different lengths")

    def locally_valid(index: int) -> bool:
        return (
            (index == 0 or plan[index] != plan[index - 1])
            and (index == len(plan) - 1 or plan[index] != plan[index + 1])
        )

    for _ in range(len(plan)):
        duplicate = next(
            (index for index in range(1, len(plan)) if plan[index] == plan[index - 1]),
            None,
        )
        if duplicate is None:
            return tuple(plan)
        repaired = False
        for candidate in range(len(plan) - 1):
            if candidate in (duplicate - 1, duplicate):
                continue
            if anchors[candidate] != anchors[duplicate]:
                continue
            plan[candidate], plan[duplicate] = plan[duplicate], plan[candidate]
            affected = {
                index
                for center in (candidate, duplicate)
                for index in range(max(0, center - 1), min(len(plan), center + 2))
            }
            if all(locally_valid(index) for index in affected):
                repaired = True
                break
            plan[candidate], plan[duplicate] = plan[duplicate], plan[candidate]
        if not repaired:
            raise AssertionError("cannot remove adjacent relay duplicate")
    raise AssertionError("relay-word repair did not converge")


def _paper_anchor_balanced_relay_word(
    quotas: Mapping[str, int],
    follower_ids: tuple[str, ...],
    anchor_indices: Sequence[int],
) -> tuple[tuple[str, ...], dict[str, int], dict[str, int]]:
    """Spread each relay quota nearly equally over A- and B-anchor blocks."""

    anchor_indices = tuple(int(value) for value in anchor_indices)
    anchor_counts = (
        sum(value == 0 for value in anchor_indices),
        sum(value == 1 for value in anchor_indices),
    )
    quota_a, quota_b = _split_quotas_by_block(
        quotas, anchor_counts, follower_ids
    )
    words = (
        iter(_weighted_round_robin(quota_a, follower_ids)),
        iter(_weighted_round_robin(quota_b, follower_ids)),
    )
    relay_word = tuple(next(words[anchor]) for anchor in anchor_indices)
    relay_word = _repair_anchor_balanced_relay_word(relay_word, anchor_indices)
    return relay_word, quota_a, quota_b


def _uniform_macro_selection(count: int) -> tuple[int, ...]:
    if not 0 <= count <= MACROS_PER_ANCHOR_BLOCK:
        raise ValueError("Joint macro count lies outside one anchor block")
    selected = tuple(
        index
        for index in range(MACROS_PER_ANCHOR_BLOCK)
        if math.floor((index + 1) * count / MACROS_PER_ANCHOR_BLOCK)
        - math.floor(index * count / MACROS_PER_ANCHOR_BLOCK)
        == 1
    )
    if len(selected) != count:
        raise AssertionError("balanced-word macro selection has the wrong size")
    return selected


def _weak_macro_selection(
    block_id: int,
    count: int,
    weak_window_scores: Mapping[tuple[int, int], float] | None,
) -> tuple[int, ...]:
    if weak_window_scores is None:
        raise ValueError(
            "weak-window candidates require 32 deterministic nominal scores"
        )
    scores: dict[int, float] = {}
    for macro_index in range(MACROS_PER_ANCHOR_BLOCK):
        key = (block_id, macro_index)
        if key not in weak_window_scores:
            raise ValueError(f"missing weak-window score for block/macro {key}")
        value = float(weak_window_scores[key])
        if not math.isfinite(value):
            raise ValueError("weak-window scores must be finite")
        scores[macro_index] = value
    selected = sorted(
        sorted(scores, key=lambda item: (-scores[item], item))[:count]
    )
    return tuple(selected)


def _macro_selection(
    definition: FixedCandidateDefinition,
    weak_window_scores: Mapping[tuple[int, int], float] | None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if definition.joint_macros_by_block is None:
        return (), ()
    selected: list[tuple[int, ...]] = []
    for block_id, count in enumerate(definition.joint_macros_by_block):
        if definition.timing_policy == TIMING_UNIFORM:
            selected.append(_uniform_macro_selection(count))
        elif definition.timing_policy == TIMING_WEAK_WINDOWS:
            selected.append(
                _weak_macro_selection(block_id, count, weak_window_scores)
            )
        else:
            raise ValueError("unknown timing policy")
    return selected[0], selected[1]


def _action_skeletons(
    definition: FixedCandidateDefinition,
    selected_macros: tuple[tuple[int, ...], tuple[int, ...]],
) -> list[ScheduledAcousticAction]:
    actions: list[ScheduledAcousticAction] = []
    # The deployable dual-master baseline is the already frozen 100-cycle
    # A->B protocol over the complete horizon.  Only Joint/hybrid candidates
    # reverse their anchor roles in the second 100 s block.
    roles = (
        (("A", "B"), ("A", "B"))
        if definition.candidate_id == CANDIDATE_D100
        else (("A", "B"), ("B", "A"))
    )
    if definition.joint_macros_by_block is None:
        for block_id, (anchor, second) in enumerate(roles):
            block_start = block_id * ANCHOR_BLOCK_DURATION_S
            for local_index in range(33):
                actions.append(
                    ScheduledAcousticAction(
                        sequence_index=-1,
                        action_type=ACTION_JOINT,
                        start_time_s=block_start + 3.0 * local_index,
                        block_id=block_id,
                        anchor_master_id=anchor,
                        second_master_id=second,
                        relay_id=None,
                        macro_index=None,
                        nominal_wave_cost=3,
                    )
                )
        return actions

    for block_id, (anchor, second) in enumerate(roles):
        block_start = block_id * ANCHOR_BLOCK_DURATION_S
        joint_macros = set(selected_macros[block_id])
        for macro_index in range(MACROS_PER_ANCHOR_BLOCK):
            macro_start = block_start + macro_index * MACRO_DURATION_S
            if macro_index in joint_macros:
                offsets = (0.0, 3.0)
                action_type = ACTION_JOINT
                wave_cost = 3
            else:
                offsets = (0.0, 2.0, 4.0)
                action_type = ACTION_DUAL_MASTER
                wave_cost = 2
            for offset in offsets:
                actions.append(
                    ScheduledAcousticAction(
                        sequence_index=-1,
                        action_type=action_type,
                        start_time_s=macro_start + offset,
                        block_id=block_id,
                        anchor_master_id=anchor,
                        second_master_id=second,
                        relay_id=None,
                        macro_index=macro_index,
                        nominal_wave_cost=wave_cost,
                    )
                )
        for offset in (96.0, 98.0):
            actions.append(
                ScheduledAcousticAction(
                    sequence_index=-1,
                    action_type=ACTION_DUAL_MASTER,
                    start_time_s=block_start + offset,
                    block_id=block_id,
                    anchor_master_id=anchor,
                    second_master_id=second,
                    relay_id=None,
                    macro_index=None,
                    nominal_wave_cost=2,
                )
            )
    return actions


def _fingerprint_payload(
    definition: FixedCandidateDefinition,
    follower_ids: tuple[str, ...],
    angles: tuple[float, ...],
    baseline_angle: float,
    actions: tuple[ScheduledAcousticAction, ...],
    quotas: tuple[tuple[str, int], ...],
    quotas_by_block: tuple[tuple[str, int, int], ...],
) -> str:
    payload = {
        "definition": asdict(definition),
        "followers": follower_ids,
        "sector_angles_rad": angles,
        "master_baseline_angle_rad": baseline_angle,
        "actions": [asdict(action) for action in actions],
        "relay_quotas": quotas,
        "relay_quotas_by_block": quotas_by_block,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def build_fixed_candidate_schedule(
    candidate_id: str,
    *,
    follower_ids: Sequence[str] = DEFAULT_FOLLOWER_IDS,
    sector_angles_rad: Mapping[str, float] | Sequence[float] | None = None,
    master_baseline_angle_rad: float = math.pi / 2.0,
    weak_window_scores: Mapping[tuple[int, int], float] | None = None,
    duration_s: float = FIXED_DURATION_S,
    wave_budget: int = FIXED_WAVE_BUDGET,
) -> FixedCandidateSchedule:
    """Build one preregistered schedule without sampling any random quantity.

    ``weak_window_scores`` are required only by ``*-DW`` candidates.  They must
    be produced by deterministic nominal covariance prediction and indexed by
    ``(anchor_block_id, macro_index)`` for two blocks and sixteen macros.
    """

    if candidate_id not in _DEFINITIONS:
        raise ValueError(f"unknown fixed candidate: {candidate_id}")
    if float(duration_s) != FIXED_DURATION_S:
        raise ValueError("fixed candidate generator currently supports 200 s only")
    if int(wave_budget) != FIXED_WAVE_BUDGET:
        raise ValueError("fixed candidate generator currently supports 200 waves only")
    if not math.isfinite(master_baseline_angle_rad):
        raise ValueError("master baseline angle must be finite")

    definition = _DEFINITIONS[candidate_id]
    followers = _validate_followers(follower_ids)
    angles = _sector_angles(followers, sector_angles_rad)
    selected_macros = _macro_selection(definition, weak_window_scores)
    quotas = _relay_quotas(
        definition, followers, angles, float(master_baseline_angle_rad)
    )

    if definition.joint_macros_by_block is None:
        block_joint_counts = (33, 33)
    else:
        block_joint_counts = tuple(
            2 * count for count in definition.joint_macros_by_block
        )
    block_0_quota, block_1_quota = _split_quotas_by_block(
        quotas, block_joint_counts, followers
    )
    relay_by_block = (
        iter(_weighted_round_robin(block_0_quota, followers)),
        iter(_weighted_round_robin(block_1_quota, followers)),
    )

    skeletons = sorted(
        _action_skeletons(definition, selected_macros),
        key=lambda action: action.start_time_s,
    )
    actions: list[ScheduledAcousticAction] = []
    for sequence_index, action in enumerate(skeletons):
        relay_id = (
            next(relay_by_block[action.block_id])
            if action.action_type == ACTION_JOINT
            else None
        )
        actions.append(
            replace(
                action,
                sequence_index=sequence_index,
                relay_id=relay_id,
            )
        )
    frozen_actions = tuple(actions)
    frozen_quotas = tuple((item, quotas[item]) for item in followers)
    frozen_block_quotas = tuple(
        (item, block_0_quota[item], block_1_quota[item]) for item in followers
    )
    fingerprint = _fingerprint_payload(
        definition,
        followers,
        angles,
        float(master_baseline_angle_rad),
        frozen_actions,
        frozen_quotas,
        frozen_block_quotas,
    )
    schedule = FixedCandidateSchedule(
        definition=definition,
        follower_ids=followers,
        sector_angles_rad=angles,
        master_baseline_angle_rad=float(master_baseline_angle_rad),
        actions=frozen_actions,
        relay_quotas=frozen_quotas,
        relay_quotas_by_block=frozen_block_quotas,
        selected_joint_macros_by_block=selected_macros,
        duration_s=float(duration_s),
        wave_budget=int(wave_budget),
        fingerprint=fingerprint,
    )
    schedule.validate()
    return schedule


def build_all_fixed_candidate_schedules(
    *,
    follower_ids: Sequence[str] = DEFAULT_FOLLOWER_IDS,
    sector_angles_rad: Mapping[str, float] | Sequence[float] | None = None,
    master_baseline_angle_rad: float = math.pi / 2.0,
    weak_window_scores: Mapping[tuple[int, int], float] | None,
) -> tuple[FixedCandidateSchedule, ...]:
    """Build the complete fixed candidate set in stable report order."""

    return tuple(
        build_fixed_candidate_schedule(
            candidate_id,
            follower_ids=follower_ids,
            sector_angles_rad=sector_angles_rad,
            master_baseline_angle_rad=master_baseline_angle_rad,
            weak_window_scores=weak_window_scores,
        )
        for candidate_id in FIXED_CANDIDATE_IDS
    )


def _paper_directional_actions(
    candidate_id: str,
    followers: tuple[str, ...],
) -> tuple[
    FixedCandidateDefinition,
    tuple[ScheduledAcousticAction, ...],
    tuple[tuple[str, int], ...],
    tuple[tuple[str, int, int], ...],
]:
    """Construct one finite rendering of an indefinitely repeatable word."""

    if candidate_id == CANDIDATE_D_ONLY:
        definition = FixedCandidateDefinition(
            candidate_id=candidate_id,
            ordinary_action_count=100,
            joint_action_count=0,
            joint_macros_by_block=(0, 0),
            relay_policy=RELAY_NONE,
            timing_policy="periodic",
        )
        actions = tuple(
            ScheduledAcousticAction(
                sequence_index=index,
                action_type=ACTION_DUAL_MASTER,
                start_time_s=2.0 * index,
                block_id=0 if index < 50 else 1,
                anchor_master_id="A",
                second_master_id="B",
                relay_id=None,
                macro_index=None,
                nominal_wave_cost=2,
            )
            for index in range(100)
        )
        quotas = {agent: 0 for agent in followers}
    else:
        if candidate_id not in DIRECTIONAL_JOINT_RELAY_QUOTAS:
            raise ValueError(f"unknown paper directional schedule: {candidate_id}")
        values = DIRECTIONAL_JOINT_RELAY_QUOTAS[candidate_id]
        quotas = dict(zip(followers, values))
        anchor_indices = tuple(
            block_id % 2
            for block_id, cycle_count in enumerate(PAPER_JOINT_BLOCK_LENGTHS)
            for _ in range(cycle_count)
        )
        relay_word, quota_a, quota_b = _paper_anchor_balanced_relay_word(
            quotas, followers, anchor_indices
        )
        if len(relay_word) != PAPER_JOINT_ACTION_COUNT:
            raise AssertionError(
                "paper Joint relay word has the wrong action count"
            )
        definition = FixedCandidateDefinition(
            candidate_id=candidate_id,
            ordinary_action_count=0,
            joint_action_count=PAPER_JOINT_ACTION_COUNT,
            joint_macros_by_block=None,
            relay_policy=(RELAY_UNIFORM if candidate_id == CANDIDATE_J_RR
                          else RELAY_DIRECTIONAL),
            timing_policy="periodic",
            boundary_waves_by_block=(1,) * PAPER_JOINT_BLOCK_COUNT,
        )
        action_list: list[ScheduledAcousticAction] = []
        sequence_index = 0
        block_start_s = 0.0
        for block_id, cycle_count in enumerate(PAPER_JOINT_BLOCK_LENGTHS):
            anchor, second = (("A", "B") if block_id % 2 == 0 else ("B", "A"))
            for local_index in range(cycle_count):
                action_list.append(
                    ScheduledAcousticAction(
                        sequence_index=sequence_index,
                        action_type=ACTION_JOINT,
                        start_time_s=block_start_s + 3.0 * local_index,
                        block_id=block_id,
                        anchor_master_id=anchor,
                        second_master_id=second,
                        relay_id=relay_word[sequence_index],
                        macro_index=sequence_index % PAPER_JOINT_ACTION_COUNT,
                        nominal_wave_cost=3,
                    )
                )
                sequence_index += 1
            # A reversed anchor cannot reuse the preceding terminal wave.  One
            # additional physical-wave slot separates adjacent chain blocks.
            block_start_s += 3.0 * cycle_count + 1.0
        actions = tuple(action_list)
        if sequence_index != PAPER_JOINT_ACTION_COUNT or block_start_s != 200.0:
            raise AssertionError("paper Joint blocks do not fill the 200 s ledger")

    by_anchor = {
        agent: [0, 0]
        for agent in followers
    }
    for action in actions:
        if action.action_type == ACTION_JOINT:
            assert action.relay_id is not None
            anchor_index = 0 if action.anchor_master_id == "A" else 1
            by_anchor[action.relay_id][anchor_index] += 1
    frozen_quotas = tuple((agent, quotas[agent]) for agent in followers)
    frozen_block_quotas = tuple(
        (agent, by_anchor[agent][0], by_anchor[agent][1]) for agent in followers
    )
    return definition, actions, frozen_quotas, frozen_block_quotas


def build_paper_directional_schedule(
    candidate_id: str,
    *,
    follower_ids: Sequence[str] = DEFAULT_FOLLOWER_IDS,
    sector_angles_rad: Mapping[str, float] | Sequence[float] | None = None,
    master_baseline_angle_rad: float = math.pi / 2.0,
) -> FixedCandidateSchedule:
    """Build a paper schedule with no weak-window or posterior dependency.

    ``J-RR`` and ``J-D1``--``J-D3`` share all 62 Joint action times.  Their
    sole difference is the deterministic, evenly interleaved relay word.
    ``D-only`` retains the original two-wave/one-row master factor.
    """

    if candidate_id not in DIRECTIONAL_JOINT_CANDIDATE_IDS:
        raise ValueError(f"unknown paper schedule: {candidate_id}")
    followers = _validate_followers(follower_ids)
    angles = _sector_angles(followers, sector_angles_rad)
    definition, actions, quotas, block_quotas = _paper_directional_actions(
        candidate_id, followers
    )
    fingerprint = _fingerprint_payload(
        definition, followers, angles, float(master_baseline_angle_rad),
        actions, quotas, block_quotas,
    )
    schedule = FixedCandidateSchedule(
        definition=definition,
        follower_ids=followers,
        sector_angles_rad=angles,
        master_baseline_angle_rad=float(master_baseline_angle_rad),
        actions=actions,
        relay_quotas=quotas,
        relay_quotas_by_block=block_quotas,
        selected_joint_macros_by_block=((), ()),
        duration_s=FIXED_DURATION_S,
        wave_budget=FIXED_WAVE_BUDGET,
        fingerprint=fingerprint,
    )
    schedule.validate()
    return schedule


def build_all_paper_directional_schedules(
    *,
    follower_ids: Sequence[str] = DEFAULT_FOLLOWER_IDS,
    sector_angles_rad: Mapping[str, float] | Sequence[float] | None = None,
    master_baseline_angle_rad: float = math.pi / 2.0,
) -> tuple[FixedCandidateSchedule, ...]:
    """Return D-only, J-RR and the three directional relay schedules."""

    return tuple(
        build_paper_directional_schedule(
            candidate_id,
            follower_ids=follower_ids,
            sector_angles_rad=sector_angles_rad,
            master_baseline_angle_rad=master_baseline_angle_rad,
        )
        for candidate_id in DIRECTIONAL_JOINT_CANDIDATE_IDS
    )
