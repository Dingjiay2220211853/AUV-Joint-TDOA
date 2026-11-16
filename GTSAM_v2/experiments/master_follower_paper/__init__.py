"""Legacy five-AUV validation and the formal controlled/radial paper suite."""

from .model import (
    FOLLOWER_IDS,
    MASTER_IDS,
    FormalExperimentConfig,
    FormalExperimentDataset,
    build_default_config,
    generate_dataset,
)
from .formal_protocol import (
    FORMAL_SEEDS,
    build_controlled_spec,
    build_radial_spec,
    build_radial_v2_spec,
    generate_formal_world,
)
from .formal_motion import (
    ConstantUnicycleMotion,
    SinusoidalHeadingMotion,
    build_random_nonstraight_case,
    sample_random_path_parameters,
)
from .formal_sector_sweep_motion import (
    SectorSweepParameter,
    StructuredSectorSweepMotion,
    build_structured_sector_sweep_case,
    sample_sector_sweep_parameters,
)
from .formal_candidate_schedules import (
    DIRECTIONAL_JOINT_CANDIDATE_IDS,
    DIRECTIONAL_JOINT_RELAY_QUOTAS,
    FIXED_CANDIDATE_IDS,
    FixedCandidateSchedule,
    build_all_paper_directional_schedules,
    build_all_fixed_candidate_schedules,
    build_paper_directional_schedule,
    build_fixed_candidate_schedule,
    fixed_candidate_definitions,
)

__all__ = [
    "FOLLOWER_IDS",
    "MASTER_IDS",
    "FormalExperimentConfig",
    "FormalExperimentDataset",
    "build_default_config",
    "generate_dataset",
    "FORMAL_SEEDS",
    "build_controlled_spec",
    "build_radial_spec",
    "build_radial_v2_spec",
    "generate_formal_world",
    "ConstantUnicycleMotion",
    "SinusoidalHeadingMotion",
    "build_random_nonstraight_case",
    "sample_random_path_parameters",
    "SectorSweepParameter",
    "StructuredSectorSweepMotion",
    "build_structured_sector_sweep_case",
    "sample_sector_sweep_parameters",
    "DIRECTIONAL_JOINT_CANDIDATE_IDS",
    "DIRECTIONAL_JOINT_RELAY_QUOTAS",
    "FIXED_CANDIDATE_IDS",
    "FixedCandidateSchedule",
    "build_all_paper_directional_schedules",
    "build_all_fixed_candidate_schedules",
    "build_paper_directional_schedule",
    "build_fixed_candidate_schedule",
    "fixed_candidate_definitions",
]
