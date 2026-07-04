"""Centralised configuration dataclasses for the street-view filter."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class ThresholdConfig:
    # darkness
    dark_mean_v: float = 38.0
    dark_low_ratio: float = 0.55
    dark_hard_mean_v: float = 24.0
    dark_hard_score: float = 0.90
    dark_soft_score: float = 0.72

    # glare / brightness
    glare_soft_score: float = 0.62
    glare_hard_score: float = 0.88
    bright_hard_pixel_ratio: float = 0.42
    clip_hard_ratio: float = 0.12

    # blur
    blur_soft_severity: float = 0.56
    blur_hard_severity: float = 0.78

    # building / facade
    building_min_score: float = 0.34
    building_soft_score: float = 0.42

    # open scene family
    sky_dominance: float = 0.45
    vegetation_dominance: float = 0.45
    road_corridor: float = 0.50
    open_scene_combined: float = 0.56

    # interior
    interior_soft: float = 0.72
    interior_hard: float = 0.84
    interior_outdoor_evidence_soft_max: float = 0.42
    interior_outdoor_evidence_hard_max: float = 0.35

    # occlusion / vehicle-like blocking
    occlusion_soft: float = 0.42
    occlusion_hard: float = 0.58

    # non-google (demoted signal)
    non_google_soft: float = 0.78

    # logo / template support
    logo_presence_expected: float = 0.55

    # placeholder similarity
    placeholder_soft_similarity: float = 0.88
    placeholder_hard_similarity: float = 0.94

    # final combined decision
    invalid_score_threshold: float = 0.62
    borderline_margin: float = 0.08


@dataclass
class WeightConfig:
    low_building: float = 0.34
    open_scene: float = 0.20
    blur: float = 0.14
    road_corridor: float = 0.10
    occlusion: float = 0.08
    interior: float = 0.10
    non_google: float = 0.03
    glare: float = 0.07
    dark: float = 0.04


@dataclass
class RuntimeConfig:
    max_side: int = 640
    recursive: bool = False
    supported_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

    invalid_action: str = "copy"  # copy | move | none
    organize_by_primary_reason: bool = True
    preserve_relative_structure: bool = True

    resume: bool = True
    random_seed: int = 1337
    log_level: str = "INFO"
    progress_every_n: int = 250

    save_debug_images: bool = True
    debug_max_per_reason: int = 20
    debug_save_valid_borderline: bool = True

    write_csv: bool = True
    write_json: bool = True
    write_jsonl: bool = True

    review_random_invalid_count: int = 40
    review_borderline_count: int = 30
    review_per_reason_count: int = 20
    review_copy_images: bool = False


@dataclass
class FilterConfig:
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    weights: WeightConfig = field(default_factory=WeightConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    placeholder_reference_paths: List[str] = field(default_factory=list)
    logo_template_paths: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = FilterConfig()
