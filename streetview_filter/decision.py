"""Decision engine for the street-view image filter."""

from __future__ import annotations

from typing import Any, Dict, List

from streetview_filter.config import FilterConfig
from streetview_filter.utils import clamp, normalize


class DecisionEngine:
    def __init__(self, config: FilterConfig):
        self.config = config

    def decide(self, scores: Dict[str, float]) -> Dict[str, Any]:
        t = self.config.thresholds
        w = self.config.weights

        hard_reasons: List[str] = []
        soft_reasons: List[str] = []

        # Hard rejects
        if scores["placeholder_similarity"] >= t.placeholder_hard_similarity:
            hard_reasons.append("placeholder_no_imagery")

        if (
            scores["glare_score"] >= t.glare_hard_score
            or scores["bright_pixel_ratio"] >= t.bright_hard_pixel_ratio
            or scores["clip_ratio"] >= t.clip_hard_ratio
        ):
            hard_reasons.append("extreme_glare_or_washout")

        if (
            scores["brightness_mean_v"] <= t.dark_hard_mean_v
            and scores["dark_score"] >= t.dark_hard_score
        ):
            hard_reasons.append("extreme_darkness")

        if (
            scores["interior_score"] >= t.interior_hard
            and scores["interior_outdoor_evidence"] <= t.interior_outdoor_evidence_hard_max
        ):
            hard_reasons.append("strong_interior_likelihood")

        if (
            scores["blur_severity"] >= t.blur_hard_severity
            and scores["building_score"] <= t.building_soft_score
        ):
            hard_reasons.append("severe_blur_low_facade_detail")

        low_building = clamp(
            (t.building_min_score - scores["building_score"]) / max(t.building_min_score, 1e-6)
        )

        open_scene_pressure = clamp(
            0.50 * normalize(scores["open_scene_score"], t.open_scene_combined, 1.0)
            + 0.25 * normalize(scores["sky_ratio"], t.sky_dominance, 1.0)
            + 0.25 * normalize(scores["vegetation_ratio"], t.vegetation_dominance, 1.0)
        )

        contributions = {
            "low_building": w.low_building * low_building,
            "open_scene": w.open_scene * open_scene_pressure,
            "blur": w.blur * normalize(scores["blur_severity"], t.blur_soft_severity, 1.0),
            "road_corridor": w.road_corridor * normalize(scores["road_corridor_score"], t.road_corridor, 1.0),
            "occlusion": w.occlusion * normalize(scores["occlusion_score"], t.occlusion_soft, 1.0),
            "interior": w.interior * normalize(scores["interior_score"], t.interior_soft, 1.0),
            "non_google": w.non_google * normalize(scores["non_google_score"], t.non_google_soft, 1.0),
            "glare": w.glare * normalize(scores["glare_score"], t.glare_soft_score, 1.0),
            "dark": w.dark * normalize(scores["dark_score"], t.dark_soft_score, 1.0),
        }

        synergy_bonus = 0.0

        if low_building > 0.35 and open_scene_pressure > 0.45:
            synergy_bonus += 0.10
            soft_reasons.append("low_facade_plus_open_scene")

        if low_building > 0.35 and scores["road_corridor_score"] >= t.road_corridor:
            synergy_bonus += 0.08
            soft_reasons.append("street_corridor_not_facade_view")

        if low_building > 0.30 and scores["vegetation_ratio"] >= t.vegetation_dominance:
            synergy_bonus += 0.06
            soft_reasons.append("vegetation_dominance_low_facade")

        if low_building > 0.25 and scores["blur_severity"] >= t.blur_soft_severity:
            synergy_bonus += 0.07
            soft_reasons.append("blur_with_low_facade_detail")

        if low_building > 0.25 and scores["occlusion_score"] >= t.occlusion_soft:
            synergy_bonus += 0.07
            soft_reasons.append("occlusion_with_low_facade_visibility")

        if scores["placeholder_similarity"] >= t.placeholder_soft_similarity:
            soft_reasons.append("placeholder_similarity_high")

        if scores["interior_score"] >= t.interior_soft and scores["interior_outdoor_evidence"] <= t.interior_outdoor_evidence_soft_max:
            soft_reasons.append("possible_interior")

        invalid_score = clamp(sum(contributions.values()) + synergy_bonus)

        hard_reject = len(hard_reasons) > 0
        is_invalid = hard_reject or invalid_score >= t.invalid_score_threshold

        rescued_by_building = False
        if (
            is_invalid
            and not hard_reject
            and scores["building_score"] >= (t.building_min_score + 0.10)
            and scores["open_scene_score"] < 0.72
            and scores["road_corridor_score"] < 0.72
        ):
            is_invalid = False
            rescued_by_building = True
            soft_reasons.append("rescued_by_building_presence")

        # Build reason list and pick primary
        if hard_reasons:
            reasons = hard_reasons + [r for r in soft_reasons if r not in hard_reasons]
            primary_reason = hard_reasons[0]
        else:
            reasons = soft_reasons.copy()
            if not reasons:
                reasons = ["valid_facade_view"]

            if is_invalid:
                reason_map = {
                    "low_building": "insufficient_facade_presence",
                    "open_scene": "open_scene_dominance",
                    "road_corridor": "street_corridor_view",
                    "blur": "blurred_facade",
                    "occlusion": "vehicle_or_object_occlusion",
                    "interior": "interior_likelihood",
                    "glare": "glare_or_overexposure",
                    "dark": "dark_or_night",
                    "non_google": "non_google_format_signal",
                }
                top_component = max(contributions.items(), key=lambda kv: kv[1])[0]
                primary_reason = reason_map.get(top_component, reasons[0])
            else:
                primary_reason = "valid"

        borderline = abs(invalid_score - t.invalid_score_threshold) <= t.borderline_margin

        if hard_reject:
            confidence = clamp(max(
                scores["placeholder_similarity"],
                scores["glare_score"],
                scores["dark_score"],
                scores["interior_score"],
                scores["blur_severity"],
            ))
        else:
            confidence = clamp(0.5 + abs(invalid_score - t.invalid_score_threshold) * 0.9)

        return {
            "is_invalid": bool(is_invalid),
            "hard_reject": bool(hard_reject),
            "borderline": bool(borderline),
            "invalid_score": float(invalid_score),
            "confidence": float(confidence),
            "primary_reason": primary_reason,
            "reasons": reasons,
            "secondary_reasons": [r for r in reasons if r != primary_reason],
            "rescued_by_building": bool(rescued_by_building),
            "contrib_low_building": float(contributions["low_building"]),
            "contrib_open_scene": float(contributions["open_scene"]),
            "contrib_blur": float(contributions["blur"]),
            "contrib_road_corridor": float(contributions["road_corridor"]),
            "contrib_occlusion": float(contributions["occlusion"]),
            "contrib_interior": float(contributions["interior"]),
            "contrib_non_google": float(contributions["non_google"]),
            "contrib_glare": float(contributions["glare"]),
            "contrib_dark": float(contributions["dark"]),
            "synergy_bonus": float(synergy_bonus),
        }
