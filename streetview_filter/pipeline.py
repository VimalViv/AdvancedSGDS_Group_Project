"""Pipeline runner and evaluation helpers for the street-view filter."""

from __future__ import annotations

import json
import logging
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from streetview_filter.config import FilterConfig
from streetview_filter.decision import DecisionEngine
from streetview_filter.detectors import MultiCueDetectors
from streetview_filter.utils import (
    discover_images,
    draw_debug_overlay,
    ensure_dir,
    sample_from_dataframe,
    utc_now_iso,
)


class StreetViewInvalidFilterPipeline:
    def __init__(self, input_folder: Path, output_folder: Path, config: FilterConfig):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.config = config

        self.detectors = MultiCueDetectors(config)
        self.engine = DecisionEngine(config)

        self.logger = logging.getLogger("sv_filter")
        self.logger.setLevel(getattr(logging, config.runtime.log_level.upper(), logging.INFO))
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            self.logger.addHandler(handler)

        random.seed(config.runtime.random_seed)
        np.random.seed(config.runtime.random_seed)

        self.reports_dir = self.output_folder / "reports"
        self.invalid_dir = self.output_folder / "invalid"
        self.debug_dir = self.output_folder / "debug"
        self.review_dir = self.output_folder / "review_samples"

        ensure_dir(self.output_folder)
        ensure_dir(self.reports_dir)
        ensure_dir(self.invalid_dir)
        ensure_dir(self.debug_dir)
        ensure_dir(self.review_dir)

        self.results_jsonl_path = self.reports_dir / "results_v3.jsonl"
        self.results_csv_path = self.reports_dir / "results_v3.csv"
        self.results_json_path = self.reports_dir / "results_v3.json"
        self.summary_json_path = self.reports_dir / "summary_v3.json"
        self.config_json_path = self.reports_dir / "effective_config_v3.json"

        self.debug_reason_counts: Dict[str, int] = {}

    def _load_existing_records(self) -> Dict[str, Dict[str, Any]]:
        records: Dict[str, Dict[str, Any]] = {}
        if not self.config.runtime.resume:
            return records
        if not self.results_jsonl_path.exists():
            return records

        with self.results_jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                path_key = row.get("image_path")
                if path_key:
                    records[path_key] = row
        return records

    def _write_jsonl_rows(self, rows: List[Dict[str, Any]]) -> None:
        if not self.config.runtime.write_jsonl or not rows:
            return
        mode = "a" if self.results_jsonl_path.exists() else "w"
        with self.results_jsonl_path.open(mode, encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

    def _route_invalid_image(self, src: Path, primary_reason: str) -> Optional[Path]:
        action = self.config.runtime.invalid_action.lower().strip()
        if action == "none":
            return None

        reason_dir = self.invalid_dir / primary_reason if self.config.runtime.organize_by_primary_reason else self.invalid_dir

        if self.config.runtime.preserve_relative_structure:
            try:
                rel_parent = src.relative_to(self.input_folder).parent
            except Exception:
                rel_parent = Path("")
            reason_dir = reason_dir / rel_parent

        ensure_dir(reason_dir)
        dest = reason_dir / src.name

        if dest.exists():
            stem = dest.stem
            suffix = dest.suffix
            counter = 1
            while True:
                candidate = reason_dir / f"{stem}_{counter}{suffix}"
                if not candidate.exists():
                    dest = candidate
                    break
                counter += 1

        if action == "copy":
            shutil.copy2(src, dest)
        elif action == "move":
            shutil.move(str(src), str(dest))
        else:
            return None

        return dest

    def _maybe_write_debug(self, image_bgr: np.ndarray, row: Dict[str, Any]) -> None:
        if not self.config.runtime.save_debug_images:
            return

        reason = row["primary_reason"] if row["is_invalid"] else "valid"

        if row["is_invalid"]:
            count = self.debug_reason_counts.get(reason, 0)
            if count >= self.config.runtime.debug_max_per_reason:
                return
            self.debug_reason_counts[reason] = count + 1
        else:
            if not row["borderline"] or not self.config.runtime.debug_save_valid_borderline:
                return
            count = self.debug_reason_counts.get("borderline_valid", 0)
            if count >= self.config.runtime.debug_max_per_reason:
                return
            self.debug_reason_counts["borderline_valid"] = count + 1
            reason = "borderline_valid"

        overlay = draw_debug_overlay(
            image_bgr,
            decision={
                "is_invalid": row["is_invalid"],
                "primary_reason": row["primary_reason"],
                "confidence": row["confidence"],
                "invalid_score": row["invalid_score"],
                "hard_reject": row["hard_reject"],
                "borderline": row["borderline"],
                "reasons": row["reasons"].split(",") if row["reasons"] else [],
            },
            scores={
                "building_score": row["building_score"],
                "open_scene_score": row["open_scene_score"],
                "road_corridor_score": row["road_corridor_score"],
                "blur_severity": row["blur_severity"],
                "glare_score": row["glare_score"],
                "dark_score": row["dark_score"],
                "interior_score": row["interior_score"],
                "occlusion_score": row["occlusion_score"],
                "placeholder_similarity": row["placeholder_similarity"],
            },
        )

        out_dir = self.debug_dir / reason
        ensure_dir(out_dir)
        out_path = out_dir / Path(row["image_path"]).name
        cv2.imwrite(str(out_path), overlay)

    def _row_from_scores(
        self,
        image_path: Path,
        decision: Dict[str, Any],
        scores: Dict[str, float],
        routed_path: Optional[Path],
    ) -> Dict[str, Any]:
        t = self.config.thresholds

        row: Dict[str, Any] = {
            "image_path": str(image_path),
            "filename": image_path.name,
            "is_invalid": bool(decision["is_invalid"]),
            "hard_reject": bool(decision["hard_reject"]),
            "borderline": bool(decision["borderline"]),
            "invalid_score": float(decision["invalid_score"]),
            "confidence": float(decision["confidence"]),
            "primary_reason": str(decision["primary_reason"]),
            "secondary_reasons": ",".join(decision["secondary_reasons"]),
            "reasons": ",".join(decision["reasons"]),
            "rescued_by_building": bool(decision["rescued_by_building"]),
            "routed_invalid_path": str(routed_path) if routed_path else "",
            "processed_at": utc_now_iso(),

            # detector scores
            "brightness_mean_v": float(scores["brightness_mean_v"]),
            "dark_score": float(scores["dark_score"]),
            "bright_pixel_ratio": float(scores["bright_pixel_ratio"]),
            "clip_ratio": float(scores["clip_ratio"]),
            "glare_score": float(scores["glare_score"]),
            "laplacian_var_roi": float(scores["laplacian_var_roi"]),
            "tenengrad_mean_roi": float(scores["tenengrad_mean_roi"]),
            "edge_density_roi": float(scores["edge_density_roi"]),
            "blur_severity": float(scores["blur_severity"]),
            "building_candidate_ratio": float(scores["building_candidate_ratio"]),
            "building_edge_density": float(scores["building_edge_density"]),
            "building_texture_energy": float(scores["building_texture_energy"]),
            "building_side_presence": float(scores["building_side_presence"]),
            "building_hv_line_ratio": float(scores["building_hv_line_ratio"]),
            "building_score": float(scores["building_score"]),
            "sky_ratio": float(scores["sky_ratio"]),
            "sky_top_ratio": float(scores["sky_top_ratio"]),
            "vegetation_ratio": float(scores["vegetation_ratio"]),
            "road_ratio": float(scores["road_ratio"]),
            "open_scene_score": float(scores["open_scene_score"]),
            "road_center_bottom_ratio": float(scores["road_center_bottom_ratio"]),
            "road_side_facade_ratio": float(scores["road_side_facade_ratio"]),
            "road_corridor_score": float(scores["road_corridor_score"]),
            "interior_warm_ratio": float(scores["interior_warm_ratio"]),
            "interior_outdoor_evidence": float(scores["interior_outdoor_evidence"]),
            "interior_score": float(scores["interior_score"]),
            "occlusion_largest_component_ratio": float(scores["occlusion_largest_component_ratio"]),
            "occlusion_center_cover_ratio": float(scores["occlusion_center_cover_ratio"]),
            "occlusion_score": float(scores["occlusion_score"]),
            "non_google_score": float(scores["non_google_score"]),
            "logo_match_score": float(scores["logo_match_score"]),
            "placeholder_similarity": float(scores["placeholder_similarity"]),
            "placeholder_best_reference": str(scores.get("placeholder_best_reference", "")),

            # contributions
            "contrib_low_building": float(decision["contrib_low_building"]),
            "contrib_open_scene": float(decision["contrib_open_scene"]),
            "contrib_blur": float(decision["contrib_blur"]),
            "contrib_road_corridor": float(decision["contrib_road_corridor"]),
            "contrib_occlusion": float(decision["contrib_occlusion"]),
            "contrib_interior": float(decision["contrib_interior"]),
            "contrib_non_google": float(decision["contrib_non_google"]),
            "contrib_glare": float(decision["contrib_glare"]),
            "contrib_dark": float(decision["contrib_dark"]),
            "synergy_bonus": float(decision["synergy_bonus"]),

            # key thresholds used
            "threshold_invalid_score": float(t.invalid_score_threshold),
            "threshold_building_min": float(t.building_min_score),
            "threshold_glare_hard": float(t.glare_hard_score),
            "threshold_blur_hard": float(t.blur_hard_severity),
            "threshold_placeholder_hard": float(t.placeholder_hard_similarity),
            "threshold_interior_hard": float(t.interior_hard),
        }
        return row

    # ------------------------------------------------------------------
    # Review sampling  (uses sample_from_dataframe to eliminate 3x duplication)
    # ------------------------------------------------------------------

    def generate_review_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config.runtime
        rng = np.random.default_rng(cfg.random_seed)

        picks: List[pd.DataFrame] = []

        invalid_df = df[df["is_invalid"] == True]  # noqa: E712
        borderline_df = df[df["borderline"] == True]  # noqa: E712

        for source_df, count, bucket in [
            (invalid_df, cfg.review_random_invalid_count, "random_invalid"),
            (borderline_df, cfg.review_borderline_count, "borderline"),
        ]:
            sampled = sample_from_dataframe(source_df, count, rng, bucket)
            if sampled is not None:
                picks.append(sampled)

        if not invalid_df.empty:
            for reason, group in invalid_df.groupby("primary_reason"):
                sampled = sample_from_dataframe(
                    group, cfg.review_per_reason_count, rng, f"reason_{reason}",
                )
                if sampled is not None:
                    picks.append(sampled)

        if picks:
            review_df = pd.concat(picks, ignore_index=True).drop_duplicates(subset=["image_path", "review_bucket"])
        else:
            review_df = pd.DataFrame(columns=list(df.columns) + ["review_bucket"])

        ensure_dir(self.review_dir)
        review_manifest = self.review_dir / "review_manifest_v3.csv"
        review_df.to_csv(review_manifest, index=False)

        if cfg.review_copy_images and not review_df.empty:
            for _, row in review_df.iterrows():
                src = Path(row["image_path"])
                if not src.exists():
                    continue
                bucket_dir = self.review_dir / row["review_bucket"]
                ensure_dir(bucket_dir)
                dst = bucket_dir / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)

        self.logger.info("Review manifest saved: %s (%d rows)", review_manifest, len(review_df))
        return review_df

    def run(self, max_images: Optional[int] = None) -> pd.DataFrame:
        if not self.input_folder.exists():
            raise FileNotFoundError(f"Input folder does not exist: {self.input_folder}")

        ensure_dir(self.reports_dir)
        self.config_json_path.write_text(json.dumps(self.config.to_dict(), indent=2))

        existing_map = self._load_existing_records()
        processed_paths = set(existing_map.keys()) if self.config.runtime.resume else set()

        image_paths = discover_images(self.input_folder, self.config.runtime)
        if max_images is not None:
            image_paths = image_paths[:max_images]

        self.logger.info("Discovered %d images", len(image_paths))
        self.logger.info("Resume enabled: %s (already processed: %d)", self.config.runtime.resume, len(processed_paths))

        rows_to_append: List[Dict[str, Any]] = []
        processed_count = 0
        skipped_count = 0

        pbar = tqdm(image_paths, desc="Filtering images")
        for idx, image_path in enumerate(pbar, start=1):
            path_key = str(image_path)
            if path_key in processed_paths:
                skipped_count += 1
                continue

            image_bgr = cv2.imread(path_key, cv2.IMREAD_COLOR)
            if image_bgr is None:
                skipped_count += 1
                continue

            scores, aux = self.detectors.analyze(image_bgr, image_path)
            decision = self.engine.decide(scores)

            routed_path = None
            if decision["is_invalid"]:
                routed_path = self._route_invalid_image(image_path, decision["primary_reason"])

            row = self._row_from_scores(image_path, decision, scores, routed_path)
            rows_to_append.append(row)
            existing_map[path_key] = row

            self._maybe_write_debug(aux["image_bgr"], row)

            processed_count += 1
            if (
                self.config.runtime.progress_every_n > 0
                and processed_count % self.config.runtime.progress_every_n == 0
            ):
                self.logger.info(
                    "Processed %d new images | skipped %d | last=%s",
                    processed_count,
                    skipped_count,
                    image_path.name,
                )

        self._write_jsonl_rows(rows_to_append)

        all_rows = list(existing_map.values())
        df = pd.DataFrame(all_rows)
        if not df.empty:
            df = df.sort_values("image_path").reset_index(drop=True)

        if self.config.runtime.write_csv:
            df.to_csv(self.results_csv_path, index=False)
        if self.config.runtime.write_json:
            self.results_json_path.write_text(json.dumps(all_rows, indent=2))

        summary = {
            "generated_at_utc": utc_now_iso(),
            "input_folder": str(self.input_folder),
            "output_folder": str(self.output_folder),
            "images_discovered": int(len(image_paths)),
            "new_processed": int(processed_count),
            "skipped": int(skipped_count),
            "total_records": int(len(df)),
            "invalid_count": int(df["is_invalid"].sum()) if "is_invalid" in df else 0,
            "valid_count": int((~df["is_invalid"]).sum()) if "is_invalid" in df else 0,
            "invalid_rate": float(df["is_invalid"].mean()) if "is_invalid" in df and len(df) else 0.0,
            "hard_reject_count": int(df["hard_reject"].sum()) if "hard_reject" in df else 0,
            "borderline_count": int(df["borderline"].sum()) if "borderline" in df else 0,
            "primary_reason_counts": (
                df["primary_reason"].value_counts().to_dict() if "primary_reason" in df else {}
            ),
            "average_scores": {
                k: float(df[k].mean())
                for k in [
                    "invalid_score",
                    "building_score",
                    "open_scene_score",
                    "blur_severity",
                    "glare_score",
                    "dark_score",
                    "interior_score",
                    "occlusion_score",
                    "placeholder_similarity",
                ]
                if k in df and len(df)
            },
        }

        self.summary_json_path.write_text(json.dumps(summary, indent=2))

        self.logger.info("Run complete. Results CSV: %s", self.results_csv_path)
        self.logger.info("Summary JSON: %s", self.summary_json_path)

        self.generate_review_samples(df)
        return df


def evaluate_with_labels(
    results_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    results_path_col: str = "image_path",
    labels_path_col: str = "image_path",
    labels_target_col: str = "is_invalid",
) -> pd.DataFrame:
    merged = results_df.merge(
        labels_df[[labels_path_col, labels_target_col]],
        left_on=results_path_col,
        right_on=labels_path_col,
        how="inner",
    )
    if merged.empty:
        print("No overlap between results and labels.")
        return merged

    y_true = merged[labels_target_col].astype(bool)
    y_pred = merged["is_invalid"].astype(bool)

    tp = int(np.sum((y_true == True) & (y_pred == True)))   # noqa: E712
    tn = int(np.sum((y_true == False) & (y_pred == False)))  # noqa: E712
    fp = int(np.sum((y_true == False) & (y_pred == True)))   # noqa: E712
    fn = int(np.sum((y_true == True) & (y_pred == False)))   # noqa: E712

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    print(f"Matched labels: {len(merged)}")
    print(f"TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f"Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f}")

    return merged
