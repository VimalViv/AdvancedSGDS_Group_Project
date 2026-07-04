"""Multi-cue image detectors for the street-view filter."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from streetview_filter.config import FilterConfig
from streetview_filter.utils import (
    clamp,
    compute_dhash,
    compute_phash,
    corr_similarity,
    hash_similarity,
    inverse_normalize,
    load_grayscale_images,
    logo_absence_score,
    morphology_open_bool,
    normalize,
    resize_keep_aspect,
    safe_template_match,
    weighted_sum,
)


class MultiCueDetectors:
    def __init__(self, config: FilterConfig):
        self.config = config
        self.placeholder_refs = self._load_placeholder_refs(config.placeholder_reference_paths)
        self.logo_templates = self._load_logo_templates(config.logo_template_paths)

    # ------------------------------------------------------------------
    # Reference image loading  (unified via load_grayscale_images)
    # ------------------------------------------------------------------

    def _load_placeholder_refs(self, paths: List[str]) -> List[Dict[str, Any]]:
        refs: List[Dict[str, Any]] = []
        for path_str, gray in load_grayscale_images(paths):
            refs.append({
                "path": path_str,
                "gray": gray,
                "dhash": compute_dhash(gray),
                "phash": compute_phash(gray),
            })
        return refs

    def _load_logo_templates(self, paths: List[str]) -> List[np.ndarray]:
        return [gray for _, gray in load_grayscale_images(paths)]

    # ------------------------------------------------------------------
    # Mask computation  (morphology_open_bool replaces 3x inline calls)
    # ------------------------------------------------------------------

    def _compute_masks(self, hsv: np.ndarray, gray: np.ndarray) -> Dict[str, np.ndarray]:
        h, s, v = cv2.split(hsv)
        h = h.astype(np.int16)
        s = s.astype(np.int16)
        v = v.astype(np.int16)

        sky = (((h >= 85) & (h <= 130) & (s <= 140) & (v >= 70)) | ((s <= 25) & (v >= 210)))
        vegetation = (h >= 28) & (h <= 95) & (s >= 40) & (v >= 30)

        hh, ww = gray.shape
        lower = np.zeros((hh, ww), dtype=bool)
        lower[int(0.45 * hh):, :] = True

        road = (s <= 60) & (v >= 45) & (v <= 200) & lower
        road = road & (~sky) & (~vegetation)

        kernel = np.ones((3, 3), np.uint8)
        sky = morphology_open_bool(sky, kernel)
        vegetation = morphology_open_bool(vegetation, kernel)
        road = morphology_open_bool(road, kernel)

        facade = (~sky) & (~vegetation) & (gray > 28)

        return {
            "sky": sky,
            "vegetation": vegetation,
            "road": road,
            "facade_candidate": facade,
        }

    # ------------------------------------------------------------------
    # Line features
    # ------------------------------------------------------------------

    def _line_features(self, canny: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        h, w = canny.shape
        working = canny.copy()
        if mask is not None:
            working = cv2.bitwise_and(working, working, mask=mask.astype(np.uint8) * 255)

        min_len = max(24, int(0.08 * min(h, w)))
        lines = cv2.HoughLinesP(
            working,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=min_len,
            maxLineGap=8,
        )

        if lines is None:
            return {
                "line_count": 0.0,
                "horizontal_ratio": 0.0,
                "vertical_ratio": 0.0,
                "diagonal_ratio": 0.0,
                "hv_ratio": 0.0,
                "vanish_like": 0.0,
            }

        # HoughLinesP returns (N,1,4) or (N,4) depending on OpenCV version
        if lines.ndim == 3:
            lines = lines[:, 0, :]

        total = float(len(lines))
        horizontal = 0
        vertical = 0
        diagonal = 0
        vanish_like = 0

        for line in lines:
            x1, y1, x2, y2 = [int(v) for v in line]
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            if length < min_len:
                continue

            angle = abs(math.degrees(math.atan2(dy, dx)))
            if angle < 20 or angle > 160:
                horizontal += 1
            elif 70 < angle < 110:
                vertical += 1
            elif 20 <= angle <= 70 or 110 <= angle <= 160:
                diagonal += 1

            top_y = min(y1, y2)
            bottom_y = max(y1, y2)
            mid_x = 0.5 * (x1 + x2)
            if (
                bottom_y >= 0.85 * h
                and top_y <= 0.65 * h
                and abs(mid_x - (w / 2.0)) <= 0.35 * w
                and (20 <= angle <= 70 or 110 <= angle <= 160)
            ):
                vanish_like += 1

        hv_ratio = (horizontal + vertical) / max(total, 1.0)

        return {
            "line_count": total,
            "horizontal_ratio": horizontal / max(total, 1.0),
            "vertical_ratio": vertical / max(total, 1.0),
            "diagonal_ratio": diagonal / max(total, 1.0),
            "hv_ratio": hv_ratio,
            "vanish_like": vanish_like / max(total, 1.0),
        }

    # ------------------------------------------------------------------
    # Individual detectors  (weighted_sum replaces inline clamp(w1*v1+...))
    # ------------------------------------------------------------------

    def _score_darkness(self, v_channel: np.ndarray) -> Dict[str, float]:
        mean_v = float(v_channel.mean())
        low_ratio = float(np.mean(v_channel < 40))
        very_low_ratio = float(np.mean(v_channel < 20))

        score = weighted_sum([
            (0.55, inverse_normalize(mean_v, 25.0, 95.0)),
            (0.35, normalize(low_ratio, 0.20, 0.92)),
            (0.10, normalize(very_low_ratio, 0.05, 0.70)),
        ])
        return {
            "brightness_mean_v": mean_v,
            "dark_low_ratio": low_ratio,
            "dark_very_low_ratio": very_low_ratio,
            "dark_score": score,
        }

    def _score_glare(self, bgr: np.ndarray, hsv: np.ndarray, gray: np.ndarray) -> Dict[str, float]:
        v = hsv[:, :, 2]
        s = hsv[:, :, 1]
        max_channel = np.max(bgr, axis=2)

        bright_mask = v > 245
        bright_ratio = float(np.mean(bright_mask))
        clip_ratio = float(np.mean(max_channel > 250))
        bright_low_sat_ratio = float(np.mean((v > 240) & (s < 30)))

        local_std = float(np.std(gray[bright_mask])) if np.any(bright_mask) else 50.0
        washout_score = inverse_normalize(local_std, 10.0, 45.0)

        hotspot_ratio = 0.0
        if np.any(bright_mask):
            num, _, stats, _ = cv2.connectedComponentsWithStats(bright_mask.astype(np.uint8), connectivity=8)
            if num > 1:
                hotspot_ratio = float(np.max(stats[1:, cv2.CC_STAT_AREA]) / bright_mask.size)

        glare_score = weighted_sum([
            (0.38, normalize(bright_ratio, 0.02, 0.55)),
            (0.24, normalize(clip_ratio, 0.01, 0.20)),
            (0.20, normalize(bright_low_sat_ratio, 0.01, 0.35)),
            (0.18, max(washout_score, normalize(hotspot_ratio, 0.02, 0.40))),
        ])

        return {
            "bright_pixel_ratio": bright_ratio,
            "clip_ratio": clip_ratio,
            "bright_low_sat_ratio": bright_low_sat_ratio,
            "glare_hotspot_ratio": hotspot_ratio,
            "glare_score": glare_score,
        }

    def _score_blur(
        self,
        gray: np.ndarray,
        canny: np.ndarray,
        facade_mask: np.ndarray,
    ) -> Dict[str, float]:
        h, w = gray.shape
        roi = facade_mask.copy()
        roi[:int(0.10 * h), :] = False
        roi[int(0.93 * h):, :] = False

        if float(np.mean(roi)) < 0.06:
            roi = np.zeros_like(facade_mask, dtype=bool)
            roi[int(0.20 * h):int(0.90 * h), int(0.10 * w):int(0.90 * w)] = True

        lap = cv2.Laplacian(gray, cv2.CV_32F)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(gx, gy)

        roi_values = roi if np.any(roi) else np.ones_like(roi, dtype=bool)

        lap_var = float(np.var(lap[roi_values]))
        tenengrad_mean = float(np.mean(grad_mag[roi_values]))
        edge_density = float(np.mean(canny[roi_values] > 0))

        blur_severity = weighted_sum([
            (0.46, inverse_normalize(lap_var, 45.0, 250.0)),
            (0.34, inverse_normalize(tenengrad_mean, 8.0, 30.0)),
            (0.20, inverse_normalize(edge_density, 0.04, 0.20)),
        ])

        return {
            "laplacian_var_roi": lap_var,
            "tenengrad_mean_roi": tenengrad_mean,
            "edge_density_roi": edge_density,
            "blur_severity": blur_severity,
        }

    def _score_building_presence(
        self,
        gray: np.ndarray,
        canny: np.ndarray,
        facade_mask: np.ndarray,
    ) -> Dict[str, float]:
        h, w = gray.shape

        mid_band = np.zeros_like(facade_mask, dtype=bool)
        mid_band[int(0.12 * h):int(0.88 * h), :] = True

        candidate = facade_mask & mid_band
        candidate_ratio = float(np.mean(candidate))

        if np.any(candidate):
            edge_density = float(np.mean(canny[candidate] > 0))
            texture_energy = float(np.mean(np.abs(cv2.Laplacian(gray, cv2.CV_32F)[candidate])))
        else:
            edge_density = float(np.mean(canny[mid_band] > 0))
            texture_energy = float(np.mean(np.abs(cv2.Laplacian(gray, cv2.CV_32F)[mid_band])))

        left_side = float(np.mean(candidate[:, :w // 3]))
        right_side = float(np.mean(candidate[:, (2 * w) // 3:]))
        side_presence = normalize((left_side + right_side) / 2.0, 0.06, 0.28)

        line_features = self._line_features(canny, candidate)
        vh_balance = 1.0 - abs(line_features["vertical_ratio"] - line_features["horizontal_ratio"])
        line_score = 0.5 * vh_balance + 0.5 * normalize(line_features["hv_ratio"], 0.20, 0.75)

        building_score = weighted_sum([
            (0.28, normalize(candidate_ratio, 0.10, 0.55)),
            (0.24, normalize(edge_density, 0.025, 0.16)),
            (0.18, normalize(texture_energy, 3.0, 20.0)),
            (0.16, side_presence),
            (0.14, clamp(line_score)),
        ])

        return {
            "building_candidate_ratio": candidate_ratio,
            "building_edge_density": edge_density,
            "building_texture_energy": texture_energy,
            "building_side_presence": side_presence,
            "building_hv_line_ratio": line_features["hv_ratio"],
            "building_vh_balance": vh_balance,
            "building_score": building_score,
            "line_vanish_like": line_features["vanish_like"],
            "line_diagonal_ratio": line_features["diagonal_ratio"],
            "line_horizontal_ratio": line_features["horizontal_ratio"],
            "line_vertical_ratio": line_features["vertical_ratio"],
        }

    def _score_open_scene(
        self,
        masks: Dict[str, np.ndarray],
        canny: np.ndarray,
        building_score: float,
    ) -> Dict[str, float]:
        sky = masks["sky"]
        vegetation = masks["vegetation"]
        road = masks["road"]

        h, _ = sky.shape
        sky_ratio = float(np.mean(sky))
        top_sky_ratio = float(np.mean(sky[:int(0.50 * h), :]))
        vegetation_ratio = float(np.mean(vegetation))
        road_ratio = float(np.mean(road))
        global_edge_density = float(np.mean(canny > 0))

        open_scene_score = weighted_sum([
            (0.34, normalize(top_sky_ratio, 0.18, 0.75)),
            (0.24, normalize(vegetation_ratio, 0.12, 0.62)),
            (0.18, normalize(road_ratio, 0.12, 0.55)),
            (0.14, inverse_normalize(global_edge_density, 0.05, 0.17)),
            (0.10, inverse_normalize(building_score, 0.28, 0.66)),
        ])

        return {
            "sky_ratio": sky_ratio,
            "sky_top_ratio": top_sky_ratio,
            "vegetation_ratio": vegetation_ratio,
            "road_ratio": road_ratio,
            "global_edge_density": global_edge_density,
            "open_scene_score": open_scene_score,
        }

    def _score_road_corridor(
        self,
        masks: Dict[str, np.ndarray],
        line_vanish_like: float,
    ) -> Dict[str, float]:
        road = masks["road"]
        facade = masks["facade_candidate"]
        h, w = road.shape

        center_bottom = road[int(0.55 * h):, int(0.25 * w):int(0.75 * w)]
        center_bottom_road_ratio = float(np.mean(center_bottom)) if center_bottom.size else 0.0

        left_facade = float(np.mean(facade[int(0.20 * h):int(0.90 * h), :int(0.22 * w)]))
        right_facade = float(np.mean(facade[int(0.20 * h):int(0.90 * h), int(0.78 * w):]))
        side_facade_ratio = 0.5 * (left_facade + right_facade)

        distant_center_facade = float(
            np.mean(facade[:int(0.35 * h), int(0.35 * w):int(0.65 * w)])
        )

        corridor_score = weighted_sum([
            (0.45, normalize(center_bottom_road_ratio, 0.08, 0.62)),
            (0.23, normalize(line_vanish_like, 0.08, 0.55)),
            (0.20, inverse_normalize(side_facade_ratio, 0.08, 0.35)),
            (0.12, inverse_normalize(distant_center_facade, 0.04, 0.20)),
        ])

        return {
            "road_center_bottom_ratio": center_bottom_road_ratio,
            "road_side_facade_ratio": side_facade_ratio,
            "road_distant_center_facade_ratio": distant_center_facade,
            "road_corridor_score": corridor_score,
        }

    def _score_occlusion(self, masks: Dict[str, np.ndarray], gray: np.ndarray) -> Dict[str, float]:
        sky = masks["sky"]
        vegetation = masks["vegetation"]
        road = masks["road"]

        h, w = gray.shape
        foreground = (~sky) & (~vegetation) & (~road) & (gray > 30)

        lower = np.zeros_like(foreground, dtype=bool)
        lower[int(0.35 * h):, :] = True
        obj = (foreground & lower).astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)
        obj = cv2.morphologyEx(obj, cv2.MORPH_OPEN, kernel, iterations=1)
        obj = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, kernel, iterations=1)

        largest_ratio = 0.0
        num, _, stats, centroids = cv2.connectedComponentsWithStats(obj, connectivity=8)
        if num > 1:
            for i in range(1, num):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area <= 0:
                    continue
                cx = float(centroids[i, 0])
                cy = float(centroids[i, 1])
                if cy < 0.45 * h:
                    continue
                if not (0.20 * w <= cx <= 0.80 * w):
                    continue
                largest_ratio = max(largest_ratio, area / float(h * w))

        center_region = obj[int(0.45 * h):, int(0.20 * w):int(0.80 * w)]
        center_cover_ratio = float(np.mean(center_region > 0)) if center_region.size else 0.0

        occlusion_score = weighted_sum([
            (0.55, normalize(largest_ratio, 0.04, 0.30)),
            (0.45, normalize(center_cover_ratio, 0.07, 0.50)),
        ])

        return {
            "occlusion_largest_component_ratio": largest_ratio,
            "occlusion_center_cover_ratio": center_cover_ratio,
            "occlusion_score": occlusion_score,
        }

    def _score_logo_presence(self, gray: np.ndarray) -> Dict[str, float]:
        if not self.logo_templates:
            return {
                "logo_templates_available": 0.0,
                "logo_match_score": 0.5,
            }

        h, w = gray.shape
        left_crop = gray[int(0.82 * h):, :int(0.36 * w)]
        right_crop = gray[int(0.82 * h):, int(0.64 * w):]

        max_val = 0.0
        for templ in self.logo_templates:
            max_val = max(max_val, safe_template_match(left_crop, templ))
            max_val = max(max_val, safe_template_match(right_crop, templ))

        return {
            "logo_templates_available": 1.0,
            "logo_match_score": clamp(max_val),
        }

    def _score_placeholder_similarity(self, gray: np.ndarray) -> Dict[str, float]:
        if not self.placeholder_refs:
            return {
                "placeholder_similarity": 0.0,
                "placeholder_best_reference": "",
            }

        d_hash = compute_dhash(gray)
        p_hash = compute_phash(gray)

        best_similarity = 0.0
        best_ref = ""

        for ref in self.placeholder_refs:
            sim_d = hash_similarity(d_hash, ref["dhash"])
            sim_p = hash_similarity(p_hash, ref["phash"])
            sim_corr = corr_similarity(gray, ref["gray"])
            sim = 0.45 * sim_p + 0.35 * sim_d + 0.20 * sim_corr
            if sim > best_similarity:
                best_similarity = sim
                best_ref = ref["path"]

        return {
            "placeholder_similarity": clamp(best_similarity),
            "placeholder_best_reference": best_ref,
        }

    def _score_non_google(self, gray: np.ndarray, logo_score: float, logo_available: bool) -> Dict[str, float]:
        h, w = gray.shape

        strip = max(2, int(round(0.05 * min(h, w))))
        strips = [
            gray[:strip, :],
            gray[h - strip:, :],
            gray[:, :strip],
            gray[:, w - strip:],
        ]
        border_extreme_ratio = float(
            np.mean(
                np.concatenate([((x < 15) | (x > 245)).reshape(-1) for x in strips], axis=0)
            )
        )

        aspect_ratio = w / float(max(h, 1))
        aspect_score = normalize(abs(aspect_ratio - 1.0), 0.35, 1.0)

        non_google_score = weighted_sum([
            (0.45, normalize(border_extreme_ratio, 0.05, 0.45)),
            (0.25, aspect_score),
            (0.30, logo_absence_score(logo_score, logo_available)),
        ])

        return {
            "border_extreme_ratio": border_extreme_ratio,
            "aspect_ratio": aspect_ratio,
            "non_google_score": non_google_score,
        }

    def _score_interior(
        self,
        hsv: np.ndarray,
        building_score: float,
        sky_ratio: float,
        vegetation_ratio: float,
        center_road_ratio: float,
        logo_score: float,
        logo_available: bool,
        line_horizontal_ratio: float,
    ) -> Dict[str, float]:
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        warm_ratio = float(np.mean((h <= 25) & (s >= 35) & (v >= 35)))

        outdoor_evidence = clamp(
            0.40 * building_score
            + 0.25 * sky_ratio
            + 0.20 * center_road_ratio
            + 0.15 * vegetation_ratio
        )

        interior_score = weighted_sum([
            (0.48, inverse_normalize(outdoor_evidence, 0.18, 0.58)),
            (0.22, normalize(warm_ratio, 0.12, 0.55)),
            (0.16, normalize(line_horizontal_ratio, 0.25, 0.85)),
            (0.14, logo_absence_score(logo_score, logo_available)),
        ])

        return {
            "interior_warm_ratio": warm_ratio,
            "interior_outdoor_evidence": outdoor_evidence,
            "interior_score": interior_score,
        }

    # ------------------------------------------------------------------
    # Main analysis entry-point
    # ------------------------------------------------------------------

    def analyze(self, image_bgr: np.ndarray, image_path: Path) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        image_bgr, _ = resize_keep_aspect(image_bgr, self.config.runtime.max_side)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        canny = cv2.Canny(gray, 80, 180)

        masks = self._compute_masks(hsv, gray)

        scores: Dict[str, float] = {}

        scores.update(self._score_darkness(hsv[:, :, 2]))
        scores.update(self._score_glare(image_bgr, hsv, gray))
        scores.update(self._score_blur(gray, canny, masks["facade_candidate"]))
        scores.update(self._score_building_presence(gray, canny, masks["facade_candidate"]))
        scores.update(self._score_open_scene(masks, canny, scores["building_score"]))
        scores.update(self._score_road_corridor(masks, scores["line_vanish_like"]))
        scores.update(self._score_occlusion(masks, gray))
        scores.update(self._score_logo_presence(gray))
        scores.update(self._score_placeholder_similarity(gray))

        logo_available = bool(scores["logo_templates_available"] > 0.5)

        scores.update(self._score_non_google(
            gray,
            scores["logo_match_score"],
            logo_available,
        ))
        scores.update(self._score_interior(
            hsv=hsv,
            building_score=scores["building_score"],
            sky_ratio=scores["sky_ratio"],
            vegetation_ratio=scores["vegetation_ratio"],
            center_road_ratio=scores["road_center_bottom_ratio"],
            logo_score=scores["logo_match_score"],
            logo_available=logo_available,
            line_horizontal_ratio=scores["line_horizontal_ratio"],
        ))

        scores["image_height"] = float(image_bgr.shape[0])
        scores["image_width"] = float(image_bgr.shape[1])

        return scores, {"image_bgr": image_bgr, **masks}
