"""Lightweight detector self-tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from streetview_filter.config import FilterConfig
from streetview_filter.decision import DecisionEngine
from streetview_filter.detectors import MultiCueDetectors
from streetview_filter.utils import compute_dhash, compute_phash


def run_detector_self_tests() -> None:
    cfg = FilterConfig()
    det = MultiCueDetectors(cfg)
    eng = DecisionEngine(cfg)

    # 1) Dark frame should score high on dark
    dark_img = np.full((256, 256, 3), 10, dtype=np.uint8)
    s_dark, _ = det.analyze(dark_img, Path("dark.jpg"))
    assert s_dark["dark_score"] > 0.80, f"dark_score too low: {s_dark['dark_score']}"

    # 2) Fully white frame should score high on glare
    bright_img = np.full((256, 256, 3), 255, dtype=np.uint8)
    s_bright, _ = det.analyze(bright_img, Path("bright.jpg"))
    assert s_bright["glare_score"] > 0.80, f"glare_score too low: {s_bright['glare_score']}"

    # 3) Blurred checkerboard should be blurrier than sharp checkerboard
    sharp = np.zeros((256, 256, 3), dtype=np.uint8)
    for x in range(0, 256, 16):
        cv2.line(sharp, (x, 0), (x, 255), (255, 255, 255), 1)
    for y in range(0, 256, 16):
        cv2.line(sharp, (0, y), (255, y), (255, 255, 255), 1)
    blurred = cv2.GaussianBlur(sharp, (17, 17), 0)

    s_sharp, _ = det.analyze(sharp, Path("sharp.jpg"))
    s_blur, _ = det.analyze(blurred, Path("blurred.jpg"))
    assert s_blur["blur_severity"] > s_sharp["blur_severity"], (
        s_blur["blur_severity"],
        s_sharp["blur_severity"],
    )

    # 4) Placeholder matching sanity (self-similarity)
    ref_gray = cv2.cvtColor(bright_img, cv2.COLOR_BGR2GRAY)
    det.placeholder_refs = [
        {
            "path": "synthetic_ref",
            "gray": ref_gray,
            "dhash": compute_dhash(ref_gray),
            "phash": compute_phash(ref_gray),
        }
    ]
    s_ph, _ = det.analyze(bright_img, Path("same_as_placeholder.jpg"))
    assert s_ph["placeholder_similarity"] > 0.95, s_ph["placeholder_similarity"]

    # 5) Open scene + low facade should trend toward invalid/no-facade
    scene = np.full((256, 256, 3), (230, 230, 200), dtype=np.uint8)
    scene[130:, :] = (60, 150, 60)
    s_scene, _ = det.analyze(scene, Path("open_scene.jpg"))
    d_scene = eng.decide(s_scene)
    assert s_scene["open_scene_score"] > 0.60
    assert d_scene["is_invalid"] is True

    print("All detector self-tests passed.")


if __name__ == "__main__":
    run_detector_self_tests()
