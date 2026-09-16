"""Bright-region threshold detector."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThresholdDetectionResult:
    """Threshold-detector output in image-pixel coordinates."""

    mask: np.ndarray
    threshold_value: float


class ThresholdDetector:
    """Detect bright regions using an absolute or percentile threshold."""

    def __init__(
        self,
        threshold: float | None = None,
        percentile: float | None = 99.5,
        gaussian_sigma: float = 0.0,
    ):
        if threshold is None and percentile is None:
            raise ValueError("Either threshold or percentile must be provided")
        self.threshold = threshold
        self.percentile = percentile
        self.gaussian_sigma = gaussian_sigma

    def detect(self, image: np.ndarray) -> ThresholdDetectionResult:
        image_2d = np.asarray(image)
        if image_2d.ndim != 2:
            raise ValueError(f"Expected a 2D image, got shape {image_2d.shape}")

        working = image_2d.astype(np.float32, copy=False)
        if self.gaussian_sigma > 0:
            from scipy import ndimage

            working = ndimage.gaussian_filter(working, sigma=self.gaussian_sigma)

        if self.threshold is None:
            threshold_value = float(np.percentile(working, self.percentile))
        else:
            threshold_value = float(self.threshold)

        return ThresholdDetectionResult(
            mask=working >= threshold_value,
            threshold_value=threshold_value,
        )
