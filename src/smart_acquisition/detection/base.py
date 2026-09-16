"""Common detection contracts for the smart-acquisition workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from smart_acquisition.detection.normalized_RF_detector import (
    ComponentMeasurement,
    NapariTilingPreview,
)
from smart_acquisition.models import OverviewImageMetadata


SelectionCallback = Callable[
    [tuple[ComponentMeasurement, ...], int, np.ndarray, np.ndarray, np.ndarray],
    tuple[ComponentMeasurement, ...],
]
PreviewCallback = Callable[
    [np.ndarray, tuple[ComponentMeasurement, ...], np.ndarray],
    NapariTilingPreview,
]


@dataclass(frozen=True)
class DetectionContext:
    """Runtime context passed from the stable workflow into a detector."""

    image_path: Path
    output_prefix: Path
    metadata: OverviewImageMetadata
    pixel_size_x_um: float
    pixel_size_y_um: float
    interactive: bool = False
    selection_callback: SelectionCallback | None = None
    preview_callback: PreviewCallback | None = None
    argmax_z: np.ndarray | None = None
    z_positions_um: tuple[float, ...] | None = None


@dataclass(frozen=True)
class DetectionResult:
    """Detector output in processing-image coordinates.

    Detectors may return only a mask and measurements, or richer artifacts such
    as class labels, positive-class probabilities, and a normalized image. The
    workflow only requires ``mask`` and ``measurements``; the richer artifacts
    are used for QC outputs, napari previews, and Z-aware planning.
    """

    mask: np.ndarray
    measurements: tuple[ComponentMeasurement, ...]
    labels: np.ndarray | None = None
    positive_probability: np.ndarray | None = None
    normalized_image: np.ndarray | None = None

    @property
    def planning_labels(self) -> np.ndarray:
        """Labels used by planning when a detector has no native label image."""

        if self.labels is not None:
            return np.asarray(self.labels)
        return np.asarray(self.mask, dtype=np.uint8)


class ObjectDetector(Protocol):
    """Protocol implemented by swappable object detectors."""

    def detect(self, image: np.ndarray, context: DetectionContext) -> DetectionResult:
        """Return selected positive detections for ``image``."""

