"""Detector adapters used by the generic workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from smart_acquisition.detection.base import DetectionContext, DetectionResult
from smart_acquisition.detection.normalized_RF_detector import (
    ComponentFilterLimits,
    NormalizedRandomForestDetector,
    filter_component_labels,
    filter_components,
    measure_positive_components,
    run_napari_filter,
)
from smart_acquisition.detection.threshold_detector import ThresholdDetector


@dataclass(frozen=True)
class ComponentFilterConfig:
    """Reusable component-filter settings for mask-like detectors."""

    min_area_px: int = 100
    max_area_px: int | None = None
    min_elongation: float = 0.0
    max_elongation: float = 1.0
    max_mean_3nn_distance_um: float | None = None
    min_size_weighted_confidence: float | None = None
    max_selected_objects: int | None = None


@dataclass(frozen=True)
class RandomForestDetectorConfig:
    """Settings for the normalized random-forest detector adapter."""

    model_path: Path
    lower_absolute: float = 100.0
    normalization_search_min: float | None = None
    normalization_search_max: float | None = None
    write_normalization_histogram: bool = True
    ignore_zero_pixels: bool = True
    positive_label: int = 2
    prediction_tile_size_px: int = 1024
    prediction_overlap_px: int = 12
    filters: ComponentFilterConfig = field(default_factory=ComponentFilterConfig)


@dataclass(frozen=True)
class ThresholdDetectorConfig:
    """Settings for a simple threshold detector adapter."""

    threshold: float | None = None
    percentile: float | None = 99.5
    gaussian_sigma: float = 0.0
    filters: ComponentFilterConfig = field(
        default_factory=lambda: ComponentFilterConfig(min_area_px=25)
    )


class RandomForestObjectDetector:
    """Workflow adapter around ``NormalizedRandomForestDetector``."""

    def __init__(self, config: RandomForestDetectorConfig):
        self.config = config

    def detect(self, image: np.ndarray, context: DetectionContext) -> DetectionResult:
        histogram_path = None
        if self.config.write_normalization_histogram:
            histogram_path = context.output_prefix.with_name(
                f"{context.output_prefix.name}_normalization_histogram.png"
            )

        detector = NormalizedRandomForestDetector(
            self.config.model_path,
            lower_absolute=self.config.lower_absolute,
            normalization_search_min=self.config.normalization_search_min,
            normalization_search_max=self.config.normalization_search_max,
            normalization_histogram_output_path=histogram_path,
            ignore_zero_pixels=self.config.ignore_zero_pixels,
            positive_label=self.config.positive_label,
            tile_size=self.config.prediction_tile_size_px,
            overlap=self.config.prediction_overlap_px,
        )
        result = detector.detect(
            image,
            min_area_px=self.config.filters.min_area_px,
            max_area_px=self.config.filters.max_area_px,
            min_elongation=self.config.filters.min_elongation,
            max_elongation=self.config.filters.max_elongation,
            max_mean_3nn_distance_um=(
                self.config.filters.max_mean_3nn_distance_um
            ),
            min_size_weighted_confidence=(
                self.config.filters.min_size_weighted_confidence
            ),
            max_selected_objects=self.config.filters.max_selected_objects,
            pixel_size_x_um=context.pixel_size_x_um,
            pixel_size_y_um=context.pixel_size_y_um,
            interactive=context.interactive,
            selection_callback=context.selection_callback,
            preview_callback=context.preview_callback,
        )
        return DetectionResult(
            mask=result.mask,
            measurements=result.measurements,
            labels=result.labels,
            positive_probability=result.positive_probability,
            normalized_image=result.normalized_image,
        )


class ThresholdObjectDetector:
    """Workflow adapter for absolute/percentile threshold detection."""

    def __init__(self, config: ThresholdDetectorConfig):
        self.config = config

    def detect(self, image: np.ndarray, context: DetectionContext) -> DetectionResult:
        threshold_result = ThresholdDetector(
            threshold=self.config.threshold,
            percentile=(
                self.config.percentile if self.config.threshold is None else None
            ),
            gaussian_sigma=self.config.gaussian_sigma,
        ).detect(image)
        prediction_labels = np.asarray(threshold_result.mask, dtype=np.uint8)

        if context.interactive:
            mask, measurements = run_napari_filter(
                image,
                prediction_labels,
                positive_label=1,
                min_area_px=self.config.filters.min_area_px,
                max_area_px=self.config.filters.max_area_px,
                min_elongation=self.config.filters.min_elongation,
                max_elongation=self.config.filters.max_elongation,
                max_mean_3nn_distance_um=(
                    self.config.filters.max_mean_3nn_distance_um
                ),
                min_size_weighted_confidence=(
                    self.config.filters.min_size_weighted_confidence
                ),
                max_selected_objects=self.config.filters.max_selected_objects,
                pixel_size_x_um=context.pixel_size_x_um,
                pixel_size_y_um=context.pixel_size_y_um,
                selection_callback=context.selection_callback,
                preview_callback=context.preview_callback,
            )
            return DetectionResult(
                mask=mask,
                measurements=measurements,
                labels=prediction_labels,
            )

        component_labels, measurements = measure_positive_components(
            prediction_labels,
            positive_label=1,
            pixel_size_x_um=context.pixel_size_x_um,
            pixel_size_y_um=context.pixel_size_y_um,
        )
        mask, measurements = filter_component_labels(
            component_labels,
            measurements,
            ComponentFilterLimits(
                min_area_px=self.config.filters.min_area_px,
                max_area_px=self.config.filters.max_area_px,
                min_elongation=self.config.filters.min_elongation,
                max_elongation=self.config.filters.max_elongation,
                max_mean_3nn_distance_um=(
                    self.config.filters.max_mean_3nn_distance_um
                ),
                min_size_weighted_confidence=(
                    self.config.filters.min_size_weighted_confidence
                ),
            ),
            pixel_size_x_um=context.pixel_size_x_um,
            pixel_size_y_um=context.pixel_size_y_um,
        )
        mask, measurements = _limit_measurements_by_area(
            mask,
            component_labels,
            measurements,
            self.config.filters.max_selected_objects,
        )
        return DetectionResult(mask=mask, measurements=measurements, labels=prediction_labels)


def _limit_measurements_by_area(
    mask: np.ndarray,
    component_labels: np.ndarray,
    measurements,
    max_selected_objects: int | None,
):
    if max_selected_objects is None or max_selected_objects <= 0:
        return mask, measurements
    if len(measurements) <= max_selected_objects:
        return mask, measurements

    kept = tuple(
        sorted(measurements, key=lambda item: item.area_px, reverse=True)[
            :max_selected_objects
        ]
    )
    kept_labels = [measurement.label for measurement in kept]
    return np.isin(component_labels, kept_labels), kept
