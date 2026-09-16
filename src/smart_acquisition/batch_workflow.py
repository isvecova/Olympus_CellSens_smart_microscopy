"""RF batch-workflow compatibility helpers.

The stable orchestration now lives in :mod:`smart_acquisition.workflow`. This
module keeps the previous RF-specific API used by the GUI and older scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from smart_acquisition.detection.adapters import (
    ComponentFilterConfig,
    RandomForestDetectorConfig,
    RandomForestObjectDetector,
)
from smart_acquisition.detection.base import DetectionResult
from smart_acquisition.models import AcquisitionTile
from smart_acquisition.targeting.coordinate_transform import AffinePixelToStage
from smart_acquisition.workflow import (
    ProcessedTargetPlan,
    ReviewConfig,
    TargetPlanningConfig,
    VsiInputConfig,
    WorkflowConfig,
    plan_targets_from_detection as _plan_targets_from_detection,
    process_vsi_to_target_plan,
)


@dataclass(frozen=True)
class BatchWorkflowConfig:
    """Settings for one or more reviewed RF detection runs."""

    model_path: Path
    output_dir: Path
    planning_mode: str = "optimized_z_aware_irregular_mosaics"
    scene: int | str | None = 0
    channel: int = 0
    time: int = 0
    metadata_position: str = "origin"
    overview_pixel_size_x_um: float | None = 1.3
    overview_pixel_size_y_um: float | None = 1.3
    fallback_z_step_um: float | None = None
    classification_pixel_size_um: float = 1.3
    build_z_cache_at_classification_pixel_size: bool = True
    use_z_projection_cache: bool = True
    write_z_projection_cache: bool = True
    ignore_zero_pixels: bool = True
    lower_absolute: float = 100.0
    normalization_search_min: float | None = None
    normalization_search_max: float | None = None
    positive_label: int = 2
    prediction_tile_size_px: int = 1024
    prediction_overlap_px: int = 12
    use_napari_filter: bool = True
    min_area_px: int = 100
    max_area_px: int | None = None
    min_elongation: float = 0.0
    max_elongation: float = 1.0
    max_mean_3nn_distance_um: float | None = None
    min_size_weighted_confidence: float | None = None
    max_selected_objects: int | None = None
    show_napari_tiling_preview: bool = True
    invert_x: bool = False
    invert_y: bool = False
    target_image_width_px: int = 2304
    target_image_height_px: int = 2304
    target_pixel_size_x_um: float = 0.325
    target_pixel_size_y_um: float = 0.325
    target_tile_overlap_fraction: float = 0.1
    group_merge_distance_factor: float = 1.0
    max_group_z_difference_um: float = 10.0
    max_extra_tile_fraction: float = 0.10
    coverage_margin_um: float = 0.0
    polygon_simplify_tolerance_um: float = 10.0
    estimate_tile_scan_z: bool = True
    write_tile_scan_z_histograms: bool = True
    z_estimation_labels: tuple[int, ...] = (1, 2)


def process_zstack_to_target_plan(
    image_path: str | Path,
    config: BatchWorkflowConfig,
) -> ProcessedTargetPlan:
    """Process one VSI file with the default RF detector and save a target plan."""

    detector = RandomForestObjectDetector(
        RandomForestDetectorConfig(
            model_path=config.model_path,
            lower_absolute=config.lower_absolute,
            normalization_search_min=config.normalization_search_min,
            normalization_search_max=config.normalization_search_max,
            ignore_zero_pixels=config.ignore_zero_pixels,
            positive_label=config.positive_label,
            prediction_tile_size_px=config.prediction_tile_size_px,
            prediction_overlap_px=config.prediction_overlap_px,
            filters=ComponentFilterConfig(
                min_area_px=config.min_area_px,
                max_area_px=config.max_area_px,
                min_elongation=config.min_elongation,
                max_elongation=config.max_elongation,
                max_mean_3nn_distance_um=config.max_mean_3nn_distance_um,
                min_size_weighted_confidence=config.min_size_weighted_confidence,
                max_selected_objects=config.max_selected_objects,
            ),
        )
    )
    return process_vsi_to_target_plan(
        image_path,
        detector,
        _workflow_config_from_batch_config(config),
    )


def plan_targets_from_detection(
    *,
    detection_mask: np.ndarray,
    detection_labels: np.ndarray,
    measurements,
    classifier_metadata_z_um: float,
    transform: AffinePixelToStage,
    target_tile: AcquisitionTile,
    config: BatchWorkflowConfig,
    argmax_z: np.ndarray | None,
    z_positions_um: tuple[float, ...] | None,
) -> tuple[list, list, list, list]:
    """Compatibility wrapper for older callers."""

    from smart_acquisition.models import OverviewImageMetadata

    metadata = OverviewImageMetadata(
        stage_x_um=0.0,
        stage_y_um=0.0,
        stage_z_um=classifier_metadata_z_um,
        pixel_size_x_um=1.0,
        pixel_size_y_um=1.0,
        size_x_px=int(np.asarray(detection_mask).shape[1]),
        size_y_px=int(np.asarray(detection_mask).shape[0]),
    )
    return _plan_targets_from_detection(
        detection=DetectionResult(
            mask=detection_mask,
            measurements=measurements,
            labels=detection_labels,
        ),
        transform=transform,
        metadata=metadata,
        planning=_planning_config_from_batch_config(config, target_tile=target_tile),
        argmax_z=argmax_z,
        z_positions_um=z_positions_um,
    )


def _workflow_config_from_batch_config(config: BatchWorkflowConfig) -> WorkflowConfig:
    return WorkflowConfig(
        output_dir=config.output_dir,
        input=VsiInputConfig(
            scene=config.scene,
            channel=config.channel,
            time=config.time,
            metadata_position=config.metadata_position,
            overview_pixel_size_x_um=config.overview_pixel_size_x_um,
            overview_pixel_size_y_um=config.overview_pixel_size_y_um,
            fallback_z_step_um=config.fallback_z_step_um,
            processing_pixel_size_um=config.classification_pixel_size_um,
            build_z_cache_at_processing_pixel_size=(
                config.build_z_cache_at_classification_pixel_size
            ),
            use_z_projection_cache=config.use_z_projection_cache,
            write_z_projection_cache=config.write_z_projection_cache,
            invert_x=config.invert_x,
            invert_y=config.invert_y,
        ),
        review=ReviewConfig(
            enabled=config.use_napari_filter,
            show_tiling_preview=config.show_napari_tiling_preview,
        ),
        planning=_planning_config_from_batch_config(
            config,
            target_tile=AcquisitionTile(
                width_px=config.target_image_width_px,
                height_px=config.target_image_height_px,
                pixel_size_x_um=config.target_pixel_size_x_um,
                pixel_size_y_um=config.target_pixel_size_y_um,
                overlap_fraction=config.target_tile_overlap_fraction,
            ),
        ),
    )


def _planning_config_from_batch_config(
    config: BatchWorkflowConfig,
    *,
    target_tile: AcquisitionTile,
) -> TargetPlanningConfig:
    return TargetPlanningConfig(
        planning_mode=config.planning_mode,
        target_tile=target_tile,
        group_merge_distance_factor=config.group_merge_distance_factor,
        max_group_z_difference_um=config.max_group_z_difference_um,
        max_extra_tile_fraction=config.max_extra_tile_fraction,
        coverage_margin_um=config.coverage_margin_um,
        polygon_simplify_tolerance_um=config.polygon_simplify_tolerance_um,
        estimate_tile_scan_z=config.estimate_tile_scan_z,
        write_tile_scan_z_histograms=config.write_tile_scan_z_histograms,
        z_estimation_labels=config.z_estimation_labels,
    )

