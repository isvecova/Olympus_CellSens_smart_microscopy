"""Detector-agnostic smart-acquisition workflow orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from smart_acquisition.cellsens.target_plan_io import (
    TargetPlan,
    save_target_plan,
    write_combined_cellsens_xml,
)
from smart_acquisition.detection.base import (
    DetectionContext,
    DetectionResult,
    ObjectDetector,
)
from smart_acquisition.detection.normalized_RF_detector import ComponentMeasurement
from smart_acquisition.image_input.bioio_vsi import read_vsi_zstack
from smart_acquisition.image_input.resampling import resample_to_pixel_size
from smart_acquisition.interactive_review import (
    ReviewPlanningOptions,
    build_napari_tiling_preview,
    select_measurements_minimizing_tiles,
)
from smart_acquisition.models import (
    AcquisitionTile,
    OverviewImageMetadata,
    PolygonRegion,
    RectangularRegion,
    StagePosition,
)
from smart_acquisition.outputs import (
    write_labels_if_requested,
    write_mask,
    write_measurements_csv,
    write_polygon_regions_csv,
    write_probability_if_requested,
    write_rectangular_regions_csv,
    write_region_confidences_csv,
    write_stage_positions_csv,
)
from smart_acquisition.targeting.coordinate_transform import AffinePixelToStage
from smart_acquisition.targeting.polygon_regions import (
    plan_mixed_positions_and_polygon_regions_from_mask,
    plan_optimized_z_aware_positions_and_polygon_regions_from_mask,
    plan_z_aware_positions_and_polygon_regions_from_mask,
    plan_z_aware_tile_positions_from_mask,
)
from smart_acquisition.targeting.tile_planner import (
    plan_mixed_positions_and_tilescans,
    plan_positions_for_high_mag_tiles,
)
from smart_acquisition.targeting.z_estimation import (
    resample_argmax_z_to_shape,
    update_tile_scan_region_z_positions,
    write_region_z_histograms_csv,
)


@dataclass(frozen=True)
class VsiInputConfig:
    """Settings for loading and preparing one VSI overview or z-stack."""

    scene: int | str | None = 0
    channel: int = 0
    time: int = 0
    metadata_position: str = "origin"
    overview_pixel_size_x_um: float | None = 1.3
    overview_pixel_size_y_um: float | None = 1.3
    fallback_z_step_um: float | None = None
    processing_pixel_size_um: float | tuple[float, float] | None = 1.3
    build_z_cache_at_processing_pixel_size: bool = True
    use_z_projection_cache: bool = True
    write_z_projection_cache: bool = True
    invert_x: bool = False
    invert_y: bool = False


@dataclass(frozen=True)
class ReviewConfig:
    """Settings for optional user review of detected positive areas."""

    enabled: bool = True
    show_tiling_preview: bool = True


@dataclass(frozen=True)
class TargetPlanningConfig:
    """Settings that convert selected detections into CellSens targets."""

    planning_mode: str = "optimized_z_aware_irregular_mosaics"
    target_tile: AcquisitionTile = field(
        default_factory=lambda: AcquisitionTile(
            width_px=2304,
            height_px=2304,
            pixel_size_x_um=0.325,
            pixel_size_y_um=0.325,
            overlap_fraction=0.1,
        )
    )
    group_merge_distance_factor: float = 1.0
    max_group_z_difference_um: float = 10.0
    max_extra_tile_fraction: float = 0.10
    coverage_margin_um: float = 0.0
    polygon_simplify_tolerance_um: float = 10.0
    estimate_tile_scan_z: bool = True
    write_tile_scan_z_histograms: bool = True
    z_estimation_labels: tuple[int, ...] = (1, 2)
    detected_name_prefix: str = "Detected"
    planned_name_prefix: str = "Planned"


@dataclass(frozen=True)
class WorkflowConfig:
    """Stable workflow settings independent of the detection implementation."""

    output_dir: Path
    input: VsiInputConfig = field(default_factory=VsiInputConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
    planning: TargetPlanningConfig = field(default_factory=TargetPlanningConfig)


@dataclass(frozen=True)
class ProcessedTargetPlan:
    """Outputs created for one processed source image."""

    plan: TargetPlan
    plan_path: Path
    mask_path: Path
    labels_path: Path
    probability_path: Path
    positions_path: Path
    planned_positions_path: Path
    rectangular_regions_path: Path
    polygon_regions_path: Path
    region_confidence_path: Path
    z_histograms_path: Path | None = None


def process_vsi_to_target_plan(
    image_path: str | Path,
    detector: ObjectDetector,
    config: WorkflowConfig,
) -> ProcessedTargetPlan:
    """Process one VSI file with a pluggable detector and save a target plan."""

    image_path = Path(image_path)
    output_prefix = _unique_output_prefix(config.output_dir, image_path.stem)
    cache_dir = config.output_dir / "z_projection_cache"

    read_result = read_vsi_zstack(
        image_path,
        scene=config.input.scene,
        channel=config.input.channel,
        time=config.input.time,
        source_position_is_center=config.input.metadata_position == "center",
        fallback_pixel_size_x_um=config.input.overview_pixel_size_x_um,
        fallback_pixel_size_y_um=config.input.overview_pixel_size_y_um,
        fallback_z_step_um=config.input.fallback_z_step_um,
        projection_pixel_size_um=(
            config.input.processing_pixel_size_um
            if config.input.build_z_cache_at_processing_pixel_size
            else None
        ),
        cache_dir=cache_dir,
        use_cache=config.input.use_z_projection_cache,
        write_cache=config.input.write_z_projection_cache,
    )

    if config.input.build_z_cache_at_processing_pixel_size:
        processing_image = np.asarray(read_result.image)
        processing_metadata = read_result.metadata
    else:
        processing_image, processing_metadata = resample_to_pixel_size(
            read_result.image,
            read_result.metadata,
            config.input.processing_pixel_size_um,
        )

    transform = AffinePixelToStage.from_overview_metadata(
        processing_metadata,
        invert_x=config.input.invert_x,
        invert_y=config.input.invert_y,
    )
    review_argmax_z = (
        None
        if read_result.argmax_z is None
        else resample_argmax_z_to_shape(read_result.argmax_z, processing_image.shape)
    )
    review_options = ReviewPlanningOptions(
        planning_mode=config.planning.planning_mode,
        transform=transform,
        z_um=processing_metadata.stage_z_um,
        tile=config.planning.target_tile,
        merge_distance_factor=config.planning.group_merge_distance_factor,
        coverage_margin_um=config.planning.coverage_margin_um,
        polygon_simplify_tolerance_um=(
            config.planning.polygon_simplify_tolerance_um
        ),
        argmax_z=review_argmax_z,
        z_positions_um=read_result.z_positions_um,
        z_estimation_labels=config.planning.z_estimation_labels,
        max_group_z_difference_um=config.planning.max_group_z_difference_um,
        max_extra_tile_fraction=config.planning.max_extra_tile_fraction,
        name_prefix=config.planning.planned_name_prefix,
    )

    selection_callback = None
    preview_callback = None
    if config.review.enabled:

        def selection_callback(
            measurements,
            count,
            candidate_mask,
            component_labels,
            labels,
        ):
            return select_measurements_minimizing_tiles(
                measurements,
                count,
                candidate_mask,
                component_labels,
                labels,
                options=review_options,
            )

        if config.review.show_tiling_preview:

            def preview_callback(mask, measurements, labels):
                return build_napari_tiling_preview(
                    mask,
                    measurements,
                    labels,
                    options=review_options,
                )

    detection = detector.detect(
        processing_image,
        DetectionContext(
            image_path=image_path,
            output_prefix=output_prefix,
            metadata=processing_metadata,
            pixel_size_x_um=processing_metadata.pixel_size_x_um,
            pixel_size_y_um=processing_metadata.pixel_size_y_um,
            interactive=config.review.enabled,
            selection_callback=selection_callback,
            preview_callback=preview_callback,
            argmax_z=review_argmax_z,
            z_positions_um=read_result.z_positions_um,
        ),
    )

    planned_positions, rectangular_regions, polygon_regions, _group_count = (
        plan_targets_from_detection(
            detection=detection,
            transform=transform,
            metadata=processing_metadata,
            planning=config.planning,
            argmax_z=read_result.argmax_z,
            z_positions_um=read_result.z_positions_um,
        )
    )

    z_histograms_path = output_prefix.with_name(
        f"{output_prefix.name}_tile_scan_z_histograms.csv"
    )
    wrote_z_histograms = False
    if (
        config.planning.estimate_tile_scan_z
        and config.planning.planning_mode
        not in ("z_aware_irregular_mosaics", "optimized_z_aware_irregular_mosaics")
        and (polygon_regions or rectangular_regions)
    ):
        if read_result.argmax_z is None or read_result.z_positions_um is None:
            raise ValueError("Tile-scan Z estimation requires argmax-Z metadata")
        classifier_argmax_z = resample_argmax_z_to_shape(
            read_result.argmax_z,
            detection.planning_labels.shape,
        )
        polygon_regions, rectangular_regions, z_estimates = (
            update_tile_scan_region_z_positions(
                polygon_regions=polygon_regions,
                rectangular_regions=rectangular_regions,
                classifier_labels=detection.planning_labels,
                argmax_z=classifier_argmax_z,
                z_positions_um=read_result.z_positions_um,
                transform=transform,
                include_labels=config.planning.z_estimation_labels,
            )
        )
        if config.planning.write_tile_scan_z_histograms:
            write_region_z_histograms_csv(
                z_histograms_path,
                z_estimates,
                read_result.z_positions_um,
            )
            wrote_z_histograms = True

    mask_path = output_prefix.with_name(f"{output_prefix.name}_mask.tif")
    labels_path = output_prefix.with_name(f"{output_prefix.name}_raw_labels.tif")
    probability_path = output_prefix.with_name(
        f"{output_prefix.name}_positive_probability.tif"
    )
    positions_path = output_prefix.with_name(
        f"{output_prefix.name}_detected_positions.csv"
    )
    planned_positions_path = output_prefix.with_name(
        f"{output_prefix.name}_planned.csv"
    )
    rectangular_regions_path = output_prefix.with_name(
        f"{output_prefix.name}_rectangular_tilescans.csv"
    )
    polygon_regions_path = output_prefix.with_name(
        f"{output_prefix.name}_polygon_mosaics.csv"
    )
    region_confidence_path = output_prefix.with_name(
        f"{output_prefix.name}_region_confidence.csv"
    )
    plan_path = output_prefix.with_name(f"{output_prefix.name}_target_plan.json")

    positions = positions_from_measurements(
        detection.measurements,
        transform=transform,
        z_um=processing_metadata.stage_z_um,
        name_prefix=config.planning.detected_name_prefix,
    )
    write_mask(mask_path, detection.mask)
    write_labels_if_requested(labels_path, detection.labels)
    write_probability_if_requested(probability_path, detection.positive_probability)
    write_measurements_csv(positions_path, detection.measurements, positions)
    write_stage_positions_csv(planned_positions_path, planned_positions)
    write_rectangular_regions_csv(rectangular_regions_path, rectangular_regions)
    write_polygon_regions_csv(polygon_regions_path, polygon_regions)
    write_region_confidences_csv(
        region_confidence_path,
        positive_probability=detection.positive_probability,
        detection_mask=detection.mask,
        transform=transform,
        rectangular_regions=rectangular_regions,
        polygon_regions=polygon_regions,
    )

    plan = TargetPlan(
        source_image=str(image_path),
        planning_mode=config.planning.planning_mode,
        point_positions=planned_positions,
        rectangular_regions=rectangular_regions,
        polygon_regions=polygon_regions,
        overview_metadata=processing_metadata,
        selected_object_count=len(detection.measurements),
    )
    save_target_plan(plan_path, plan)
    return ProcessedTargetPlan(
        plan=plan,
        plan_path=plan_path,
        mask_path=mask_path,
        labels_path=labels_path,
        probability_path=probability_path,
        positions_path=positions_path,
        planned_positions_path=planned_positions_path,
        rectangular_regions_path=rectangular_regions_path,
        polygon_regions_path=polygon_regions_path,
        region_confidence_path=region_confidence_path,
        z_histograms_path=z_histograms_path if wrote_z_histograms else None,
    )


def process_vsi_files_to_cellsens_xml(
    image_paths: Iterable[str | Path],
    detector: ObjectDetector,
    config: WorkflowConfig,
    *,
    template_xml: str | Path,
    output_xml: str | Path,
) -> list[ProcessedTargetPlan]:
    """Process multiple VSI files and write one combined CellSens XML."""

    processed = [
        process_vsi_to_target_plan(image_path, detector, config)
        for image_path in image_paths
    ]
    write_combined_cellsens_xml(
        template_xml=template_xml,
        output_xml=output_xml,
        plans=[item.plan for item in processed],
    )
    return processed


def plan_targets_from_detection(
    *,
    detection: DetectionResult,
    transform: AffinePixelToStage,
    metadata: OverviewImageMetadata,
    planning: TargetPlanningConfig,
    argmax_z: np.ndarray | None,
    z_positions_um: tuple[float, ...] | None,
) -> tuple[
    list[StagePosition],
    list[RectangularRegion],
    list[PolygonRegion],
    list,
]:
    """Convert selected detections into final planned target objects."""

    positions = positions_from_measurements(
        detection.measurements,
        transform=transform,
        z_um=metadata.stage_z_um,
        name_prefix=planning.detected_name_prefix,
    )
    if planning.planning_mode == "mixed_tilescans":
        mixed_plan = plan_mixed_positions_and_tilescans(
            positions,
            tile=planning.target_tile,
            merge_distance_factor=planning.group_merge_distance_factor,
            coverage_margin_um=planning.coverage_margin_um,
            name_prefix=planning.planned_name_prefix,
        )
        return mixed_plan.point_positions, mixed_plan.rectangular_regions, [], mixed_plan.groups

    if planning.planning_mode == "mixed_irregular_mosaics":
        polygon_plan = plan_mixed_positions_and_polygon_regions_from_mask(
            detection.mask,
            transform=transform,
            z_um=metadata.stage_z_um,
            tile=planning.target_tile,
            min_area_px=1,
            merge_distance_factor=planning.group_merge_distance_factor,
            simplify_tolerance_um=planning.polygon_simplify_tolerance_um,
            name_prefix=planning.planned_name_prefix,
        )
        return polygon_plan.point_positions, [], polygon_plan.polygon_regions, []

    if planning.planning_mode in (
        "z_aware_irregular_mosaics",
        "optimized_z_aware_irregular_mosaics",
        "z_aware_tiles",
    ):
        if argmax_z is None or z_positions_um is None:
            raise ValueError(f"{planning.planning_mode} requires argmax-Z metadata")
        classifier_argmax_z = resample_argmax_z_to_shape(
            argmax_z,
            detection.planning_labels.shape,
        )
        if planning.planning_mode == "z_aware_irregular_mosaics":
            z_plan = plan_z_aware_positions_and_polygon_regions_from_mask(
                detection.mask,
                classifier_labels=detection.planning_labels,
                argmax_z=classifier_argmax_z,
                z_positions_um=z_positions_um,
                transform=transform,
                fallback_z_um=metadata.stage_z_um,
                tile=planning.target_tile,
                include_labels=planning.z_estimation_labels,
                min_area_px=1,
                merge_distance_factor=planning.group_merge_distance_factor,
                max_group_z_difference_um=planning.max_group_z_difference_um,
                simplify_tolerance_um=planning.polygon_simplify_tolerance_um,
                name_prefix=planning.planned_name_prefix,
            )
            return z_plan.point_positions, [], z_plan.polygon_regions, z_plan.groups

        if planning.planning_mode == "optimized_z_aware_irregular_mosaics":
            z_plan = plan_optimized_z_aware_positions_and_polygon_regions_from_mask(
                detection.mask,
                classifier_labels=detection.planning_labels,
                argmax_z=classifier_argmax_z,
                z_positions_um=z_positions_um,
                transform=transform,
                fallback_z_um=metadata.stage_z_um,
                tile=planning.target_tile,
                include_labels=planning.z_estimation_labels,
                min_area_px=1,
                merge_distance_factor=planning.group_merge_distance_factor,
                max_group_z_difference_um=planning.max_group_z_difference_um,
                max_extra_tile_fraction=planning.max_extra_tile_fraction,
                simplify_tolerance_um=planning.polygon_simplify_tolerance_um,
                name_prefix=planning.planned_name_prefix,
            )
            return z_plan.point_positions, [], z_plan.polygon_regions, z_plan.groups

        z_tile_plan = plan_z_aware_tile_positions_from_mask(
            detection.mask,
            classifier_labels=detection.planning_labels,
            argmax_z=classifier_argmax_z,
            z_positions_um=z_positions_um,
            transform=transform,
            fallback_z_um=metadata.stage_z_um,
            tile=planning.target_tile,
            include_labels=planning.z_estimation_labels,
            min_area_px=1,
            merge_distance_factor=planning.group_merge_distance_factor,
            max_group_z_difference_um=planning.max_group_z_difference_um,
            coverage_margin_um=planning.coverage_margin_um,
            name_prefix=planning.planned_name_prefix,
        )
        return z_tile_plan.point_positions, [], [], z_tile_plan.groups

    planned_positions, groups = plan_positions_for_high_mag_tiles(
        positions,
        tile=planning.target_tile,
        mode=planning.planning_mode,
        merge_distance_factor=planning.group_merge_distance_factor,
        coverage_margin_um=planning.coverage_margin_um,
        name_prefix=planning.planned_name_prefix,
    )
    return planned_positions, [], [], groups


def positions_from_measurements(
    measurements: tuple[ComponentMeasurement, ...],
    *,
    transform: AffinePixelToStage,
    z_um: float,
    name_prefix: str,
) -> list[StagePosition]:
    """Convert component centroids into stage positions."""

    positions: list[StagePosition] = []
    for index, measurement in enumerate(measurements, start=1):
        x_um, y_um = transform.apply(
            x_px=measurement.centroid_x_px,
            y_px=measurement.centroid_y_px,
        )
        positions.append(
            StagePosition(
                x_um=x_um,
                y_um=y_um,
                z_um=z_um,
                name=f"{name_prefix} {index}",
            )
        )
    return positions


def _unique_output_prefix(output_dir: Path, source_stem: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = "".join(
        char if char.isalnum() or char in ("-", "_") else "_"
        for char in source_stem
    )
    candidate = output_dir / safe_stem
    if not candidate.with_name(f"{candidate.name}_target_plan.json").exists():
        return candidate
    for index in range(2, 10_000):
        candidate = output_dir / f"{safe_stem}_{index}"
        if not candidate.with_name(f"{candidate.name}_target_plan.json").exists():
            return candidate
    raise RuntimeError(f"Could not create a unique output prefix for {source_stem}")
