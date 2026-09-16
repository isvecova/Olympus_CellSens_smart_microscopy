"""Example BioIO VSI random-forest workflow with optional napari filtering."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from smart_acquisition.cellsens.xml_writer import (
    DEFAULT_CELLSENS_TEMPLATE,
    CellSensStageNavigatorWriter,
)
from smart_acquisition.detection.normalized_RF_detector import (
    ComponentMeasurement,
    NormalizedRandomForestDetector,
)
from smart_acquisition.image_input.bioio_vsi import read_vsi_overview
from smart_acquisition.image_input.resampling import resample_to_pixel_size
from smart_acquisition.models import (
    AcquisitionTile,
    OverviewImageMetadata,
    PolygonRegion,
    RectangularRegion,
    StagePosition,
)
from smart_acquisition.targeting.coordinate_transform import (
    AffinePixelToStage,
    overview_stage_edge_positions,
)
from smart_acquisition.targeting.polygon_regions import (
    plan_mixed_positions_and_polygon_regions_from_mask,
)
from smart_acquisition.targeting.tile_planner import (
    plan_mixed_positions_and_tilescans,
    plan_positions_for_high_mag_tiles,
)


def run(args: Any) -> int:
    read_result = read_vsi_overview(
        args.image,
        scene=args.scene,
        channel=args.channel,
        z=args.z,
        time=args.time,
        z_projection=args.z_projection,
        source_position_is_center=args.metadata_position == "center",
        fallback_pixel_size_x_um=args.overview_pixel_size_x_um,
        fallback_pixel_size_y_um=args.overview_pixel_size_y_um,
    )
    classifier_image, classifier_metadata = resample_to_pixel_size(
        read_result.image,
        read_result.metadata,
        args.classification_pixel_size_um,
    )
    detector = NormalizedRandomForestDetector(
        args.model,
        lower_absolute=args.lower_absolute,
        ignore_zero_pixels=not args.include_zero_pixels,
        positive_label=args.positive_label,
        tile_size=args.tile_size,
        overlap=args.overlap,
    )
    detection = detector.detect(
        classifier_image,
        min_area_px=args.min_area_px,
        max_area_px=args.max_area_px,
        min_elongation=args.min_elongation,
        max_elongation=args.max_elongation,
        max_mean_3nn_distance_um=args.max_mean_3nn_distance_um,
        min_size_weighted_confidence=args.min_size_weighted_confidence,
        max_selected_objects=args.max_selected_objects,
        pixel_size_x_um=classifier_metadata.pixel_size_x_um,
        pixel_size_y_um=classifier_metadata.pixel_size_y_um,
        interactive=args.napari_filter,
    )

    transform = AffinePixelToStage.from_overview_metadata(
        classifier_metadata,
        invert_x=args.invert_x,
        invert_y=args.invert_y,
    )
    positions = _positions_from_measurements(
        detection.measurements,
        transform=transform,
        z_um=classifier_metadata.stage_z_um,
        name_prefix=args.name_prefix,
    )

    target_tile = AcquisitionTile(
        width_px=args.target_width_px,
        height_px=args.target_height_px,
        pixel_size_x_um=args.target_pixel_size_x_um,
        pixel_size_y_um=args.target_pixel_size_y_um,
        overlap_fraction=args.target_tile_overlap_fraction,
    )
    planned_positions, rectangular_regions, polygon_regions, group_count = _plan_targets(
        args,
        detection.mask,
        positions,
        transform,
        classifier_metadata.stage_z_um,
        target_tile,
    )

    _write_mask(args.mask_output, detection.mask)
    _write_labels_if_requested(args.labels_output, detection.labels)
    _write_probability_if_requested(args.probability_output, detection.positive_probability)
    _write_measurements_csv(args.positions_output, detection.measurements, positions)
    if args.planned_positions_output:
        _write_stage_positions_csv(args.planned_positions_output, planned_positions)
    if args.rectangular_regions_output:
        _write_rectangular_regions_csv(
            args.rectangular_regions_output,
            rectangular_regions,
        )
    if args.polygon_regions_output:
        _write_polygon_regions_csv(args.polygon_regions_output, polygon_regions)
    if args.region_confidence_output:
        _write_region_confidences_csv(
            args.region_confidence_output,
            positive_probability=detection.positive_probability,
            detection_mask=detection.mask,
            transform=transform,
            rectangular_regions=rectangular_regions,
            polygon_regions=polygon_regions,
        )

    if args.cellsens_output or args.cellsens_template or args.use_default_cellsens_template:
        _write_cellsens_xml(
            args,
            read_result.metadata,
            planned_positions,
            rectangular_regions,
            polygon_regions,
        )

    print(f"Read image: {args.image}")
    print(f"Image shape YX: {tuple(np.asarray(read_result.image).shape)}")
    print(f"Classifier image shape YX: {tuple(np.asarray(classifier_image).shape)}")
    print(
        "Classifier pixel size: "
        f"{classifier_metadata.pixel_size_x_um:.6g} x "
        f"{classifier_metadata.pixel_size_y_um:.6g} um"
    )
    print(f"Model: {args.model}")
    print(f"Kept RF components: {len(detection.measurements)}")
    print(f"Nearby groups: {group_count}")
    print(f"Planned cellSens positions: {len(planned_positions)}")
    print(f"Planned rectangular tile scans: {len(rectangular_regions)}")
    print(f"Planned irregular mosaic tile scans: {len(polygon_regions)}")
    print(f"Mask output: {args.mask_output}")
    if args.probability_output:
        print(f"Positive probability output: {args.probability_output}")
    print(f"Positions output: {args.positions_output}")
    if args.region_confidence_output:
        print(f"Region confidence output: {args.region_confidence_output}")
    if args.cellsens_output:
        print(f"cellSens XML output: {args.cellsens_output}")
    return 0


def _write_cellsens_xml(
    args: Any,
    metadata: OverviewImageMetadata,
    planned_positions: list[StagePosition],
    rectangular_regions: list[RectangularRegion],
    polygon_regions: list[PolygonRegion],
) -> None:
    if not args.cellsens_output:
        raise ValueError("--cellsens-output is required when writing cellSens XML")

    if args.use_default_cellsens_template:
        template = Path(args.default_cellsens_template)
    else:
        if not args.cellsens_template:
            raise ValueError(
                "--cellsens-template is required unless "
                "--use-default-cellsens-template is set"
            )
        template = Path(args.cellsens_template)

    writer = CellSensStageNavigatorWriter(template)
    if args.use_default_cellsens_template:
        writer.set_overview_stage_edges(
            **overview_stage_edge_positions(
                metadata,
                invert_x=args.invert_x,
                invert_y=args.invert_y,
            )
        )

    if rectangular_regions or polygon_regions:
        writer.replace_targets(
            positions=planned_positions,
            rectangular_regions=rectangular_regions,
            polygon_regions=polygon_regions,
        )
    else:
        writer.add_positions(planned_positions)
    writer.save(args.cellsens_output)


def _plan_targets(
    args: Any,
    mask: np.ndarray,
    positions: list[StagePosition],
    transform: AffinePixelToStage,
    z_um: float,
    target_tile: AcquisitionTile,
) -> tuple[list[StagePosition], list[RectangularRegion], list[PolygonRegion], int]:
    if args.planning_mode == "raw":
        return positions, [], [], len(positions)

    if args.planning_mode == "mixed_tilescans":
        mixed_plan = plan_mixed_positions_and_tilescans(
            positions,
            tile=target_tile,
            merge_distance_factor=args.group_merge_distance_factor,
            coverage_margin_um=args.coverage_margin_um,
            name_prefix="Planned",
        )
        return (
            mixed_plan.point_positions,
            mixed_plan.rectangular_regions,
            [],
            len(mixed_plan.groups),
        )

    if args.planning_mode == "mixed_irregular_mosaics":
        polygon_plan = plan_mixed_positions_and_polygon_regions_from_mask(
            mask,
            transform=transform,
            z_um=z_um,
            tile=target_tile,
            min_area_px=1,
            merge_distance_factor=args.group_merge_distance_factor,
            simplify_tolerance_um=args.polygon_simplify_tolerance_um,
            name_prefix="Planned",
        )
        return (
            polygon_plan.point_positions,
            [],
            polygon_plan.polygon_regions,
            len(polygon_plan.point_positions) + len(polygon_plan.polygon_regions),
        )

    planned_positions, groups = plan_positions_for_high_mag_tiles(
        positions,
        tile=target_tile,
        mode=args.planning_mode,
        merge_distance_factor=args.group_merge_distance_factor,
        coverage_margin_um=args.coverage_margin_um,
        name_prefix="Planned",
    )
    return planned_positions, [], [], len(groups)


def _positions_from_measurements(
    measurements: tuple[ComponentMeasurement, ...],
    transform: AffinePixelToStage,
    z_um: float,
    name_prefix: str,
) -> list[StagePosition]:
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


def _write_mask(path: str | Path, mask: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.asarray(mask, dtype=np.uint8) * 255)


def _write_labels_if_requested(path: str | Path | None, labels: np.ndarray) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.asarray(labels))


def _write_probability_if_requested(
    path: str | Path | None,
    positive_probability: np.ndarray,
) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.asarray(positive_probability, dtype=np.float32))


def _write_measurements_csv(
    path: str | Path,
    measurements: tuple[ComponentMeasurement, ...],
    positions: list[StagePosition],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "name",
                "x_um",
                "y_um",
                "z_um",
                "area_px",
                "elongation",
                "axis_ratio",
                "mean_3nn_distance_um",
                "centroid_x_px",
                "centroid_y_px",
                "mean_positive_probability",
                "min_positive_probability",
                "positive_probability_sum",
                "size_weighted_confidence",
                "source_label",
            ]
        )
        for measurement, position in zip(measurements, positions):
            writer.writerow(
                [
                    position.name,
                    repr(float(position.x_um)),
                    repr(float(position.y_um)),
                    repr(float(position.z_um)),
                    measurement.area_px,
                    repr(float(measurement.elongation)),
                    repr(float(measurement.axis_ratio)),
                    repr(float(measurement.mean_3nn_distance_um)),
                    repr(float(measurement.centroid_x_px)),
                    repr(float(measurement.centroid_y_px)),
                    repr(float(measurement.mean_positive_probability)),
                    repr(float(measurement.min_positive_probability)),
                    repr(float(measurement.positive_probability_sum)),
                    repr(float(measurement.size_weighted_confidence)),
                    measurement.label,
                ]
            )


def _write_stage_positions_csv(path: str | Path, positions: list[StagePosition]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "x_um", "y_um", "z_um"])
        for position in positions:
            writer.writerow(
                [
                    position.name,
                    repr(float(position.x_um)),
                    repr(float(position.y_um)),
                    repr(float(position.z_um)),
                ]
            )


def _write_rectangular_regions_csv(
    path: str | Path,
    regions: list[RectangularRegion],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["name", "center_x_um", "center_y_um", "z_um", "width_um", "height_um"]
        )
        for region in regions:
            writer.writerow(
                [
                    region.name,
                    repr(float(region.center_x_um)),
                    repr(float(region.center_y_um)),
                    repr(float(region.z_um)),
                    repr(float(region.width_um)),
                    repr(float(region.height_um)),
                ]
            )


def _write_polygon_regions_csv(path: str | Path, regions: list[PolygonRegion]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "name",
                "z_um",
                "vertex_count",
                "min_x_um",
                "max_x_um",
                "min_y_um",
                "max_y_um",
            ]
        )
        for region in regions:
            xs = [vertex[0] for vertex in region.vertices_xy_um]
            ys = [vertex[1] for vertex in region.vertices_xy_um]
            writer.writerow(
                [
                    region.name,
                    repr(float(region.z_um)),
                    len(region.vertices_xy_um),
                    repr(float(min(xs))),
                    repr(float(max(xs))),
                    repr(float(min(ys))),
                    repr(float(max(ys))),
                ]
            )


def _write_region_confidences_csv(
    path: str | Path,
    *,
    positive_probability: np.ndarray,
    detection_mask: np.ndarray,
    transform: AffinePixelToStage,
    rectangular_regions: list[RectangularRegion],
    polygon_regions: list[PolygonRegion],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    positive_probability = np.asarray(positive_probability, dtype=np.float32)
    detection_mask = np.asarray(detection_mask, dtype=bool)
    if positive_probability.shape != detection_mask.shape:
        raise ValueError(
            f"positive_probability shape {positive_probability.shape} does not "
            f"match detection_mask shape {detection_mask.shape}"
        )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "target_type",
                "name",
                "roi_area_px",
                "detected_area_px",
                "detected_fraction",
                "mean_positive_probability",
                "min_positive_probability",
                "positive_probability_sum",
                "size_weighted_confidence",
            ]
        )
        for region in rectangular_regions:
            roi_mask = _rectangular_region_mask(positive_probability.shape, region, transform)
            writer.writerow(
                _region_confidence_row(
                    "rectangle",
                    region.name,
                    roi_mask,
                    detection_mask,
                    positive_probability,
                )
            )
        for region in polygon_regions:
            roi_mask = _polygon_region_mask(positive_probability.shape, region, transform)
            writer.writerow(
                _region_confidence_row(
                    "polygon",
                    region.name,
                    roi_mask,
                    detection_mask,
                    positive_probability,
                )
            )


def _region_confidence_row(
    target_type: str,
    name: str | None,
    roi_mask: np.ndarray,
    detection_mask: np.ndarray,
    positive_probability: np.ndarray,
) -> list[str | int | float | None]:
    roi_area_px = int(np.count_nonzero(roi_mask))
    support_mask = roi_mask & detection_mask & np.isfinite(positive_probability)
    detected_area_px = int(np.count_nonzero(support_mask))
    detected_fraction = detected_area_px / roi_area_px if roi_area_px else 0.0
    if detected_area_px == 0:
        nan = float("nan")
        return [
            target_type,
            name,
            roi_area_px,
            detected_area_px,
            repr(float(detected_fraction)),
            repr(nan),
            repr(nan),
            repr(0.0),
            repr(nan),
        ]

    values = positive_probability[support_mask]
    mean_probability = float(np.mean(values))
    min_probability = float(np.min(values))
    probability_sum = float(np.sum(values))
    size_weighted = mean_probability * float(np.sqrt(detected_area_px))
    return [
        target_type,
        name,
        roi_area_px,
        detected_area_px,
        repr(float(detected_fraction)),
        repr(mean_probability),
        repr(min_probability),
        repr(probability_sum),
        repr(size_weighted),
    ]


def _rectangular_region_mask(
    shape: tuple[int, int],
    region: RectangularRegion,
    transform: AffinePixelToStage,
) -> np.ndarray:
    half_width = region.width_um / 2.0
    half_height = region.height_um / 2.0
    vertices = [
        (region.center_x_um - half_width, region.center_y_um - half_height),
        (region.center_x_um + half_width, region.center_y_um - half_height),
        (region.center_x_um + half_width, region.center_y_um + half_height),
        (region.center_x_um - half_width, region.center_y_um + half_height),
    ]
    return _vertices_mask(shape, vertices, transform)


def _polygon_region_mask(
    shape: tuple[int, int],
    region: PolygonRegion,
    transform: AffinePixelToStage,
) -> np.ndarray:
    return _vertices_mask(shape, region.vertices_xy_um, transform)


def _vertices_mask(
    shape: tuple[int, int],
    vertices_xy_um: list[tuple[float, float]],
    transform: AffinePixelToStage,
) -> np.ndarray:
    from skimage.draw import polygon

    vertices_px = [
        transform.apply_inverse(x_um=x_um, y_um=y_um) for x_um, y_um in vertices_xy_um
    ]
    x_values = np.asarray([vertex[0] for vertex in vertices_px], dtype=np.float32)
    y_values = np.asarray([vertex[1] for vertex in vertices_px], dtype=np.float32)
    rr, cc = polygon(y_values, x_values, shape=shape)
    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True
    return mask


def _normalize_scene(scene: str) -> int | str:
    try:
        return int(scene)
    except ValueError:
        return scene
