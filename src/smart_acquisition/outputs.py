"""Output writers shared by workflows, GUIs, and scripts."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from smart_acquisition.detection.normalized_RF_detector import ComponentMeasurement
from smart_acquisition.models import PolygonRegion, RectangularRegion, StagePosition
from smart_acquisition.targeting.coordinate_transform import AffinePixelToStage


def write_mask(path: str | Path, mask: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import tifffile

    tifffile.imwrite(path, np.asarray(mask, dtype=np.uint8) * 255)


def write_labels_if_requested(path: str | Path | None, labels: np.ndarray | None) -> None:
    if path is None or labels is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import tifffile

    tifffile.imwrite(path, np.asarray(labels))


def write_probability_if_requested(
    path: str | Path | None,
    positive_probability: np.ndarray | None,
) -> None:
    if path is None or positive_probability is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import tifffile

    tifffile.imwrite(path, np.asarray(positive_probability, dtype=np.float32))


def write_measurements_csv(
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


def write_stage_positions_csv(path: str | Path, positions: list[StagePosition]) -> None:
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


def write_rectangular_regions_csv(
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


def write_polygon_regions_csv(path: str | Path, regions: list[PolygonRegion]) -> None:
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


def write_region_confidences_csv(
    path: str | Path,
    *,
    positive_probability: np.ndarray | None,
    detection_mask: np.ndarray,
    transform: AffinePixelToStage,
    rectangular_regions: list[RectangularRegion],
    polygon_regions: list[PolygonRegion],
) -> None:
    if positive_probability is None:
        return

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

