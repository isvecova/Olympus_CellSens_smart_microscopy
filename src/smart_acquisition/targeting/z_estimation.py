"""Estimate one Z position per planned tile-scan region."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import csv

import numpy as np

from smart_acquisition.models import PolygonRegion, RectangularRegion
from smart_acquisition.targeting.coordinate_transform import AffinePixelToStage


@dataclass(frozen=True)
class RegionZEstimate:
    """Z histogram and selected Z plane for one planned tile-scan region."""

    region_type: str
    region_index: int
    region_name: str
    selected_z_index: int | None
    selected_z_um: float | None
    pixel_count: int
    histogram: tuple[int, ...]


def resample_argmax_z_to_shape(
    argmax_z: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Nearest-neighbor resample of an argmax-Z label image."""

    argmax_array = np.asarray(argmax_z)
    if argmax_array.shape == output_shape:
        return argmax_array

    from skimage.transform import resize

    return resize(
        argmax_array,
        output_shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(argmax_array.dtype, copy=False)


def update_tile_scan_region_z_positions(
    *,
    polygon_regions: list[PolygonRegion],
    rectangular_regions: list[RectangularRegion],
    classifier_labels: np.ndarray,
    argmax_z: np.ndarray,
    z_positions_um: tuple[float, ...],
    transform: AffinePixelToStage,
    include_labels: tuple[int, ...] = (1, 2),
) -> tuple[list[PolygonRegion], list[RectangularRegion], list[RegionZEstimate]]:
    """Return tile-scan regions with Z estimated from label 1/2 argmax-Z pixels."""

    labels = np.asarray(classifier_labels)
    argmax = np.asarray(argmax_z)
    if labels.shape != argmax.shape:
        raise ValueError(
            f"classifier_labels and argmax_z must have same shape, got "
            f"{labels.shape} and {argmax.shape}"
        )
    if not z_positions_um:
        raise ValueError("z_positions_um must contain at least one Z position")

    valid_label_mask = np.isin(labels, include_labels)
    updated_polygons: list[PolygonRegion] = []
    updated_rectangles: list[RectangularRegion] = []
    estimates: list[RegionZEstimate] = []

    for index, region in enumerate(polygon_regions, start=1):
        mask = _polygon_region_mask(region, labels.shape, transform)
        estimate = _estimate_region_z(
            region_type="polygon",
            region_index=index,
            region_name=region.name or f"polygon {index}",
            region_mask=mask,
            valid_label_mask=valid_label_mask,
            argmax_z=argmax,
            z_positions_um=z_positions_um,
        )
        estimates.append(estimate)
        updated_polygons.append(
            replace(region, z_um=estimate.selected_z_um)
            if estimate.selected_z_um is not None
            else region
        )

    for index, region in enumerate(rectangular_regions, start=1):
        mask = _rectangular_region_mask(region, labels.shape, transform)
        estimate = _estimate_region_z(
            region_type="rectangle",
            region_index=index,
            region_name=region.name or f"rectangle {index}",
            region_mask=mask,
            valid_label_mask=valid_label_mask,
            argmax_z=argmax,
            z_positions_um=z_positions_um,
        )
        estimates.append(estimate)
        updated_rectangles.append(
            replace(region, z_um=estimate.selected_z_um)
            if estimate.selected_z_um is not None
            else region
        )

    return updated_polygons, updated_rectangles, estimates


def write_region_z_histograms_csv(
    path: str | Path,
    estimates: list[RegionZEstimate],
    z_positions_um: tuple[float, ...],
) -> None:
    """Write one QC CSV row per region and Z plane."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "region_type",
                "region_index",
                "region_name",
                "selected_z_index",
                "selected_z_um",
                "pixel_count",
                "z_index",
                "z_um",
                "count",
            ]
        )
        for estimate in estimates:
            for z_index, count in enumerate(estimate.histogram):
                z_um = z_positions_um[z_index] if z_index < len(z_positions_um) else ""
                writer.writerow(
                    [
                        estimate.region_type,
                        estimate.region_index,
                        estimate.region_name,
                        estimate.selected_z_index,
                        estimate.selected_z_um,
                        estimate.pixel_count,
                        z_index,
                        z_um,
                        count,
                    ]
                )


def _estimate_region_z(
    *,
    region_type: str,
    region_index: int,
    region_name: str,
    region_mask: np.ndarray,
    valid_label_mask: np.ndarray,
    argmax_z: np.ndarray,
    z_positions_um: tuple[float, ...],
) -> RegionZEstimate:
    valid_region_mask = region_mask & valid_label_mask
    z_values = np.asarray(argmax_z[valid_region_mask], dtype=np.int64)
    z_values = z_values[(z_values >= 0) & (z_values < len(z_positions_um))]
    histogram = np.bincount(z_values, minlength=len(z_positions_um))
    pixel_count = int(z_values.size)
    if pixel_count == 0:
        return RegionZEstimate(
            region_type=region_type,
            region_index=region_index,
            region_name=region_name,
            selected_z_index=None,
            selected_z_um=None,
            pixel_count=0,
            histogram=tuple(int(value) for value in histogram),
        )

    selected_z_index = int(np.argmax(histogram))
    return RegionZEstimate(
        region_type=region_type,
        region_index=region_index,
        region_name=region_name,
        selected_z_index=selected_z_index,
        selected_z_um=float(z_positions_um[selected_z_index]),
        pixel_count=pixel_count,
        histogram=tuple(int(value) for value in histogram),
    )


def _polygon_region_mask(
    region: PolygonRegion,
    shape_yx: tuple[int, int],
    transform: AffinePixelToStage,
) -> np.ndarray:
    from skimage.draw import polygon

    vertices_px = [
        transform.apply_inverse(x_um=x_um, y_um=y_um)
        for x_um, y_um in region.vertices_xy_um
    ]
    x_values = np.asarray([vertex[0] for vertex in vertices_px], dtype=np.float32)
    y_values = np.asarray([vertex[1] for vertex in vertices_px], dtype=np.float32)
    rr, cc = polygon(y_values, x_values, shape=shape_yx)
    mask = np.zeros(shape_yx, dtype=bool)
    mask[rr, cc] = True
    return mask


def _rectangular_region_mask(
    region: RectangularRegion,
    shape_yx: tuple[int, int],
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
    polygon_region = PolygonRegion(vertices_xy_um=vertices, z_um=region.z_um)
    return _polygon_region_mask(polygon_region, shape_yx, transform)
