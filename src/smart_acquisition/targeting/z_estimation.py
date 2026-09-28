"""Estimate one Z position per planned tile-scan region."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import csv

import numpy as np

from smart_acquisition.models import AcquisitionTile, PolygonRegion, RectangularRegion
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


@dataclass(frozen=True)
class _RegionBounds:
    min_x_um: float
    max_x_um: float
    min_y_um: float
    max_y_um: float

    @property
    def center_x_um(self) -> float:
        return (self.min_x_um + self.max_x_um) / 2.0

    @property
    def center_y_um(self) -> float:
        return (self.min_y_um + self.max_y_um) / 2.0

    @property
    def width_um(self) -> float:
        return max(0.0, self.max_x_um - self.min_x_um)

    @property
    def height_um(self) -> float:
        return max(0.0, self.max_y_um - self.min_y_um)


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
    tile: AcquisitionTile | None = None,
    include_labels: tuple[int, ...] = (1, 2),
) -> tuple[list[PolygonRegion], list[RectangularRegion], list[RegionZEstimate]]:
    """Return regions with Z estimated from valid pixels in final scan coverage."""

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
        if tile is not None:
            mask = _tile_scan_coverage_mask(mask, transform=transform, tile=tile)
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
        if tile is not None:
            mask = _tile_scan_coverage_mask(mask, transform=transform, tile=tile)
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
    """Select the modal argmax-Z plane within one planned ROI."""

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
    """Rasterize a polygon ROI into the classifier image grid."""

    vertices_px = [
        transform.apply_inverse(x_um=x_um, y_um=y_um)
        for x_um, y_um in region.vertices_xy_um
    ]
    mask = np.zeros(shape_yx, dtype=bool)
    if len(vertices_px) < 3:
        return mask

    x_values = np.asarray([vertex[0] for vertex in vertices_px], dtype=np.float64)
    y_values = np.asarray([vertex[1] for vertex in vertices_px], dtype=np.float64)
    min_x = max(0, int(np.floor(float(np.min(x_values)))))
    max_x = min(shape_yx[1], int(np.ceil(float(np.max(x_values)))) + 1)
    min_y = max(0, int(np.floor(float(np.min(y_values)))))
    max_y = min(shape_yx[0], int(np.ceil(float(np.max(y_values)))) + 1)
    if min_x >= max_x or min_y >= max_y:
        return mask

    yy, xx = np.mgrid[min_y:max_y, min_x:max_x]
    mask[min_y:max_y, min_x:max_x] = _points_in_polygon(
        xx.astype(np.float64, copy=False),
        yy.astype(np.float64, copy=False),
        x_values,
        y_values,
    )
    return mask


def _points_in_polygon(
    x_points: np.ndarray,
    y_points: np.ndarray,
    polygon_x: np.ndarray,
    polygon_y: np.ndarray,
) -> np.ndarray:
    """Return a mask of points inside or on a polygon boundary."""

    inside = np.zeros(x_points.shape, dtype=bool)
    on_boundary = np.zeros(x_points.shape, dtype=bool)
    previous_index = len(polygon_x) - 1
    eps = 1e-9

    for index in range(len(polygon_x)):
        x0 = polygon_x[previous_index]
        y0 = polygon_y[previous_index]
        x1 = polygon_x[index]
        y1 = polygon_y[index]

        crosses = (y0 > y_points) != (y1 > y_points)
        with np.errstate(divide="ignore", invalid="ignore"):
            intersection_x = (x1 - x0) * (y_points - y0) / (y1 - y0) + x0
        inside ^= crosses & (x_points < intersection_x)

        cross_product = (x_points - x0) * (y1 - y0) - (y_points - y0) * (x1 - x0)
        within_x = (np.minimum(x0, x1) - eps <= x_points) & (
            x_points <= np.maximum(x0, x1) + eps
        )
        within_y = (np.minimum(y0, y1) - eps <= y_points) & (
            y_points <= np.maximum(y0, y1) + eps
        )
        on_boundary |= (np.abs(cross_product) <= eps) & within_x & within_y
        previous_index = index

    return inside | on_boundary


def _tile_scan_coverage_mask(
    region_mask: np.ndarray,
    *,
    transform: AffinePixelToStage,
    tile: AcquisitionTile,
) -> np.ndarray:
    """Return the union of high-mag tile footprints covering a region mask."""

    mask_array = np.asarray(region_mask, dtype=bool)
    if not np.any(mask_array):
        return mask_array

    bounds = _mask_stage_bounds(mask_array, transform)
    count_x = _tile_count_for_span(bounds.width_um, tile.width_um, tile.step_x_um)
    count_y = _tile_count_for_span(bounds.height_um, tile.height_um, tile.step_y_um)
    span_x = max(0.0, (count_x - 1) * tile.step_x_um)
    span_y = max(0.0, (count_y - 1) * tile.step_y_um)
    start_x = bounds.center_x_um - span_x / 2.0
    start_y = bounds.center_y_um - span_y / 2.0

    coverage = np.zeros_like(mask_array, dtype=bool)
    for y_index in range(count_y):
        for x_index in range(count_x):
            tile_slice = _tile_slices(
                mask_array.shape,
                transform=transform,
                center_x_um=start_x + x_index * tile.step_x_um,
                center_y_um=start_y + y_index * tile.step_y_um,
                tile=tile,
            )
            if tile_slice is not None and np.any(mask_array[tile_slice]):
                coverage[tile_slice] = True
    return coverage


def _mask_stage_bounds(
    mask: np.ndarray,
    transform: AffinePixelToStage,
) -> _RegionBounds:
    """Return stage-space bounds of positive pixels in a mask."""

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Cannot create stage bounds for an empty mask")
    y_slice = slice(int(ys.min()), int(ys.max()) + 1)
    x_slice = slice(int(xs.min()), int(xs.max()) + 1)
    return _slice_stage_bounds((y_slice, x_slice), mask.shape, transform)


def _slice_stage_bounds(
    object_slice: tuple[slice, slice],
    shape_yx: tuple[int, int],
    transform: AffinePixelToStage,
) -> _RegionBounds:
    """Convert an image slice into stage-space bounds using corner pixels."""

    min_y_px = max(0.0, float(object_slice[0].start) - 0.5)
    max_y_px = min(float(shape_yx[0]), float(object_slice[0].stop) - 0.5)
    min_x_px = max(0.0, float(object_slice[1].start) - 0.5)
    max_x_px = min(float(shape_yx[1]), float(object_slice[1].stop) - 0.5)
    vertices = [
        transform.apply(min_x_px, min_y_px),
        transform.apply(max_x_px, min_y_px),
        transform.apply(max_x_px, max_y_px),
        transform.apply(min_x_px, max_y_px),
    ]
    xs = [vertex[0] for vertex in vertices]
    ys = [vertex[1] for vertex in vertices]
    return _RegionBounds(
        min_x_um=min(xs),
        max_x_um=max(xs),
        min_y_um=min(ys),
        max_y_um=max(ys),
    )


def _tile_slices(
    shape_yx: tuple[int, int],
    *,
    transform: AffinePixelToStage,
    center_x_um: float,
    center_y_um: float,
    tile: AcquisitionTile,
) -> tuple[slice, slice] | None:
    """Return the image slice covered by a stage-space tile footprint."""

    half_width_um = tile.width_um / 2.0
    half_height_um = tile.height_um / 2.0
    stage_vertices = [
        (center_x_um - half_width_um, center_y_um - half_height_um),
        (center_x_um + half_width_um, center_y_um - half_height_um),
        (center_x_um + half_width_um, center_y_um + half_height_um),
        (center_x_um - half_width_um, center_y_um + half_height_um),
    ]
    pixel_vertices = [
        transform.apply_inverse(x_um=x_um, y_um=y_um)
        for x_um, y_um in stage_vertices
    ]
    x_values = np.asarray([vertex[0] for vertex in pixel_vertices], dtype=np.float64)
    y_values = np.asarray([vertex[1] for vertex in pixel_vertices], dtype=np.float64)

    min_x = max(0, int(np.floor(float(np.min(x_values)))))
    max_x = min(shape_yx[1], int(np.ceil(float(np.max(x_values)))) + 1)
    min_y = max(0, int(np.floor(float(np.min(y_values)))))
    max_y = min(shape_yx[0], int(np.ceil(float(np.max(y_values)))) + 1)

    if min_x >= max_x or min_y >= max_y:
        return None
    return slice(min_y, max_y), slice(min_x, max_x)


def _tile_count_for_span(span_um: float, tile_size_um: float, step_um: float) -> int:
    """Return how many overlapping tiles cover one stage-space span."""

    remaining_after_first_tile = max(0.0, span_um - tile_size_um)
    return 1 + int(np.ceil(remaining_after_first_tile / step_um))


def _rectangular_region_mask(
    region: RectangularRegion,
    shape_yx: tuple[int, int],
    transform: AffinePixelToStage,
) -> np.ndarray:
    """Rasterize a rectangular ROI by converting it to polygon vertices."""

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
