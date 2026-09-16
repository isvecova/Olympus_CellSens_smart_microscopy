"""Interactive selection helpers for napari detection review."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np

from smart_acquisition.detection.normalized_RF_detector import (
    ComponentMeasurement,
    NapariTilingPreview,
)
from smart_acquisition.models import (
    AcquisitionTile,
    PolygonRegion,
    RectangularRegion,
    StagePosition,
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


@dataclass(frozen=True)
class ReviewPlanningOptions:
    """Planning settings needed during interactive napari review."""

    planning_mode: str
    transform: AffinePixelToStage
    z_um: float
    tile: AcquisitionTile
    merge_distance_factor: float = 1.0
    coverage_margin_um: float = 0.0
    polygon_simplify_tolerance_um: float = 10.0
    argmax_z: np.ndarray | None = None
    z_positions_um: tuple[float, ...] | None = None
    z_estimation_labels: tuple[int, ...] = (1, 2)
    max_group_z_difference_um: float = 10.0
    max_extra_tile_fraction: float = 0.25
    name_prefix: str = "Planned"


@dataclass(frozen=True)
class ReviewPlan:
    """Planned targets plus lightweight tile-count summary."""

    point_positions: list[StagePosition]
    rectangular_regions: list[RectangularRegion]
    polygon_regions: list[PolygonRegion]
    group_count: int
    tile_count: int


def plan_review_targets(
    mask: np.ndarray,
    measurements: tuple[ComponentMeasurement, ...],
    *,
    classifier_labels: np.ndarray,
    options: ReviewPlanningOptions,
) -> ReviewPlan:
    """Plan targets for the current interactive selection."""

    if not measurements:
        return ReviewPlan([], [], [], 0, 0)

    positions = _positions_from_measurements(
        measurements,
        transform=options.transform,
        z_um=options.z_um,
        name_prefix=options.name_prefix,
    )

    if options.planning_mode == "mixed_tilescans":
        mixed_plan = plan_mixed_positions_and_tilescans(
            positions,
            tile=options.tile,
            merge_distance_factor=options.merge_distance_factor,
            coverage_margin_um=options.coverage_margin_um,
            name_prefix=options.name_prefix,
        )
        return _make_review_plan(
            mixed_plan.point_positions,
            mixed_plan.rectangular_regions,
            [],
            len(mixed_plan.groups),
            options.tile,
        )

    if options.planning_mode == "mixed_irregular_mosaics":
        polygon_plan = plan_mixed_positions_and_polygon_regions_from_mask(
            mask,
            transform=options.transform,
            z_um=options.z_um,
            tile=options.tile,
            min_area_px=1,
            merge_distance_factor=options.merge_distance_factor,
            simplify_tolerance_um=options.polygon_simplify_tolerance_um,
            name_prefix=options.name_prefix,
        )
        return _make_review_plan(
            polygon_plan.point_positions,
            [],
            polygon_plan.polygon_regions,
            len(polygon_plan.point_positions) + len(polygon_plan.polygon_regions),
            options.tile,
        )

    if options.planning_mode == "z_aware_irregular_mosaics":
        z_plan = plan_z_aware_positions_and_polygon_regions_from_mask(
            mask,
            classifier_labels=classifier_labels,
            argmax_z=_require_argmax_z(options),
            z_positions_um=_require_z_positions(options),
            transform=options.transform,
            fallback_z_um=options.z_um,
            tile=options.tile,
            include_labels=options.z_estimation_labels,
            min_area_px=1,
            merge_distance_factor=options.merge_distance_factor,
            max_group_z_difference_um=options.max_group_z_difference_um,
            simplify_tolerance_um=options.polygon_simplify_tolerance_um,
            name_prefix=options.name_prefix,
        )
        return _make_review_plan(
            z_plan.point_positions,
            [],
            z_plan.polygon_regions,
            len(z_plan.groups),
            options.tile,
        )

    if options.planning_mode == "optimized_z_aware_irregular_mosaics":
        z_plan = plan_optimized_z_aware_positions_and_polygon_regions_from_mask(
            mask,
            classifier_labels=classifier_labels,
            argmax_z=_require_argmax_z(options),
            z_positions_um=_require_z_positions(options),
            transform=options.transform,
            fallback_z_um=options.z_um,
            tile=options.tile,
            include_labels=options.z_estimation_labels,
            min_area_px=1,
            merge_distance_factor=options.merge_distance_factor,
            max_group_z_difference_um=options.max_group_z_difference_um,
            max_extra_tile_fraction=options.max_extra_tile_fraction,
            simplify_tolerance_um=options.polygon_simplify_tolerance_um,
            name_prefix=options.name_prefix,
        )
        return _make_review_plan(
            z_plan.point_positions,
            [],
            z_plan.polygon_regions,
            len(z_plan.groups),
            options.tile,
        )

    if options.planning_mode == "z_aware_tiles":
        z_plan = plan_z_aware_tile_positions_from_mask(
            mask,
            classifier_labels=classifier_labels,
            argmax_z=_require_argmax_z(options),
            z_positions_um=_require_z_positions(options),
            transform=options.transform,
            fallback_z_um=options.z_um,
            tile=options.tile,
            include_labels=options.z_estimation_labels,
            min_area_px=1,
            merge_distance_factor=options.merge_distance_factor,
            max_group_z_difference_um=options.max_group_z_difference_um,
            coverage_margin_um=options.coverage_margin_um,
            name_prefix=options.name_prefix,
        )
        return _make_review_plan(z_plan.point_positions, [], [], len(z_plan.groups), options.tile)

    planned_positions, groups = plan_positions_for_high_mag_tiles(
        positions,
        tile=options.tile,
        mode=options.planning_mode,
        merge_distance_factor=options.merge_distance_factor,
        coverage_margin_um=options.coverage_margin_um,
        name_prefix=options.name_prefix,
    )
    return _make_review_plan(planned_positions, [], [], len(groups), options.tile)


def select_measurements_minimizing_tiles(
    measurements: tuple[ComponentMeasurement, ...],
    requested_count: int,
    candidate_mask: np.ndarray,
    component_labels: np.ndarray,
    classifier_labels: np.ndarray,
    *,
    options: ReviewPlanningOptions,
    max_seed_count: int = 24,
) -> tuple[ComponentMeasurement, ...]:
    """Select a compact subset, using tile count as the main objective.

    Exhaustively testing all combinations would explode quickly. This uses a
    bounded multi-start greedy search: each seed grows a compact neighborhood,
    then the full planned tile count is evaluated once for that candidate set.
    """

    if requested_count <= 0 or len(measurements) <= requested_count:
        return measurements

    candidates = tuple(measurements)
    scores = {
        measurement.label: _finite_score(measurement.size_weighted_confidence)
        for measurement in candidates
    }
    seed_measurements = tuple(
        sorted(candidates, key=lambda item: scores[item.label], reverse=True)[
            :max_seed_count
        ]
    )

    best_selection: tuple[ComponentMeasurement, ...] | None = None
    best_key: tuple[int, float, float] | None = None
    for seed in seed_measurements:
        selected = _grow_compact_selection(
            seed,
            candidates,
            requested_count,
            scores=scores,
            options=options,
        )
        selected_labels = [measurement.label for measurement in selected]
        selected_mask = candidate_mask & np.isin(component_labels, selected_labels)
        plan = plan_review_targets(
            selected_mask,
            selected,
            classifier_labels=classifier_labels,
            options=options,
        )
        total_score = sum(scores[measurement.label] for measurement in selected)
        mean_score = total_score / max(1, len(selected))
        key = (plan.tile_count, -total_score, -mean_score)
        if best_key is None or key < best_key:
            best_key = key
            best_selection = selected

    if best_selection is None:
        return tuple(candidates[:requested_count])
    return best_selection


def build_napari_tiling_preview(
    mask: np.ndarray,
    measurements: tuple[ComponentMeasurement, ...],
    classifier_labels: np.ndarray,
    *,
    options: ReviewPlanningOptions,
) -> NapariTilingPreview:
    """Create napari shape geometry for the current interactive plan."""

    plan = plan_review_targets(
        mask,
        measurements,
        classifier_labels=classifier_labels,
        options=options,
    )
    tile_shapes: list[np.ndarray] = []
    region_shapes: list[np.ndarray] = []
    point_centers: list[tuple[float, float]] = []

    for position in plan.point_positions:
        tile_shapes.append(
            _stage_vertices_to_yx(
                _tile_vertices(position.x_um, position.y_um, options.tile),
                options.transform,
            )
        )
        point_centers.append(
            _stage_point_to_yx(position.x_um, position.y_um, options.transform)
        )

    for region in plan.rectangular_regions:
        region_vertices = _rectangular_region_vertices(region)
        region_shapes.append(_stage_vertices_to_yx(region_vertices, options.transform))
        for x_um, y_um in _tile_centers_for_rectangle(region, options.tile):
            tile_shapes.append(
                _stage_vertices_to_yx(
                    _tile_vertices(x_um, y_um, options.tile),
                    options.transform,
                )
            )

    for region in plan.polygon_regions:
        region_shapes.append(
            _stage_vertices_to_yx(region.vertices_xy_um, options.transform)
        )
        for x_um, y_um in _tile_centers_for_polygon(region, options.tile):
            tile_shapes.append(
                _stage_vertices_to_yx(
                    _tile_vertices(x_um, y_um, options.tile),
                    options.transform,
                )
            )

    return NapariTilingPreview(
        tile_rectangles_yx=tuple(tile_shapes),
        region_outlines_yx=tuple(region_shapes),
        point_centers_yx=tuple(point_centers),
        tile_count=len(tile_shapes),
        target_count=(
            len(plan.point_positions)
            + len(plan.rectangular_regions)
            + len(plan.polygon_regions)
        ),
        group_count=plan.group_count,
    )


def _make_review_plan(
    point_positions: list[StagePosition],
    rectangular_regions: list[RectangularRegion],
    polygon_regions: list[PolygonRegion],
    group_count: int,
    tile: AcquisitionTile,
) -> ReviewPlan:
    tile_count = (
        len(point_positions)
        + sum(len(_tile_centers_for_rectangle(region, tile)) for region in rectangular_regions)
        + sum(len(_tile_centers_for_polygon(region, tile)) for region in polygon_regions)
    )
    return ReviewPlan(
        point_positions=point_positions,
        rectangular_regions=rectangular_regions,
        polygon_regions=polygon_regions,
        group_count=group_count,
        tile_count=tile_count,
    )


def _positions_from_measurements(
    measurements: tuple[ComponentMeasurement, ...],
    *,
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


def _grow_compact_selection(
    seed: ComponentMeasurement,
    candidates: tuple[ComponentMeasurement, ...],
    requested_count: int,
    *,
    scores: dict[int, float],
    options: ReviewPlanningOptions,
) -> tuple[ComponentMeasurement, ...]:
    selected = [seed]
    remaining = [candidate for candidate in candidates if candidate.label != seed.label]
    while len(selected) < requested_count and remaining:
        remaining.sort(
            key=lambda candidate: (
                _normalized_distance_to_selection(candidate, selected, options),
                -scores[candidate.label],
            )
        )
        selected.append(remaining.pop(0))
    return tuple(selected)


def _normalized_distance_to_selection(
    candidate: ComponentMeasurement,
    selected: list[ComponentMeasurement],
    options: ReviewPlanningOptions,
) -> float:
    candidate_x, candidate_y = _measurement_stage_xy(candidate, options.transform)
    selected_xy = [
        _measurement_stage_xy(measurement, options.transform) for measurement in selected
    ]
    return min(
        max(
            abs(candidate_x - selected_x) / max(options.tile.width_um, 1e-12),
            abs(candidate_y - selected_y) / max(options.tile.height_um, 1e-12),
        )
        for selected_x, selected_y in selected_xy
    )


def _measurement_stage_xy(
    measurement: ComponentMeasurement,
    transform: AffinePixelToStage,
) -> tuple[float, float]:
    return transform.apply(
        x_px=measurement.centroid_x_px,
        y_px=measurement.centroid_y_px,
    )


def _finite_score(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


def _require_argmax_z(options: ReviewPlanningOptions) -> np.ndarray:
    if options.argmax_z is None:
        raise ValueError("Z-aware napari preview requires argmax-Z data")
    return options.argmax_z


def _require_z_positions(options: ReviewPlanningOptions) -> tuple[float, ...]:
    if options.z_positions_um is None:
        raise ValueError("Z-aware napari preview requires z-position metadata")
    return options.z_positions_um


def _tile_centers_for_rectangle(
    region: RectangularRegion,
    tile: AcquisitionTile,
) -> list[tuple[float, float]]:
    count_x = _tile_count_for_span(region.width_um, tile.width_um, tile.step_x_um)
    count_y = _tile_count_for_span(region.height_um, tile.height_um, tile.step_y_um)
    span_x = max(0.0, (count_x - 1) * tile.step_x_um)
    span_y = max(0.0, (count_y - 1) * tile.step_y_um)
    start_x = region.center_x_um - span_x / 2.0
    start_y = region.center_y_um - span_y / 2.0
    return [
        (start_x + x_index * tile.step_x_um, start_y + y_index * tile.step_y_um)
        for y_index in range(count_y)
        for x_index in range(count_x)
    ]


def _tile_centers_for_polygon(
    region: PolygonRegion,
    tile: AcquisitionTile,
) -> list[tuple[float, float]]:
    vertices = np.asarray(region.vertices_xy_um, dtype=float)
    if vertices.ndim != 2 or vertices.shape[0] < 3 or vertices.shape[1] != 2:
        return []

    min_x = float(np.min(vertices[:, 0]))
    max_x = float(np.max(vertices[:, 0]))
    min_y = float(np.min(vertices[:, 1]))
    max_y = float(np.max(vertices[:, 1]))
    count_x = _tile_count_for_span(max_x - min_x, tile.width_um, tile.step_x_um)
    count_y = _tile_count_for_span(max_y - min_y, tile.height_um, tile.step_y_um)
    span_x = max(0.0, (count_x - 1) * tile.step_x_um)
    span_y = max(0.0, (count_y - 1) * tile.step_y_um)
    start_x = (min_x + max_x) / 2.0 - span_x / 2.0
    start_y = (min_y + max_y) / 2.0 - span_y / 2.0

    candidates = [
        (start_x + x_index * tile.step_x_um, start_y + y_index * tile.step_y_um)
        for y_index in range(count_y)
        for x_index in range(count_x)
    ]
    selected = [
        center
        for center in candidates
        if _tile_footprint_intersects_polygon(center, tile, vertices)
    ]
    if selected:
        return selected
    return [(float(np.mean(vertices[:, 0])), float(np.mean(vertices[:, 1])))]


def _tile_footprint_intersects_polygon(
    center_xy_um: tuple[float, float],
    tile: AcquisitionTile,
    vertices: np.ndarray,
) -> bool:
    center_x, center_y = center_xy_um
    left = center_x - tile.width_um / 2.0
    right = center_x + tile.width_um / 2.0
    bottom = center_y - tile.height_um / 2.0
    top = center_y + tile.height_um / 2.0
    rectangle = np.asarray(
        [[left, bottom], [right, bottom], [right, top], [left, top]],
        dtype=float,
    )
    if np.any(_points_in_polygon(rectangle, vertices)):
        return True
    if any(_point_in_rectangle(vertex, left, right, bottom, top) for vertex in vertices):
        return True
    rectangle_edges = tuple(_polygon_edges(rectangle))
    return any(
        _segments_intersect(poly_start, poly_end, rect_start, rect_end)
        for poly_start, poly_end in _polygon_edges(vertices)
        for rect_start, rect_end in rectangle_edges
    )


def _points_in_polygon(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    polygon_x = vertices[:, 0]
    polygon_y = vertices[:, 1]
    inside = np.zeros(points.shape[0], dtype=bool)
    previous = len(vertices) - 1
    for current in range(len(vertices)):
        yi = polygon_y[current]
        yj = polygon_y[previous]
        xi = polygon_x[current]
        xj = polygon_x[previous]
        crosses_y = (yi > y) != (yj > y)
        x_intersection = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        inside ^= crosses_y & (x < x_intersection)
        previous = current
    return inside


def _point_in_rectangle(
    point: np.ndarray,
    left: float,
    right: float,
    bottom: float,
    top: float,
) -> bool:
    eps = 1e-9
    x, y = point
    return left - eps <= x <= right + eps and bottom - eps <= y <= top + eps


def _polygon_edges(vertices: np.ndarray):
    for index in range(len(vertices)):
        yield vertices[index], vertices[(index + 1) % len(vertices)]


def _segments_intersect(
    a_start: np.ndarray,
    a_end: np.ndarray,
    b_start: np.ndarray,
    b_end: np.ndarray,
) -> bool:
    o1 = _orientation(a_start, a_end, b_start)
    o2 = _orientation(a_start, a_end, b_end)
    o3 = _orientation(b_start, b_end, a_start)
    o4 = _orientation(b_start, b_end, a_end)
    if o1 == 0 and _point_on_segment(b_start, a_start, a_end):
        return True
    if o2 == 0 and _point_on_segment(b_end, a_start, a_end):
        return True
    if o3 == 0 and _point_on_segment(a_start, b_start, b_end):
        return True
    if o4 == 0 and _point_on_segment(a_end, b_start, b_end):
        return True
    return o1 != o2 and o3 != o4


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> int:
    value = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if abs(value) <= 1e-9:
        return 0
    return 1 if value > 0 else -1


def _point_on_segment(
    point: np.ndarray,
    segment_start: np.ndarray,
    segment_end: np.ndarray,
) -> bool:
    eps = 1e-9
    return (
        min(segment_start[0], segment_end[0]) - eps
        <= point[0]
        <= max(segment_start[0], segment_end[0]) + eps
        and min(segment_start[1], segment_end[1]) - eps
        <= point[1]
        <= max(segment_start[1], segment_end[1]) + eps
    )


def _tile_count_for_span(span_um: float, tile_size_um: float, step_um: float) -> int:
    remaining_after_first_tile = max(0.0, span_um - tile_size_um)
    return 1 + ceil(remaining_after_first_tile / step_um)


def _rectangular_region_vertices(region: RectangularRegion) -> list[tuple[float, float]]:
    half_width = region.width_um / 2.0
    half_height = region.height_um / 2.0
    return [
        (region.center_x_um - half_width, region.center_y_um - half_height),
        (region.center_x_um + half_width, region.center_y_um - half_height),
        (region.center_x_um + half_width, region.center_y_um + half_height),
        (region.center_x_um - half_width, region.center_y_um + half_height),
    ]


def _tile_vertices(
    center_x_um: float,
    center_y_um: float,
    tile: AcquisitionTile,
) -> list[tuple[float, float]]:
    half_width = tile.width_um / 2.0
    half_height = tile.height_um / 2.0
    return [
        (center_x_um - half_width, center_y_um - half_height),
        (center_x_um + half_width, center_y_um - half_height),
        (center_x_um + half_width, center_y_um + half_height),
        (center_x_um - half_width, center_y_um + half_height),
    ]


def _stage_vertices_to_yx(
    vertices_xy_um: list[tuple[float, float]],
    transform: AffinePixelToStage,
) -> np.ndarray:
    return np.asarray(
        [_stage_point_to_yx(x_um, y_um, transform) for x_um, y_um in vertices_xy_um],
        dtype=np.float32,
    )


def _stage_point_to_yx(
    x_um: float,
    y_um: float,
    transform: AffinePixelToStage,
) -> tuple[float, float]:
    x_px, y_px = transform.apply_inverse(x_um=x_um, y_um=y_um)
    return float(y_px), float(x_px)
