"""Plan CellSens targets from positive masks.

The public entry point is `plan_composed_targets_from_mask`. It accepts the
independent planning choices parsed by `planning_modes.py` and emits point
positions, rectangular tile scans, and/or polygon mosaic ROIs. Internally the
module first measures connected mask components, then groups them, estimates Z
when requested, and finally converts each group into CellSens target geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from math import ceil, hypot
from time import perf_counter

import numpy as np
from scipy import ndimage

from smart_acquisition.models import (
    AcquisitionTile,
    PolygonRegion,
    RectangularRegion,
    StagePosition,
)
from smart_acquisition.targeting.coordinate_transform import AffinePixelToStage
from smart_acquisition.targeting.tile_planner import (
    PositionGroup,
    _find,
    _tile_count_for_span,
    _union,
    _validate_tile,
    rectangular_region_for_group,
    tile_positions_for_group,
)
from smart_acquisition.targeting.z_estimation import (
    update_tile_scan_region_z_positions,
)


@dataclass(frozen=True)
class ZAwareTileGroup:
    """A set of detections planned on one aligned XY grid and one Z band."""

    name: str
    component_labels: tuple[int, ...]
    z_um: float
    tile_count: int


@dataclass(frozen=True)
class ComposedTargetPlan:
    """Targets planned from independent output/grouping/Z/optimization choices."""

    point_positions: list[StagePosition]
    rectangular_regions: list[RectangularRegion]
    polygon_regions: list[PolygonRegion]
    groups: list[ZAwareTileGroup]
    grouped_mask: np.ndarray


@dataclass(frozen=True)
class _MaskComponent:
    """Connected mask component measured without Z-stack metadata."""

    label: int
    area_px: int
    centroid_x_px: float
    centroid_y_px: float
    position: StagePosition
    bounds: "_StageBounds"
    slice_yx: tuple[slice, slice]


@dataclass(frozen=True)
class _StageBounds:
    """Axis-aligned stage-space bounds for a component or planned group."""

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

    def with_margin(self, margin_um: float) -> "_StageBounds":
        return _StageBounds(
            min_x_um=self.min_x_um - margin_um,
            max_x_um=self.max_x_um + margin_um,
            min_y_um=self.min_y_um - margin_um,
            max_y_um=self.max_y_um + margin_um,
        )


@dataclass(frozen=True)
class _ZMaskComponent:
    """Connected mask component with the Z value used for Z-aware grouping."""

    label: int
    bounds: _StageBounds
    z_um: float
    position: StagePosition
    slice_yx: tuple[slice, slice]


def plan_composed_targets_from_mask(
    mask: np.ndarray,
    *,
    classifier_labels: np.ndarray,
    argmax_z: np.ndarray | None,
    z_positions_um: tuple[float, ...] | None,
    transform: AffinePixelToStage,
    fallback_z_um: float,
    tile: AcquisitionTile,
    output_choice: str,
    grouping_choice: str,
    z_choice: str,
    optimization_choice: str,
    include_labels: tuple[int, ...] = (1, 2),
    min_area_px: int = 1,
    merge_distance_factor: float = 1.0,
    max_group_z_difference_um: float = 10.0,
    max_extra_tile_fraction: float = 0.25,
    coverage_margin_um: float = 0.0,
    simplify_tolerance_um: float = 10.0,
    polygon_boundary_mode: str = "alpha_shape",
    alpha_radius_tile_fraction: float = 0.75,
    name_prefix: str = "Planned",
    estimate_final_roi_z: bool = True,
    estimate_group_tile_counts: bool = True,
    timing_logs_enabled: bool = False,
) -> ComposedTargetPlan:
    """Plan final acquisition targets from a selected positive mask.

    `output_choice`, `grouping_choice`, `z_choice`, and `optimization_choice`
    are intentionally independent so GUIs and scripts can mix them freely.
    """

    _validate_composed_choices(output_choice, grouping_choice, z_choice, optimization_choice)
    _validate_tile(tile)
    needs_z_metadata = grouping_choice == "xy_z" or z_choice in ("per_roi", "per_tile")
    if z_choice == "per_tile" and output_choice != "tile_point_grid":
        raise ValueError(
            "Per-tile Z can only be represented as tile point grid output; "
            "CellSens rectangle and polygon ROIs store one Z per ROI."
        )
    if needs_z_metadata:
        if argmax_z is None or z_positions_um is None:
            raise ValueError(
                "This planning combination requires z-stack argmax-Z metadata."
            )
        phase_start = perf_counter()
        labels, components = _measure_z_components(
            mask,
            classifier_labels=classifier_labels,
            argmax_z=argmax_z,
            z_positions_um=z_positions_um,
            transform=transform,
            fallback_z_um=fallback_z_um,
            include_labels=include_labels,
            min_area_px=min_area_px,
        )
        _log_planner_timing(
            "measure Z components",
            phase_start,
            enabled=timing_logs_enabled,
            extra=f"{len(components)} components",
        )
    else:
        phase_start = perf_counter()
        labels, measured_components = _measure_components(
            mask,
            transform,
            fallback_z_um,
            min_area_px,
        )
        components = [
            _ZMaskComponent(
                label=component.label,
                bounds=component.bounds,
                z_um=fallback_z_um,
                position=component.position,
                slice_yx=component.slice_yx,
            )
            for component in measured_components
        ]
        _log_planner_timing(
            "measure components",
            phase_start,
            enabled=timing_logs_enabled,
            extra=f"{len(components)} components",
        )

    if not components:
        return ComposedTargetPlan([], [], [], [], np.zeros_like(mask, dtype=bool))

    phase_start = perf_counter()
    groups = _compose_component_groups(
        labels,
        components,
        transform=transform,
        tile=tile,
        output_choice=output_choice,
        grouping_choice=grouping_choice,
        optimization_choice=optimization_choice,
        merge_distance_factor=merge_distance_factor,
        max_group_z_difference_um=max_group_z_difference_um,
        max_extra_tile_fraction=max_extra_tile_fraction,
        simplify_tolerance_um=simplify_tolerance_um,
        polygon_boundary_mode=polygon_boundary_mode,
        alpha_radius_tile_fraction=alpha_radius_tile_fraction,
    )
    _log_planner_timing(
        "compose component groups",
        phase_start,
        enabled=timing_logs_enabled,
        extra=f"{len(groups)} groups",
    )

    point_positions: list[StagePosition] = []
    rectangular_regions: list[RectangularRegion] = []
    polygon_regions: list[PolygonRegion] = []
    planned_groups: list[ZAwareTileGroup] = []
    grouped_mask = np.zeros_like(labels, dtype=bool)

    phase_start = perf_counter()
    for group_index, component_indexes in enumerate(groups, start=1):
        group_components = [components[index] for index in component_indexes]
        group_labels = tuple(component.label for component in group_components)
        group_mask = np.isin(labels, group_labels)
        grouped_mask |= group_mask
        group_z_um = _composed_group_z_um(
            group_components,
            fallback_z_um=fallback_z_um,
            z_choice=z_choice,
        )
        group_tile_count = 0

        if output_choice == "detection_points":
            for component in group_components:
                z_um = (
                    fallback_z_um
                    if z_choice == "overview"
                    else component.z_um
                )
                point_positions.append(
                    StagePosition(
                        x_um=component.position.x_um,
                        y_um=component.position.y_um,
                        z_um=z_um,
                        name=f"{name_prefix} position {len(point_positions) + 1}",
                    )
                )
            group_tile_count = len(group_components)

        elif output_choice == "group_center_points":
            bounds = _combined_bounds([component.bounds for component in group_components])
            point_positions.append(
                StagePosition(
                    x_um=bounds.center_x_um,
                    y_um=bounds.center_y_um,
                    z_um=group_z_um,
                    name=f"{name_prefix} group {group_index}",
                )
            )
            group_tile_count = 1

        elif output_choice == "tile_point_grid":
            if z_choice == "per_tile":
                if argmax_z is None or z_positions_um is None:
                    raise ValueError("Per-tile Z requires argmax-Z metadata")
                group_positions = _tile_positions_for_z_group(
                    group_mask,
                    classifier_labels=classifier_labels,
                    argmax_z=argmax_z,
                    z_positions_um=z_positions_um,
                    transform=transform,
                    fallback_z_um=group_z_um,
                    tile=tile,
                    include_labels=include_labels,
                    margin_um=coverage_margin_um,
                    name_prefix=f"{name_prefix} group {group_index}",
                )
            else:
                group_positions = tile_positions_for_group(
                    _position_group_for_components(
                        group_components,
                        z_um=group_z_um,
                        name=f"Group {group_index}",
                    ),
                    tile=tile,
                    margin_um=coverage_margin_um,
                    name_prefix=f"{name_prefix} group {group_index}",
                )
            point_positions.extend(group_positions)
            group_tile_count = len(group_positions)

        elif output_choice == "rectangular_regions":
            rectangle_plan = rectangular_region_for_group(
                _position_group_for_components(
                    group_components,
                    z_um=group_z_um,
                    name=f"Group {group_index}",
                ),
                tile=tile,
                margin_um=coverage_margin_um,
                name=f"{name_prefix} tilescan {len(rectangular_regions) + 1}",
            )
            rectangular_regions.append(rectangle_plan.region)
            group_tile_count = rectangle_plan.tile_count

        elif output_choice == "polygon_regions":
            vertex_parts = _mask_to_polygon_vertex_parts(
                group_mask,
                transform=transform,
                tile=tile,
                simplify_tolerance_um=simplify_tolerance_um,
                polygon_boundary_mode=polygon_boundary_mode,
                alpha_radius_tile_fraction=alpha_radius_tile_fraction,
                split_disconnected_alpha=True,
            )
            for vertices in vertex_parts:
                polygon_regions.append(
                    PolygonRegion(
                        vertices_xy_um=vertices,
                        z_um=group_z_um,
                        name=f"{name_prefix} mosaic {len(polygon_regions) + 1}",
                    )
                )
            if estimate_group_tile_counts:
                group_tile_count = sum(
                    _estimated_polygon_tile_count_from_vertices(
                        vertices,
                        shape_yx=group_mask.shape,
                        transform=transform,
                        tile=tile,
                    )
                    for vertices in vertex_parts
                )
            else:
                group_tile_count = 0

        planned_groups.append(
            ZAwareTileGroup(
                name=f"Group {group_index}",
                component_labels=group_labels,
                z_um=group_z_um,
                tile_count=group_tile_count,
            )
        )

    _log_planner_timing(
        "build output targets",
        phase_start,
        enabled=timing_logs_enabled,
        extra=(
            f"{len(point_positions)} points, {len(rectangular_regions)} rectangles, "
            f"{len(polygon_regions)} polygons"
        ),
    )

    if estimate_final_roi_z and z_choice == "per_roi" and (
        polygon_regions or rectangular_regions
    ):
        if argmax_z is None or z_positions_um is None:
            raise ValueError("Per-ROI Z requires argmax-Z metadata")
        phase_start = perf_counter()
        polygon_regions, rectangular_regions, _z_estimates = (
            update_tile_scan_region_z_positions(
                polygon_regions=polygon_regions,
                rectangular_regions=rectangular_regions,
                classifier_labels=classifier_labels,
                argmax_z=argmax_z,
                z_positions_um=z_positions_um,
                transform=transform,
                tile=tile,
                include_labels=include_labels,
            )
        )
        _log_planner_timing(
            "final per-ROI Z",
            phase_start,
            enabled=timing_logs_enabled,
            extra=f"{len(rectangular_regions) + len(polygon_regions)} regions",
        )

    return ComposedTargetPlan(
        point_positions=point_positions,
        rectangular_regions=rectangular_regions,
        polygon_regions=polygon_regions,
        groups=planned_groups,
        grouped_mask=grouped_mask,
    )


def _validate_composed_choices(
    output_choice: str,
    grouping_choice: str,
    z_choice: str,
    optimization_choice: str,
) -> None:
    """Validate the four independent planning choice keys."""

    if output_choice not in {
        "detection_points",
        "group_center_points",
        "tile_point_grid",
        "rectangular_regions",
        "polygon_regions",
    }:
        raise ValueError(f"Unknown planning output choice: {output_choice}")
    if grouping_choice not in {"none", "xy", "xy_z"}:
        raise ValueError(f"Unknown grouping choice: {grouping_choice}")
    if z_choice not in {"overview", "per_roi", "per_tile"}:
        raise ValueError(f"Unknown Z handling choice: {z_choice}")
    if optimization_choice not in {"direct", "tile_count"}:
        raise ValueError(f"Unknown optimization choice: {optimization_choice}")


def _compose_component_groups(
    labels: np.ndarray,
    components: list[_ZMaskComponent],
    *,
    transform: AffinePixelToStage,
    tile: AcquisitionTile,
    output_choice: str,
    grouping_choice: str,
    optimization_choice: str,
    merge_distance_factor: float,
    max_group_z_difference_um: float,
    max_extra_tile_fraction: float,
    simplify_tolerance_um: float,
    polygon_boundary_mode: str,
    alpha_radius_tile_fraction: float,
) -> list[list[int]]:
    """Return component-index groups for the selected grouping/optimization mode."""

    if grouping_choice == "none":
        return [[index] for index in range(len(components))]
    if optimization_choice == "tile_count":
        # Pre-partitioning keeps the greedy optimizer from comparing components
        # that can never merge under the XY/Z distance rules.
        candidate_pools = _group_z_components(
            components,
            tile=tile,
            merge_distance_factor=merge_distance_factor,
            max_group_z_difference_um=(
                max_group_z_difference_um if grouping_choice == "xy_z" else float("inf")
            ),
        )
        groups: list[list[int]] = []
        for pool in candidate_pools:
            if len(pool) == 1:
                groups.append(pool)
                continue
            groups.extend(
                _optimize_composed_groups(
                    labels,
                    components,
                    transform=transform,
                    tile=tile,
                    output_choice=output_choice,
                    grouping_choice=grouping_choice,
                    merge_distance_factor=merge_distance_factor,
                    max_group_z_difference_um=max_group_z_difference_um,
                    max_extra_tile_fraction=max_extra_tile_fraction,
                    simplify_tolerance_um=simplify_tolerance_um,
                    polygon_boundary_mode=polygon_boundary_mode,
                    alpha_radius_tile_fraction=alpha_radius_tile_fraction,
                    initial_groups=[[index] for index in pool],
                )
            )
        return _sort_component_groups(groups, components)
    return _group_z_components(
        components,
        tile=tile,
        merge_distance_factor=merge_distance_factor,
        max_group_z_difference_um=(
            max_group_z_difference_um
            if grouping_choice == "xy_z"
            else float("inf")
        ),
    )


def _optimize_composed_groups(
    labels: np.ndarray,
    components: list[_ZMaskComponent],
    *,
    transform: AffinePixelToStage,
    tile: AcquisitionTile,
    output_choice: str,
    grouping_choice: str,
    merge_distance_factor: float,
    max_group_z_difference_um: float,
    max_extra_tile_fraction: float,
    simplify_tolerance_um: float,
    polygon_boundary_mode: str,
    alpha_radius_tile_fraction: float,
    initial_groups: list[list[int]] | None = None,
) -> list[list[int]]:
    """Greedily merge nearby groups when doing so reduces estimated tile count."""

    initial_group_list = (
        [list(group) for group in initial_groups]
        if initial_groups is not None
        else [[index] for index in range(len(components))]
    )
    bounds_cache: dict[tuple[int, ...], _StageBounds] = {}
    z_span_cache: dict[tuple[int, ...], float] = {}
    output_count_cache: dict[tuple[int, ...], int] = {}
    detection_count_cache: dict[tuple[int, ...], int] = {}
    merge_x_um = tile.width_um * merge_distance_factor
    merge_y_um = tile.height_um * merge_distance_factor
    active_groups: dict[int, list[int]] = {
        index: group for index, group in enumerate(initial_group_list)
    }
    active_ids = set(active_groups)
    next_group_id = len(active_groups)
    merge_heap: list[tuple[int, int, int, int]] = []

    def group_key(group: list[int]) -> tuple[int, ...]:
        return tuple(sorted(group))

    def group_bounds(group: list[int]) -> _StageBounds:
        key = group_key(group)
        if key not in bounds_cache:
            bounds_cache[key] = _combined_bounds(
                [components[index].bounds for index in key]
            )
        return bounds_cache[key]

    def group_z_span(group: list[int]) -> float:
        key = group_key(group)
        if key not in z_span_cache:
            z_values = [components[index].z_um for index in key]
            z_span_cache[key] = max(z_values) - min(z_values)
        return z_span_cache[key]

    def output_tile_count(group: list[int]) -> int:
        key = group_key(group)
        if key not in output_count_cache:
            # During search, polygon groups use a cheap bounding-box estimate.
            # Exact polygon geometry is generated only once final groups exist.
            output_count_cache[key] = _estimated_group_output_tile_count(
                labels,
                components,
                list(key),
                transform=transform,
                tile=tile,
                output_choice=output_choice,
                simplify_tolerance_um=simplify_tolerance_um,
                polygon_boundary_mode=polygon_boundary_mode,
                alpha_radius_tile_fraction=alpha_radius_tile_fraction,
                fast_polygon_estimate=True,
            )
        return output_count_cache[key]

    def detection_tile_count(group: list[int]) -> int:
        key = group_key(group)
        if key not in detection_count_cache:
            detection_count_cache[key] = _estimated_detection_tile_count(
                labels,
                components,
                list(key),
                transform=transform,
                tile=tile,
            )
        return detection_count_cache[key]

    def merge_candidate(
        group_a_id: int,
        group_b_id: int,
    ) -> tuple[int, int, int, int] | None:
        group_a = active_groups[group_a_id]
        group_b = active_groups[group_b_id]
        merged_group = group_a + group_b
        if (
            grouping_choice == "xy_z"
            and group_z_span(merged_group) > max_group_z_difference_um
        ):
            return None
        bounds_a = group_bounds(group_a)
        bounds_b = group_bounds(group_b)
        if (
            _bounds_gap_um(bounds_a, bounds_b, axis="x") > merge_x_um
            or _bounds_gap_um(bounds_a, bounds_b, axis="y") > merge_y_um
        ):
            return None

        separate_count = output_tile_count(group_a) + output_tile_count(group_b)
        merged_count = output_tile_count(merged_group)
        saving = separate_count - merged_count
        if saving < 0:
            return None
        detection_count = detection_tile_count(merged_group)
        extra_fraction = max(0, merged_count - detection_count) / max(
            1,
            merged_count,
        )
        if extra_fraction > max_extra_tile_fraction:
            return None
        first_id, second_id = sorted((group_a_id, group_b_id))
        return (-saving, merged_count, first_id, second_id)

    def push_candidate(group_a_id: int, group_b_id: int) -> None:
        candidate = merge_candidate(group_a_id, group_b_id)
        if candidate is not None:
            heapq.heappush(merge_heap, candidate)

    group_ids = sorted(active_ids)
    for index, group_a_id in enumerate(group_ids):
        for group_b_id in group_ids[index + 1 :]:
            push_candidate(group_a_id, group_b_id)

    while merge_heap:
        _negative_saving, _merged_count, group_a_id, group_b_id = heapq.heappop(
            merge_heap
        )
        if group_a_id not in active_ids or group_b_id not in active_ids:
            continue

        merged_group = active_groups[group_a_id] + active_groups[group_b_id]
        del active_groups[group_a_id]
        del active_groups[group_b_id]
        active_ids.remove(group_a_id)
        active_ids.remove(group_b_id)

        merged_group_id = next_group_id
        next_group_id += 1
        active_groups[merged_group_id] = merged_group
        active_ids.add(merged_group_id)

        for other_group_id in sorted(active_ids):
            if other_group_id != merged_group_id:
                push_candidate(merged_group_id, other_group_id)

    return _sort_component_groups(list(active_groups.values()), components)


def _sort_component_groups(
    groups: list[list[int]],
    components: list[_ZMaskComponent],
) -> list[list[int]]:
    """Sort groups top-to-bottom, then left-to-right for stable output names."""

    return sorted(
        groups,
        key=lambda indexes: (
            min(components[index].bounds.min_y_um for index in indexes),
            min(components[index].bounds.min_x_um for index in indexes),
        ),
    )


def _composed_group_merge_candidate(
    components: list[_ZMaskComponent],
    group_a: list[int],
    group_b: list[int],
    *,
    tile: AcquisitionTile,
    grouping_choice: str,
    merge_distance_factor: float,
    max_group_z_difference_um: float,
) -> bool:
    """Return whether two groups are close enough to be considered for merging."""

    if (
        grouping_choice == "xy_z"
        and _z_group_span_um(components, group_a + group_b)
        > max_group_z_difference_um
    ):
        return False
    bounds_a = _combined_bounds([components[index].bounds for index in group_a])
    bounds_b = _combined_bounds([components[index].bounds for index in group_b])
    merge_x_um = tile.width_um * merge_distance_factor
    merge_y_um = tile.height_um * merge_distance_factor
    return (
        _bounds_gap_um(bounds_a, bounds_b, axis="x") <= merge_x_um
        and _bounds_gap_um(bounds_a, bounds_b, axis="y") <= merge_y_um
    )


def _estimated_group_output_tile_count(
    labels: np.ndarray,
    components: list[_ZMaskComponent],
    group: list[int],
    *,
    transform: AffinePixelToStage,
    tile: AcquisitionTile,
    output_choice: str,
    simplify_tolerance_um: float,
    polygon_boundary_mode: str,
    alpha_radius_tile_fraction: float,
    fast_polygon_estimate: bool = False,
) -> int:
    """Estimate output tiles for a group under the selected output style."""

    if output_choice == "detection_points":
        return len(group)
    if output_choice == "group_center_points":
        return 1
    bounds = _combined_bounds([components[index].bounds for index in group])
    if output_choice in {"tile_point_grid", "rectangular_regions"}:
        return _estimated_tile_count_for_bounds(bounds, tile)
    if output_choice == "polygon_regions" and fast_polygon_estimate:
        return _estimated_tile_count_for_bounds(bounds, tile)
    group_mask, group_transform = _component_group_mask(
        labels,
        components,
        group,
        transform,
    )
    return _estimated_polygon_tile_count(
        group_mask,
        transform=group_transform,
        tile=tile,
        simplify_tolerance_um=simplify_tolerance_um,
        polygon_boundary_mode=polygon_boundary_mode,
        alpha_radius_tile_fraction=alpha_radius_tile_fraction,
    )


def _position_group_for_components(
    components: list[_ZMaskComponent],
    *,
    z_um: float,
    name: str,
) -> PositionGroup:
    """Convert measured components into the simpler tile-planner group object."""

    return PositionGroup(
        name=name,
        positions=tuple(
            StagePosition(
                x_um=component.position.x_um,
                y_um=component.position.y_um,
                z_um=z_um,
                name=component.position.name,
            )
            for component in components
        ),
    )


def _composed_group_z_um(
    components: list[_ZMaskComponent],
    *,
    fallback_z_um: float,
    z_choice: str,
) -> float:
    """Return the Z value assigned to one planned output group."""

    if z_choice == "overview":
        return float(fallback_z_um)
    return _median_z_um(components)


def _measure_components(
    mask: np.ndarray,
    transform: AffinePixelToStage,
    z_um: float,
    min_area_px: int,
) -> tuple[np.ndarray, list[_MaskComponent]]:
    """Label connected positive mask components and measure their centroids."""

    labels, label_count = ndimage.label(np.asarray(mask, dtype=bool))
    object_slices = ndimage.find_objects(labels)
    components: list[_MaskComponent] = []

    for label_index in range(1, label_count + 1):
        object_slice = object_slices[label_index - 1]
        if object_slice is None:
            continue

        component_mask = labels[object_slice] == label_index
        area_px = int(np.count_nonzero(component_mask))
        if area_px < min_area_px:
            labels[labels == label_index] = 0
            continue

        y_local, x_local = ndimage.center_of_mass(component_mask)
        y_px = float(y_local + object_slice[0].start)
        x_px = float(x_local + object_slice[1].start)
        x_um, y_um = transform.apply(x_px=x_px, y_px=y_px)
        components.append(
            _MaskComponent(
                label=label_index,
                area_px=area_px,
                centroid_x_px=x_px,
                centroid_y_px=y_px,
                position=StagePosition(
                    x_um=x_um,
                    y_um=y_um,
                    z_um=z_um,
                    name=f"Component {len(components) + 1}",
                ),
                bounds=_slice_stage_bounds(object_slice, mask.shape, transform),
                slice_yx=object_slice,
            )
        )

    return labels, components


def _group_component_indexes(
    components: list[_MaskComponent],
    tile: AcquisitionTile,
    merge_distance_factor: float,
) -> list[list[int]]:
    """Group non-Z-aware components by tile-sized XY proximity."""

    merge_x_um = tile.width_um * merge_distance_factor
    merge_y_um = tile.height_um * merge_distance_factor
    parent = list(range(len(components)))

    for index_a, component_a in enumerate(components):
        for index_b in range(index_a + 1, len(components)):
            component_b = components[index_b]
            if (
                abs(component_a.position.x_um - component_b.position.x_um) <= merge_x_um
                and abs(component_a.position.y_um - component_b.position.y_um)
                <= merge_y_um
            ):
                _union(parent, index_a, index_b)

    groups: dict[int, list[int]] = {}
    for index in range(len(components)):
        groups.setdefault(_find(parent, index), []).append(index)

    return sorted(
        groups.values(),
        key=lambda indexes: (
            min(components[index].position.y_um for index in indexes),
            min(components[index].position.x_um for index in indexes),
        ),
    )


def _measure_z_components(
    mask: np.ndarray,
    *,
    classifier_labels: np.ndarray,
    argmax_z: np.ndarray,
    z_positions_um: tuple[float, ...],
    transform: AffinePixelToStage,
    fallback_z_um: float,
    include_labels: tuple[int, ...],
    min_area_px: int,
) -> tuple[np.ndarray, list[_ZMaskComponent]]:
    """Measure components and assign each one an argmax-Z-derived stage Z."""

    mask_array = np.asarray(mask, dtype=bool)
    label_array = np.asarray(classifier_labels)
    argmax_array = np.asarray(argmax_z)
    if label_array.shape != mask_array.shape:
        raise ValueError(
            f"classifier_labels shape {label_array.shape} does not match "
            f"mask shape {mask_array.shape}"
        )
    if argmax_array.shape != mask_array.shape:
        raise ValueError(
            f"argmax_z shape {argmax_array.shape} does not match "
            f"mask shape {mask_array.shape}"
        )
    if not z_positions_um:
        raise ValueError("z_positions_um must contain at least one Z position")

    labels, label_count = ndimage.label(mask_array)
    object_slices = ndimage.find_objects(labels)
    components: list[_ZMaskComponent] = []

    for label_index in range(1, label_count + 1):
        object_slice = object_slices[label_index - 1]
        if object_slice is None:
            continue

        component_mask = labels[object_slice] == label_index
        area_px = int(np.count_nonzero(component_mask))
        if area_px < min_area_px:
            labels[labels == label_index] = 0
            continue

        component_labels = label_array[object_slice]
        component_argmax_z = argmax_array[object_slice]
        valid_z_mask = component_mask & np.isin(component_labels, include_labels)
        selected_z_um = _estimate_z_from_argmax(
            component_argmax_z[valid_z_mask],
            z_positions_um=z_positions_um,
            fallback_z_um=fallback_z_um,
        )
        y_local, x_local = ndimage.center_of_mass(component_mask)
        x_um, y_um = transform.apply(
            x_px=float(x_local + object_slice[1].start),
            y_px=float(y_local + object_slice[0].start),
        )
        components.append(
            _ZMaskComponent(
                label=label_index,
                bounds=_slice_stage_bounds(object_slice, mask_array.shape, transform),
                z_um=selected_z_um,
                position=StagePosition(
                    x_um=x_um,
                    y_um=y_um,
                    z_um=selected_z_um,
                    name=f"Component {len(components) + 1}",
                ),
                slice_yx=object_slice,
            )
        )

    return labels, components


def _group_z_components(
    components: list[_ZMaskComponent],
    *,
    tile: AcquisitionTile,
    merge_distance_factor: float,
    max_group_z_difference_um: float,
) -> list[list[int]]:
    """Group components by XY proximity while respecting maximum Z span."""

    if max_group_z_difference_um < 0:
        raise ValueError("max_group_z_difference_um must be non-negative")

    merge_x_um = tile.width_um * merge_distance_factor
    merge_y_um = tile.height_um * merge_distance_factor
    parent = list(range(len(components)))

    for index_a, component_a in enumerate(components):
        for index_b in range(index_a + 1, len(components)):
            component_b = components[index_b]
            if (
                _bounds_gap_um(component_a.bounds, component_b.bounds, axis="x")
                <= merge_x_um
                and _bounds_gap_um(component_a.bounds, component_b.bounds, axis="y")
                <= merge_y_um
                and abs(component_a.z_um - component_b.z_um)
                <= max_group_z_difference_um
                and _merged_z_span_um(
                    components,
                    parent,
                    index_a,
                    index_b,
                )
                <= max_group_z_difference_um
            ):
                _union(parent, index_a, index_b)

    groups: dict[int, list[int]] = {}
    for index in range(len(components)):
        groups.setdefault(_find(parent, index), []).append(index)

    return sorted(
        groups.values(),
        key=lambda indexes: (
            min(components[index].bounds.min_y_um for index in indexes),
            min(components[index].bounds.min_x_um for index in indexes),
        ),
    )


def _merged_z_span_um(
    components: list[_ZMaskComponent],
    parent: list[int],
    index_a: int,
    index_b: int,
) -> float:
    """Return the Z span that would result from merging two union-find groups."""

    root_a = _find(parent, index_a)
    root_b = _find(parent, index_b)
    z_values = [
        component.z_um
        for index, component in enumerate(components)
        if _find(parent, index) in (root_a, root_b)
    ]
    return max(z_values) - min(z_values)


def _z_group_merge_candidate(
    components: list[_ZMaskComponent],
    group_a: list[int],
    group_b: list[int],
    *,
    tile: AcquisitionTile,
    merge_distance_factor: float,
    max_group_z_difference_um: float,
) -> bool:
    """Legacy-style pairwise Z-aware merge predicate."""

    if _z_group_span_um(components, group_a + group_b) > max_group_z_difference_um:
        return False

    bounds_a = _combined_bounds([components[index].bounds for index in group_a])
    bounds_b = _combined_bounds([components[index].bounds for index in group_b])
    merge_x_um = tile.width_um * merge_distance_factor
    merge_y_um = tile.height_um * merge_distance_factor
    return (
        _bounds_gap_um(bounds_a, bounds_b, axis="x") <= merge_x_um
        and _bounds_gap_um(bounds_a, bounds_b, axis="y") <= merge_y_um
    )


def _z_group_span_um(
    components: list[_ZMaskComponent],
    group: list[int],
) -> float:
    """Return max minus min component Z for a candidate group."""

    z_values = [components[index].z_um for index in group]
    return max(z_values) - min(z_values)


def _combined_bounds(bounds_list: list[_StageBounds]) -> _StageBounds:
    """Combine several stage-space bounding boxes into one box."""

    if not bounds_list:
        raise ValueError("Cannot combine an empty bounds list")
    return _StageBounds(
        min_x_um=min(bounds.min_x_um for bounds in bounds_list),
        max_x_um=max(bounds.max_x_um for bounds in bounds_list),
        min_y_um=min(bounds.min_y_um for bounds in bounds_list),
        max_y_um=max(bounds.max_y_um for bounds in bounds_list),
    )


def _component_group_mask(
    labels: np.ndarray,
    components: list[_ZMaskComponent],
    group: list[int],
    transform: AffinePixelToStage,
) -> tuple[np.ndarray, AffinePixelToStage]:
    """Build a cropped boolean mask for a component group.

    The returned transform is shifted so local cropped pixels still map to the
    same stage coordinates as the original full-size image.
    """

    object_slice = _combined_component_slice(
        [components[index].slice_yx for index in group],
        labels.shape,
    )
    group_labels = [components[index].label for index in group]
    group_mask = np.isin(labels[object_slice], group_labels)
    return group_mask, _offset_transform(transform, object_slice)


def _combined_component_slice(
    slices_yx: list[tuple[slice, slice]],
    shape_yx: tuple[int, ...],
) -> tuple[slice, slice]:
    """Return the smallest image slice covering several component slices."""

    if not slices_yx:
        raise ValueError("Cannot combine an empty slice list")
    min_y = max(0, min(item[0].start for item in slices_yx))
    max_y = min(shape_yx[0], max(item[0].stop for item in slices_yx))
    min_x = max(0, min(item[1].start for item in slices_yx))
    max_x = min(shape_yx[1], max(item[1].stop for item in slices_yx))
    return slice(min_y, max_y), slice(min_x, max_x)


def _offset_transform(
    transform: AffinePixelToStage,
    object_slice: tuple[slice, slice],
) -> AffinePixelToStage:
    """Shift a pixel-to-stage transform to coordinates of a cropped image."""

    x_offset = float(object_slice[1].start)
    y_offset = float(object_slice[0].start)
    return AffinePixelToStage(
        a=transform.a,
        b=transform.b,
        tx=transform.tx + transform.a * x_offset + transform.b * y_offset,
        c=transform.c,
        d=transform.d,
        ty=transform.ty + transform.c * x_offset + transform.d * y_offset,
    )


def _estimated_detection_tile_count(
    labels: np.ndarray,
    components: list[_ZMaskComponent],
    group: list[int],
    *,
    transform: AffinePixelToStage,
    tile: AcquisitionTile,
) -> int:
    """Count how many high-mag tiles intersect the detected pixels in a group."""

    group_mask, group_transform = _component_group_mask(
        labels,
        components,
        group,
        transform,
    )
    return _estimated_tile_count_for_mask(
        group_mask,
        transform=group_transform,
        tile=tile,
    )


def _estimated_polygon_tile_count(
    mask: np.ndarray,
    *,
    transform: AffinePixelToStage,
    tile: AcquisitionTile,
    simplify_tolerance_um: float,
    polygon_boundary_mode: str,
    alpha_radius_tile_fraction: float,
) -> int:
    """Estimate tile count after filling the polygon boundary for a mask."""

    filled_mask = _filled_polygon_mask(
        mask,
        transform=transform,
        tile=tile,
        simplify_tolerance_um=simplify_tolerance_um,
        polygon_boundary_mode=polygon_boundary_mode,
        alpha_radius_tile_fraction=alpha_radius_tile_fraction,
    )
    return _estimated_tile_count_for_mask(filled_mask, transform=transform, tile=tile)


def _filled_polygon_mask(
    mask: np.ndarray,
    *,
    transform: AffinePixelToStage,
    tile: AcquisitionTile,
    simplify_tolerance_um: float,
    polygon_boundary_mode: str,
    alpha_radius_tile_fraction: float,
) -> np.ndarray:
    """Rasterize the selected polygon boundary into a filled boolean mask."""

    mask_array = np.asarray(mask, dtype=bool)
    try:
        vertices = _mask_to_polygon_vertices(
            mask_array,
            transform=transform,
            tile=tile,
            simplify_tolerance_um=simplify_tolerance_um,
            polygon_boundary_mode=polygon_boundary_mode,
            alpha_radius_tile_fraction=alpha_radius_tile_fraction,
        )
    except ModuleNotFoundError:
        return _bounding_box_mask(mask_array)

    filled = _polygon_vertices_to_mask(
        vertices,
        shape_yx=mask_array.shape,
        transform=transform,
    )
    if filled is None:
        return _bounding_box_mask(mask_array)
    return filled


def _estimated_polygon_tile_count_from_vertices(
    vertices_xy_um: list[tuple[float, float]],
    *,
    shape_yx: tuple[int, int],
    transform: AffinePixelToStage,
    tile: AcquisitionTile,
) -> int:
    """Estimate tile count from already-computed polygon vertices."""

    filled_mask = _polygon_vertices_to_mask(
        vertices_xy_um,
        shape_yx=shape_yx,
        transform=transform,
    )
    if filled_mask is None:
        return _estimated_tile_count_for_bounds(
            _stage_bounds_for_vertices(vertices_xy_um),
            tile,
        )
    return _estimated_tile_count_for_mask(filled_mask, transform=transform, tile=tile)


def _polygon_vertices_to_mask(
    vertices_xy_um: list[tuple[float, float]],
    *,
    shape_yx: tuple[int, int],
    transform: AffinePixelToStage,
) -> np.ndarray | None:
    """Rasterize stage-space polygon vertices, returning None without skimage."""

    try:
        from skimage.draw import polygon
    except ModuleNotFoundError:
        return None

    pixel_vertices = [
        transform.apply_inverse(x_um=x_um, y_um=y_um) for x_um, y_um in vertices_xy_um
    ]
    x_values = np.asarray([vertex[0] for vertex in pixel_vertices], dtype=np.float32)
    y_values = np.asarray([vertex[1] for vertex in pixel_vertices], dtype=np.float32)
    rr, cc = polygon(y_values, x_values, shape=shape_yx)
    filled = np.zeros(shape_yx, dtype=bool)
    filled[rr, cc] = True
    return filled


def _stage_bounds_for_vertices(vertices_xy_um: list[tuple[float, float]]) -> _StageBounds:
    """Return stage-space bounds around polygon vertices."""

    if not vertices_xy_um:
        raise ValueError("Cannot create stage bounds for empty vertices")
    x_values = [vertex[0] for vertex in vertices_xy_um]
    y_values = [vertex[1] for vertex in vertices_xy_um]
    return _StageBounds(
        min_x_um=min(x_values),
        max_x_um=max(x_values),
        min_y_um=min(y_values),
        max_y_um=max(y_values),
    )


def _bounding_box_mask(mask: np.ndarray) -> np.ndarray:
    """Return a mask filled over the positive pixels' bounding box."""

    mask_array = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(mask_array)
    if len(xs) == 0:
        return np.zeros_like(mask_array, dtype=bool)
    result = np.zeros_like(mask_array, dtype=bool)
    result[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] = True
    return result


def _estimated_tile_count_for_mask(
    mask: np.ndarray,
    *,
    transform: AffinePixelToStage,
    tile: AcquisitionTile,
) -> int:
    """Count tile centers whose footprint overlaps a positive mask."""

    mask_array = np.asarray(mask, dtype=bool)
    if not np.any(mask_array):
        return 0

    bounds = _mask_stage_bounds(mask_array, transform)
    count_x = _tile_count_for_span(bounds.width_um, tile.width_um, tile.step_x_um)
    count_y = _tile_count_for_span(bounds.height_um, tile.height_um, tile.step_y_um)
    span_x = max(0.0, (count_x - 1) * tile.step_x_um)
    span_y = max(0.0, (count_y - 1) * tile.step_y_um)
    start_x = bounds.center_x_um - span_x / 2.0
    start_y = bounds.center_y_um - span_y / 2.0

    tile_count = 0
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
                tile_count += 1
    return tile_count


def _tile_positions_for_z_group(
    group_mask: np.ndarray,
    *,
    classifier_labels: np.ndarray,
    argmax_z: np.ndarray,
    z_positions_um: tuple[float, ...],
    transform: AffinePixelToStage,
    fallback_z_um: float,
    tile: AcquisitionTile,
    include_labels: tuple[int, ...],
    margin_um: float,
    name_prefix: str,
) -> list[StagePosition]:
    """Create tile-center point targets and estimate Z separately for each tile."""

    label_array = np.asarray(classifier_labels)
    argmax_array = np.asarray(argmax_z)
    bounds = _mask_stage_bounds(group_mask, transform).with_margin(margin_um)
    count_x = _tile_count_for_span(bounds.width_um, tile.width_um, tile.step_x_um)
    count_y = _tile_count_for_span(bounds.height_um, tile.height_um, tile.step_y_um)
    span_x = max(0.0, (count_x - 1) * tile.step_x_um)
    span_y = max(0.0, (count_y - 1) * tile.step_y_um)
    start_x = bounds.center_x_um - span_x / 2.0
    start_y = bounds.center_y_um - span_y / 2.0

    positions: list[StagePosition] = []
    for y_index in range(count_y):
        for x_index in range(count_x):
            center_x_um = start_x + x_index * tile.step_x_um
            center_y_um = start_y + y_index * tile.step_y_um
            tile_slice = _tile_slices(
                group_mask.shape,
                transform=transform,
                center_x_um=center_x_um,
                center_y_um=center_y_um,
                tile=tile,
            )
            if tile_slice is None:
                continue

            detection_tile_mask = group_mask[tile_slice]
            if not np.any(detection_tile_mask):
                continue

            valid_z_mask = detection_tile_mask & np.isin(
                label_array[tile_slice],
                include_labels,
            )
            tile_z_um = _estimate_z_from_argmax(
                argmax_array[tile_slice][valid_z_mask],
                z_positions_um=z_positions_um,
                fallback_z_um=fallback_z_um,
            )
            positions.append(
                StagePosition(
                    x_um=center_x_um,
                    y_um=center_y_um,
                    z_um=tile_z_um,
                    name=f"{name_prefix} tile {len(positions) + 1}",
                )
            )

    return positions


def _estimate_z_from_argmax(
    z_indexes: np.ndarray,
    *,
    z_positions_um: tuple[float, ...],
    fallback_z_um: float,
) -> float:
    """Convert argmax-Z indexes into the most frequent stage Z value."""

    valid_indexes = np.asarray(z_indexes, dtype=np.int64)
    valid_indexes = valid_indexes[
        (valid_indexes >= 0) & (valid_indexes < len(z_positions_um))
    ]
    if valid_indexes.size == 0:
        return float(fallback_z_um)
    histogram = np.bincount(valid_indexes, minlength=len(z_positions_um))
    return float(z_positions_um[int(np.argmax(histogram))])


def _median_z_um(components: list[_ZMaskComponent]) -> float:
    """Return the median component Z for one group."""

    if not components:
        raise ValueError("Cannot estimate group Z without components")
    return float(np.median([component.z_um for component in components]))


def _estimated_tile_count_for_bounds(bounds: _StageBounds, tile: AcquisitionTile) -> int:
    """Estimate rectangular tile coverage for stage-space bounds."""

    count_x = _tile_count_for_span(bounds.width_um, tile.width_um, tile.step_x_um)
    count_y = _tile_count_for_span(bounds.height_um, tile.height_um, tile.step_y_um)
    return count_x * count_y


def _bounds_gap_um(a: _StageBounds, b: _StageBounds, *, axis: str) -> float:
    """Return the non-overlapping gap between two bounds on one axis."""

    if axis == "x":
        return max(0.0, a.min_x_um - b.max_x_um, b.min_x_um - a.max_x_um)
    if axis == "y":
        return max(0.0, a.min_y_um - b.max_y_um, b.min_y_um - a.max_y_um)
    raise ValueError(f"Unknown axis: {axis}")


def _mask_stage_bounds(
    mask: np.ndarray,
    transform: AffinePixelToStage,
) -> _StageBounds:
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
) -> _StageBounds:
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
    return _StageBounds(
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
    x_values = np.asarray([vertex[0] for vertex in pixel_vertices], dtype=np.float32)
    y_values = np.asarray([vertex[1] for vertex in pixel_vertices], dtype=np.float32)

    min_x = max(0, int(np.floor(float(np.min(x_values)))))
    max_x = min(shape_yx[1], int(np.ceil(float(np.max(x_values)))) + 1)
    min_y = max(0, int(np.floor(float(np.min(y_values)))))
    max_y = min(shape_yx[0], int(np.ceil(float(np.max(y_values)))) + 1)

    if min_x >= max_x or min_y >= max_y:
        return None
    return slice(min_y, max_y), slice(min_x, max_x)


def _mask_to_polygon_vertices(
    mask: np.ndarray,
    transform: AffinePixelToStage,
    tile: AcquisitionTile | None,
    simplify_tolerance_um: float,
    polygon_boundary_mode: str,
    alpha_radius_tile_fraction: float,
) -> list[tuple[float, float]]:
    """Return one polygon boundary for callers that cannot handle split parts."""

    return _mask_to_polygon_vertex_parts(
        mask,
        transform=transform,
        tile=tile,
        simplify_tolerance_um=simplify_tolerance_um,
        polygon_boundary_mode=polygon_boundary_mode,
        alpha_radius_tile_fraction=alpha_radius_tile_fraction,
        split_disconnected_alpha=False,
    )[0]


def _mask_to_polygon_vertex_parts(
    mask: np.ndarray,
    *,
    transform: AffinePixelToStage,
    tile: AcquisitionTile | None,
    simplify_tolerance_um: float,
    polygon_boundary_mode: str,
    alpha_radius_tile_fraction: float,
    split_disconnected_alpha: bool,
) -> list[list[tuple[float, float]]]:
    """Convert a mask into one or more polygon boundaries.

    In split mode, disconnected alpha-shape islands become separate polygon
    ROIs instead of falling back to one stretched convex hull.
    """

    if polygon_boundary_mode not in ("mask_contour", "alpha_shape"):
        raise ValueError(
            "polygon_boundary_mode must be 'mask_contour' or 'alpha_shape'"
        )
    if alpha_radius_tile_fraction <= 0:
        raise ValueError("alpha_radius_tile_fraction must be positive")

    object_slices = ndimage.find_objects(mask.astype(np.uint8))
    object_slice = next((item for item in object_slices if item is not None), None)
    if object_slice is None:
        raise ValueError("Cannot create polygon for an empty mask")

    fallback_parts = _fallback_polygon_vertex_parts(
        mask,
        transform=transform,
        split_disconnected=split_disconnected_alpha
        and polygon_boundary_mode == "alpha_shape",
    )
    cropped = mask[object_slice]
    if cropped.shape[0] < 2 or cropped.shape[1] < 2:
        return fallback_parts

    try:
        from skimage import measure
    except ModuleNotFoundError:
        return fallback_parts

    contours = measure.find_contours(cropped.astype(float), level=0.5)
    if not contours:
        return fallback_parts

    contour_points = []
    for contour in contours:
        if len(contour) < 3:
            continue
        for y_local, x_local in contour:
            contour_points.append(
                [
                    float(x_local + object_slice[1].start),
                    float(y_local + object_slice[0].start),
                ]
            )

    if len(contour_points) < 3:
        return fallback_parts

    tolerance_px = _stage_tolerance_to_pixels(transform, simplify_tolerance_um)
    if len(contours) == 1:
        vertices = _contour_to_polygon_vertices(
            np.asarray(contour_points),
            mask=mask,
            transform=transform,
            measure=measure,
            tolerance_px=tolerance_px,
        )
        return [vertices] if vertices is not None else [_bounding_box_vertices(mask, transform)]
    elif polygon_boundary_mode == "alpha_shape" and tile is not None:
        alpha_contours = _alpha_shape_contour_parts(
            np.asarray(contour_points, dtype=np.float64),
            mask_shape_yx=mask.shape,
            transform=transform,
            tile=tile,
            alpha_radius_tile_fraction=alpha_radius_tile_fraction,
        )
        if alpha_contours is None:
            if split_disconnected_alpha:
                return fallback_parts
            contour = _convex_hull_contour(contour_points, mask, transform)
        elif len(alpha_contours) == 1 or split_disconnected_alpha:
            vertex_parts = [
                vertices
                for contour_part in alpha_contours
                if (
                    vertices := _contour_to_polygon_vertices(
                        contour_part,
                        mask=mask,
                        transform=transform,
                        measure=measure,
                        tolerance_px=tolerance_px,
                    )
                )
                is not None
            ]
            if vertex_parts:
                return vertex_parts
            if split_disconnected_alpha:
                return fallback_parts
            contour = _convex_hull_contour(contour_points, mask, transform)
        else:
            contour = _convex_hull_contour(contour_points, mask, transform)
    else:
        contour = _convex_hull_contour(contour_points, mask, transform)

    vertices = _contour_to_polygon_vertices(
        contour,
        mask=mask,
        transform=transform,
        measure=measure,
        tolerance_px=tolerance_px,
    )
    return [vertices] if vertices is not None else [_bounding_box_vertices(mask, transform)]


def _contour_to_polygon_vertices(
    contour: np.ndarray,
    *,
    mask: np.ndarray,
    transform: AffinePixelToStage,
    measure: object,
    tolerance_px: float,
) -> list[tuple[float, float]] | None:
    """Simplify a pixel-space contour and convert it to stage vertices."""

    simplified = measure.approximate_polygon(contour, tolerance=tolerance_px)
    if _is_closed(simplified):
        simplified = simplified[:-1]
    if len(simplified) < 3:
        return None

    vertices = [transform.apply(x_px=float(x), y_px=float(y)) for x, y in simplified]
    if len(vertices) > 1 and _same_point(vertices[0], vertices[-1]):
        vertices = vertices[:-1]
    if len(vertices) < 3:
        return None
    return vertices


def _fallback_polygon_vertex_parts(
    mask: np.ndarray,
    *,
    transform: AffinePixelToStage,
    split_disconnected: bool,
) -> list[list[tuple[float, float]]]:
    """Fallback polygon parts when contour/alpha-shape dependencies are absent."""

    if not split_disconnected:
        return [_bounding_box_vertices(mask, transform)]

    labels, component_count = ndimage.label(np.asarray(mask, dtype=bool))
    if component_count <= 1:
        return [_bounding_box_vertices(mask, transform)]

    vertex_parts: list[list[tuple[float, float]]] = []
    object_slices = ndimage.find_objects(labels)
    for label_index in range(1, component_count + 1):
        object_slice = object_slices[label_index - 1]
        if object_slice is None:
            continue
        component_mask = labels[object_slice] == label_index
        if not np.any(component_mask):
            continue
        vertex_parts.append(
            _bounding_box_vertices(
                component_mask,
                _offset_transform(transform, object_slice),
            )
        )

    return sorted(vertex_parts, key=_vertices_sort_key) or [
        _bounding_box_vertices(mask, transform)
    ]


def _convex_hull_contour(
    contour_points: list[list[float]],
    mask: np.ndarray,
    transform: AffinePixelToStage,
) -> np.ndarray:
    """Return a convex-hull contour for multi-contour masks."""

    from scipy.spatial import ConvexHull, QhullError

    try:
        hull = ConvexHull(np.asarray(contour_points))
    except QhullError:
        return np.asarray(
            [
                transform.apply_inverse(x_um=x_um, y_um=y_um)
                for x_um, y_um in _bounding_box_vertices(mask, transform)
            ],
            dtype=np.float64,
        )
    return np.asarray(contour_points, dtype=np.float64)[hull.vertices]


def _alpha_shape_contour_points(
    points_xy: np.ndarray,
    *,
    mask_shape_yx: tuple[int, ...],
    transform: AffinePixelToStage,
    tile: AcquisitionTile,
    alpha_radius_tile_fraction: float,
) -> np.ndarray | None:
    """Return a single alpha-shape contour, or None if it would be disconnected."""

    contours = _alpha_shape_contour_parts(
        points_xy,
        mask_shape_yx=mask_shape_yx,
        transform=transform,
        tile=tile,
        alpha_radius_tile_fraction=alpha_radius_tile_fraction,
    )
    if contours is None or len(contours) != 1:
        return None
    return contours[0]


def _alpha_shape_contour_parts(
    points_xy: np.ndarray,
    *,
    mask_shape_yx: tuple[int, ...],
    transform: AffinePixelToStage,
    tile: AcquisitionTile,
    alpha_radius_tile_fraction: float,
) -> list[np.ndarray] | None:
    """Build alpha-shape contours and preserve disconnected contour parts."""

    if len(points_xy) < 4:
        return None

    from scipy.spatial import Delaunay, QhullError
    try:
        from skimage import draw, measure
    except ModuleNotFoundError:
        return None

    points_xy = _sample_alpha_shape_points(points_xy)
    alpha_radius_um = min(tile.width_um, tile.height_um) * alpha_radius_tile_fraction
    alpha_radius_px = _stage_tolerance_to_pixels(transform, alpha_radius_um)
    if alpha_radius_px <= 0:
        return None

    try:
        triangles = Delaunay(points_xy).simplices
    except QhullError:
        return None

    kept_triangles: list[np.ndarray] = []
    for triangle_indexes in triangles:
        triangle = points_xy[triangle_indexes]
        if _triangle_circumradius_px(triangle) <= alpha_radius_px:
            kept_triangles.append(triangle)
    if not kept_triangles:
        return None

    margin_px = int(ceil(alpha_radius_px)) + 3
    min_x = max(0, int(np.floor(float(np.min(points_xy[:, 0])))) - margin_px)
    max_x = min(
        mask_shape_yx[1],
        int(np.ceil(float(np.max(points_xy[:, 0])))) + margin_px + 1,
    )
    min_y = max(0, int(np.floor(float(np.min(points_xy[:, 1])))) - margin_px)
    max_y = min(
        mask_shape_yx[0],
        int(np.ceil(float(np.max(points_xy[:, 1])))) + margin_px + 1,
    )
    if min_x >= max_x or min_y >= max_y:
        return None

    alpha_mask = np.zeros((max_y - min_y, max_x - min_x), dtype=bool)
    offset = np.asarray([min_x, min_y], dtype=np.float64)
    for triangle in kept_triangles:
        local_triangle = triangle - offset
        rr, cc = draw.polygon(
            local_triangle[:, 1],
            local_triangle[:, 0],
            shape=alpha_mask.shape,
        )
        alpha_mask[rr, cc] = True

    labels, component_count = ndimage.label(alpha_mask)
    if component_count < 1:
        return None

    contour_parts: list[np.ndarray] = []
    for component_index in range(1, component_count + 1):
        component_mask = labels == component_index
        contours = measure.find_contours(component_mask.astype(float), level=0.5)
        if not contours:
            continue
        contour_yx = max(contours, key=_contour_area_yx)
        contour_xy = np.asarray(
            [[float(x + min_x), float(y + min_y)] for y, x in contour_yx],
            dtype=np.float64,
        )
        if len(contour_xy) >= 3:
            contour_parts.append(contour_xy)

    if not contour_parts:
        return None
    return sorted(contour_parts, key=_contour_sort_key)


def _sample_alpha_shape_points(
    points_xy: np.ndarray,
    max_points: int = 4000,
) -> np.ndarray:
    """Downsample contour points to keep Delaunay triangulation bounded."""

    if len(points_xy) <= max_points:
        return points_xy
    step = int(ceil(len(points_xy) / max_points))
    return points_xy[::step]


def _triangle_circumradius_px(triangle_xy: np.ndarray) -> float:
    """Return triangle circumradius in pixels for alpha-shape filtering."""

    a = float(np.linalg.norm(triangle_xy[1] - triangle_xy[0]))
    b = float(np.linalg.norm(triangle_xy[2] - triangle_xy[1]))
    c = float(np.linalg.norm(triangle_xy[0] - triangle_xy[2]))
    twice_area = abs(
        (triangle_xy[1, 0] - triangle_xy[0, 0])
        * (triangle_xy[2, 1] - triangle_xy[0, 1])
        - (triangle_xy[2, 0] - triangle_xy[0, 0])
        * (triangle_xy[1, 1] - triangle_xy[0, 1])
    )
    if twice_area <= 1e-12:
        return float("inf")
    return (a * b * c) / (2.0 * twice_area)


def _contour_area_yx(contour_yx: np.ndarray) -> float:
    """Return unsigned area of a contour stored as Y/X coordinates."""

    if len(contour_yx) < 3:
        return 0.0
    y_values = contour_yx[:, 0]
    x_values = contour_yx[:, 1]
    return abs(
        0.5
        * float(
            np.dot(x_values, np.roll(y_values, -1))
            - np.dot(y_values, np.roll(x_values, -1))
        )
    )


def _contour_sort_key(contour_xy: np.ndarray) -> tuple[float, float]:
    """Sort contours by top-left position for stable ROI order."""

    if len(contour_xy) == 0:
        return (float("inf"), float("inf"))
    return (
        float(np.min(contour_xy[:, 1])),
        float(np.min(contour_xy[:, 0])),
    )


def _vertices_sort_key(vertices_xy_um: list[tuple[float, float]]) -> tuple[float, float]:
    """Sort stage-space vertex lists by top-left position."""

    if not vertices_xy_um:
        return (float("inf"), float("inf"))
    return (
        min(y_um for _x_um, y_um in vertices_xy_um),
        min(x_um for x_um, _y_um in vertices_xy_um),
    )


def _bounding_box_vertices(
    mask: np.ndarray,
    transform: AffinePixelToStage,
) -> list[tuple[float, float]]:
    """Return rectangular stage vertices around positive pixels."""

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Cannot create bounding box for an empty mask")
    min_x = max(0.0, float(xs.min()) - 0.5)
    max_x = min(float(mask.shape[1]), float(xs.max()) + 0.5)
    min_y = max(0.0, float(ys.min()) - 0.5)
    max_y = min(float(mask.shape[0]), float(ys.max()) + 0.5)
    return [
        transform.apply(min_x, min_y),
        transform.apply(max_x, min_y),
        transform.apply(max_x, max_y),
        transform.apply(min_x, max_y),
    ]


def _stage_tolerance_to_pixels(
    transform: AffinePixelToStage,
    simplify_tolerance_um: float,
) -> float:
    """Convert a stage-space simplification tolerance into pixels."""

    if simplify_tolerance_um <= 0:
        return 0.0
    pixel_x_um = hypot(transform.a, transform.c)
    pixel_y_um = hypot(transform.b, transform.d)
    pixel_um = max(pixel_x_um, pixel_y_um, 1e-12)
    return simplify_tolerance_um / pixel_um


def _is_closed(points: np.ndarray) -> bool:
    """Return whether the first and last contour points are effectively equal."""

    if len(points) < 2:
        return False
    return bool(np.allclose(points[0], points[-1]))


def _same_point(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """Return whether two stage-space points are equal within numeric tolerance."""

    return abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9


def _log_planner_timing(
    label: str,
    start: float,
    *,
    enabled: bool,
    extra: str | None = None,
) -> None:
    """Print elapsed time for an internal planning phase when requested."""

    if not enabled:
        return
    elapsed_s = perf_counter() - start
    suffix = f" ({extra})" if extra else ""
    print(f"[timing] planner: {label}: {elapsed_s:.3f} s{suffix}")
