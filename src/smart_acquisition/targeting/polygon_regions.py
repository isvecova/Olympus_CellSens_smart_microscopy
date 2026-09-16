"""Plan polygon tile-scan regions from positive masks."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

import numpy as np
from scipy import ndimage

from smart_acquisition.models import AcquisitionTile, PolygonRegion, StagePosition
from smart_acquisition.targeting.coordinate_transform import AffinePixelToStage
from smart_acquisition.targeting.tile_planner import (
    _find,
    _tile_count_for_span,
    _union,
    _validate_tile,
)


@dataclass(frozen=True)
class MixedPolygonPlan:
    """Ordinary positions plus irregular polygon mosaic regions."""

    point_positions: list[StagePosition]
    polygon_regions: list[PolygonRegion]
    grouped_mask: np.ndarray


@dataclass(frozen=True)
class ZAwareTileGroup:
    """A set of detections planned on one aligned XY grid and one Z band."""

    name: str
    component_labels: tuple[int, ...]
    z_um: float
    tile_count: int


@dataclass(frozen=True)
class ZAwareTilePlan:
    """Individual tile positions with per-tile Z estimates."""

    point_positions: list[StagePosition]
    groups: list[ZAwareTileGroup]
    grouped_mask: np.ndarray


@dataclass(frozen=True)
class ZAwarePolygonPlan:
    """Ordinary positions plus Z-aware irregular polygon mosaic regions."""

    point_positions: list[StagePosition]
    polygon_regions: list[PolygonRegion]
    groups: list[ZAwareTileGroup]
    grouped_mask: np.ndarray


@dataclass(frozen=True)
class _MaskComponent:
    label: int
    area_px: int
    centroid_x_px: float
    centroid_y_px: float
    position: StagePosition


@dataclass(frozen=True)
class _StageBounds:
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
    label: int
    bounds: _StageBounds
    z_um: float
    position: StagePosition


def plan_mixed_positions_and_polygon_regions_from_mask(
    mask: np.ndarray,
    transform: AffinePixelToStage,
    z_um: float,
    tile: AcquisitionTile,
    min_area_px: int = 1,
    merge_distance_factor: float = 1.0,
    simplify_tolerance_um: float = 10.0,
    name_prefix: str = "Planned",
) -> MixedPolygonPlan:
    """Use ordinary positions for isolated detections and polygons for groups."""

    _validate_tile(tile)
    labels, components = _measure_components(mask, transform, z_um, min_area_px)
    if not components:
        return MixedPolygonPlan([], [], np.zeros_like(mask, dtype=bool))

    grouped_indexes = _group_component_indexes(
        components,
        tile=tile,
        merge_distance_factor=merge_distance_factor,
    )

    point_positions: list[StagePosition] = []
    polygon_regions: list[PolygonRegion] = []
    grouped_mask = np.zeros_like(labels, dtype=bool)

    for group_index, component_indexes in enumerate(grouped_indexes, start=1):
        group_components = [components[index] for index in component_indexes]
        group_labels = [component.label for component in group_components]
        group_mask = np.isin(labels, group_labels)

        if len(group_components) == 1:
            component = group_components[0]
            point_positions.append(
                StagePosition(
                    x_um=component.position.x_um,
                    y_um=component.position.y_um,
                    z_um=component.position.z_um,
                    name=f"{name_prefix} position {len(point_positions) + 1}",
                )
            )
            continue

        grouped_mask |= group_mask
        vertices = _mask_to_polygon_vertices(
            group_mask,
            transform=transform,
            simplify_tolerance_um=simplify_tolerance_um,
        )
        polygon_regions.append(
            PolygonRegion(
                vertices_xy_um=vertices,
                z_um=z_um,
                name=f"{name_prefix} mosaic {len(polygon_regions) + 1}",
            )
        )

    return MixedPolygonPlan(
        point_positions=point_positions,
        polygon_regions=polygon_regions,
        grouped_mask=grouped_mask,
    )


def plan_z_aware_tile_positions_from_mask(
    mask: np.ndarray,
    *,
    classifier_labels: np.ndarray,
    argmax_z: np.ndarray,
    z_positions_um: tuple[float, ...],
    transform: AffinePixelToStage,
    fallback_z_um: float,
    tile: AcquisitionTile,
    include_labels: tuple[int, ...] = (1, 2),
    min_area_px: int = 1,
    merge_distance_factor: float = 1.0,
    max_group_z_difference_um: float = 10.0,
    coverage_margin_um: float = 0.0,
    name_prefix: str = "Planned",
) -> ZAwareTilePlan:
    """Plan individual high-mag tiles while preserving local Z estimates.

    This mode avoids merged CellSens ROIs because those carry one Z value for
    the whole region. Instead, every selected high-mag tile is written as a
    point position with its own Z estimated from the argmax-Z image.
    """

    _validate_tile(tile)
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
    if not components:
        return ZAwareTilePlan([], [], np.zeros_like(mask, dtype=bool))

    grouped_indexes = _group_z_components(
        components,
        tile=tile,
        merge_distance_factor=merge_distance_factor,
        max_group_z_difference_um=max_group_z_difference_um,
    )

    point_positions: list[StagePosition] = []
    groups: list[ZAwareTileGroup] = []
    grouped_mask = np.zeros_like(labels, dtype=bool)

    for group_index, component_indexes in enumerate(grouped_indexes, start=1):
        group_components = [components[index] for index in component_indexes]
        group_labels = tuple(component.label for component in group_components)
        group_mask = np.isin(labels, group_labels)
        grouped_mask |= group_mask

        group_z_um = _median_z_um(group_components)
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
        point_positions.extend(group_positions)
        groups.append(
            ZAwareTileGroup(
                name=f"Group {group_index}",
                component_labels=group_labels,
                z_um=group_z_um,
                tile_count=len(group_positions),
            )
        )

    return ZAwareTilePlan(
        point_positions=point_positions,
        groups=groups,
        grouped_mask=grouped_mask,
    )


def plan_z_aware_positions_and_polygon_regions_from_mask(
    mask: np.ndarray,
    *,
    classifier_labels: np.ndarray,
    argmax_z: np.ndarray,
    z_positions_um: tuple[float, ...],
    transform: AffinePixelToStage,
    fallback_z_um: float,
    tile: AcquisitionTile,
    include_labels: tuple[int, ...] = (1, 2),
    min_area_px: int = 1,
    merge_distance_factor: float = 1.0,
    max_group_z_difference_um: float = 10.0,
    simplify_tolerance_um: float = 10.0,
    name_prefix: str = "Planned",
) -> ZAwarePolygonPlan:
    """Plan irregular mosaic ROIs without merging across large Z differences."""

    _validate_tile(tile)
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
    if not components:
        return ZAwarePolygonPlan([], [], [], np.zeros_like(mask, dtype=bool))

    grouped_indexes = _group_z_components(
        components,
        tile=tile,
        merge_distance_factor=merge_distance_factor,
        max_group_z_difference_um=max_group_z_difference_um,
    )

    point_positions: list[StagePosition] = []
    polygon_regions: list[PolygonRegion] = []
    groups: list[ZAwareTileGroup] = []
    grouped_mask = np.zeros_like(labels, dtype=bool)

    for group_index, component_indexes in enumerate(grouped_indexes, start=1):
        group_components = [components[index] for index in component_indexes]
        group_labels = tuple(component.label for component in group_components)
        group_mask = np.isin(labels, group_labels)
        grouped_mask |= group_mask
        group_z_um = _median_z_um(group_components)

        if len(group_components) == 1:
            component = group_components[0]
            point_positions.append(
                StagePosition(
                    x_um=component.position.x_um,
                    y_um=component.position.y_um,
                    z_um=component.z_um,
                    name=f"{name_prefix} position {len(point_positions) + 1}",
                )
            )
            tile_count = 1
        else:
            vertices = _mask_to_polygon_vertices(
                group_mask,
                transform=transform,
                simplify_tolerance_um=simplify_tolerance_um,
            )
            polygon_regions.append(
                PolygonRegion(
                    vertices_xy_um=vertices,
                    z_um=group_z_um,
                    name=f"{name_prefix} mosaic {len(polygon_regions) + 1}",
                )
            )
            tile_count = _estimated_tile_count_for_bounds(
                _mask_stage_bounds(group_mask, transform),
                tile,
            )

        groups.append(
            ZAwareTileGroup(
                name=f"Group {group_index}",
                component_labels=group_labels,
                z_um=group_z_um,
                tile_count=tile_count,
            )
        )

    return ZAwarePolygonPlan(
        point_positions=point_positions,
        polygon_regions=polygon_regions,
        groups=groups,
        grouped_mask=grouped_mask,
    )


def plan_optimized_z_aware_positions_and_polygon_regions_from_mask(
    mask: np.ndarray,
    *,
    classifier_labels: np.ndarray,
    argmax_z: np.ndarray,
    z_positions_um: tuple[float, ...],
    transform: AffinePixelToStage,
    fallback_z_um: float,
    tile: AcquisitionTile,
    include_labels: tuple[int, ...] = (1, 2),
    min_area_px: int = 1,
    merge_distance_factor: float = 1.0,
    max_group_z_difference_um: float = 10.0,
    max_extra_tile_fraction: float = 0.25,
    simplify_tolerance_um: float = 10.0,
    name_prefix: str = "Planned",
) -> ZAwarePolygonPlan:
    """Plan irregular mosaic ROIs using tile-count and Z-aware merge costs."""

    _validate_tile(tile)
    if max_extra_tile_fraction < 0:
        raise ValueError("max_extra_tile_fraction must be non-negative")

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
    if not components:
        return ZAwarePolygonPlan([], [], [], np.zeros_like(mask, dtype=bool))

    groups = [[index] for index in range(len(components))]
    while True:
        best_merge: tuple[int, int, int, int] | None = None
        for group_a_index, group_a in enumerate(groups):
            for group_b_index in range(group_a_index + 1, len(groups)):
                group_b = groups[group_b_index]
                if not _z_group_merge_candidate(
                    components,
                    group_a,
                    group_b,
                    tile=tile,
                    merge_distance_factor=merge_distance_factor,
                    max_group_z_difference_um=max_group_z_difference_um,
                ):
                    continue

                separate_count = _estimated_group_tile_count(
                    labels,
                    components,
                    group_a,
                    transform=transform,
                    tile=tile,
                    simplify_tolerance_um=simplify_tolerance_um,
                ) + _estimated_group_tile_count(
                    labels,
                    components,
                    group_b,
                    transform=transform,
                    tile=tile,
                    simplify_tolerance_um=simplify_tolerance_um,
                )
                merged_group = group_a + group_b
                merged_count = _estimated_group_tile_count(
                    labels,
                    components,
                    merged_group,
                    transform=transform,
                    tile=tile,
                    simplify_tolerance_um=simplify_tolerance_um,
                )
                detection_count = _estimated_detection_tile_count(
                    labels,
                    components,
                    merged_group,
                    transform=transform,
                    tile=tile,
                )
                extra_fraction = max(0, merged_count - detection_count) / max(
                    1,
                    merged_count,
                )
                saving = separate_count - merged_count
                if saving < 0 or extra_fraction > max_extra_tile_fraction:
                    continue
                candidate = (
                    saving,
                    -merged_count,
                    group_a_index,
                    group_b_index,
                )
                if best_merge is None or candidate > best_merge:
                    best_merge = candidate

        if best_merge is None:
            break

        _saving, _negative_count, group_a_index, group_b_index = best_merge
        groups[group_a_index] = groups[group_a_index] + groups[group_b_index]
        del groups[group_b_index]

    groups = sorted(
        groups,
        key=lambda indexes: (
            min(components[index].bounds.min_y_um for index in indexes),
            min(components[index].bounds.min_x_um for index in indexes),
        ),
    )
    return _z_groups_to_polygon_plan(
        labels,
        components,
        classifier_labels=classifier_labels,
        argmax_z=argmax_z,
        z_positions_um=z_positions_um,
        transform=transform,
        fallback_z_um=fallback_z_um,
        tile=tile,
        include_labels=include_labels,
        groups=groups,
        simplify_tolerance_um=simplify_tolerance_um,
        name_prefix=name_prefix,
    )


def _measure_components(
    mask: np.ndarray,
    transform: AffinePixelToStage,
    z_um: float,
    min_area_px: int,
) -> tuple[np.ndarray, list[_MaskComponent]]:
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
            )
        )

    return labels, components


def _group_component_indexes(
    components: list[_MaskComponent],
    tile: AcquisitionTile,
    merge_distance_factor: float,
) -> list[list[int]]:
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
    z_values = [components[index].z_um for index in group]
    return max(z_values) - min(z_values)


def _combined_bounds(bounds_list: list[_StageBounds]) -> _StageBounds:
    if not bounds_list:
        raise ValueError("Cannot combine an empty bounds list")
    return _StageBounds(
        min_x_um=min(bounds.min_x_um for bounds in bounds_list),
        max_x_um=max(bounds.max_x_um for bounds in bounds_list),
        min_y_um=min(bounds.min_y_um for bounds in bounds_list),
        max_y_um=max(bounds.max_y_um for bounds in bounds_list),
    )


def _z_groups_to_polygon_plan(
    labels: np.ndarray,
    components: list[_ZMaskComponent],
    *,
    classifier_labels: np.ndarray,
    argmax_z: np.ndarray,
    z_positions_um: tuple[float, ...],
    transform: AffinePixelToStage,
    fallback_z_um: float,
    tile: AcquisitionTile,
    include_labels: tuple[int, ...],
    groups: list[list[int]],
    simplify_tolerance_um: float,
    name_prefix: str,
) -> ZAwarePolygonPlan:
    point_positions: list[StagePosition] = []
    polygon_regions: list[PolygonRegion] = []
    planned_groups: list[ZAwareTileGroup] = []
    grouped_mask = np.zeros_like(labels, dtype=bool)

    for group_index, component_indexes in enumerate(groups, start=1):
        group_components = [components[index] for index in component_indexes]
        group_labels = tuple(component.label for component in group_components)
        group_mask = np.isin(labels, group_labels)
        grouped_mask |= group_mask
        group_z_um = _estimate_group_z_um(
            group_mask,
            classifier_labels=classifier_labels,
            argmax_z=argmax_z,
            z_positions_um=z_positions_um,
            fallback_z_um=fallback_z_um,
            include_labels=include_labels,
        )

        if len(group_components) == 1:
            component = group_components[0]
            point_positions.append(
                StagePosition(
                    x_um=component.position.x_um,
                    y_um=component.position.y_um,
                    z_um=component.z_um,
                    name=f"{name_prefix} position {len(point_positions) + 1}",
                )
            )
            tile_count = 1
        else:
            vertices = _mask_to_polygon_vertices(
                group_mask,
                transform=transform,
                simplify_tolerance_um=simplify_tolerance_um,
            )
            polygon_regions.append(
                PolygonRegion(
                    vertices_xy_um=vertices,
                    z_um=group_z_um,
                    name=f"{name_prefix} mosaic {len(polygon_regions) + 1}",
                )
            )
            tile_count = _estimated_group_tile_count(
                labels,
                components,
                component_indexes,
                transform=transform,
                tile=tile,
                simplify_tolerance_um=simplify_tolerance_um,
            )

        planned_groups.append(
            ZAwareTileGroup(
                name=f"Group {group_index}",
                component_labels=group_labels,
                z_um=group_z_um,
                tile_count=tile_count,
            )
        )

    return ZAwarePolygonPlan(
        point_positions=point_positions,
        polygon_regions=polygon_regions,
        groups=planned_groups,
        grouped_mask=grouped_mask,
    )


def _estimate_group_z_um(
    group_mask: np.ndarray,
    *,
    classifier_labels: np.ndarray,
    argmax_z: np.ndarray,
    z_positions_um: tuple[float, ...],
    fallback_z_um: float,
    include_labels: tuple[int, ...],
) -> float:
    valid_z_mask = np.asarray(group_mask, dtype=bool) & np.isin(
        classifier_labels,
        include_labels,
    )
    return _estimate_z_from_argmax(
        np.asarray(argmax_z)[valid_z_mask],
        z_positions_um=z_positions_um,
        fallback_z_um=fallback_z_um,
    )


def _estimated_group_tile_count(
    labels: np.ndarray,
    components: list[_ZMaskComponent],
    group: list[int],
    *,
    transform: AffinePixelToStage,
    tile: AcquisitionTile,
    simplify_tolerance_um: float,
) -> int:
    group_labels = [components[index].label for index in group]
    group_mask = np.isin(labels, group_labels)
    return _estimated_polygon_tile_count(
        group_mask,
        transform=transform,
        tile=tile,
        simplify_tolerance_um=simplify_tolerance_um,
    )


def _estimated_detection_tile_count(
    labels: np.ndarray,
    components: list[_ZMaskComponent],
    group: list[int],
    *,
    transform: AffinePixelToStage,
    tile: AcquisitionTile,
) -> int:
    group_labels = [components[index].label for index in group]
    group_mask = np.isin(labels, group_labels)
    return _estimated_tile_count_for_mask(group_mask, transform=transform, tile=tile)


def _estimated_polygon_tile_count(
    mask: np.ndarray,
    *,
    transform: AffinePixelToStage,
    tile: AcquisitionTile,
    simplify_tolerance_um: float,
) -> int:
    filled_mask = _filled_polygon_mask(
        mask,
        transform=transform,
        simplify_tolerance_um=simplify_tolerance_um,
    )
    return _estimated_tile_count_for_mask(filled_mask, transform=transform, tile=tile)


def _filled_polygon_mask(
    mask: np.ndarray,
    *,
    transform: AffinePixelToStage,
    simplify_tolerance_um: float,
) -> np.ndarray:
    mask_array = np.asarray(mask, dtype=bool)
    try:
        vertices = _mask_to_polygon_vertices(
            mask_array,
            transform=transform,
            simplify_tolerance_um=simplify_tolerance_um,
        )
        from skimage.draw import polygon
    except ModuleNotFoundError:
        return _bounding_box_mask(mask_array)

    pixel_vertices = [
        transform.apply_inverse(x_um=x_um, y_um=y_um) for x_um, y_um in vertices
    ]
    x_values = np.asarray([vertex[0] for vertex in pixel_vertices], dtype=np.float32)
    y_values = np.asarray([vertex[1] for vertex in pixel_vertices], dtype=np.float32)
    rr, cc = polygon(y_values, x_values, shape=mask_array.shape)
    filled = np.zeros_like(mask_array, dtype=bool)
    filled[rr, cc] = True
    return filled


def _bounding_box_mask(mask: np.ndarray) -> np.ndarray:
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
    valid_indexes = np.asarray(z_indexes, dtype=np.int64)
    valid_indexes = valid_indexes[
        (valid_indexes >= 0) & (valid_indexes < len(z_positions_um))
    ]
    if valid_indexes.size == 0:
        return float(fallback_z_um)
    histogram = np.bincount(valid_indexes, minlength=len(z_positions_um))
    return float(z_positions_um[int(np.argmax(histogram))])


def _median_z_um(components: list[_ZMaskComponent]) -> float:
    if not components:
        raise ValueError("Cannot estimate group Z without components")
    return float(np.median([component.z_um for component in components]))


def _estimated_tile_count_for_bounds(bounds: _StageBounds, tile: AcquisitionTile) -> int:
    count_x = _tile_count_for_span(bounds.width_um, tile.width_um, tile.step_x_um)
    count_y = _tile_count_for_span(bounds.height_um, tile.height_um, tile.step_y_um)
    return count_x * count_y


def _bounds_gap_um(a: _StageBounds, b: _StageBounds, *, axis: str) -> float:
    if axis == "x":
        return max(0.0, a.min_x_um - b.max_x_um, b.min_x_um - a.max_x_um)
    if axis == "y":
        return max(0.0, a.min_y_um - b.max_y_um, b.min_y_um - a.max_y_um)
    raise ValueError(f"Unknown axis: {axis}")


def _mask_stage_bounds(
    mask: np.ndarray,
    transform: AffinePixelToStage,
) -> _StageBounds:
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
    simplify_tolerance_um: float,
) -> list[tuple[float, float]]:
    object_slices = ndimage.find_objects(mask.astype(np.uint8))
    object_slice = next((item for item in object_slices if item is not None), None)
    if object_slice is None:
        raise ValueError("Cannot create polygon for an empty mask")

    cropped = mask[object_slice]
    if cropped.shape[0] < 2 or cropped.shape[1] < 2:
        return _bounding_box_vertices(mask, transform)

    from skimage import measure

    contours = measure.find_contours(cropped.astype(float), level=0.5)
    if not contours:
        return _bounding_box_vertices(mask, transform)

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
        return _bounding_box_vertices(mask, transform)

    if len(contours) == 1:
        contour = np.asarray(contour_points)
    else:
        from scipy.spatial import ConvexHull, QhullError

        try:
            hull = ConvexHull(np.asarray(contour_points))
        except QhullError:
            return _bounding_box_vertices(mask, transform)
        contour = np.asarray(contour_points)[hull.vertices]

    tolerance_px = _stage_tolerance_to_pixels(transform, simplify_tolerance_um)
    simplified = measure.approximate_polygon(contour, tolerance=tolerance_px)
    if _is_closed(simplified):
        simplified = simplified[:-1]
    if len(simplified) < 3:
        return _bounding_box_vertices(mask, transform)

    vertices = [transform.apply(x_px=float(x), y_px=float(y)) for x, y in simplified]
    if len(vertices) > 1 and _same_point(vertices[0], vertices[-1]):
        vertices = vertices[:-1]
    if len(vertices) < 3:
        return _bounding_box_vertices(mask, transform)
    return vertices


def _bounding_box_vertices(
    mask: np.ndarray,
    transform: AffinePixelToStage,
) -> list[tuple[float, float]]:
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
    if simplify_tolerance_um <= 0:
        return 0.0
    pixel_x_um = hypot(transform.a, transform.c)
    pixel_y_um = hypot(transform.b, transform.d)
    pixel_um = max(pixel_x_um, pixel_y_um, 1e-12)
    return simplify_tolerance_um / pixel_um


def _is_closed(points: np.ndarray) -> bool:
    if len(points) < 2:
        return False
    return bool(np.allclose(points[0], points[-1]))


def _same_point(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9
