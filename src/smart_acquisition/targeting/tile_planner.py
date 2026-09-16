"""Plan high-magnification position lists from detected low-mag targets."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Literal

from smart_acquisition.models import AcquisitionTile, RectangularRegion, StagePosition

PlanningMode = Literal["raw", "group_centers", "group_tiles", "mixed_tilescans"]


@dataclass(frozen=True)
class PositionGroup:
    """A group of detections close enough to plan together."""

    name: str
    positions: tuple[StagePosition, ...]

    @property
    def min_x_um(self) -> float:
        return min(position.x_um for position in self.positions)

    @property
    def max_x_um(self) -> float:
        return max(position.x_um for position in self.positions)

    @property
    def min_y_um(self) -> float:
        return min(position.y_um for position in self.positions)

    @property
    def max_y_um(self) -> float:
        return max(position.y_um for position in self.positions)

    @property
    def z_um(self) -> float:
        return self.positions[0].z_um

    @property
    def center_x_um(self) -> float:
        return (self.min_x_um + self.max_x_um) / 2.0

    @property
    def center_y_um(self) -> float:
        return (self.min_y_um + self.max_y_um) / 2.0


@dataclass(frozen=True)
class MixedTilePlan:
    """Planned ordinary positions plus native rectangular tile-scan regions."""

    point_positions: list[StagePosition]
    rectangular_regions: list[RectangularRegion]
    groups: list[PositionGroup]


def plan_positions_for_high_mag_tiles(
    positions: list[StagePosition],
    tile: AcquisitionTile,
    mode: PlanningMode = "group_centers",
    merge_distance_factor: float = 1.0,
    coverage_margin_um: float = 0.0,
    name_prefix: str = "Planned",
) -> tuple[list[StagePosition], list[PositionGroup]]:
    """Plan output positions from raw detected positions.

    ``raw`` keeps every detected centroid.
    ``group_centers`` emits one position at the center of each nearby group.
    ``group_tiles`` emits a grid of tile centers covering each nearby group.
    """

    _validate_tile(tile)
    if mode == "raw":
        return positions, [
            PositionGroup(name=f"Group {index}", positions=(position,))
            for index, position in enumerate(positions, start=1)
        ]

    groups = group_positions_by_tile_size(
        positions,
        tile=tile,
        merge_distance_factor=merge_distance_factor,
        name_prefix="Group",
    )
    if mode == "group_centers":
        planned = [
            StagePosition(
                x_um=group.center_x_um,
                y_um=group.center_y_um,
                z_um=group.z_um,
                name=f"{name_prefix} group {index}",
            )
            for index, group in enumerate(groups, start=1)
        ]
        return planned, groups

    if mode == "group_tiles":
        planned = []
        for group_index, group in enumerate(groups, start=1):
            planned.extend(
                tile_positions_for_group(
                    group,
                    tile=tile,
                    margin_um=coverage_margin_um,
                    name_prefix=f"{name_prefix} group {group_index}",
                )
            )
        return planned, groups

    if mode == "mixed_tilescans":
        mixed_plan = plan_mixed_positions_and_tilescans(
            positions,
            tile=tile,
            merge_distance_factor=merge_distance_factor,
            coverage_margin_um=coverage_margin_um,
            name_prefix=name_prefix,
        )
        return (
            mixed_plan.point_positions
            + rectangles_to_center_positions(mixed_plan.rectangular_regions),
            mixed_plan.groups,
        )

    raise ValueError(f"Unknown planning mode: {mode}")


def plan_mixed_positions_and_tilescans(
    positions: list[StagePosition],
    tile: AcquisitionTile,
    merge_distance_factor: float = 1.0,
    coverage_margin_um: float = 0.0,
    name_prefix: str = "Planned",
) -> MixedTilePlan:
    """Use single point positions for one-tile groups and rectangles otherwise."""

    _validate_tile(tile)
    groups = group_positions_by_tile_size(
        positions,
        tile=tile,
        merge_distance_factor=merge_distance_factor,
        name_prefix="Group",
    )

    point_positions: list[StagePosition] = []
    rectangular_regions: list[RectangularRegion] = []
    for group_index, group in enumerate(groups, start=1):
        plan = rectangular_region_for_group(
            group,
            tile=tile,
            margin_um=coverage_margin_um,
            name=f"{name_prefix} tilescan {group_index}",
        )
        if plan.tile_count == 1:
            point_positions.append(
                StagePosition(
                    x_um=group.center_x_um,
                    y_um=group.center_y_um,
                    z_um=group.z_um,
                    name=f"{name_prefix} position {len(point_positions) + 1}",
                )
            )
        else:
            rectangular_regions.append(plan.region)

    return MixedTilePlan(
        point_positions=point_positions,
        rectangular_regions=rectangular_regions,
        groups=groups,
    )


def group_positions_by_tile_size(
    positions: list[StagePosition],
    tile: AcquisitionTile,
    merge_distance_factor: float = 1.0,
    name_prefix: str = "Group",
) -> list[PositionGroup]:
    """Group positions whose X/Y distance is within one tile field of view."""

    _validate_tile(tile)
    if not positions:
        return []

    merge_x_um = tile.width_um * merge_distance_factor
    merge_y_um = tile.height_um * merge_distance_factor
    parent = list(range(len(positions)))

    for index_a, position_a in enumerate(positions):
        for index_b in range(index_a + 1, len(positions)):
            position_b = positions[index_b]
            if (
                abs(position_a.x_um - position_b.x_um) <= merge_x_um
                and abs(position_a.y_um - position_b.y_um) <= merge_y_um
            ):
                _union(parent, index_a, index_b)

    grouped_indexes: dict[int, list[int]] = {}
    for index in range(len(positions)):
        root = _find(parent, index)
        grouped_indexes.setdefault(root, []).append(index)

    groups = []
    sorted_groups = sorted(
        grouped_indexes.values(),
        key=lambda indexes: (
            min(positions[index].y_um for index in indexes),
            min(positions[index].x_um for index in indexes),
        ),
    )
    for group_index, indexes in enumerate(sorted_groups, start=1):
        group_positions = tuple(positions[index] for index in indexes)
        groups.append(
            PositionGroup(
                name=f"{name_prefix} {group_index}",
                positions=group_positions,
            )
        )
    return groups


def tile_positions_for_group(
    group: PositionGroup,
    tile: AcquisitionTile,
    margin_um: float = 0.0,
    name_prefix: str = "Tile",
) -> list[StagePosition]:
    """Return tile-center positions covering a grouped detection bounding box."""

    _validate_tile(tile)
    min_x = group.min_x_um - margin_um
    max_x = group.max_x_um + margin_um
    min_y = group.min_y_um - margin_um
    max_y = group.max_y_um + margin_um

    count_x = _tile_count_for_span(max_x - min_x, tile.width_um, tile.step_x_um)
    count_y = _tile_count_for_span(max_y - min_y, tile.height_um, tile.step_y_um)

    span_x = max(0.0, (count_x - 1) * tile.step_x_um)
    span_y = max(0.0, (count_y - 1) * tile.step_y_um)
    start_x = (min_x + max_x) / 2.0 - span_x / 2.0
    start_y = (min_y + max_y) / 2.0 - span_y / 2.0

    positions: list[StagePosition] = []
    for y_index in range(count_y):
        for x_index in range(count_x):
            positions.append(
                StagePosition(
                    x_um=start_x + x_index * tile.step_x_um,
                    y_um=start_y + y_index * tile.step_y_um,
                    z_um=group.z_um,
                    name=f"{name_prefix} tile {len(positions) + 1}",
                )
            )
    return positions


@dataclass(frozen=True)
class RectangularTilePlan:
    """Rectangular coverage for a group, including implied tile counts."""

    region: RectangularRegion
    tile_count_x: int
    tile_count_y: int

    @property
    def tile_count(self) -> int:
        return self.tile_count_x * self.tile_count_y


def rectangular_region_for_group(
    group: PositionGroup,
    tile: AcquisitionTile,
    margin_um: float = 0.0,
    name: str | None = None,
) -> RectangularTilePlan:
    """Return the smallest rectangular scan region implied by tile coverage."""

    _validate_tile(tile)
    min_x = group.min_x_um - margin_um
    max_x = group.max_x_um + margin_um
    min_y = group.min_y_um - margin_um
    max_y = group.max_y_um + margin_um

    count_x = _tile_count_for_span(max_x - min_x, tile.width_um, tile.step_x_um)
    count_y = _tile_count_for_span(max_y - min_y, tile.height_um, tile.step_y_um)
    covered_width_um = tile.width_um + (count_x - 1) * tile.step_x_um
    covered_height_um = tile.height_um + (count_y - 1) * tile.step_y_um

    region = RectangularRegion(
        center_x_um=(min_x + max_x) / 2.0,
        center_y_um=(min_y + max_y) / 2.0,
        z_um=group.z_um,
        width_um=covered_width_um,
        height_um=covered_height_um,
        name=name,
    )
    return RectangularTilePlan(
        region=region,
        tile_count_x=count_x,
        tile_count_y=count_y,
    )


def rectangles_to_center_positions(
    rectangular_regions: list[RectangularRegion],
) -> list[StagePosition]:
    """Represent rectangles by their centers for CSV/debug views."""

    return [
        StagePosition(
            x_um=region.center_x_um,
            y_um=region.center_y_um,
            z_um=region.z_um,
            name=region.name,
        )
        for region in rectangular_regions
    ]


def _tile_count_for_span(span_um: float, tile_size_um: float, step_um: float) -> int:
    remaining_after_first_tile = max(0.0, span_um - tile_size_um)
    return 1 + ceil(remaining_after_first_tile / step_um)


def _validate_tile(tile: AcquisitionTile) -> None:
    if tile.width_px <= 0 or tile.height_px <= 0:
        raise ValueError("Tile pixel dimensions must be positive")
    if tile.pixel_size_x_um <= 0 or tile.pixel_size_y_um <= 0:
        raise ValueError("Tile pixel sizes must be positive")
    if not 0.0 <= tile.overlap_fraction < 1.0:
        raise ValueError("Tile overlap fraction must be >= 0 and < 1")


def _find(parent: list[int], index: int) -> int:
    while parent[index] != index:
        parent[index] = parent[parent[index]]
        index = parent[index]
    return index


def _union(parent: list[int], index_a: int, index_b: int) -> None:
    root_a = _find(parent, index_a)
    root_b = _find(parent, index_b)
    if root_a != root_b:
        parent[root_b] = root_a
