"""Save and combine reviewed acquisition target plans."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

from smart_acquisition.cellsens.xml_writer import CellSensStageNavigatorWriter
from smart_acquisition.models import (
    OverviewImageMetadata,
    PolygonRegion,
    RectangularRegion,
    StagePosition,
)


@dataclass(frozen=True)
class TargetPlan:
    """Reviewed targets from one source overview/z-stack."""

    source_image: str
    planning_mode: str
    point_positions: list[StagePosition]
    rectangular_regions: list[RectangularRegion]
    polygon_regions: list[PolygonRegion]
    overview_metadata: OverviewImageMetadata | None = None
    selected_object_count: int | None = None


def save_target_plan(path: str | Path, plan: TargetPlan) -> None:
    """Write a target plan JSON sidecar."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(_target_plan_to_payload(plan), handle, indent=2)


def load_target_plan(path: str | Path) -> TargetPlan:
    """Read a target plan JSON sidecar."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return _target_plan_from_payload(payload)


def combine_target_plans(
    plans: Iterable[TargetPlan],
) -> tuple[list[StagePosition], list[RectangularRegion], list[PolygonRegion]]:
    """Concatenate reviewed targets while giving each source a stable prefix."""

    all_positions: list[StagePosition] = []
    all_rectangles: list[RectangularRegion] = []
    all_polygons: list[PolygonRegion] = []

    for plan_index, plan in enumerate(plans, start=1):
        prefix = f"Source {plan_index}"
        all_positions.extend(
            StagePosition(
                x_um=position.x_um,
                y_um=position.y_um,
                z_um=position.z_um,
                name=_prefixed_name(prefix, position.name, "position", index),
            )
            for index, position in enumerate(plan.point_positions, start=1)
        )
        all_rectangles.extend(
            RectangularRegion(
                center_x_um=region.center_x_um,
                center_y_um=region.center_y_um,
                z_um=region.z_um,
                width_um=region.width_um,
                height_um=region.height_um,
                name=_prefixed_name(prefix, region.name, "tilescan", index),
            )
            for index, region in enumerate(plan.rectangular_regions, start=1)
        )
        all_polygons.extend(
            PolygonRegion(
                vertices_xy_um=list(region.vertices_xy_um),
                z_um=region.z_um,
                name=_prefixed_name(prefix, region.name, "mosaic", index),
            )
            for index, region in enumerate(plan.polygon_regions, start=1)
        )

    return all_positions, all_rectangles, all_polygons


def write_combined_cellsens_xml(
    *,
    template_xml: str | Path,
    output_xml: str | Path,
    plans: Iterable[TargetPlan],
) -> None:
    """Write one CellSens XML containing targets from all reviewed plans."""

    positions, rectangles, polygons = combine_target_plans(plans)
    writer = CellSensStageNavigatorWriter(template_xml)
    if rectangles or polygons:
        writer.replace_targets(
            positions=positions,
            rectangular_regions=rectangles,
            polygon_regions=polygons,
        )
    else:
        writer.add_positions(positions)
    writer.save(output_xml)


def _target_plan_to_payload(plan: TargetPlan) -> dict:
    return {
        "source_image": plan.source_image,
        "planning_mode": plan.planning_mode,
        "point_positions": [asdict(position) for position in plan.point_positions],
        "rectangular_regions": [
            asdict(region) for region in plan.rectangular_regions
        ],
        "polygon_regions": [asdict(region) for region in plan.polygon_regions],
        "overview_metadata": (
            None if plan.overview_metadata is None else asdict(plan.overview_metadata)
        ),
        "selected_object_count": plan.selected_object_count,
    }


def _target_plan_from_payload(payload: dict) -> TargetPlan:
    return TargetPlan(
        source_image=str(payload["source_image"]),
        planning_mode=str(payload["planning_mode"]),
        point_positions=[
            StagePosition(**position)
            for position in payload.get("point_positions", [])
        ],
        rectangular_regions=[
            RectangularRegion(**region)
            for region in payload.get("rectangular_regions", [])
        ],
        polygon_regions=[
            PolygonRegion(
                vertices_xy_um=[tuple(vertex) for vertex in region["vertices_xy_um"]],
                z_um=float(region["z_um"]),
                name=region.get("name"),
            )
            for region in payload.get("polygon_regions", [])
        ],
        overview_metadata=(
            None
            if payload.get("overview_metadata") is None
            else OverviewImageMetadata(**payload["overview_metadata"])
        ),
        selected_object_count=payload.get("selected_object_count"),
    )


def _prefixed_name(
    prefix: str,
    current_name: str | None,
    fallback_kind: str,
    index: int,
) -> str:
    name = current_name or f"{fallback_kind} {index}"
    return f"{prefix} - {name}"
