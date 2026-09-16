"""Editable example: threshold a VSI overview and export target positions.

Open this file, change the values in the CONFIGURATION section, and run it in
the ``irbisPythonJupyter`` conda environment. No PowerShell script is needed.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from smart_acquisition.cellsens.xml_writer import (
    DEFAULT_CELLSENS_TEMPLATE,
    CellSensStageNavigatorWriter,
)
from smart_acquisition.detection.threshold_detector import ThresholdDetector
from smart_acquisition.examples.process_vsi_threshold import (
    _write_mask,
    _write_positions_csv,
)
from smart_acquisition.image_input.bioio_vsi import read_vsi_overview
from smart_acquisition.models import (
    AcquisitionTile,
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
from smart_acquisition.targeting.positions import mask_to_centroid_positions
from smart_acquisition.targeting.tile_planner import (
    plan_mixed_positions_and_tilescans,
    plan_positions_for_high_mag_tiles,
)


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# IMAGE_PATH = Path("data/overview_scan.vsi")

# MASK_OUTPUT = Path("data/overview_scan_threshold_mask.tif")
# POSITIONS_OUTPUT = Path("data/overview_scan_threshold_positions.csv")

IMAGE_PATH = Path(r"N:\02_instrument_management\SpinSR10\260827_AppData_cellSens_config\260829\overview_1.vsi")

MASK_OUTPUT = Path(r"N:\02_instrument_management\SpinSR10\260827_AppData_cellSens_config\260829\processed_mixed_1\overview_260829_after_threshold_mask.tif")
POSITIONS_OUTPUT = Path(r"N:\02_instrument_management\SpinSR10\260827_AppData_cellSens_config\260829\processed_mixed_1\overview_260829_after_threshold_positions.csv")
PLANNED_POSITIONS_OUTPUT = POSITIONS_OUTPUT.with_name(
    f"{POSITIONS_OUTPUT.stem}_planned.csv"
)
RECTANGULAR_REGIONS_OUTPUT = POSITIONS_OUTPUT.with_name(
    f"{POSITIONS_OUTPUT.stem}_rectangular_tilescans.csv"
)
POLYGON_REGIONS_OUTPUT = POSITIONS_OUTPUT.with_name(
    f"{POSITIONS_OUTPUT.stem}_polygon_mosaics.csv"
)

# Set both of these to None if you only want the mask and CSV.
CELLSENS_TEMPLATE = Path(r"N:\02_instrument_management\SpinSR10\260827_AppData_cellSens_config\260829\overview_260829_before.xml")
CELLSENS_OUTPUT = Path(r"N:\02_instrument_management\SpinSR10\260827_AppData_cellSens_config\260829\processed_mixed_1\overview_260829_after_mixed.xml")
# CELLSENS_TEMPLATE = Path("xml_templates/overview_before_scan.xml")
# CELLSENS_OUTPUT = Path("data/overview_scan_threshold_positions.xml")

# Use this when the current overview image does not have its own matching XML.
# The fallback template will be patched with the current image edge positions
# from BioIO metadata before generated targets are written.
USE_DEFAULT_CELLSENS_TEMPLATE = False
DEFAULT_CELLSENS_TEMPLATE_PATH = DEFAULT_CELLSENS_TEMPLATE

# BioIO image selection.
SCENE = 0
CHANNEL = 0
Z = 0
TIME = 0

# Optional Z projection. Use None, "max", or "mean".
Z_PROJECTION = None

# Bright-region detection.
# If ABSOLUTE_THRESHOLD is None, the script uses PERCENTILE_THRESHOLD.
ABSOLUTE_THRESHOLD = None
PERCENTILE_THRESHOLD = 99.5
GAUSSIAN_SIGMA_PX = 10
MIN_AREA_PX = 2000

# Coordinate interpretation.
# The sample overview_scan.vsi looks like BioIO position_x/y is the image origin.
# Change to "center" if your metadata describes the image center instead.
METADATA_POSITION = "origin"  # "origin" or "center"
INVERT_X = False
INVERT_Y = False

NAME_PREFIX = "Detected"

# High-magnification acquisition planning.
# "raw" keeps every detected centroid as one cellSens position.
# "group_centers" merges nearby detections into one position per nearby group.
# "group_tiles" writes a grid of positions covering each nearby group.
# "mixed_tilescans" writes isolated/one-tile groups as ordinary positions and
# larger groups as native rectangular tile-scan regions.
# "mixed_irregular_mosaics" writes isolated detections as ordinary positions
# and nearby grouped detections as polygon/irregular mosaic regions.
PLANNING_MODE = "mixed_irregular_mosaics"

# Set these to the camera image size and pixel size of the NEW high-mag
# acquisition, not the low-mag overview. Example: 2048 px * 0.108 um/px.
TARGET_IMAGE_WIDTH_PX = 2304
TARGET_IMAGE_HEIGHT_PX = 2304
TARGET_PIXEL_SIZE_X_UM = 0.325
TARGET_PIXEL_SIZE_Y_UM = 0.325
TARGET_TILE_OVERLAP_FRACTION = 0.1  # 10% overlap

# 1.0 means detections are grouped when their X/Y separation is within one
# high-mag tile field of view. Increase slightly, e.g. 1.2, to merge more.
GROUP_MERGE_DISTANCE_FACTOR = 1.0

# Only used for PLANNING_MODE = "group_tiles". It expands each grouped bounding
# box before tile centers are generated.
COVERAGE_MARGIN_UM = 0.0

# Only used for PLANNING_MODE = "mixed_irregular_mosaics".
POLYGON_SIMPLIFY_TOLERANCE_UM = 10.0


# ---------------------------------------------------------------------------
# SCRIPT
# ---------------------------------------------------------------------------


def main() -> None:
    read_result = read_vsi_overview(
        IMAGE_PATH,
        scene=SCENE,
        channel=CHANNEL,
        z=Z,
        time=TIME,
        z_projection=Z_PROJECTION,
        source_position_is_center=METADATA_POSITION == "center",
    )

    detector = ThresholdDetector(
        threshold=ABSOLUTE_THRESHOLD,
        percentile=PERCENTILE_THRESHOLD if ABSOLUTE_THRESHOLD is None else None,
        gaussian_sigma=GAUSSIAN_SIGMA_PX,
    )
    detection = detector.detect(read_result.image)

    transform = AffinePixelToStage.from_overview_metadata(
        read_result.metadata,
        invert_x=INVERT_X,
        invert_y=INVERT_Y,
    )
    positions, measurements, cleaned_mask = mask_to_centroid_positions(
        detection.mask,
        transform=transform,
        z_um=read_result.metadata.stage_z_um,
        min_area_px=MIN_AREA_PX,
        name_prefix=NAME_PREFIX,
    )

    target_tile = AcquisitionTile(
        width_px=TARGET_IMAGE_WIDTH_PX,
        height_px=TARGET_IMAGE_HEIGHT_PX,
        pixel_size_x_um=TARGET_PIXEL_SIZE_X_UM,
        pixel_size_y_um=TARGET_PIXEL_SIZE_Y_UM,
        overlap_fraction=TARGET_TILE_OVERLAP_FRACTION,
    )
    if PLANNING_MODE == "mixed_tilescans":
        mixed_plan = plan_mixed_positions_and_tilescans(
            positions,
            tile=target_tile,
            merge_distance_factor=GROUP_MERGE_DISTANCE_FACTOR,
            coverage_margin_um=COVERAGE_MARGIN_UM,
            name_prefix="Planned",
        )
        planned_positions = mixed_plan.point_positions
        rectangular_regions = mixed_plan.rectangular_regions
        polygon_regions = []
        position_groups = mixed_plan.groups
    elif PLANNING_MODE == "mixed_irregular_mosaics":
        polygon_plan = plan_mixed_positions_and_polygon_regions_from_mask(
            cleaned_mask,
            transform=transform,
            z_um=read_result.metadata.stage_z_um,
            tile=target_tile,
            min_area_px=MIN_AREA_PX,
            merge_distance_factor=GROUP_MERGE_DISTANCE_FACTOR,
            simplify_tolerance_um=POLYGON_SIMPLIFY_TOLERANCE_UM,
            name_prefix="Planned",
        )
        planned_positions = polygon_plan.point_positions
        rectangular_regions = []
        polygon_regions = polygon_plan.polygon_regions
        position_groups = []
    else:
        planned_positions, position_groups = plan_positions_for_high_mag_tiles(
            positions,
            tile=target_tile,
            mode=PLANNING_MODE,
            merge_distance_factor=GROUP_MERGE_DISTANCE_FACTOR,
            coverage_margin_um=COVERAGE_MARGIN_UM,
            name_prefix="Planned",
        )
        rectangular_regions = []
        polygon_regions = []

    _write_mask(MASK_OUTPUT, cleaned_mask)
    _write_positions_csv(POSITIONS_OUTPUT, measurements)
    _write_stage_positions_csv(PLANNED_POSITIONS_OUTPUT, planned_positions)
    _write_rectangular_regions_csv(RECTANGULAR_REGIONS_OUTPUT, rectangular_regions)
    _write_polygon_regions_csv(POLYGON_REGIONS_OUTPUT, polygon_regions)

    if CELLSENS_OUTPUT is not None and (
        CELLSENS_TEMPLATE is not None or USE_DEFAULT_CELLSENS_TEMPLATE
    ):
        template = (
            DEFAULT_CELLSENS_TEMPLATE_PATH
            if USE_DEFAULT_CELLSENS_TEMPLATE
            else CELLSENS_TEMPLATE
        )
        writer = CellSensStageNavigatorWriter(template)
        if USE_DEFAULT_CELLSENS_TEMPLATE:
            writer.set_overview_stage_edges(
                **overview_stage_edge_positions(
                    read_result.metadata,
                    invert_x=INVERT_X,
                    invert_y=INVERT_Y,
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
        writer.save(CELLSENS_OUTPUT)

    print(f"Read image: {IMAGE_PATH}")
    print(f"Image shape YX: {tuple(np.asarray(read_result.image).shape)}")
    print(f"Stage origin/position: {read_result.metadata}")
    print(f"Threshold: {detection.threshold_value}")
    print(f"Raw detected components: {len(positions)}")
    print(f"Nearby groups: {len(position_groups)}")
    print(f"Planned cellSens positions: {len(planned_positions)}")
    print(f"Planned rectangular tile scans: {len(rectangular_regions)}")
    print(f"Planned irregular mosaic tile scans: {len(polygon_regions)}")
    print(
        "High-mag tile FOV: "
        f"{target_tile.width_um:.3f} x {target_tile.height_um:.3f} um"
    )
    print(f"Mask output: {MASK_OUTPUT}")
    print(f"Raw positions output: {POSITIONS_OUTPUT}")
    print(f"Planned positions output: {PLANNED_POSITIONS_OUTPUT}")
    print(f"Rectangular tile-scan output: {RECTANGULAR_REGIONS_OUTPUT}")
    print(f"Polygon mosaic output: {POLYGON_REGIONS_OUTPUT}")
    if CELLSENS_OUTPUT is not None:
        print(f"cellSens XML output: {CELLSENS_OUTPUT}")


def _write_stage_positions_csv(path: Path, positions: list[StagePosition]) -> None:
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
    path: Path, regions: list[RectangularRegion]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "center_x_um", "center_y_um", "z_um", "width_um", "height_um"])
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


def _write_polygon_regions_csv(path: Path, regions: list[PolygonRegion]) -> None:
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


if __name__ == "__main__":
    main()
