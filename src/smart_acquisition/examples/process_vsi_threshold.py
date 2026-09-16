"""Example BioIO VSI thresholding workflow.

Reads an Olympus VSI overview image, thresholds bright regions, writes a
positive-area mask for visual control, and writes centroid stage positions.
"""

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
from smart_acquisition.detection.threshold_detector import ThresholdDetector
from smart_acquisition.image_input.bioio_vsi import read_vsi_overview
from smart_acquisition.models import OverviewImageMetadata, StagePosition
from smart_acquisition.targeting.coordinate_transform import (
    AffinePixelToStage,
    overview_stage_edge_positions,
)
from smart_acquisition.targeting.positions import ComponentMeasurement, mask_to_centroid_positions


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
    detector = ThresholdDetector(
        threshold=args.threshold,
        percentile=args.percentile if args.threshold is None else None,
        gaussian_sigma=args.gaussian_sigma,
    )
    detection = detector.detect(read_result.image)

    transform = AffinePixelToStage.from_overview_metadata(
        read_result.metadata,
        invert_x=args.invert_x,
        invert_y=args.invert_y,
    )
    positions, measurements, cleaned_mask = mask_to_centroid_positions(
        detection.mask,
        transform=transform,
        z_um=read_result.metadata.stage_z_um,
        min_area_px=args.min_area_px,
        name_prefix=args.name_prefix,
    )

    _write_mask(args.mask_output, cleaned_mask)
    _write_positions_csv(args.positions_output, measurements)

    if args.cellsens_output or args.cellsens_template or args.use_default_cellsens_template:
        _write_cellsens_xml(args, read_result.metadata, positions)

    print(f"Read image: {args.image}")
    print(f"Image shape YX: {tuple(np.asarray(read_result.image).shape)}")
    print(f"Threshold: {detection.threshold_value}")
    print(f"Kept components: {len(positions)}")
    print(f"Mask output: {args.mask_output}")
    print(f"Positions output: {args.positions_output}")
    if args.cellsens_output:
        print(f"cellSens XML output: {args.cellsens_output}")
    return 0


def _write_cellsens_xml(
    args: Any,
    metadata: OverviewImageMetadata,
    positions: list[StagePosition],
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
    writer.add_positions(positions)
    writer.save(args.cellsens_output)


def _write_mask(path: str | Path, mask: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.asarray(mask, dtype=np.uint8) * 255)


def _write_positions_csv(
    path: str | Path, measurements: list[ComponentMeasurement]
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
                "centroid_x_px",
                "centroid_y_px",
                "source_label",
            ]
        )
        for measurement in measurements:
            position = measurement.position
            writer.writerow(
                [
                    position.name,
                    repr(float(position.x_um)),
                    repr(float(position.y_um)),
                    repr(float(position.z_um)),
                    measurement.area_px,
                    repr(float(measurement.centroid_x_px)),
                    repr(float(measurement.centroid_y_px)),
                    measurement.label,
                ]
            )


def _normalize_scene(scene: str) -> int | str:
    try:
        return int(scene)
    except ValueError:
        return scene
