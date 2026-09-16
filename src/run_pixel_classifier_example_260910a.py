"""Editable example: pixel-classify a VSI overview and export targets.

Open this file, change the values in the CONFIGURATION section, and run it in
the ``irbisPythonJupyter`` conda environment. No PowerShell script is needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from smart_acquisition.cellsens.xml_writer import (
    DEFAULT_CELLSENS_TEMPLATE,
    CellSensStageNavigatorWriter,
)
from smart_acquisition.detection.normalized_RF_detector import (
    NormalizedRandomForestDetector,
)
from smart_acquisition.examples.process_vsi_random_forest import (
    _positions_from_measurements,
    _write_labels_if_requested,
    _write_mask,
    _write_measurements_csv,
    _write_polygon_regions_csv,
    _write_probability_if_requested,
    _write_rectangular_regions_csv,
    _write_region_confidences_csv,
    _write_stage_positions_csv,
)
from smart_acquisition.image_input.bioio_vsi import read_vsi_zstack
from smart_acquisition.image_input.resampling import resample_to_pixel_size
from smart_acquisition.interactive_review import (
    ReviewPlanningOptions,
    build_napari_tiling_preview,
    select_measurements_minimizing_tiles,
)
from smart_acquisition.models import AcquisitionTile
from smart_acquisition.targeting.coordinate_transform import (
    AffinePixelToStage,
    overview_stage_edge_positions,
)
from smart_acquisition.targeting.polygon_regions import (
    plan_optimized_z_aware_positions_and_polygon_regions_from_mask,
    plan_mixed_positions_and_polygon_regions_from_mask,
    plan_z_aware_positions_and_polygon_regions_from_mask,
    plan_z_aware_tile_positions_from_mask,
)
from smart_acquisition.targeting.tile_planner import (
    plan_mixed_positions_and_tilescans,
    plan_positions_for_high_mag_tiles,
)
from smart_acquisition.targeting.z_estimation import (
    resample_argmax_z_to_shape,
    update_tile_scan_region_z_positions,
    write_region_z_histograms_csv,
)


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# IMAGE_PATH = Path(r"N:\02_instrument_management\SpinSR10\260827_AppData_cellSens_config\MK_muscle_focus_map\muscle_overview_focus_map_02.vsi")
# IMAGE_PATH = Path(r"N:\02_instrument_management\SpinSR10\260827_AppData_cellSens_config\MK_muscle_focus_map\MAX_muscle_overview_focus_map_02.tif")
IMAGE_PATH = Path(r"N:\02_instrument_management\SpinSR10\260827_AppData_cellSens_config\260910_MK_muscle_data\10xScan_01_2nd_column.vsi")

MODEL_PATH = Path(
    r"L:\0_Service\SpinSR10_autoDetect\260829_pixel_segmentation_napari\napari_models\v3_RF_normalized_background_to_muscle_peak.joblib"
)

CELLSENS_TEMPLATE = Path(r"L:\0_Service\SpinSR10_autoDetect\260827_SpinSR10_automatic_position_detection\xml_templates\260910_01_muscle_2nd_column.xml")

SHARED_OUTPUT_DIR = r"N:\02_instrument_management\SpinSR10\260827_AppData_cellSens_config\260910\06"

MASK_OUTPUT = Path(rf"{SHARED_OUTPUT_DIR}\mask.tif")
LABELS_OUTPUT = MASK_OUTPUT.with_name(f"{MASK_OUTPUT.stem}_raw_labels.tif")
PROBABILITY_OUTPUT = MASK_OUTPUT.with_name(
    f"{MASK_OUTPUT.stem}_positive_probability.tif"
)
POSITIONS_OUTPUT = Path(rf"{SHARED_OUTPUT_DIR}\pixel_classifier_positions.csv")
PLANNED_POSITIONS_OUTPUT = POSITIONS_OUTPUT.with_name(
    f"{POSITIONS_OUTPUT.stem}_planned.csv"
)
RECTANGULAR_REGIONS_OUTPUT = POSITIONS_OUTPUT.with_name(
    f"{POSITIONS_OUTPUT.stem}_rectangular_tilescans.csv"
)
POLYGON_REGIONS_OUTPUT = POSITIONS_OUTPUT.with_name(
    f"{POSITIONS_OUTPUT.stem}_polygon_mosaics.csv"
)
Z_HISTOGRAMS_OUTPUT = POSITIONS_OUTPUT.with_name(
    f"{POSITIONS_OUTPUT.stem}_tile_scan_z_histograms.csv"
)
REGION_CONFIDENCE_OUTPUT = POSITIONS_OUTPUT.with_name(
    f"{POSITIONS_OUTPUT.stem}_region_confidence.csv"
)
# Set to None to disable the normalization QC histogram.
NORMALIZATION_HISTOGRAM_OUTPUT = POSITIONS_OUTPUT.with_name(
    f"{POSITIONS_OUTPUT.stem}_normalization_histogram.png"
)

# Use the default fallback template only when the overview does not have a
# matching cellSens XML. That fallback has placeholder overview boundaries, so
# the script patches them from the current image metadata.
USE_DEFAULT_CELLSENS_TEMPLATE = True
DEFAULT_CELLSENS_TEMPLATE_PATH = DEFAULT_CELLSENS_TEMPLATE

# Used when USE_DEFAULT_CELLSENS_TEMPLATE is False. In this mode the template's
# existing overview boundaries are preserved.
CELLSENS_TEMPLATE_PATH = CELLSENS_TEMPLATE
CELLSENS_OUTPUT = Path(rf"{SHARED_OUTPUT_DIR}\positions_260910e.xml")

# BioIO image selection.
SCENE = 0
CHANNEL = 0
TIME = 0

# Some VSI exports do not expose physical pixel size through BioIO. Set these
# to the actual input image pixel size before resampling for classification.
# For an already-binned 1.3 um overview, set both values to 1.3. If BioIO
# exposes physical pixel sizes, its metadata is used before these fallbacks.
OVERVIEW_PIXEL_SIZE_X_UM = 1.3
OVERVIEW_PIXEL_SIZE_Y_UM = 1.3

# The first z-stack run can cache its max projection, per-pixel argmax-Z plane,
# and z metadata here. Later runs reuse these files without rereading the VSI.
Z_PROJECTION_CACHE_DIR = MASK_OUTPUT.parent / "z_projection_cache"
USE_Z_PROJECTION_CACHE = True
WRITE_Z_PROJECTION_CACHE = True
BUILD_Z_CACHE_AT_CLASSIFICATION_PIXEL_SIZE = True

# Use this only if BioIO does not expose per-plane Z positions. If metadata
# exposes them, those values are used instead.
FALLBACK_Z_STEP_UM = None

# Pixel-classifier prediction.
CLASSIFICATION_PIXEL_SIZE_UM = 1.3
# Zero-valued pixels are treated as non-image/padded pixels: they are excluded
# from normalization and forced to background in the RF labels.
IGNORE_ZERO_PIXELS = True
# Used only when IGNORE_ZERO_PIXELS is False.
LOWER_ABSOLUTE = 100.0
NORMALIZATION_SEARCH_MIN = None
NORMALIZATION_SEARCH_MAX = None
POSITIVE_LABEL = 2
PREDICTION_TILE_SIZE_PX = 1024
PREDICTION_OVERLAP_PX = 12

# Interactive napari review.
# The script pauses here until the user presses OK in the filter widget.
USE_NAPARI_FILTER = True

# Component filters. Elongation is 0 for round objects and approaches 1 for
# long, thin objects.
MIN_AREA_PX = 100
MAX_AREA_PX = None
MIN_ELONGATION = 0.0
MAX_ELONGATION = 1.0
MAX_MEAN_3NN_DISTANCE_UM = None
MIN_SIZE_WEIGHTED_CONFIDENCE = None

# Optional initial object-count limit in napari. Set to None or 0 to keep all
# filtered objects until the user changes the count in the napari widget.
MAX_SELECTED_OBJECTS = None
SHOW_NAPARI_TILING_PREVIEW = True

# Coordinate interpretation.
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
# "z_aware_irregular_mosaics" is like mixed_irregular_mosaics, but detections
# are grouped only when they are close in XY and Z.
# "optimized_z_aware_irregular_mosaics" additionally merges only when the
# estimated CellSens high-mag tile count is improved without too much extra
# empty coverage from a filled polygon ROI.
# "z_aware_tiles" writes high-mag tiles as ordinary positions with per-tile Z;
# detections are grouped only when they are close in XY and Z.
PLANNING_MODE = "mixed_irregular_mosaics"

# Set these to the camera image size and pixel size of the NEW high-mag
# acquisition, not the low-mag overview.
TARGET_IMAGE_WIDTH_PX = 2304
TARGET_IMAGE_HEIGHT_PX = 2304
TARGET_PIXEL_SIZE_X_UM = 0.325
TARGET_PIXEL_SIZE_Y_UM = 0.325
TARGET_TILE_OVERLAP_FRACTION = 0.1

# 1.0 means detections are grouped when their X/Y separation is within one
# high-mag tile field of view. Increase slightly, e.g. 1.2, to merge more.
GROUP_MERGE_DISTANCE_FACTOR = 1.0

# Only used for Z-aware planning modes. Detections farther apart in Z than this
# value are kept in separate tile groups/mosaics.
MAX_GROUP_Z_DIFFERENCE_UM = 10.0

# Only used for PLANNING_MODE = "optimized_z_aware_irregular_mosaics". A merge
# is rejected if the estimated merged polygon tile grid has more extra tiles
# than this fraction because CellSens polygon ROIs cannot represent holes.
MAX_EXTRA_TILE_FRACTION = 0.10

# Only used for PLANNING_MODE = "group_tiles". It expands each grouped bounding
# box before tile centers are generated.
COVERAGE_MARGIN_UM = 0.0

# Only used for PLANNING_MODE = "mixed_irregular_mosaics".
POLYGON_SIMPLIFY_TOLERANCE_UM = 10.0

# Estimate one Z coordinate per native tile-scan region from argmax-Z values
# inside classifier labels 1 and 2. Histogram CSV is useful for QC but can be
# disabled for routine use.
ESTIMATE_TILE_SCAN_Z = True
WRITE_TILE_SCAN_Z_HISTOGRAMS = True
Z_ESTIMATION_LABELS = (1, 2)


# ---------------------------------------------------------------------------
# SCRIPT
# ---------------------------------------------------------------------------


def main() -> None:
    print("Starting pixel-classifier example...")
    read_result = read_vsi_zstack(
        IMAGE_PATH,
        scene=SCENE,
        channel=CHANNEL,
        time=TIME,
        source_position_is_center=METADATA_POSITION == "center",
        fallback_pixel_size_x_um=OVERVIEW_PIXEL_SIZE_X_UM,
        fallback_pixel_size_y_um=OVERVIEW_PIXEL_SIZE_Y_UM,
        fallback_z_step_um=FALLBACK_Z_STEP_UM,
        projection_pixel_size_um=(
            CLASSIFICATION_PIXEL_SIZE_UM
            if BUILD_Z_CACHE_AT_CLASSIFICATION_PIXEL_SIZE
            else None
        ),
        cache_dir=Z_PROJECTION_CACHE_DIR,
        use_cache=USE_Z_PROJECTION_CACHE,
        write_cache=WRITE_Z_PROJECTION_CACHE,
    )
    print(f"Read image: {IMAGE_PATH}")
    if BUILD_Z_CACHE_AT_CLASSIFICATION_PIXEL_SIZE:
        classifier_image = np.asarray(read_result.image)
        classifier_metadata = read_result.metadata
        if not (
            np.isclose(classifier_metadata.pixel_size_x_um, CLASSIFICATION_PIXEL_SIZE_UM)
            and np.isclose(
                classifier_metadata.pixel_size_y_um,
                CLASSIFICATION_PIXEL_SIZE_UM,
            )
        ):
            raise ValueError(
                "Expected read_vsi_zstack to return the classifier pixel size, "
                f"but got {classifier_metadata.pixel_size_x_um:.6g} x "
                f"{classifier_metadata.pixel_size_y_um:.6g} um"
            )
    else:
        classifier_image, classifier_metadata = resample_to_pixel_size(
            read_result.image,
            read_result.metadata,
            CLASSIFICATION_PIXEL_SIZE_UM,
        )

    detector = NormalizedRandomForestDetector(
        MODEL_PATH,
        lower_absolute=LOWER_ABSOLUTE,
        normalization_search_min=NORMALIZATION_SEARCH_MIN,
        normalization_search_max=NORMALIZATION_SEARCH_MAX,
        normalization_histogram_output_path=NORMALIZATION_HISTOGRAM_OUTPUT,
        ignore_zero_pixels=IGNORE_ZERO_PIXELS,
        positive_label=POSITIVE_LABEL,
        tile_size=PREDICTION_TILE_SIZE_PX,
        overlap=PREDICTION_OVERLAP_PX,
    )

    transform = AffinePixelToStage.from_overview_metadata(
        classifier_metadata,
        invert_x=INVERT_X,
        invert_y=INVERT_Y,
    )
    target_tile = AcquisitionTile(
        width_px=TARGET_IMAGE_WIDTH_PX,
        height_px=TARGET_IMAGE_HEIGHT_PX,
        pixel_size_x_um=TARGET_PIXEL_SIZE_X_UM,
        pixel_size_y_um=TARGET_PIXEL_SIZE_Y_UM,
        overlap_fraction=TARGET_TILE_OVERLAP_FRACTION,
    )
    review_argmax_z = (
        None
        if read_result.argmax_z is None
        else resample_argmax_z_to_shape(read_result.argmax_z, classifier_image.shape)
    )
    review_options = ReviewPlanningOptions(
        planning_mode=PLANNING_MODE,
        transform=transform,
        z_um=classifier_metadata.stage_z_um,
        tile=target_tile,
        merge_distance_factor=GROUP_MERGE_DISTANCE_FACTOR,
        coverage_margin_um=COVERAGE_MARGIN_UM,
        polygon_simplify_tolerance_um=POLYGON_SIMPLIFY_TOLERANCE_UM,
        argmax_z=review_argmax_z,
        z_positions_um=read_result.z_positions_um,
        z_estimation_labels=Z_ESTIMATION_LABELS,
        max_group_z_difference_um=MAX_GROUP_Z_DIFFERENCE_UM,
        max_extra_tile_fraction=MAX_EXTRA_TILE_FRACTION,
        name_prefix=NAME_PREFIX,
    )
    preview_callback = None
    selection_callback = None
    if USE_NAPARI_FILTER:
        def selection_callback(
            measurements,
            count,
            candidate_mask,
            component_labels,
            labels,
        ):
            return select_measurements_minimizing_tiles(
                measurements,
                count,
                candidate_mask,
                component_labels,
                labels,
                options=review_options,
            )

        if SHOW_NAPARI_TILING_PREVIEW:
            def preview_callback(mask, measurements, labels):
                return build_napari_tiling_preview(
                    mask,
                    measurements,
                    labels,
                    options=review_options,
                )

    detection = detector.detect(
        classifier_image,
        min_area_px=MIN_AREA_PX,
        max_area_px=MAX_AREA_PX,
        min_elongation=MIN_ELONGATION,
        max_elongation=MAX_ELONGATION,
        max_mean_3nn_distance_um=MAX_MEAN_3NN_DISTANCE_UM,
        min_size_weighted_confidence=MIN_SIZE_WEIGHTED_CONFIDENCE,
        max_selected_objects=MAX_SELECTED_OBJECTS,
        pixel_size_x_um=classifier_metadata.pixel_size_x_um,
        pixel_size_y_um=classifier_metadata.pixel_size_y_um,
        interactive=USE_NAPARI_FILTER,
        selection_callback=selection_callback,
        preview_callback=preview_callback,
    )

    positions = _positions_from_measurements(
        detection.measurements,
        transform=transform,
        z_um=classifier_metadata.stage_z_um,
        name_prefix=NAME_PREFIX,
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
            detection.mask,
            transform=transform,
            z_um=classifier_metadata.stage_z_um,
            tile=target_tile,
            min_area_px=1,
            merge_distance_factor=GROUP_MERGE_DISTANCE_FACTOR,
            simplify_tolerance_um=POLYGON_SIMPLIFY_TOLERANCE_UM,
            name_prefix="Planned",
        )
        planned_positions = polygon_plan.point_positions
        rectangular_regions = []
        polygon_regions = polygon_plan.polygon_regions
        position_groups = []
    elif PLANNING_MODE == "z_aware_irregular_mosaics":
        if read_result.argmax_z is None or read_result.z_positions_um is None:
            raise ValueError(
                "Z-aware irregular mosaic planning requires argmax-Z and "
                "z-position metadata"
            )
        classifier_argmax_z = resample_argmax_z_to_shape(
            read_result.argmax_z,
            detection.labels.shape,
        )
        z_polygon_plan = plan_z_aware_positions_and_polygon_regions_from_mask(
            detection.mask,
            classifier_labels=detection.labels,
            argmax_z=classifier_argmax_z,
            z_positions_um=read_result.z_positions_um,
            transform=transform,
            fallback_z_um=classifier_metadata.stage_z_um,
            tile=target_tile,
            include_labels=Z_ESTIMATION_LABELS,
            min_area_px=1,
            merge_distance_factor=GROUP_MERGE_DISTANCE_FACTOR,
            max_group_z_difference_um=MAX_GROUP_Z_DIFFERENCE_UM,
            simplify_tolerance_um=POLYGON_SIMPLIFY_TOLERANCE_UM,
            name_prefix="Planned",
        )
        planned_positions = z_polygon_plan.point_positions
        rectangular_regions = []
        polygon_regions = z_polygon_plan.polygon_regions
        position_groups = z_polygon_plan.groups
    elif PLANNING_MODE == "optimized_z_aware_irregular_mosaics":
        if read_result.argmax_z is None or read_result.z_positions_um is None:
            raise ValueError(
                "Optimized Z-aware irregular mosaic planning requires argmax-Z "
                "and z-position metadata"
            )
        classifier_argmax_z = resample_argmax_z_to_shape(
            read_result.argmax_z,
            detection.labels.shape,
        )
        z_polygon_plan = plan_optimized_z_aware_positions_and_polygon_regions_from_mask(
            detection.mask,
            classifier_labels=detection.labels,
            argmax_z=classifier_argmax_z,
            z_positions_um=read_result.z_positions_um,
            transform=transform,
            fallback_z_um=classifier_metadata.stage_z_um,
            tile=target_tile,
            include_labels=Z_ESTIMATION_LABELS,
            min_area_px=1,
            merge_distance_factor=GROUP_MERGE_DISTANCE_FACTOR,
            max_group_z_difference_um=MAX_GROUP_Z_DIFFERENCE_UM,
            max_extra_tile_fraction=MAX_EXTRA_TILE_FRACTION,
            simplify_tolerance_um=POLYGON_SIMPLIFY_TOLERANCE_UM,
            name_prefix="Planned",
        )
        planned_positions = z_polygon_plan.point_positions
        rectangular_regions = []
        polygon_regions = z_polygon_plan.polygon_regions
        position_groups = z_polygon_plan.groups
    elif PLANNING_MODE == "z_aware_tiles":
        if read_result.argmax_z is None or read_result.z_positions_um is None:
            raise ValueError(
                "Z-aware tile planning requires argmax-Z and z-position metadata"
            )
        classifier_argmax_z = resample_argmax_z_to_shape(
            read_result.argmax_z,
            detection.labels.shape,
        )
        z_tile_plan = plan_z_aware_tile_positions_from_mask(
            detection.mask,
            classifier_labels=detection.labels,
            argmax_z=classifier_argmax_z,
            z_positions_um=read_result.z_positions_um,
            transform=transform,
            fallback_z_um=classifier_metadata.stage_z_um,
            tile=target_tile,
            include_labels=Z_ESTIMATION_LABELS,
            min_area_px=1,
            merge_distance_factor=GROUP_MERGE_DISTANCE_FACTOR,
            max_group_z_difference_um=MAX_GROUP_Z_DIFFERENCE_UM,
            coverage_margin_um=COVERAGE_MARGIN_UM,
            name_prefix="Planned",
        )
        planned_positions = z_tile_plan.point_positions
        rectangular_regions = []
        polygon_regions = []
        position_groups = z_tile_plan.groups
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

    z_estimates = []
    if (
        ESTIMATE_TILE_SCAN_Z
        and PLANNING_MODE
        not in ("z_aware_irregular_mosaics", "optimized_z_aware_irregular_mosaics")
        and (polygon_regions or rectangular_regions)
    ):
        if read_result.argmax_z is None or read_result.z_positions_um is None:
            raise ValueError(
                "Tile-scan Z estimation requires argmax-Z and z-position metadata"
            )
        classifier_argmax_z = resample_argmax_z_to_shape(
            read_result.argmax_z,
            detection.labels.shape,
        )
        polygon_regions, rectangular_regions, z_estimates = (
            update_tile_scan_region_z_positions(
                polygon_regions=polygon_regions,
                rectangular_regions=rectangular_regions,
                classifier_labels=detection.labels,
                argmax_z=classifier_argmax_z,
                z_positions_um=read_result.z_positions_um,
                transform=transform,
                include_labels=Z_ESTIMATION_LABELS,
            )
        )
        if WRITE_TILE_SCAN_Z_HISTOGRAMS:
            write_region_z_histograms_csv(
                Z_HISTOGRAMS_OUTPUT,
                z_estimates,
                read_result.z_positions_um,
            )

    _write_mask(MASK_OUTPUT, detection.mask)
    _write_labels_if_requested(LABELS_OUTPUT, detection.labels)
    _write_probability_if_requested(PROBABILITY_OUTPUT, detection.positive_probability)
    _write_measurements_csv(POSITIONS_OUTPUT, detection.measurements, positions)
    _write_stage_positions_csv(PLANNED_POSITIONS_OUTPUT, planned_positions)
    _write_rectangular_regions_csv(RECTANGULAR_REGIONS_OUTPUT, rectangular_regions)
    _write_polygon_regions_csv(POLYGON_REGIONS_OUTPUT, polygon_regions)
    _write_region_confidences_csv(
        REGION_CONFIDENCE_OUTPUT,
        positive_probability=detection.positive_probability,
        detection_mask=detection.mask,
        transform=transform,
        rectangular_regions=rectangular_regions,
        polygon_regions=polygon_regions,
    )

    if CELLSENS_OUTPUT is not None:
        template = (
            DEFAULT_CELLSENS_TEMPLATE_PATH
            if USE_DEFAULT_CELLSENS_TEMPLATE
            else CELLSENS_TEMPLATE_PATH
        )
        if not USE_DEFAULT_CELLSENS_TEMPLATE and template is None:
            raise ValueError("Set CELLSENS_TEMPLATE_PATH to a user-provided XML template")

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
    print(f"Classifier image shape YX: {tuple(np.asarray(classifier_image).shape)}")
    print(
        "Classifier pixel size: "
        f"{classifier_metadata.pixel_size_x_um:.6g} x "
        f"{classifier_metadata.pixel_size_y_um:.6g} um"
    )
    print(f"Stage origin/position: {read_result.metadata}")
    print(f"Model: {MODEL_PATH}")
    print(f"Kept classifier components: {len(detection.measurements)}")
    print(f"Nearby groups: {len(position_groups)}")
    print(f"Planned cellSens positions: {len(planned_positions)}")
    print(f"Planned rectangular tile scans: {len(rectangular_regions)}")
    print(f"Planned irregular mosaic tile scans: {len(polygon_regions)}")
    print(f"Tile-scan Z estimates: {len(z_estimates)}")
    print(
        "High-mag tile FOV: "
        f"{target_tile.width_um:.3f} x {target_tile.height_um:.3f} um"
    )
    print(f"Mask output: {MASK_OUTPUT}")
    print(f"Raw labels output: {LABELS_OUTPUT}")
    print(f"Positive probability output: {PROBABILITY_OUTPUT}")
    print(f"Raw positions output: {POSITIONS_OUTPUT}")
    print(f"Planned positions output: {PLANNED_POSITIONS_OUTPUT}")
    print(f"Rectangular tile-scan output: {RECTANGULAR_REGIONS_OUTPUT}")
    print(f"Polygon mosaic output: {POLYGON_REGIONS_OUTPUT}")
    if NORMALIZATION_HISTOGRAM_OUTPUT is not None:
        print(f"Normalization histogram output: {NORMALIZATION_HISTOGRAM_OUTPUT}")
    if WRITE_TILE_SCAN_Z_HISTOGRAMS:
        print(f"Tile-scan Z histogram output: {Z_HISTOGRAMS_OUTPUT}")
    print(f"Region confidence output: {REGION_CONFIDENCE_OUTPUT}")
    if CELLSENS_OUTPUT is not None:
        print(f"cellSens XML output: {CELLSENS_OUTPUT}")


if __name__ == "__main__":
    main()
