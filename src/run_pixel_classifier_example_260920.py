"""Editable researcher script for the random-forest VSI workflow.

This is intentionally a script rather than a CLI. Update the paths and planning
settings below, then run it to process one or more VSI files through the generic
workflow with the RF detector adapter.
"""

from pathlib import Path

from smart_acquisition.detection.adapters import (
    ComponentFilterConfig,
    RandomForestDetectorConfig,
    RandomForestObjectDetector,
)
from smart_acquisition.models import AcquisitionTile
from smart_acquisition.planning_modes import planning_mode_from_choices
from smart_acquisition.workflow import (
    ReviewConfig,
    TargetPlanningConfig,
    VsiInputConfig,
    WorkflowConfig,
    process_vsi_files_to_cellsens_xml,
)

MODEL_PATH = Path(r"L:\0_Service\SpinSR10_autoDetect\napari_models\v3_RF_normalized_background_to_muscle_peak.joblib")
IMAGE_PATH = Path(r"N:\02_instrument_management\SpinSR10\260827_AppData_cellSens_config\260910_MK_muscle_data\10xScan_01_2nd_column.vsi")

CELLSENS_TEMPLATE = Path(r"L:\0_Service\SpinSR10_autoDetect\260827_SpinSR10_automatic_position_detection\xml_templates\260910_01_muscle_2nd_column.xml")

SHARED_OUTPUT_DIR = r"N:\02_instrument_management\SpinSR10\260827_AppData_cellSens_config\260917"
PLANNING_MODE = planning_mode_from_choices(
    output_choice="polygon_regions",
    grouping_choice="xy",
    z_choice="per_roi",
    optimization_choice="direct",
)


detector = RandomForestObjectDetector(
    RandomForestDetectorConfig(
        model_path=MODEL_PATH,
        filters=ComponentFilterConfig(min_area_px=100),
    )
)

config = WorkflowConfig(
    output_dir=Path(SHARED_OUTPUT_DIR),
    input=VsiInputConfig(
        processing_pixel_size_um=1.3,
        overview_pixel_size_x_um=1.3,
        overview_pixel_size_y_um=1.3,
    ),
    review=ReviewConfig(enabled=True, show_tiling_preview=True),
    planning=TargetPlanningConfig(
        # See docs/PLANNING_MODES.md for all available planning modes.
        planning_mode=PLANNING_MODE,
        polygon_boundary_mode="alpha_shape",
        alpha_radius_tile_fraction=0.75,
        target_tile=AcquisitionTile(
            width_px=2304,
            height_px=2304,
            pixel_size_x_um=0.325,
            pixel_size_y_um=0.325,
            overlap_fraction=0.1,
        ),
    ),
)

process_vsi_files_to_cellsens_xml(
    [IMAGE_PATH],
    detector,
    config,
    template_xml=CELLSENS_TEMPLATE,
    output_xml=Path(SHARED_OUTPUT_DIR) / "combined_targets.xml",
)
