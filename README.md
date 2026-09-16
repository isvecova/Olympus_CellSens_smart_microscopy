# Olympus CellSens smart microscopy position detection

This repository helps detect ROIs from low-magnification Olympus .vsi overview images and convert them into tilescan positions for subsequent high-magnification acquisition. It reads VSI overview or z-stack files, segments objects using a swappable detection method, plans point targets or tile-scan regions in microscope stage coordinates and writes CellSens Stage Navigator XML that can be loaded back into CellSens.

The XML schema has been reverse-engineered from testing overview area XML files - can be saved and loaded through Stage Navigator. The schema has been tested for Olympus CellSens 4.1.1 - if you want to use it for other version, be sure to first test it and confirm that it works. 

The current main workflow is tuned for detection of neuromuscular junctions in muscle tissue sections:

1. Read an Olympus VSI overview or z-stack with BioIO.
2. Build a 2D overview image, usually a max-intensity projection for z-stacks.
3. Detect relevant regions with a trained pixel classifier.
4. Optionally review and filter detections interactively in napari.
5. Convert overview pixel coordinates into microscope stage coordinates.
6. Plan high-magnification targets as point positions, rectangular tile scans, or irregular polygon mosaics.
7. Save QC artifacts, target-plan JSON sidecars, CSV summaries, and final CellSens XML.

## Repository §ayout

```text
src/
  run_batch_pixel_classifier_gui.py      Tkinter batch GUI for z-stack processing
  run_pixel_classifier_example.py        Editable lab script for RF-based workflow
  run_threshold_example.py               Editable lab script for threshold workflow
  convert_positions_csv_to_xml.py        Convert/edit CSV targets into CellSens XML
  visualize_cellsens_tiles.py            Visualize planned CellSens tile targets
  smart_acquisition/
    workflow.py                          Stable detector-agnostic orchestration
    outputs.py                           Shared TIFF/CSV QC writers
    batch_workflow.py                    RF compatibility wrapper over workflow.py
    interactive_review.py                napari review and tile-preview helpers
    models.py                            Shared dataclasses
    cellsens/
      schema.py                          Reverse-engineered CellSens property IDs
      xml_writer.py                      Stage Navigator XML writer/validator
      target_plan_io.py                  JSON target-plan save/combine/export
    detection/
      base.py                            Detector protocol and shared result types
      adapters.py                        Threshold and RF workflow adapters
      threshold_detector.py              Simple bright-region threshold detector
      normalized_RF_detector.py          Normalization, RF prediction, components, napari UI
    image_input/
      bioio_vsi.py                       BioIO VSI reader and z-projection cache
      resampling.py                      2D image resampling helpers
    targeting/
      coordinate_transform.py            Pixel-to-stage transforms
      positions.py                       Mask components to centroid positions
      tile_planner.py                    Point/rectangle tile planning
      polygon_regions.py                 Irregular and Z-aware mosaic planning
      z_estimation.py                    Argmax-Z based region Z estimation
xml_templates/                           Known-good CellSens XML templates/examples
```

## Core concepts

### Stage targets

The package uses a few shared dataclasses from `smart_acquisition.models`:

- `StagePosition`: one X/Y/Z stage target.
- `RectangularRegion`: native rectangular CellSens tile-scan region.
- `PolygonRegion`: irregular CellSens mosaic region.
- `OverviewImageMetadata`: image size, pixel size, stage anchor, and anchor interpretation.
- `AcquisitionTile`: high-magnification camera field of view and overlap.

These objects are the internal contract between detection, planning, review, and CellSens export.

### Coordinate conversion

`AffinePixelToStage` converts overview pixel coordinates into stage micrometres. It assumes axis-aligned overview images and uses:

- BioIO plane `position_x`, `position_y`, `position_z`.
- physical pixel sizes from BioIO, or configured fallback pixel sizes.
- optional `invert_x` and `invert_y` flags.
- `metadata_position = "origin"` or `"center"` depending on whether the BioIO plane position describes the top-left image origin or the image center.

Correct metadata and axis orientation are critical. If targets are mirrored or shifted in CellSens, check `metadata_position`, `invert_x`, `invert_y`, and the fallback pixel size settings first.

### Swappable detection

The stable workflow calls any detector that implements:

```python
def detect(image: np.ndarray, context: DetectionContext) -> DetectionResult:
    ...
```

`DetectionResult` must contain a selected positive-area `mask` and component `measurements`. Detectors can also return richer artifacts such as labels, probabilities, and normalized images for QC and Z-aware planning.

Built-in adapters:

- `ThresholdObjectDetector`: wraps the bright-region threshold detector.
- `RandomForestObjectDetector`: wraps the normalized random-forest detector and its component filtering.

To support a new project-specific detection method, implement this detector protocol and keep the rest of the workflow unchanged.

### Built-in detection modes

Threshold detection is implemented by `ThresholdDetector`. It detects bright regions with either an absolute threshold or a percentile threshold, then converts connected components to centroid positions.

Random-forest detection is implemented by `NormalizedRandomForestDetector`. It:

- normalizes the overview image to a detected muscle-intensity peak;
- extracts Gaussian, structure-tensor, and local-variance features;
- predicts labels tile-by-tile to keep memory bounded;
- records positive-class probabilities;
- measures positive connected components;
- filters by area, elongation, neighbor distance, confidence, and optional object count;
- optionally opens napari for interactive confirmation.

The RF model is expected to be a joblib bundle with keys:

```python
{
    "model": fitted_classifier,
    "feature_params": {"sigmas": [...]},
}
```

The classifier must support `predict_proba` when probability outputs or confidence filtering are used.

### Planning modes

Planning converts filtered detections into final high-magnification targets.

| Mode | Output | Typical Use |
| --- | --- | --- |
| `raw` | one point per detection | Debugging or direct centroid acquisition |
| `group_centers` | one point per nearby group | One field of view per cluster |
| `group_tiles` | grid of point positions | Explicit high-mag tile positions |
| `mixed_tilescans` | points for small groups, rectangular tile scans for larger groups | Native CellSens rectangular scans |
| `mixed_irregular_mosaics` | points for isolated detections, polygon mosaics for grouped detections | Reduce empty high-mag coverage |
| `z_aware_irregular_mosaics` | points/polygon mosaics grouped by XY and Z | Avoid merging detections at different focus planes |
| `optimized_z_aware_irregular_mosaics` | Z-aware polygons with tile-count-aware merge decisions | Default batch mode for compact mosaics |
| `z_aware_tiles` | point tiles with per-tile Z estimates | Per-tile focus instead of one Z per ROI |

The high-mag field of view comes from `AcquisitionTile`: image width/height in pixels, high-mag pixel size, and tile overlap.

## Main workflows

### Batch GUI for reviewed RF z-stacks

The easiest current workflow is:

```powershell
python src\run_batch_pixel_classifier_gui.py
```

or use the wrapper:

```powershell
run_batch_pixel_classifier_gui.bat
```

In the GUI, select:

- a CellSens XML template;
- the trained RF `.joblib` model;
- an output folder;
- an output XML path;
- one or more `.vsi` z-stacks;
- planning and filtering settings.

For each z-stack, the workflow writes:

- binary detection mask TIFF;
- raw RF labels TIFF;
- positive-class probability TIFF;
- normalization histogram PNG;
- detected component CSV;
- planned point-position CSV;
- rectangular tile-scan CSV;
- polygon mosaic CSV;
- region-confidence CSV;
- target-plan JSON;
- optional tile-scan Z histogram CSV;
- cached z-projection and argmax-Z TIFF/JSON files.

After all selected z-stacks are reviewed, the JSON target plans are combined into one CellSens XML.

Internally, the GUI still uses `BatchWorkflowConfig` for compatibility, but this now delegates to the generic workflow with a `RandomForestObjectDetector`.

### Scripted workflow

Users who do not use the GUI should call the workflow API directly from a script:

```python
from pathlib import Path

from smart_acquisition.detection.adapters import (
    ComponentFilterConfig,
    RandomForestDetectorConfig,
    RandomForestObjectDetector,
)
from smart_acquisition.models import AcquisitionTile
from smart_acquisition.workflow import (
    ReviewConfig,
    TargetPlanningConfig,
    VsiInputConfig,
    WorkflowConfig,
    process_vsi_files_to_cellsens_xml,
)

detector = RandomForestObjectDetector(
    RandomForestDetectorConfig(
        model_path=Path("model.joblib"),
        filters=ComponentFilterConfig(min_area_px=100),
    )
)

config = WorkflowConfig(
    output_dir=Path("processed"),
    input=VsiInputConfig(
        processing_pixel_size_um=1.3,
        overview_pixel_size_x_um=1.3,
        overview_pixel_size_y_um=1.3,
    ),
    review=ReviewConfig(enabled=True, show_tiling_preview=True),
    planning=TargetPlanningConfig(
        planning_mode="optimized_z_aware_irregular_mosaics",
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
    [Path("overview_1.vsi"), Path("overview_2.vsi")],
    detector,
    config,
    template_xml=Path("template.xml"),
    output_xml=Path("combined_targets.xml"),
)
```

The same workflow can use a different detector without changing image loading, review, planning, or XML writing:

```python
from smart_acquisition.detection.adapters import ThresholdDetectorConfig, ThresholdObjectDetector

detector = ThresholdObjectDetector(
    ThresholdDetectorConfig(percentile=99.5)
)
```

### CSV to XML

Use `src/convert_positions_csv_to_xml.py` when you want to manually edit detected positions before creating CellSens XML. It reads a CSV with `x_um`, `y_um`, and `z_um`, plans high-mag targets, writes planned CSVs, and exports XML.

## CellSens XML strategy

`CellSensStageNavigatorWriter` edits an existing CellSens XML file instead of generating a full configuration from scratch. This is intentional: CellSens XML contains many undocumented properties, and the writer only changes the known Stage Navigator target arrays while preserving unknown template fields.

The writer supports:

- ordinary point positions;
- rectangular tile-scan regions;
- polygon/irregular mosaic regions;
- mixed point and scan-region files;
- configuration-name synchronization with the output filename;
- patching overview edge geometry when using the bundled fallback template;
- validation of generated parallel arrays and scan-region bookkeeping.

Keep new CellSens property IDs centralized in `smart_acquisition.cellsens.schema`.

## Extending the project

### Add a detection method

Add a detector module under `smart_acquisition/detection/` and implement the `ObjectDetector` protocol. The detector should return:

- a boolean selected positive-area mask;
- component measurements with centroid pixel coordinates;
- optional labels, probabilities, or normalized image artifacts.

Then pass the detector object to `process_vsi_to_target_plan` or `process_vsi_files_to_cellsens_xml`. No changes are needed in VSI loading, review callback construction, target planning, target-plan JSON writing, or CellSens XML export.

### Add a planning mode

Add the planning implementation in `smart_acquisition/targeting/`, then update:

- `BatchWorkflowConfig.planning_mode` handling in `batch_workflow.plan_targets_from_detection`;
- `interactive_review.plan_review_targets` and `build_napari_tiling_preview` if the mode should preview in napari;
- GUI mode lists and example-script choices;
- documentation in this README.

Z-aware modes must provide or require `argmax_z` and `z_positions_um`.

### Add CellSens output features

Start by adding new property IDs to `smart_acquisition/cellsens/schema.py`. Then add narrowly scoped writer methods to `xml_writer.py` and validate generated XML against at least one known-good template from `xml_templates/`.

Do not scatter raw CellSens numeric IDs through workflow code.

## Current limitations

- CellSens XML support is reverse-engineered from provided templates. Unknown template fields are preserved, but new CellSens versions may require validation.

