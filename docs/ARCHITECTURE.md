# Project Architecture

This project converts low-magnification Olympus VSI overview images into
CellSens Stage Navigator targets for high-magnification acquisition.

The stable workflow is:

1. Load one VSI overview or z-stack.
2. Build the 2D processing image and, for z-stacks, an argmax-Z image.
3. Run a swappable detector to produce a positive mask and component table.
4. Optionally let the user filter detections in napari.
5. Convert filtered detections from image pixels into stage micrometres.
6. Plan point targets, rectangular tile scans, or polygon mosaic ROIs.
7. Save QC files, JSON target plans, CSV summaries, and CellSens XML.

## Main Layers

`smart_acquisition.workflow` is the orchestration layer. It owns the stable
workflow and calls the other layers through small contracts. Most scripts and
GUIs should call this module rather than reimplementing the sequence.

`smart_acquisition.detection` is the swappable detection layer. A detector only
needs to implement `ObjectDetector.detect(image, context)` and return a
`DetectionResult`. The rest of the workflow does not need to know whether the
mask came from thresholding, a random forest, or a future model.

`smart_acquisition.image_input` reads VSI files and prepares the processing
image. Z-stacks are projected to a 2D maximum-intensity image, and argmax-Z data
is retained so planning can assign per-ROI or per-tile Z values.

`smart_acquisition.targeting` converts selected masks/components into target
geometry. This is where planning choices are interpreted: output type, grouping,
Z handling, and tile-count optimization.

`smart_acquisition.cellsens` serializes target plans into CellSens-compatible
XML. It edits a known-good XML template instead of trying to generate the full
CellSens document from scratch.

`smart_acquisition.interactive_review` bridges planning to napari. It creates
preview geometry and, when explicitly requested, can choose a compact subset of
objects that minimizes tile count.

## Extension Points

### Add A New Detector

Create a class with:

```python
def detect(self, image: np.ndarray, context: DetectionContext) -> DetectionResult:
    ...
```

Return at least:

- `mask`: boolean selected positive pixels in processing-image coordinates.
- `measurements`: component measurements for the selected objects.

Return `labels` and `positive_probability` when available. They improve
planning, previews, and QC outputs, but they are not required for a minimal
detector.

### Change Planning Behavior

Planning behavior is composed from independent choices in
`smart_acquisition.planning_modes`:

- output: point, tile grid, rectangle, or polygon;
- grouping: none, XY, or XY+Z;
- Z handling: overview, per ROI, or per tile;
- optimization: direct or tile-count optimized.

The actual implementation is in
`smart_acquisition.targeting.polygon_regions.plan_composed_targets_from_mask`.
Despite the filename, this function handles all composed target outputs.

### Change CellSens Export

CellSens XML details are isolated in `smart_acquisition.cellsens`. The writer
updates parallel arrays identified in `schema.py`, validates array lengths and
IDs, and preserves unknown XML content from the template.

## Performance Notes

The slowest planning mode is usually polygon output with tile-count
optimization. The optimizer avoids unnecessary work by:

- partitioning candidates into spatial/Z pools before greedy merging;
- caching group bounds and tile-count estimates;
- using cheap bounding-box estimates during merge scoring;
- cropping label masks to component bounds for exact detection tile counts;
- generating exact polygon geometry only for final targets.

`Update tile preview` should only rebuild the current target preview. The slower
object-count subset search belongs to the separate count-optimization action.
