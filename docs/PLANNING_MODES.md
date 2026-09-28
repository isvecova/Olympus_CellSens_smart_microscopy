# Planning Choices

Planning converts filtered detections into CellSens acquisition targets. It is
configured by independent choices instead of preset mode names.

The composed planning key stores those choices directly:

```text
custom:<output>|<grouping>|<z_handling>|<optimization>
```

Example:

```text
custom:polygon_regions|xy_z|per_roi|tile_count
```

The workflow, napari review UI, and Tk batch GUI all use
`smart_acquisition.planning_modes` for the same labels, validation, defaults,
and tooltips.

## Choices

### Output

| Key | GUI label | Meaning |
| --- | --- | --- |
| `detection_points` | Detection points | One CellSens point at each selected detection centroid. |
| `group_center_points` | Group center points | One point at each group center. |
| `tile_point_grid` | Tile point grid | Explicit point targets laid out as high-mag tile centers. |
| `rectangular_regions` | Rectangular tile scans | Native rectangular CellSens tile-scan ROIs. |
| `polygon_regions` | Polygon mosaics | Irregular CellSens mosaic ROIs following the selected mask. |

### Grouping

| Key | GUI label | Meaning |
| --- | --- | --- |
| `none` | None | Keep every component separate. |
| `xy` | XY | Group nearby components using the high-mag tile footprint. |
| `xy_z` | XY and Z | Group only when components are close in XY and have compatible estimated Z. |

### Z Handling

| Key | GUI label | Meaning |
| --- | --- | --- |
| `overview` | Overview Z | Use the overview stage Z. |
| `per_roi` | Per ROI Z | Estimate one Z for each final ROI from valid pixels covered by its tile-scan footprint. |
| `per_tile` | Per tile Z | Estimate one Z for each high-mag tile. Requires tile point grid output. |

### Optimization

| Key | GUI label | Meaning |
| --- | --- | --- |
| `direct` | Direct | Group by the selected distance/Z rules. |
| `tile_count` | Tile-count optimized | Merge groups only when the merged target does not add too many extra tiles. |

## Polygon Boundaries

Polygon outputs have an additional boundary setting:

- `alpha_shape`: uses a Delaunay alpha shape for multi-object polygon groups.
  The alpha radius is
  `min(tile.width_um, tile.height_um) * alpha_radius_tile_fraction`.
- `mask_contour`: connected masks use their traced contour; multi-contour groups
  use a convex-hull fallback.

The planner falls back to a conservative hull or bounding box when the alpha
shape cannot be computed. When alpha-shape output produces disconnected islands,
the final CellSens output is split into separate polygon ROIs instead of
stretching one convex hull across empty space.

## Z Constraints

CellSens rectangle and polygon ROI objects store one Z for the entire ROI.
Therefore true `per_tile` Z is only representable with `tile_point_grid` output.
If `per_tile` is combined with rectangle or polygon output, the workflow raises a
clear error instead of silently changing the output.

Any choice that needs estimated Z (`xy_z`, `per_roi`, or `per_tile`) requires
z-stack argmax-Z metadata from the input VSI.

## GUI Behavior

The batch GUI and napari review GUI expose Output, Grouping, Z handling, and
Optimization separately. Changing the napari planning controls changes both the
preview and the final plan for the current image.

`Update tile preview` rebuilds the visible tile/ROI overlay for the currently
filtered objects. It does not change object count and does not run the count
optimizer. `Optimize count selection` is the slower operation that uses the
optional object-count limit to choose a compact subset.
