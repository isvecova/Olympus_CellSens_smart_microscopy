"""Visualize generated cellSens Stage Navigator targets and tile footprints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from smart_acquisition.cellsens import schema
from smart_acquisition.models import AcquisitionTile


@dataclass(frozen=True)
class PointTarget:
    x_um: float
    y_um: float
    z_um: float
    name: str


@dataclass(frozen=True)
class RectangularTarget:
    center_x_um: float
    center_y_um: float
    z_um: float
    width_um: float
    height_um: float
    name: str


@dataclass(frozen=True)
class PolygonTarget:
    vertices_xy_um: list[tuple[float, float]]
    z_um: float
    name: str


@dataclass(frozen=True)
class CellSensTargets:
    points: list[PointTarget]
    rectangles: list[RectangularTarget]
    polygons: list[PolygonTarget]
    overview_bounds: tuple[float, float, float, float] | None


def visualize_cellsens_tiles(
    cellsens_xml: str | Path,
    *,
    tile: AcquisitionTile,
    output_path: str | Path | None = None,
    background: str | Path | None = None,
    show_tile_centers: bool = False,
    show_labels: bool = False,
    dpi: int = 180,
) -> Path:
    """Draw CellSens targets and approximate high-mag tile footprints."""

    cellsens_xml = Path(cellsens_xml)
    targets = _read_cellsens_targets(cellsens_xml)
    if output_path is None:
        output_path = cellsens_xml.with_name(f"{cellsens_xml.stem}_tiles.png")
    output_path = Path(output_path)

    _plot_targets(
        targets,
        tile=tile,
        output_path=output_path,
        background=None if background is None else Path(background),
        show_tile_centers=show_tile_centers,
        show_labels=show_labels,
        dpi=dpi,
    )
    print(
        "Targets: "
        f"{len(targets.points)} positions, "
        f"{len(targets.rectangles)} rectangles, "
        f"{len(targets.polygons)} mosaics"
    )
    print(f"Approximate high-mag tiles drawn: {_estimated_total_tiles(targets, tile)}")
    print(f"Wrote visualization: {output_path}")
    return output_path


def _read_cellsens_targets(path: Path) -> CellSensTargets:
    root = ET.parse(path).getroot()
    stage_navigator = _find_stage_navigator(root)

    point_names = _read_metadata_names(stage_navigator, schema.POSITION_METADATA)
    rect_names = _read_metadata_names(stage_navigator, schema.RECT_REGION_METADATA)
    polygon_names = _read_metadata_names(stage_navigator, schema.POLYGON_REGION_METADATA)

    points = [
        PointTarget(x_um=x, y_um=y, z_um=z, name=_name_at(point_names, index, "Position"))
        for index, (x, y, z) in enumerate(
            _read_vec3_array(stage_navigator, schema.POSITION_XYZ),
            start=1,
        )
    ]

    rect_anchors = _read_vec3_array(stage_navigator, schema.RECT_REGION_ANCHOR)
    rect_sizes = _read_vec2_array(stage_navigator, schema.RECT_REGION_GEOMETRY)
    rectangles = [
        RectangularTarget(
            center_x_um=x,
            center_y_um=y,
            z_um=z,
            width_um=width,
            height_um=height,
            name=_name_at(rect_names, index, "Rectangle"),
        )
        for index, ((x, y, z), (width, height)) in enumerate(
            zip(rect_anchors, rect_sizes),
            start=1,
        )
    ]

    polygons = [
        PolygonTarget(
            vertices_xy_um=vertices,
            z_um=z_um,
            name=_name_at(polygon_names, index, "Mosaic"),
        )
        for index, (vertices, z_um) in enumerate(
            zip(
                _read_polygon_vertices(stage_navigator),
                _read_polygon_z_values(stage_navigator),
            ),
            start=1,
        )
    ]

    return CellSensTargets(
        points=points,
        rectangles=rectangles,
        polygons=polygons,
        overview_bounds=_read_overview_bounds(stage_navigator),
    )


def _plot_targets(
    targets: CellSensTargets,
    *,
    tile: AcquisitionTile,
    output_path: Path,
    background: Path | None,
    show_tile_centers: bool,
    show_labels: bool,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as PolygonPatch
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(9, 11))
    if background is not None:
        _draw_background(ax, background, targets.overview_bounds)

    point_tiles = [
        (point.x_um, point.y_um, point.z_um, point.name) for point in targets.points
    ]
    rectangle_tiles = [
        (x_um, y_um, rectangle.z_um, rectangle.name)
        for rectangle in targets.rectangles
        for x_um, y_um in _tile_centers_for_rectangle(rectangle, tile)
    ]
    polygon_tiles = [
        (x_um, y_um, polygon.z_um, polygon.name)
        for polygon in targets.polygons
        for x_um, y_um in _tile_centers_for_polygon(polygon, tile)
    ]

    _draw_tile_rectangles(
        ax,
        point_tiles,
        tile,
        edgecolor="#1f77b4",
        facecolor="#1f77b422",
        label="Point tile",
    )
    _draw_tile_rectangles(
        ax,
        rectangle_tiles,
        tile,
        edgecolor="#ff7f0e",
        facecolor="#ff7f0e1c",
        label="Rectangle tiles",
    )
    _draw_tile_rectangles(
        ax,
        polygon_tiles,
        tile,
        edgecolor="#2ca02c",
        facecolor="#2ca02c18",
        label="Mosaic tiles",
    )

    for rectangle in targets.rectangles:
        ax.add_patch(
            Rectangle(
                (
                    rectangle.center_x_um - rectangle.width_um / 2.0,
                    rectangle.center_y_um - rectangle.height_um / 2.0,
                ),
                rectangle.width_um,
                rectangle.height_um,
                fill=False,
                edgecolor="#d62728",
                linewidth=1.5,
                linestyle="--",
            )
        )

    for polygon in targets.polygons:
        ax.add_patch(
            PolygonPatch(
                polygon.vertices_xy_um,
                closed=True,
                fill=False,
                edgecolor="#111111",
                linewidth=1.7,
            )
        )

    if show_tile_centers:
        all_tiles = point_tiles + rectangle_tiles + polygon_tiles
        if all_tiles:
            ax.scatter(
                [tile_info[0] for tile_info in all_tiles],
                [tile_info[1] for tile_info in all_tiles],
                s=12,
                c="#111111",
                zorder=5,
            )

    if show_labels:
        for index, point in enumerate(targets.points, start=1):
            ax.text(point.x_um, point.y_um, str(index), fontsize=7, color="#1f77b4")
        for index, polygon in enumerate(targets.polygons, start=1):
            center_x, center_y = _polygon_centroid(polygon.vertices_xy_um)
            ax.text(center_x, center_y, f"M{index}", fontsize=8, color="#111111")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Stage X (um)")
    ax.set_ylabel("Stage Y (um)")
    ax.set_title(
        "cellSens targets: "
        f"{len(targets.points)} positions, "
        f"{len(targets.rectangles)} rectangles, "
        f"{len(targets.polygons)} mosaics"
    )
    ax.grid(True, color="0.88", linewidth=0.6)
    _autoscale_axes(ax, targets, tile)
    _deduplicate_legend(ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _draw_background(
    ax: object,
    background: Path,
    overview_bounds: tuple[float, float, float, float] | None,
) -> None:
    if background.suffix.lower() in (".tif", ".tiff"):
        try:
            import tifffile
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Reading TIFF backgrounds requires tifffile."
            ) from exc

        image = np.asarray(tifffile.imread(background))
    else:
        try:
            import matplotlib.image as mpimg
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Reading non-TIFF backgrounds requires matplotlib."
            ) from exc
        image = np.asarray(mpimg.imread(background))

    if image.ndim > 2 and image.shape[-1] == 1:
        image = np.squeeze(image)
    if image.ndim not in (2, 3):
        raise ValueError(f"Expected a 2D background image, got shape {image.shape}")

    kwargs = {"alpha": 0.35, "origin": "lower"}
    if image.ndim == 2:
        kwargs["cmap"] = "gray"
    if overview_bounds is not None:
        left, bottom, right, top = overview_bounds
        kwargs["extent"] = (left, right, bottom, top)
    ax.imshow(image, **kwargs)


def _estimated_total_tiles(targets: CellSensTargets, tile: AcquisitionTile) -> int:
    return (
        len(targets.points)
        + sum(
            len(_tile_centers_for_rectangle(rectangle, tile))
            for rectangle in targets.rectangles
        )
        + sum(
            len(_tile_centers_for_polygon(polygon, tile))
            for polygon in targets.polygons
        )
    )


def _draw_tile_rectangles(
    ax: object,
    tile_centers: list[tuple[float, float, float, str]],
    tile: AcquisitionTile,
    *,
    edgecolor: str,
    facecolor: str,
    label: str,
) -> None:
    from matplotlib.patches import Rectangle

    for x_um, y_um, _z_um, _name in tile_centers:
        ax.add_patch(
            Rectangle(
                (x_um - tile.width_um / 2.0, y_um - tile.height_um / 2.0),
                tile.width_um,
                tile.height_um,
                edgecolor=edgecolor,
                facecolor=facecolor,
                linewidth=0.8,
                label=label,
            )
        )


def _tile_centers_for_rectangle(
    rectangle: RectangularTarget,
    tile: AcquisitionTile,
) -> list[tuple[float, float]]:
    count_x = _tile_count_for_span(rectangle.width_um, tile.width_um, tile.step_x_um)
    count_y = _tile_count_for_span(rectangle.height_um, tile.height_um, tile.step_y_um)
    span_x = max(0.0, (count_x - 1) * tile.step_x_um)
    span_y = max(0.0, (count_y - 1) * tile.step_y_um)
    start_x = rectangle.center_x_um - span_x / 2.0
    start_y = rectangle.center_y_um - span_y / 2.0
    return [
        (start_x + x_index * tile.step_x_um, start_y + y_index * tile.step_y_um)
        for y_index in range(count_y)
        for x_index in range(count_x)
    ]


def _tile_centers_for_polygon(
    polygon: PolygonTarget,
    tile: AcquisitionTile,
) -> list[tuple[float, float]]:
    vertices = np.asarray(polygon.vertices_xy_um, dtype=float)
    if vertices.ndim != 2 or vertices.shape[0] < 3 or vertices.shape[1] != 2:
        return []

    min_x = float(np.min(vertices[:, 0]))
    max_x = float(np.max(vertices[:, 0]))
    min_y = float(np.min(vertices[:, 1]))
    max_y = float(np.max(vertices[:, 1]))
    width = max_x - min_x
    height = max_y - min_y
    count_x = _tile_count_for_span(width, tile.width_um, tile.step_x_um)
    count_y = _tile_count_for_span(height, tile.height_um, tile.step_y_um)
    span_x = max(0.0, (count_x - 1) * tile.step_x_um)
    span_y = max(0.0, (count_y - 1) * tile.step_y_um)
    start_x = (min_x + max_x) / 2.0 - span_x / 2.0
    start_y = (min_y + max_y) / 2.0 - span_y / 2.0

    candidates = [
        (start_x + x_index * tile.step_x_um, start_y + y_index * tile.step_y_um)
        for y_index in range(count_y)
        for x_index in range(count_x)
    ]
    if not candidates:
        return []

    selected = [
        center
        for center in candidates
        if _tile_footprint_intersects_polygon(center, tile, vertices)
    ]
    if selected:
        return selected

    center = _polygon_centroid(polygon.vertices_xy_um)
    return [center]


def _tile_footprint_intersects_polygon(
    center_xy_um: tuple[float, float],
    tile: AcquisitionTile,
    vertices: np.ndarray,
) -> bool:
    center_x, center_y = center_xy_um
    left = center_x - tile.width_um / 2.0
    right = center_x + tile.width_um / 2.0
    bottom = center_y - tile.height_um / 2.0
    top = center_y + tile.height_um / 2.0

    rectangle = np.asarray(
        [
            [left, bottom],
            [right, bottom],
            [right, top],
            [left, top],
        ],
        dtype=float,
    )

    if np.any(_points_in_polygon(rectangle, vertices)):
        return True
    if any(_point_in_rectangle(vertex, left, right, bottom, top) for vertex in vertices):
        return True

    rectangle_edges = tuple(_polygon_edges(rectangle))
    return any(
        _segments_intersect(poly_start, poly_end, rect_start, rect_end)
        for poly_start, poly_end in _polygon_edges(vertices)
        for rect_start, rect_end in rectangle_edges
    )


def _point_in_rectangle(
    point: np.ndarray,
    left: float,
    right: float,
    bottom: float,
    top: float,
) -> bool:
    eps = 1e-9
    x, y = point
    return left - eps <= x <= right + eps and bottom - eps <= y <= top + eps


def _polygon_edges(vertices: np.ndarray):
    for index in range(len(vertices)):
        yield vertices[index], vertices[(index + 1) % len(vertices)]


def _segments_intersect(
    a_start: np.ndarray,
    a_end: np.ndarray,
    b_start: np.ndarray,
    b_end: np.ndarray,
) -> bool:
    o1 = _orientation(a_start, a_end, b_start)
    o2 = _orientation(a_start, a_end, b_end)
    o3 = _orientation(b_start, b_end, a_start)
    o4 = _orientation(b_start, b_end, a_end)

    if o1 == 0 and _point_on_segment(b_start, a_start, a_end):
        return True
    if o2 == 0 and _point_on_segment(b_end, a_start, a_end):
        return True
    if o3 == 0 and _point_on_segment(a_start, b_start, b_end):
        return True
    if o4 == 0 and _point_on_segment(a_end, b_start, b_end):
        return True

    return o1 != o2 and o3 != o4


def _orientation(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> int:
    value = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if abs(value) <= 1e-9:
        return 0
    return 1 if value > 0 else -1


def _point_on_segment(
    point: np.ndarray,
    segment_start: np.ndarray,
    segment_end: np.ndarray,
) -> bool:
    eps = 1e-9
    return (
        min(segment_start[0], segment_end[0]) - eps
        <= point[0]
        <= max(segment_start[0], segment_end[0]) + eps
        and min(segment_start[1], segment_end[1]) - eps
        <= point[1]
        <= max(segment_start[1], segment_end[1]) + eps
    )


def _points_in_polygon(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    polygon_x = vertices[:, 0]
    polygon_y = vertices[:, 1]
    inside = np.zeros(points.shape[0], dtype=bool)
    previous = len(vertices) - 1
    for current in range(len(vertices)):
        yi = polygon_y[current]
        yj = polygon_y[previous]
        xi = polygon_x[current]
        xj = polygon_x[previous]
        crosses_y = (yi > y) != (yj > y)
        x_intersection = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        inside ^= crosses_y & (x < x_intersection)
        previous = current
    return inside


def _tile_count_for_span(span_um: float, tile_size_um: float, step_um: float) -> int:
    remaining_after_first_tile = max(0.0, span_um - tile_size_um)
    return 1 + int(np.ceil(remaining_after_first_tile / step_um))


def _autoscale_axes(
    ax: plt.Axes,
    targets: CellSensTargets,
    tile: AcquisitionTile,
) -> None:
    xs: list[float] = []
    ys: list[float] = []

    if targets.overview_bounds is not None:
        left, bottom, right, top = targets.overview_bounds
        xs.extend((left, right))
        ys.extend((bottom, top))

    for point in targets.points:
        xs.extend((point.x_um - tile.width_um / 2.0, point.x_um + tile.width_um / 2.0))
        ys.extend((point.y_um - tile.height_um / 2.0, point.y_um + tile.height_um / 2.0))
    for rectangle in targets.rectangles:
        xs.extend((
            rectangle.center_x_um - rectangle.width_um / 2.0,
            rectangle.center_x_um + rectangle.width_um / 2.0,
        ))
        ys.extend((
            rectangle.center_y_um - rectangle.height_um / 2.0,
            rectangle.center_y_um + rectangle.height_um / 2.0,
        ))
    for polygon in targets.polygons:
        xs.extend(x for x, _y in polygon.vertices_xy_um)
        ys.extend(y for _x, y in polygon.vertices_xy_um)

    if not xs or not ys:
        return

    margin = max(tile.width_um, tile.height_um) * 0.75
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)


def _deduplicate_legend(ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    unique: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    if unique:
        ax.legend(unique.values(), unique.keys(), loc="best", fontsize=8)


def _find_stage_navigator(root: ET.Element) -> ET.Element:
    for property_set in root.iter("PropertySet"):
        if property_set.attrib.get("Key") == schema.STAGE_NAVIGATOR_KEY:
            return property_set
    raise ValueError(f"XML does not contain PropertySet Key={schema.STAGE_NAVIGATOR_KEY}")


def _find_property(parent: ET.Element, property_id: str) -> ET.Element | None:
    for prop in parent.findall("Property"):
        if prop.attrib.get("ID") == property_id:
            return prop
    return None


def _read_vec3_array(
    stage_navigator: ET.Element,
    property_id: str,
) -> list[tuple[float, float, float]]:
    prop = _find_property(stage_navigator, property_id)
    if prop is None:
        return []

    values: list[tuple[float, float, float]] = []
    for component in prop.findall("Component"):
        vec = component.find("CdVec3")
        if vec is None:
            continue
        doubles = [float(child.text) for child in vec.findall("double")]
        if len(doubles) == 3:
            values.append((doubles[0], doubles[1], doubles[2]))
    return values


def _read_vec2_array(
    stage_navigator: ET.Element,
    property_id: str,
) -> list[tuple[float, float]]:
    prop = _find_property(stage_navigator, property_id)
    if prop is None:
        return []

    values: list[tuple[float, float]] = []
    for component in prop.findall("Component"):
        vec = component.find("CdVec2")
        if vec is None:
            continue
        doubles = [float(child.text) for child in vec.findall("double")]
        if len(doubles) == 2:
            values.append((doubles[0], doubles[1]))
    return values


def _read_polygon_vertices(
    stage_navigator: ET.Element,
) -> list[list[tuple[float, float]]]:
    prop = _find_property(stage_navigator, schema.POLYGON_REGIONS)
    if prop is None:
        return []

    all_vertices: list[list[tuple[float, float]]] = []
    for component in prop.findall("Component"):
        property_set = component.find("PropertySet")
        if property_set is None:
            continue
        vertices_prop = _find_property(property_set, schema.POLYGON_VERTICES)
        if vertices_prop is None:
            continue
        vertices: list[tuple[float, float]] = []
        for vertex_component in vertices_prop.findall("Component"):
            vec = vertex_component.find("CdVec2")
            if vec is None:
                continue
            doubles = [float(child.text) for child in vec.findall("double")]
            if len(doubles) == 2:
                vertices.append((doubles[0], doubles[1]))
        if vertices:
            all_vertices.append(vertices)
    return all_vertices


def _read_polygon_z_values(stage_navigator: ET.Element) -> list[float]:
    prop = _find_property(stage_navigator, schema.POLYGON_REGIONS)
    if prop is None:
        return []

    z_values: list[float] = []
    for component in prop.findall("Component"):
        property_set = component.find("PropertySet")
        if property_set is None:
            continue
        z_prop = _find_property(property_set, schema.REGION_Z)
        if z_prop is None:
            z_values.append(0.0)
            continue
        values = [float(child.text) for child in z_prop if child.text is not None]
        z_values.append(values[0] if values else 0.0)
    return z_values


def _read_metadata_names(stage_navigator: ET.Element, property_id: str) -> list[str]:
    prop = _find_property(stage_navigator, property_id)
    if prop is None:
        return []

    names: list[str] = []
    for component in prop.findall("Component"):
        name_prop = component.find(
            f".//Property[@ID='{schema.NAME}']",
        )
        names.append(name_prop[0].text if name_prop is not None and len(name_prop) else "")
    return names


def _name_at(names: list[str], index: int, fallback_prefix: str) -> str:
    if index <= len(names) and names[index - 1]:
        return names[index - 1]
    return f"{fallback_prefix} {index}"


def _read_overview_bounds(
    stage_navigator: ET.Element,
) -> tuple[float, float, float, float] | None:
    prop = _find_property(stage_navigator, schema.OVERVIEW_BOUNDS)
    if prop is None:
        return None
    rect = prop.find("CdRect")
    if rect is None:
        return None
    values = [float(child.text) for child in rect.findall("double")]
    if len(values) != 4:
        return None
    left, bottom, right, top = values
    return left, bottom, right, top


def _polygon_centroid(vertices_xy_um: list[tuple[float, float]]) -> tuple[float, float]:
    vertices = np.asarray(vertices_xy_um, dtype=float)
    return float(np.mean(vertices[:, 0])), float(np.mean(vertices[:, 1]))

