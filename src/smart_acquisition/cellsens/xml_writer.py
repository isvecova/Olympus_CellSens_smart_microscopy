"""Minimum viable cellSens Stage Navigator XML writer.

This module intentionally edits a valid cellSens XML file as a template. It
does not attempt to generate a complete configuration from scratch.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

from smart_acquisition.cellsens import schema
from smart_acquisition.models import PolygonRegion, RectangularRegion, StagePosition


DEFAULT_CELLSENS_TEMPLATE = Path(__file__).resolve().parents[3] / "xml_templates" / "overview_before_scan.xml"


class CellSensXmlError(RuntimeError):
    """Raised when a template or generated XML is structurally invalid."""


class CellSensStageNavigatorWriter:
    """Edit Stage Navigator targets in a cellSens XML template."""

    def __init__(self, template_path: str | Path):
        self.template_path = Path(template_path)
        self.tree = ET.parse(self.template_path)
        self.root = self.tree.getroot()
        self.param_set_info = self._find_param_set_info()
        self.stage_navigator = self._find_stage_navigator()

    def clear_generated_targets(self) -> None:
        """Clear known target arrays while preserving unknown Stage Navigator XML."""

        for property_id in (
            schema.POSITION_XYZ,
            schema.POSITION_IDS,
            schema.POSITION_METADATA,
            schema.GENERAL_OBJECT_IDS,
            schema.RECT_REGION_ANCHOR,
            schema.RECT_REGION_IDS,
            schema.RECT_REGION_GEOMETRY,
            schema.RECT_REGION_METADATA,
            schema.POLYGON_REGIONS,
            schema.POLYGON_REGION_METADATA,
        ):
            self._set_empty_array_property(
                property_id, schema.ARRAY_ELEMENT_TYPES[property_id]
            )

        for property_id, value_type, _default in schema.POSITION_PARALLEL_ARRAY_DEFAULTS:
            self._set_empty_array_property(property_id, value_type)

        for property_id, value_type, _default in schema.RECT_REGION_PARALLEL_ARRAY_DEFAULTS:
            self._set_empty_array_property(property_id, value_type)

        self._clear_scan_bookkeeping()

    def add_positions(self, positions: Iterable[StagePosition]) -> None:
        """Replace ordinary point positions with ``positions``."""

        position_list = list(positions)
        self._validate_position_values(position_list)
        self.clear_generated_targets()
        self._write_position_arrays(position_list, start_id=1)
        self._set_scalar_array(
            schema.GENERAL_OBJECT_IDS, "int", range(1, len(position_list) + 1)
        )
        self.validate_positions(expected_count=len(position_list))

    def add_rectangular_regions(
        self, regions: Iterable[RectangularRegion]
    ) -> None:
        """Replace targets with rectangular tile-scan regions only."""

        region_list = list(regions)
        self._validate_rectangular_region_values(region_list)
        self.clear_generated_targets()
        self._write_rectangular_region_arrays(region_list, start_id=1)
        self._set_scalar_array(
            schema.GENERAL_OBJECT_IDS, "int", range(1, len(region_list) + 1)
        )
        self.validate_rectangular_regions(expected_count=len(region_list))

    def add_polygon_regions(self, regions: Iterable[PolygonRegion]) -> None:
        """Replace targets with polygon tile-scan regions only."""

        region_list = list(regions)
        self._validate_polygon_region_values(region_list)
        self.clear_generated_targets()
        self._write_polygon_region_arrays(region_list, start_id=1)
        self._set_scalar_array(
            schema.GENERAL_OBJECT_IDS, "int", range(1, len(region_list) + 1)
        )
        self._set_scan_bookkeeping(list(range(1, len(region_list) + 1)))
        self.validate_polygon_regions(expected_count=len(region_list))

    def replace_targets(
        self,
        positions: Iterable[StagePosition] = (),
        rectangular_regions: Iterable[RectangularRegion] = (),
        polygon_regions: Iterable[PolygonRegion] = (),
    ) -> None:
        """Replace targets with point positions, rectangle scans, and polygons."""

        position_list = list(positions)
        rect_list = list(rectangular_regions)
        polygon_list = list(polygon_regions)
        self._validate_position_values(position_list)
        self._validate_rectangular_region_values(rect_list)
        self._validate_polygon_region_values(polygon_list)
        self.clear_generated_targets()

        self._write_position_arrays(position_list, start_id=1)
        first_rect_id = len(position_list) + 1
        rect_ids = self._write_rectangular_region_arrays(
            rect_list,
            start_id=first_rect_id,
            update_bookkeeping=False,
        )
        first_polygon_id = len(position_list) + len(rect_list) + 1
        polygon_ids = self._write_polygon_region_arrays(
            polygon_list,
            start_id=first_polygon_id,
            update_bookkeeping=False,
        )

        total_count = len(position_list) + len(rect_list) + len(polygon_list)
        self._set_scalar_array(schema.GENERAL_OBJECT_IDS, "int", range(1, total_count + 1))
        self._set_scan_bookkeeping(rect_ids + polygon_ids)
        self.validate_targets(
            expected_position_count=len(position_list),
            expected_rectangular_region_count=len(rect_list),
            expected_polygon_region_count=len(polygon_list),
        )

    def _write_position_arrays(
        self, positions: list[StagePosition], start_id: int
    ) -> None:
        ids = list(range(start_id, start_id + len(positions)))

        self._set_vec3_array(
            schema.POSITION_XYZ,
            ((p.x_um, p.y_um, p.z_um) for p in positions),
        )
        self._set_scalar_array(schema.POSITION_IDS, "int", ids)

        for property_id, value_type, default in schema.POSITION_PARALLEL_ARRAY_DEFAULTS:
            self._set_scalar_array(property_id, value_type, [default] * len(positions))

        self._set_position_metadata(positions, ids)

    def _write_rectangular_region_arrays(
        self,
        regions: list[RectangularRegion],
        start_id: int,
        update_bookkeeping: bool = True,
    ) -> list[int]:
        ids = list(range(start_id, start_id + len(regions)))
        self._set_vec3_array(
            schema.RECT_REGION_ANCHOR,
            ((r.center_x_um, r.center_y_um, r.z_um) for r in regions),
        )
        self._set_scalar_array(schema.RECT_REGION_IDS, "int", ids)
        self._set_vec2_array(
            schema.RECT_REGION_GEOMETRY,
            ((r.width_um, r.height_um) for r in regions),
        )
        for property_id, value_type, default in schema.RECT_REGION_PARALLEL_ARRAY_DEFAULTS:
            self._set_scalar_array(property_id, value_type, [default] * len(regions))
        self._set_region_metadata(schema.RECT_REGION_METADATA, regions, ids)
        if update_bookkeeping:
            self._set_scan_bookkeeping(ids)
        return ids

    def _write_polygon_region_arrays(
        self,
        regions: list[PolygonRegion],
        start_id: int,
        update_bookkeeping: bool = True,
    ) -> list[int]:
        ids = list(range(start_id, start_id + len(regions)))
        self._set_polygon_regions(schema.POLYGON_REGIONS, regions, ids)
        self._set_region_metadata(schema.POLYGON_REGION_METADATA, regions, ids)
        if update_bookkeeping:
            self._set_scan_bookkeeping(ids)
        return ids

    def validate_positions(
        self,
        expected_count: int | None = None,
        check_general_object_ids: bool = True,
    ) -> None:
        """Validate the ordinary-position parallel-array representation."""

        counts = {
            schema.POSITION_XYZ: self._component_count(schema.POSITION_XYZ),
            schema.POSITION_IDS: self._component_count(schema.POSITION_IDS),
            schema.POSITION_METADATA: self._component_count(schema.POSITION_METADATA),
        }
        if check_general_object_ids:
            counts[schema.GENERAL_OBJECT_IDS] = self._component_count(
                schema.GENERAL_OBJECT_IDS
            )
        for property_id, _value_type, _default in schema.POSITION_PARALLEL_ARRAY_DEFAULTS:
            counts[property_id] = self._component_count(property_id)

        distinct_counts = set(counts.values())
        if len(distinct_counts) != 1:
            details = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
            raise CellSensXmlError(f"Position array lengths do not match: {details}")

        count = distinct_counts.pop()
        if expected_count is not None and count != expected_count:
            raise CellSensXmlError(
                f"Expected {expected_count} positions, generated {count}"
            )

        ids = self._read_scalar_array(schema.POSITION_IDS)
        expected_ids = [str(i) for i in range(1, count + 1)]
        if ids != expected_ids:
            raise CellSensXmlError(
                f"Position IDs must be 1..N; got {ids}, expected {expected_ids}"
            )

        if check_general_object_ids:
            object_ids = self._read_scalar_array(schema.GENERAL_OBJECT_IDS)
            if object_ids != expected_ids:
                raise CellSensXmlError(
                    f"Object IDs must be 1..N; got {object_ids}, expected {expected_ids}"
                )

    def validate_rectangular_regions(
        self,
        expected_count: int | None = None,
        check_scan_bookkeeping: bool = True,
    ) -> None:
        """Validate the rectangular tile-scan parallel-array representation."""

        counts = {
            schema.RECT_REGION_ANCHOR: self._component_count(
                schema.RECT_REGION_ANCHOR
            ),
            schema.RECT_REGION_IDS: self._component_count(schema.RECT_REGION_IDS),
            schema.RECT_REGION_GEOMETRY: self._component_count(
                schema.RECT_REGION_GEOMETRY
            ),
            schema.RECT_REGION_METADATA: self._component_count(
                schema.RECT_REGION_METADATA
            ),
        }
        if check_scan_bookkeeping:
            counts.update(self._scan_bookkeeping_counts())
        for property_id, _value_type, _default in schema.RECT_REGION_PARALLEL_ARRAY_DEFAULTS:
            counts[property_id] = self._component_count(property_id)

        distinct_counts = set(counts.values())
        if len(distinct_counts) != 1:
            details = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
            raise CellSensXmlError(f"Rectangular region lengths do not match: {details}")

        count = distinct_counts.pop()
        if expected_count is not None and count != expected_count:
            raise CellSensXmlError(
                f"Expected {expected_count} rectangular regions, generated {count}"
            )

        ids = self._read_scalar_array(schema.RECT_REGION_IDS)
        if len(ids) != len(set(ids)):
            raise CellSensXmlError(f"Rectangular region IDs are not unique: {ids}")

        metadata_ids = self._read_metadata_ids(schema.RECT_REGION_METADATA)
        if metadata_ids != ids:
            raise CellSensXmlError(
                f"Rectangular metadata IDs {metadata_ids} do not match "
                f"rectangle IDs {ids}"
            )

        if check_scan_bookkeeping:
            bookkeeping_ids = self._read_scan_bookkeeping_array(
                schema.SCAN_BOOKKEEPING_REGION_IDS
            )
            if bookkeeping_ids != ids:
                raise CellSensXmlError(
                    f"Scan bookkeeping IDs {bookkeeping_ids} do not match "
                    f"rectangle IDs {ids}"
                )

        for index, anchor in enumerate(
            self._read_vec3_array(schema.RECT_REGION_ANCHOR), start=1
        ):
            if not all(math.isfinite(value) for value in anchor):
                raise CellSensXmlError(
                    f"Rectangular region {index} has non-finite XYZ anchor: {anchor}"
                )

    def validate_polygon_regions(
        self,
        expected_count: int | None = None,
        check_scan_bookkeeping: bool = True,
    ) -> None:
        """Validate the polygon tile-scan representation."""

        counts = {
            schema.POLYGON_REGIONS: self._component_count(schema.POLYGON_REGIONS),
            schema.POLYGON_REGION_METADATA: self._component_count(
                schema.POLYGON_REGION_METADATA
            ),
        }
        if check_scan_bookkeeping:
            counts.update(self._scan_bookkeeping_counts())
        distinct_counts = set(counts.values())
        if len(distinct_counts) != 1:
            details = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
            raise CellSensXmlError(f"Polygon region lengths do not match: {details}")

        count = distinct_counts.pop()
        if expected_count is not None and count != expected_count:
            raise CellSensXmlError(
                f"Expected {expected_count} polygon regions, generated {count}"
            )

        ids = self._read_polygon_region_ids()
        if len(ids) != len(set(ids)):
            raise CellSensXmlError(f"Polygon region IDs are not unique: {ids}")

        metadata_ids = self._read_metadata_ids(schema.POLYGON_REGION_METADATA)
        if metadata_ids != ids:
            raise CellSensXmlError(
                f"Polygon metadata IDs {metadata_ids} do not match polygon IDs {ids}"
            )

        if check_scan_bookkeeping:
            bookkeeping_ids = self._read_scan_bookkeeping_array(
                schema.SCAN_BOOKKEEPING_REGION_IDS
            )
            if bookkeeping_ids != ids:
                raise CellSensXmlError(
                    f"Scan bookkeeping IDs {bookkeeping_ids} do not match polygon IDs {ids}"
                )

        for index, vertices in enumerate(self._read_polygon_vertices(), start=1):
            if len(vertices) < 3:
                raise CellSensXmlError(
                    f"Polygon region {index} has fewer than 3 vertices"
                )
            for vertex in vertices:
                if not all(math.isfinite(value) for value in vertex):
                    raise CellSensXmlError(
                        f"Polygon region {index} has non-finite vertex {vertex}"
                    )

    def validate_targets(
        self,
        expected_position_count: int | None = None,
        expected_rectangular_region_count: int | None = None,
        expected_polygon_region_count: int | None = None,
    ) -> None:
        """Validate mixed ordinary positions and rectangular scan regions."""

        self.validate_positions(
            expected_count=expected_position_count,
            check_general_object_ids=False,
        )
        self.validate_rectangular_regions(
            expected_count=expected_rectangular_region_count,
            check_scan_bookkeeping=False,
        )
        self.validate_polygon_regions(
            expected_count=expected_polygon_region_count,
            check_scan_bookkeeping=False,
        )
        position_count = self._component_count(schema.POSITION_XYZ)
        rect_count = self._component_count(schema.RECT_REGION_ANCHOR)
        polygon_count = self._component_count(schema.POLYGON_REGIONS)
        object_ids = self._read_scalar_array(schema.GENERAL_OBJECT_IDS)
        expected_ids = [
            str(i) for i in range(1, position_count + rect_count + polygon_count + 1)
        ]
        if object_ids != expected_ids:
            raise CellSensXmlError(
                f"Object IDs must cover all targets; got {object_ids}, "
                f"expected {expected_ids}"
            )
        expected_scan_ids = [
            str(i)
            for i in range(
                position_count + 1,
                position_count + rect_count + polygon_count + 1,
            )
        ]
        bookkeeping_ids = self._read_scan_bookkeeping_array(
            schema.SCAN_BOOKKEEPING_REGION_IDS
        )
        if bookkeeping_ids != expected_scan_ids:
            raise CellSensXmlError(
                f"Scan bookkeeping IDs must cover only scan regions; got "
                f"{bookkeeping_ids}, expected {expected_scan_ids}"
            )

    def set_configuration_name(self, name: str) -> None:
        """Set the visible cellSens configuration name in ParamSetInfo."""

        prop = self._get_or_create_param_set_property(schema.PARAM_SET_NAME)
        self._clear_property(prop)
        value = ET.SubElement(prop, "CString")
        value.text = name

    def set_overview_stage_edges(
        self,
        *,
        top_left_xy_um: tuple[float, float],
        top_right_xy_um: tuple[float, float],
        bottom_left_xy_um: tuple[float, float],
        bottom_right_xy_um: tuple[float, float] | None = None,
    ) -> None:
        """Patch the Stage Navigator overview bounds from image edge positions.

        cellSens stores three named edge positions for overview calibration. In
        the observed XML, property ``401143`` is named ``BottomRight`` but holds
        the lower-left edge vector relative to ``TopLeft`` and ``TopRight``.
        """

        points = [top_left_xy_um, top_right_xy_um, bottom_left_xy_um]
        if bottom_right_xy_um is None:
            bottom_right_xy_um = (
                top_right_xy_um[0] + bottom_left_xy_um[0] - top_left_xy_um[0],
                top_right_xy_um[1] + bottom_left_xy_um[1] - top_left_xy_um[1],
            )
        points.append(bottom_right_xy_um)
        self._validate_xy_points(points)

        self._set_vec2_property(schema.WELL_NAV_TOP_LEFT_POS, top_left_xy_um)
        self._set_vec2_property(schema.WELL_NAV_TOP_RIGHT_POS, top_right_xy_um)
        self._set_vec2_property(schema.WELL_NAV_BOTTOM_LEFT_POS, bottom_left_xy_um)

        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        left = min(xs)
        right = max(xs)
        bottom = min(ys)
        top = max(ys)
        self._set_cdrect_property(schema.OVERVIEW_BOUNDS, (left, bottom, right, top))
        self._set_vec2_property(
            schema.OVERVIEW_CENTER,
            ((left + right) / 2.0, (bottom + top) / 2.0),
        )

    def save(self, output_path: str | Path, sync_name_with_output: bool = True) -> None:
        """Write UTF-16 XML with the declaration style used by cellSens examples."""

        output_path = Path(output_path)
        if sync_name_with_output:
            self.set_configuration_name(output_path.stem)
        ET.indent(self.tree, space="\t")
        xml_body = ET.tostring(self.root, encoding="unicode", short_empty_elements=True)
        xml_text = (
            '<?xml version="1.0" encoding="UTF-16" standalone="no"?>\n'
            f"{xml_body}"
        )
        output_path.write_text(xml_text, encoding="utf-16", newline="\n")

    def _find_param_set_info(self) -> ET.Element:
        for property_set in self.root.iter("PropertySet"):
            if property_set.attrib.get("Key") == schema.PARAM_SET_INFO_KEY:
                return property_set
        raise CellSensXmlError(
            f"Template does not contain PropertySet Key={schema.PARAM_SET_INFO_KEY!r}"
        )

    def _find_stage_navigator(self) -> ET.Element:
        for property_set in self.root.iter("PropertySet"):
            if property_set.attrib.get("Key") == schema.STAGE_NAVIGATOR_KEY:
                return property_set
        raise CellSensXmlError(
            f"Template does not contain PropertySet Key={schema.STAGE_NAVIGATOR_KEY!r}"
        )

    def _find_param_set_property(self, property_id: str) -> ET.Element | None:
        for prop in self.param_set_info.findall("Property"):
            if prop.attrib.get("ID") == property_id:
                return prop
        return None

    def _get_or_create_param_set_property(self, property_id: str) -> ET.Element:
        prop = self._find_param_set_property(property_id)
        if prop is not None:
            return prop

        prop = ET.SubElement(self.param_set_info, "Property", {"ID": property_id})
        return prop

    def _find_property(self, property_id: str) -> ET.Element | None:
        for prop in self.stage_navigator.findall("Property"):
            if prop.attrib.get("ID") == property_id:
                return prop
        return None

    def _get_or_create_property(self, property_id: str) -> ET.Element:
        prop = self._find_property(property_id)
        if prop is not None:
            return prop

        prop = ET.SubElement(self.stage_navigator, "Property", {"ID": property_id})
        return prop

    def _clear_property(self, prop: ET.Element) -> None:
        for child in list(prop):
            prop.remove(child)

    def _set_empty_array_property(self, property_id: str, empty_type: str) -> None:
        prop = self._get_or_create_property(property_id)
        prop.set("IsArray", "true")
        self._clear_property(prop)
        ET.SubElement(prop, "Empty", {"Type": empty_type})

    def _set_scalar_array(
        self, property_id: str, value_type: str, values: Iterable[object]
    ) -> None:
        values = list(values)
        if not values:
            self._set_empty_array_property(property_id, value_type)
            return

        prop = self._get_or_create_property(property_id)
        prop.set("IsArray", "true")
        self._clear_property(prop)
        for value in values:
            component = ET.SubElement(prop, "Component")
            child = ET.SubElement(component, value_type)
            child.text = str(value)

    def _set_vec3_array(
        self, property_id: str, values: Iterable[tuple[float, float, float]]
    ) -> None:
        values = list(values)
        if not values:
            self._set_empty_array_property(property_id, "CdVec3")
            return

        prop = self._get_or_create_property(property_id)
        prop.set("IsArray", "true")
        self._clear_property(prop)
        for x_value, y_value, z_value in values:
            component = ET.SubElement(prop, "Component")
            vec = ET.SubElement(component, "CdVec3")
            for value in (x_value, y_value, z_value):
                child = ET.SubElement(vec, "double")
                child.text = repr(float(value))

    def _set_vec2_array(
        self, property_id: str, values: Iterable[tuple[float, float]]
    ) -> None:
        values = list(values)
        if not values:
            self._set_empty_array_property(property_id, "CdVec2")
            return

        prop = self._get_or_create_property(property_id)
        prop.set("IsArray", "true")
        self._clear_property(prop)
        for x_value, y_value in values:
            component = ET.SubElement(prop, "Component")
            vec = ET.SubElement(component, "CdVec2")
            for value in (x_value, y_value):
                child = ET.SubElement(vec, "double")
                child.text = repr(float(value))

    def _set_vec2_property(
        self,
        property_id: str,
        value: tuple[float, float],
    ) -> None:
        prop = self._get_or_create_property(property_id)
        prop.attrib.pop("IsArray", None)
        pname = prop.find("PName")
        pname_text = pname.text if pname is not None else None
        self._clear_property(prop)
        if pname_text is not None:
            pname = ET.SubElement(prop, "PName")
            pname.text = pname_text
        vec = ET.SubElement(prop, "CdVec2")
        for item in value:
            child = ET.SubElement(vec, "double")
            child.text = repr(float(item))

    def _set_cdrect_property(
        self,
        property_id: str,
        value: tuple[float, float, float, float],
    ) -> None:
        prop = self._get_or_create_property(property_id)
        prop.attrib.pop("IsArray", None)
        self._clear_property(prop)
        rect = ET.SubElement(prop, "CdRect")
        for item in value:
            child = ET.SubElement(rect, "double")
            child.text = repr(float(item))

    def _set_polygon_regions(
        self,
        property_id: str,
        regions: list[PolygonRegion],
        ids: list[int],
    ) -> None:
        if not regions:
            self._set_empty_array_property(property_id, schema.ARRAY_ELEMENT_TYPES[property_id])
            return

        prop = self._get_or_create_property(property_id)
        prop.set("IsArray", "true")
        self._clear_property(prop)

        for region, object_id in zip(regions, ids):
            component = ET.SubElement(prop, "Component")
            property_set = ET.SubElement(component, "PropertySet")

            id_prop = ET.SubElement(property_set, "Property", {"ID": schema.OBJECT_ID})
            id_value = ET.SubElement(id_prop, "int")
            id_value.text = str(object_id)

            z_prop = ET.SubElement(property_set, "Property", {"ID": schema.REGION_Z})
            z_value = ET.SubElement(z_prop, "double")
            z_value.text = repr(float(region.z_um))

            aux_prop = ET.SubElement(property_set, "Property", {"ID": schema.REGION_AUX})
            aux_value = ET.SubElement(aux_prop, "int")
            aux_value.text = "-1"

            vertices_prop = ET.SubElement(
                property_set,
                "Property",
                {"ID": schema.POLYGON_VERTICES, "IsArray": "true"},
            )
            for x_um, y_um in region.vertices_xy_um:
                vertex_component = ET.SubElement(vertices_prop, "Component")
                vec = ET.SubElement(vertex_component, "CdVec2")
                x_value = ET.SubElement(vec, "double")
                x_value.text = repr(float(x_um))
                y_value = ET.SubElement(vec, "double")
                y_value.text = repr(float(y_um))

    def _set_position_metadata(
        self, positions: list[StagePosition], ids: list[int]
    ) -> None:
        if not positions:
            self._set_empty_array_property(
                schema.POSITION_METADATA, schema.ARRAY_ELEMENT_TYPES[schema.POSITION_METADATA]
            )
            return

        prop = self._get_or_create_property(schema.POSITION_METADATA)
        prop.set("IsArray", "true")
        self._clear_property(prop)

        for index, (position, object_id) in enumerate(zip(positions, ids), start=1):
            component = ET.SubElement(prop, "Component")
            property_set = ET.SubElement(component, "PropertySet")

            name_wrapper = ET.SubElement(
                property_set, "Property", {"ID": schema.NAME_WRAPPER}
            )
            name_set = ET.SubElement(name_wrapper, "PropertySet")
            name_prop = ET.SubElement(name_set, "Property", {"ID": schema.NAME})
            name_value = ET.SubElement(name_prop, "wstring")
            name_value.text = position.name or f"Position {index}"

            id_prop = ET.SubElement(property_set, "Property", {"ID": schema.OBJECT_ID})
            id_value = ET.SubElement(id_prop, "int")
            id_value.text = str(object_id)

    def _set_region_metadata(
        self,
        property_id: str,
        regions: list[RectangularRegion],
        ids: list[int],
    ) -> None:
        if not regions:
            self._set_empty_array_property(property_id, schema.ARRAY_ELEMENT_TYPES[property_id])
            return

        prop = self._get_or_create_property(property_id)
        prop.set("IsArray", "true")
        self._clear_property(prop)

        for index, (region, object_id) in enumerate(zip(regions, ids), start=1):
            component = ET.SubElement(prop, "Component")
            property_set = ET.SubElement(component, "PropertySet")

            name_wrapper = ET.SubElement(
                property_set, "Property", {"ID": schema.NAME_WRAPPER}
            )
            name_set = ET.SubElement(name_wrapper, "PropertySet")
            name_prop = ET.SubElement(name_set, "Property", {"ID": schema.NAME})
            name_value = ET.SubElement(name_prop, "wstring")
            name_value.text = region.name or f"Tile scan {index}"

            id_prop = ET.SubElement(property_set, "Property", {"ID": schema.OBJECT_ID})
            id_value = ET.SubElement(id_prop, "int")
            id_value.text = str(object_id)

    def _clear_scan_bookkeeping(self) -> None:
        prop = self._get_or_create_property(schema.SCAN_BOOKKEEPING)
        self._clear_property(prop)
        property_set = ET.SubElement(prop, "PropertySet")
        for property_id, value_type in schema.SCAN_BOOKKEEPING_ARRAY_TYPES:
            child_prop = ET.SubElement(
                property_set, "Property", {"ID": property_id, "IsArray": "true"}
            )
            ET.SubElement(child_prop, "Empty", {"Type": value_type})

    def _set_scan_bookkeeping(self, region_ids: list[int]) -> None:
        if not region_ids:
            self._clear_scan_bookkeeping()
            return

        prop = self._get_or_create_property(schema.SCAN_BOOKKEEPING)
        self._clear_property(prop)
        property_set = ET.SubElement(prop, "PropertySet")
        arrays = {
            schema.SCAN_BOOKKEEPING_REGION_IDS: region_ids,
            schema.SCAN_BOOKKEEPING_REGION_TYPES: [3] * len(region_ids),
            schema.SCAN_BOOKKEEPING_INDEXES: list(range(len(region_ids))),
            schema.SCAN_BOOKKEEPING_PARENT_IDS: [-1] * len(region_ids),
            schema.SCAN_BOOKKEEPING_ENABLED: [1] * len(region_ids),
            schema.SCAN_BOOKKEEPING_FLAGS: [0] * len(region_ids),
        }
        types = dict(schema.SCAN_BOOKKEEPING_ARRAY_TYPES)
        for property_id, values in arrays.items():
            child_prop = ET.SubElement(
                property_set, "Property", {"ID": property_id, "IsArray": "true"}
            )
            value_type = types[property_id]
            for value in values:
                component = ET.SubElement(child_prop, "Component")
                child = ET.SubElement(component, value_type)
                child.text = str(value)

    def _component_count(self, property_id: str) -> int:
        prop = self._find_property(property_id)
        if prop is None:
            raise CellSensXmlError(f"Missing required property {property_id}")
        return len(prop.findall("Component"))

    def _read_scalar_array(self, property_id: str) -> list[str]:
        prop = self._find_property(property_id)
        if prop is None:
            raise CellSensXmlError(f"Missing required property {property_id}")

        values: list[str] = []
        for component in prop.findall("Component"):
            children = list(component)
            if len(children) != 1 or children[0].text is None:
                raise CellSensXmlError(
                    f"Expected one scalar value per Component in {property_id}"
                )
            values.append(children[0].text)
        return values

    def _read_vec3_array(self, property_id: str) -> list[tuple[float, float, float]]:
        prop = self._find_property(property_id)
        if prop is None:
            raise CellSensXmlError(f"Missing required property {property_id}")

        values: list[tuple[float, float, float]] = []
        for component in prop.findall("Component"):
            vec = component.find("CdVec3")
            if vec is None:
                raise CellSensXmlError(
                    f"Expected CdVec3 inside Component in property {property_id}"
                )
            doubles = [float(child.text) for child in vec.findall("double")]
            if len(doubles) != 3:
                raise CellSensXmlError(
                    f"Expected three doubles in CdVec3 for property {property_id}"
                )
            values.append((doubles[0], doubles[1], doubles[2]))
        return values

    def _read_polygon_region_ids(self) -> list[str]:
        prop = self._find_property(schema.POLYGON_REGIONS)
        if prop is None:
            raise CellSensXmlError(f"Missing required property {schema.POLYGON_REGIONS}")

        ids: list[str] = []
        for component in prop.findall("Component"):
            property_set = component.find("PropertySet")
            if property_set is None:
                raise CellSensXmlError("Polygon component lacks PropertySet")
            id_prop = self._find_child_property(property_set, schema.OBJECT_ID)
            values = [child.text for child in id_prop if child.text is not None]
            if len(values) != 1:
                raise CellSensXmlError("Polygon ID must have exactly one value")
            ids.append(values[0])
        return ids

    def _read_polygon_vertices(self) -> list[list[tuple[float, float]]]:
        prop = self._find_property(schema.POLYGON_REGIONS)
        if prop is None:
            raise CellSensXmlError(f"Missing required property {schema.POLYGON_REGIONS}")

        all_vertices: list[list[tuple[float, float]]] = []
        for component in prop.findall("Component"):
            property_set = component.find("PropertySet")
            if property_set is None:
                raise CellSensXmlError("Polygon component lacks PropertySet")
            vertices_prop = self._find_child_property(
                property_set, schema.POLYGON_VERTICES
            )
            vertices: list[tuple[float, float]] = []
            for vertex_component in vertices_prop.findall("Component"):
                vec = vertex_component.find("CdVec2")
                if vec is None:
                    raise CellSensXmlError("Polygon vertex lacks CdVec2")
                doubles = [float(child.text) for child in vec.findall("double")]
                if len(doubles) != 2:
                    raise CellSensXmlError("Polygon vertex must have two doubles")
                vertices.append((doubles[0], doubles[1]))
            all_vertices.append(vertices)
        return all_vertices

    def _scan_bookkeeping_counts(self) -> dict[str, int]:
        prop = self._find_property(schema.SCAN_BOOKKEEPING)
        if prop is None:
            raise CellSensXmlError(f"Missing required property {schema.SCAN_BOOKKEEPING}")
        property_set = prop.find("PropertySet")
        if property_set is None:
            raise CellSensXmlError(
                f"Property {schema.SCAN_BOOKKEEPING} does not contain a PropertySet"
            )

        counts: dict[str, int] = {}
        for property_id, _value_type in schema.SCAN_BOOKKEEPING_ARRAY_TYPES:
            child_prop = None
            for candidate in property_set.findall("Property"):
                if candidate.attrib.get("ID") == property_id:
                    child_prop = candidate
                    break
            if child_prop is None:
                raise CellSensXmlError(
                    f"Missing scan bookkeeping property {property_id}"
                )
            counts[f"{schema.SCAN_BOOKKEEPING}/{property_id}"] = len(
                child_prop.findall("Component")
            )
        return counts

    def _read_metadata_ids(self, property_id: str) -> list[str]:
        prop = self._find_property(property_id)
        if prop is None:
            raise CellSensXmlError(f"Missing required property {property_id}")

        ids: list[str] = []
        for component in prop.findall("Component"):
            property_set = component.find("PropertySet")
            if property_set is None:
                raise CellSensXmlError(
                    f"Metadata component in {property_id} lacks PropertySet"
                )
            id_prop = self._find_child_property(property_set, schema.OBJECT_ID)
            values = [child.text for child in id_prop if child.text is not None]
            if len(values) != 1:
                raise CellSensXmlError(
                    f"Metadata object ID in {property_id} must have one value"
                )
            ids.append(values[0])
        return ids

    def _read_scan_bookkeeping_array(self, property_id: str) -> list[str]:
        prop = self._find_property(schema.SCAN_BOOKKEEPING)
        if prop is None:
            raise CellSensXmlError(f"Missing required property {schema.SCAN_BOOKKEEPING}")
        property_set = prop.find("PropertySet")
        if property_set is None:
            raise CellSensXmlError(
                f"Property {schema.SCAN_BOOKKEEPING} does not contain a PropertySet"
            )
        for child_prop in property_set.findall("Property"):
            if child_prop.attrib.get("ID") == property_id:
                values: list[str] = []
                for component in child_prop.findall("Component"):
                    children = list(component)
                    if len(children) != 1 or children[0].text is None:
                        raise CellSensXmlError(
                            f"Expected one scalar value per Component in "
                            f"{schema.SCAN_BOOKKEEPING}/{property_id}"
                        )
                    values.append(children[0].text)
                return values
        raise CellSensXmlError(f"Missing scan bookkeeping property {property_id}")

    def _find_child_property(
        self, property_set: ET.Element, property_id: str
    ) -> ET.Element:
        for candidate in property_set.findall("Property"):
            if candidate.attrib.get("ID") == property_id:
                return candidate
        raise CellSensXmlError(f"Missing child property {property_id}")

    def _validate_position_values(self, positions: list[StagePosition]) -> None:
        for index, position in enumerate(positions, start=1):
            values = (position.x_um, position.y_um, position.z_um)
            if not all(math.isfinite(value) for value in values):
                raise CellSensXmlError(
                    f"Position {index} contains non-finite coordinates: {values}"
                )

    def _validate_rectangular_region_values(
        self, regions: list[RectangularRegion]
    ) -> None:
        for index, region in enumerate(regions, start=1):
            values = (
                region.center_x_um,
                region.center_y_um,
                region.z_um,
                region.width_um,
                region.height_um,
            )
            if not all(math.isfinite(value) for value in values):
                raise CellSensXmlError(
                    f"Region {index} contains non-finite values: {values}"
                )
            if region.width_um <= 0 or region.height_um <= 0:
                raise CellSensXmlError(
                    f"Region {index} must have positive size: "
                    f"{region.width_um} x {region.height_um}"
                )

    def _validate_polygon_region_values(self, regions: list[PolygonRegion]) -> None:
        for index, region in enumerate(regions, start=1):
            if not math.isfinite(region.z_um):
                raise CellSensXmlError(
                    f"Polygon region {index} has non-finite Z: {region.z_um}"
                )
            if len(region.vertices_xy_um) < 3:
                raise CellSensXmlError(
                    f"Polygon region {index} must have at least 3 vertices"
                )
            for vertex in region.vertices_xy_um:
                if len(vertex) != 2 or not all(math.isfinite(value) for value in vertex):
                    raise CellSensXmlError(
                        f"Polygon region {index} has invalid vertex: {vertex}"
                    )

    def _validate_xy_points(self, points: list[tuple[float, float]]) -> None:
        for index, point in enumerate(points, start=1):
            if len(point) != 2 or not all(math.isfinite(value) for value in point):
                raise CellSensXmlError(
                    f"Overview edge point {index} has invalid coordinates: {point}"
                )


def write_cellsens_positions(
    template_xml: str | Path,
    output_xml: str | Path,
    positions: Iterable[StagePosition],
) -> None:
    """Write ordinary Stage Navigator positions into a cellSens XML template."""

    writer = CellSensStageNavigatorWriter(template_xml)
    writer.add_positions(positions)
    writer.save(output_xml)


def write_cellsens_rectangular_regions(
    template_xml: str | Path,
    output_xml: str | Path,
    regions: Iterable[RectangularRegion],
) -> None:
    """Write rectangular tile-scan regions into a cellSens XML template."""

    writer = CellSensStageNavigatorWriter(template_xml)
    writer.add_rectangular_regions(regions)
    writer.save(output_xml)


def write_cellsens_polygon_regions(
    template_xml: str | Path,
    output_xml: str | Path,
    regions: Iterable[PolygonRegion],
) -> None:
    """Write polygon tile-scan regions into a cellSens XML template."""

    writer = CellSensStageNavigatorWriter(template_xml)
    writer.add_polygon_regions(regions)
    writer.save(output_xml)


def write_cellsens_targets(
    template_xml: str | Path,
    output_xml: str | Path,
    positions: Iterable[StagePosition] = (),
    rectangular_regions: Iterable[RectangularRegion] = (),
    polygon_regions: Iterable[PolygonRegion] = (),
) -> None:
    """Write mixed point positions, rectangles, and polygon tile-scan regions."""

    writer = CellSensStageNavigatorWriter(template_xml)
    writer.replace_targets(
        positions=positions,
        rectangular_regions=rectangular_regions,
        polygon_regions=polygon_regions,
    )
    writer.save(output_xml)

