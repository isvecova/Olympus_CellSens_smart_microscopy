"""Reverse-engineered cellSens Stage Navigator schema constants.

The numeric property IDs in this file are based on the example XML files in
``xml_templates``. Keep them centralized here so the writer code can describe
intent instead of scattering undocumented IDs through the implementation.
"""

from __future__ import annotations

PARAM_SET_INFO_KEY = "ParamSetInfo"
PARAM_SET_NAME = "1073741824"

STAGE_NAVIGATOR_KEY = "StageNavigator"

# Overview image geometry.
OVERVIEW_BOUNDS = "1073741825"
OVERVIEW_CENTER = "1073741888"
WELL_NAV_TOP_LEFT_POS = "401141"
WELL_NAV_TOP_RIGHT_POS = "401142"
WELL_NAV_BOTTOM_LEFT_POS = "401143"

# Ordinary point positions.
POSITION_XYZ = "1073741839"
POSITION_IDS = "1073741840"
GENERAL_OBJECT_IDS = "1073741885"
POSITION_METADATA = "1073741926"

# Position metadata.
OBJECT_ID = "1073741910"
NAME_WRAPPER = "1073741930"
NAME = "1073741923"
WELL_NAV_COMMENT = "401113"

# Parallel arrays observed for ordinary point positions.
POSITION_PARALLEL_ARRAY_DEFAULTS: tuple[tuple[str, str, str], ...] = (
    ("1073741845", "double", "0"),
    ("1073741846", "double", "0"),
    ("1073741849", "unsigned_char", "255"),
    ("1073741851", "unsigned_char", "0"),
    ("1073741868", "int", "-1"),
    ("1073741889", "unsigned_char", "0"),
    ("1073741890", "double", "0"),
    ("1073741892", "unsigned_char", "0"),
    ("1073741893", "double", "0"),
)

# Rectangular tile-scan regions. The exact interpretation of RECT_GEOMETRY is
# still experimental, so the MVP writer only clears these when requested.
RECT_REGION_ANCHOR = "1073741836"
RECT_REGION_IDS = "1073741837"
RECT_REGION_GEOMETRY = "1073741838"
RECT_REGION_METADATA = "1073741927"

RECT_REGION_PARALLEL_ARRAY_DEFAULTS: tuple[tuple[str, str, str], ...] = (
    ("1073741847", "double", "0"),
    ("1073741848", "double", "0"),
    ("1073741850", "unsigned_char", "255"),
    ("1073741852", "unsigned_char", "0"),
    ("1073741861", "unsigned_char", "255"),
    ("1073741869", "int", "-1"),
    ("1073741895", "unsigned_char", "0"),
    ("1073741896", "double", "0"),
    ("1073741898", "unsigned_char", "0"),
    ("1073741899", "double", "0"),
)

# Polygon tile-scan regions.
POLYGON_REGIONS = "1073741908"
POLYGON_VERTICES = "1073741909"
POLYGON_REGION_METADATA = "1073741929"
REGION_Z = "1073741912"
REGION_AUX = "1073741911"

# Scan-region bookkeeping nested under SCAN_BOOKKEEPING. These IDs overlap with
# some top-level Stage Navigator geometry fields, so keep the names explicit.
SCAN_BOOKKEEPING = "1073741884"
SCAN_BOOKKEEPING_REGION_IDS = "1073741879"
SCAN_BOOKKEEPING_REGION_TYPES = "1073741880"
SCAN_BOOKKEEPING_INDEXES = "1073741883"
SCAN_BOOKKEEPING_PARENT_IDS = "1073741882"
SCAN_BOOKKEEPING_ENABLED = "1073741881"
SCAN_BOOKKEEPING_FLAGS = "1073741907"

SCAN_BOOKKEEPING_ARRAY_TYPES: tuple[tuple[str, str], ...] = (
    (SCAN_BOOKKEEPING_REGION_IDS, "int"),
    (SCAN_BOOKKEEPING_REGION_TYPES, "int"),
    (SCAN_BOOKKEEPING_INDEXES, "int"),
    (SCAN_BOOKKEEPING_PARENT_IDS, "int"),
    (SCAN_BOOKKEEPING_ENABLED, "unsigned_char"),
    (SCAN_BOOKKEEPING_FLAGS, "unsigned_char"),
)

# Complex top-level Stage Navigator fields observed in populated examples.
# The MVP must preserve these from the template rather than generating them.
PRESERVE_TOP_LEVEL_NAV_GEOMETRY_FIELDS = frozenset(
    {
        "1073741875",
        "1073741876",
        "1073741877",
        "1073741878",
        "1073741879",
        "1073741880",
        "1073741887",
    }
)

ARRAY_ELEMENT_TYPES = {
    POSITION_XYZ: "CdVec3",
    POSITION_IDS: "int",
    GENERAL_OBJECT_IDS: "int",
    POSITION_METADATA: "CPropSet",
    RECT_REGION_ANCHOR: "CdVec3",
    RECT_REGION_IDS: "int",
    RECT_REGION_GEOMETRY: "CdVec2",
    RECT_REGION_METADATA: "CPropSet",
    POLYGON_REGIONS: "CPropSet",
    POLYGON_REGION_METADATA: "CPropSet",
    **{pid: typ for pid, typ, _ in POSITION_PARALLEL_ARRAY_DEFAULTS},
    **{pid: typ for pid, typ, _ in RECT_REGION_PARALLEL_ARRAY_DEFAULTS},
}
