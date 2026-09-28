"""Composed planning-choice definitions for target generation.

Planning is configured by independent choices:

- output shape: point targets, rectangular CellSens tile scans, or polygon mosaics
- grouping: no grouping, XY grouping, or XY+Z grouping
- Z handling: overview Z, one Z per ROI, or one Z per tile
- optimization: direct grouping or tile-count-aware grouping

The composed planning key is a compact string representation of those choices,
for example ``custom:polygon_regions|xy_z|per_roi|tile_count``.
"""

from __future__ import annotations

from dataclasses import dataclass

CUSTOM_PLANNING_PREFIX = "custom:"

OUTPUT_CHOICE_LABELS: dict[str, str] = {
    "detection_points": "Detection points",
    "group_center_points": "Group center points",
    "tile_point_grid": "Tile point grid",
    "rectangular_regions": "Rectangular tile scans",
    "polygon_regions": "Polygon mosaics",
}
GROUPING_CHOICE_LABELS: dict[str, str] = {
    "none": "None",
    "xy": "XY",
    "xy_z": "XY and Z",
}
Z_CHOICE_LABELS: dict[str, str] = {
    "overview": "Overview Z",
    "per_roi": "Per ROI Z",
    "per_tile": "Per tile Z",
}
OPTIMIZATION_CHOICE_LABELS: dict[str, str] = {
    "direct": "Direct",
    "tile_count": "Tile-count optimized",
}

DEFAULT_PLANNING_MODE = (
    f"{CUSTOM_PLANNING_PREFIX}polygon_regions|xy_z|per_roi|tile_count"
)
PLANNING_MODE_KEYS: tuple[str, ...] = (DEFAULT_PLANNING_MODE,)
PLANNING_MODE_CHOICES = PLANNING_MODE_KEYS


@dataclass(frozen=True)
class PlanningStrategy:
    """Executable planning choices parsed from a composed planning key."""

    key: str
    label: str
    output_choice: str
    grouping_choice: str
    z_choice: str
    optimization_choice: str
    description: str
    output: str
    grouping: str
    z_handling: str
    speed: str

    @property
    def estimates_z_during_planning(self) -> bool:
        return self.grouping_choice == "xy_z" or self.z_choice in (
            "per_roi",
            "per_tile",
        )


def planning_mode_from_choices(
    *,
    output_choice: str,
    grouping_choice: str,
    z_choice: str,
    optimization_choice: str,
) -> str:
    """Return a composed planning key for independent planning choices."""

    _validate_choice_key(output_choice, OUTPUT_CHOICE_LABELS, "output")
    _validate_choice_key(grouping_choice, GROUPING_CHOICE_LABELS, "grouping")
    _validate_choice_key(z_choice, Z_CHOICE_LABELS, "Z handling")
    _validate_choice_key(
        optimization_choice,
        OPTIMIZATION_CHOICE_LABELS,
        "optimization",
    )
    return CUSTOM_PLANNING_PREFIX + "|".join(
        (output_choice, grouping_choice, z_choice, optimization_choice)
    )


def normalize_planning_mode(mode: str) -> str:
    """Validate and return a composed planning key."""

    _parse_custom_planning_mode(mode)
    return mode


def get_planning_strategy(mode: str) -> PlanningStrategy:
    """Parse a composed planning key into an executable strategy."""

    normalized = normalize_planning_mode(mode)
    output_choice, grouping_choice, z_choice, optimization_choice = (
        _parse_custom_planning_mode(normalized)
    )
    output_label = OUTPUT_CHOICE_LABELS[output_choice]
    grouping_label = GROUPING_CHOICE_LABELS[grouping_choice]
    z_label = Z_CHOICE_LABELS[z_choice]
    optimization_label = OPTIMIZATION_CHOICE_LABELS[optimization_choice]
    return PlanningStrategy(
        key=normalized,
        label=(
            f"{output_label}, {grouping_label}, {z_label}, "
            f"{optimization_label}"
        ),
        output_choice=output_choice,
        grouping_choice=grouping_choice,
        z_choice=z_choice,
        optimization_choice=optimization_choice,
        description="Composed planning strategy built from independent choices.",
        output=output_label,
        grouping=grouping_label,
        z_handling=z_label,
        speed=(
            "Depends on output and optimization; tile-count optimization and "
            "polygon mosaics are slower for large selections."
        ),
    )


def planning_choices_for_mode(mode: str) -> tuple[str, str, str, str]:
    """Return output, grouping, Z, and optimization choices for a mode."""

    strategy = get_planning_strategy(mode)
    return (
        strategy.output_choice,
        strategy.grouping_choice,
        strategy.z_choice,
        strategy.optimization_choice,
    )


def planning_mode_label(mode: str) -> str:
    """Return the user-facing label for a composed planning key."""

    return get_planning_strategy(mode).label


def planning_mode_tooltip(mode: str) -> str:
    """Return a compact explanation suitable for GUI tooltips."""

    strategy = get_planning_strategy(mode)
    return (
        f"{strategy.label}\n\n"
        f"{strategy.description}\n\n"
        f"Output: {strategy.output}\n"
        f"Grouping: {strategy.grouping}\n"
        f"Z handling: {strategy.z_handling}\n"
        f"Speed: {strategy.speed}"
    )


def planning_mode_help_text() -> str:
    """Return a plain-text reference for the composed planning choices."""

    return "\n\n".join(
        (
            "Planning is composed from independent choices:",
            _choice_block("Output", OUTPUT_CHOICE_LABELS),
            _choice_block("Grouping", GROUPING_CHOICE_LABELS),
            _choice_block("Z handling", Z_CHOICE_LABELS),
            _choice_block("Optimization", OPTIMIZATION_CHOICE_LABELS),
        )
    )


def _choice_block(title: str, choices: dict[str, str]) -> str:
    """Format one choice family for plain-text help."""

    lines = [title]
    lines.extend(f"  {key}: {label}" for key, label in choices.items())
    return "\n".join(lines)


def _parse_custom_planning_mode(mode: str) -> tuple[str, str, str, str]:
    """Parse and validate the compact custom planning key."""

    if not mode.startswith(CUSTOM_PLANNING_PREFIX):
        raise ValueError(
            "Planning mode must be a composed key like "
            "'custom:polygon_regions|xy_z|per_roi|tile_count'."
        )
    parts = mode[len(CUSTOM_PLANNING_PREFIX) :].split("|")
    if len(parts) != 4:
        raise ValueError(f"Invalid composed planning mode: {mode}")
    output_choice, grouping_choice, z_choice, optimization_choice = parts
    _validate_choice_key(output_choice, OUTPUT_CHOICE_LABELS, "output")
    _validate_choice_key(grouping_choice, GROUPING_CHOICE_LABELS, "grouping")
    _validate_choice_key(z_choice, Z_CHOICE_LABELS, "Z handling")
    _validate_choice_key(
        optimization_choice,
        OPTIMIZATION_CHOICE_LABELS,
        "optimization",
    )
    return output_choice, grouping_choice, z_choice, optimization_choice


def _validate_choice_key(
    key: str,
    choices: dict[str, str],
    choice_name: str,
) -> None:
    """Raise a clear error when a composed planning choice is unknown."""

    if key not in choices:
        valid = ", ".join(choices)
        raise ValueError(f"Unknown planning {choice_name} '{key}'. Use one of: {valid}.")
