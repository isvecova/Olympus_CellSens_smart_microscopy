"""Internal target representations used across the workflow."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StagePosition:
    """A point target in microscope-stage coordinates."""

    x_um: float
    y_um: float
    z_um: float
    name: str | None = None


@dataclass(frozen=True)
class RectangularRegion:
    """A rectangular tile-scan target in microscope-stage coordinates."""

    center_x_um: float
    center_y_um: float
    z_um: float
    width_um: float
    height_um: float
    name: str | None = None


@dataclass(frozen=True)
class PolygonRegion:
    """A polygon tile-scan target in microscope-stage coordinates."""

    vertices_xy_um: list[tuple[float, float]]
    z_um: float
    name: str | None = None


@dataclass(frozen=True)
class OverviewImageMetadata:
    """Metadata needed to convert overview pixels into stage coordinates."""

    stage_x_um: float
    stage_y_um: float
    stage_z_um: float
    pixel_size_x_um: float
    pixel_size_y_um: float
    size_x_px: int
    size_y_px: int
    source_position_is_center: bool = False


@dataclass(frozen=True)
class AcquisitionTile:
    """High-resolution acquisition field of view."""

    width_px: int
    height_px: int
    pixel_size_x_um: float
    pixel_size_y_um: float
    overlap_fraction: float = 0.0

    @property
    def width_um(self) -> float:
        return self.width_px * self.pixel_size_x_um

    @property
    def height_um(self) -> float:
        return self.height_px * self.pixel_size_y_um

    @property
    def step_x_um(self) -> float:
        return self.width_um * (1.0 - self.overlap_fraction)

    @property
    def step_y_um(self) -> float:
        return self.height_um * (1.0 - self.overlap_fraction)
