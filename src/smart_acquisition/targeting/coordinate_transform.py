"""Coordinate transforms between overview pixels and microscope stage."""

from __future__ import annotations

from dataclasses import dataclass
import math

from smart_acquisition.models import OverviewImageMetadata


@dataclass(frozen=True)
class AffinePixelToStage:
    """2D affine transform from pixel coordinates to stage micrometres."""

    a: float
    b: float
    tx: float
    c: float
    d: float
    ty: float

    def apply(self, x_px: float, y_px: float) -> tuple[float, float]:
        x_stage = self.a * x_px + self.b * y_px + self.tx
        y_stage = self.c * x_px + self.d * y_px + self.ty
        return x_stage, y_stage

    def apply_inverse(self, x_um: float, y_um: float) -> tuple[float, float]:
        """Apply the inverse affine transform from stage to pixel coordinates."""

        determinant = self.a * self.d - self.b * self.c
        if not math.isfinite(determinant) or abs(determinant) < 1e-12:
            raise ValueError("Pixel-to-stage transform is not invertible")
        x_shifted = x_um - self.tx
        y_shifted = y_um - self.ty
        x_px = (self.d * x_shifted - self.b * y_shifted) / determinant
        y_px = (-self.c * x_shifted + self.a * y_shifted) / determinant
        return x_px, y_px

    @classmethod
    def from_overview_metadata(
        cls,
        metadata: OverviewImageMetadata,
        invert_x: bool = False,
        invert_y: bool = False,
    ) -> "AffinePixelToStage":
        sign_x = -1.0 if invert_x else 1.0
        sign_y = -1.0 if invert_y else 1.0
        a = sign_x * metadata.pixel_size_x_um
        d = sign_y * metadata.pixel_size_y_um
        tx = metadata.stage_x_um
        ty = metadata.stage_y_um

        if metadata.source_position_is_center:
            tx -= a * (metadata.size_x_px - 1) / 2.0
            ty -= d * (metadata.size_y_px - 1) / 2.0

        return cls(a=a, b=0.0, tx=tx, c=0.0, d=d, ty=ty)


def overview_stage_edge_positions(
    metadata: OverviewImageMetadata,
    invert_x: bool = False,
    invert_y: bool = False,
) -> dict[str, tuple[float, float]]:
    """Return stage coordinates for the current overview image corner pixels."""

    transform = AffinePixelToStage.from_overview_metadata(
        metadata,
        invert_x=invert_x,
        invert_y=invert_y,
    )
    max_x_px = max(0.0, float(metadata.size_x_px - 1))
    max_y_px = max(0.0, float(metadata.size_y_px - 1))
    return {
        "top_left_xy_um": transform.apply(x_px=0.0, y_px=0.0),
        "top_right_xy_um": transform.apply(x_px=max_x_px, y_px=0.0),
        "bottom_left_xy_um": transform.apply(x_px=0.0, y_px=max_y_px),
        "bottom_right_xy_um": transform.apply(x_px=max_x_px, y_px=max_y_px),
    }
