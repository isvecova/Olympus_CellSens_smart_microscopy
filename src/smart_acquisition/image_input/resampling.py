"""Image resampling helpers for detector-specific pixel sizes."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from smart_acquisition.models import OverviewImageMetadata


def resample_to_pixel_size(
    image: np.ndarray,
    metadata: OverviewImageMetadata,
    target_pixel_size_um: float | tuple[float, float] | None,
) -> tuple[np.ndarray, OverviewImageMetadata]:
    """Resample a 2D overview image to the requested Y/X pixel size.

    The returned metadata keeps the original stage anchor but updates image
    shape and physical pixel size so downstream pixel-to-stage transforms use
    the classifier image coordinate system.
    """

    if target_pixel_size_um is None:
        return np.asarray(image), metadata

    target_y_um, target_x_um = _normalize_pixel_size(target_pixel_size_um)
    if target_x_um <= 0 or target_y_um <= 0:
        raise ValueError("target pixel size must be positive")

    image_2d = np.asarray(image)
    if image_2d.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {image_2d.shape}")

    scale_y = metadata.pixel_size_y_um / target_y_um
    scale_x = metadata.pixel_size_x_um / target_x_um
    output_shape = (
        max(1, int(round(image_2d.shape[0] * scale_y))),
        max(1, int(round(image_2d.shape[1] * scale_x))),
    )
    if output_shape == image_2d.shape:
        return image_2d, replace(
            metadata,
            pixel_size_x_um=target_x_um,
            pixel_size_y_um=target_y_um,
        )

    from skimage.transform import resize

    resampled = resize(
        image_2d,
        output_shape,
        order=1,
        mode="reflect",
        anti_aliasing=scale_x < 1.0 or scale_y < 1.0,
        preserve_range=True,
    )
    resampled = resampled.astype(image_2d.dtype, copy=False)
    resampled_metadata = replace(
        metadata,
        pixel_size_x_um=target_x_um,
        pixel_size_y_um=target_y_um,
        size_x_px=output_shape[1],
        size_y_px=output_shape[0],
    )
    return resampled, resampled_metadata


def _normalize_pixel_size(
    target_pixel_size_um: float | tuple[float, float],
) -> tuple[float, float]:
    if isinstance(target_pixel_size_um, tuple):
        if len(target_pixel_size_um) != 2:
            raise ValueError("target pixel size tuple must be (y_um, x_um)")
        return float(target_pixel_size_um[0]), float(target_pixel_size_um[1])
    value = float(target_pixel_size_um)
    return value, value
