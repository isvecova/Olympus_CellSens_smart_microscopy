"""Convert detected masks into stage positions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from smart_acquisition.models import StagePosition
from smart_acquisition.targeting.coordinate_transform import AffinePixelToStage


@dataclass(frozen=True)
class ComponentMeasurement:
    label: int
    area_px: int
    centroid_x_px: float
    centroid_y_px: float
    position: StagePosition


def mask_to_centroid_positions(
    mask: np.ndarray,
    transform: AffinePixelToStage,
    z_um: float,
    min_area_px: int = 1,
    name_prefix: str = "Detected",
) -> tuple[list[StagePosition], list[ComponentMeasurement], np.ndarray]:
    """Return one stage position at the centroid of each positive component."""

    mask_bool = np.asarray(mask, dtype=bool)
    labels, label_count = ndimage.label(mask_bool)
    if label_count == 0:
        return [], [], np.zeros_like(mask_bool, dtype=bool)

    object_slices = ndimage.find_objects(labels)
    measurements: list[ComponentMeasurement] = []
    cleaned_mask = np.zeros_like(mask_bool, dtype=bool)

    for label_index, object_slice in enumerate(object_slices, start=1):
        if object_slice is None:
            continue

        component = labels[object_slice] == label_index
        area_px = int(np.count_nonzero(component))
        if area_px < min_area_px:
            continue

        y_local, x_local = ndimage.center_of_mass(component)
        y_px = float(y_local + object_slice[0].start)
        x_px = float(x_local + object_slice[1].start)
        x_um, y_um = transform.apply(x_px=x_px, y_px=y_px)
        name = f"{name_prefix} {len(measurements) + 1}"
        position = StagePosition(x_um=x_um, y_um=y_um, z_um=z_um, name=name)

        cleaned_mask[object_slice] |= component
        measurements.append(
            ComponentMeasurement(
                label=label_index,
                area_px=area_px,
                centroid_x_px=x_px,
                centroid_y_px=y_px,
                position=position,
            )
        )

    positions = [measurement.position for measurement in measurements]
    return positions, measurements, cleaned_mask
