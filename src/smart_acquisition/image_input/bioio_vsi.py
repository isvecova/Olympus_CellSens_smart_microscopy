"""BioIO reader for Olympus VSI overview images."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from smart_acquisition.image_input.resampling import resample_to_pixel_size
from smart_acquisition.models import OverviewImageMetadata


@dataclass(frozen=True)
class BioioReadResult:
    """A selected 2D image plane and its stage-coordinate metadata."""

    image: np.ndarray
    metadata: OverviewImageMetadata
    zstack: np.ndarray | None = None
    argmax_z: np.ndarray | None = None
    z_positions_um: tuple[float, ...] | None = None


class BioioMetadataError(RuntimeError):
    """Raised when required stage metadata is missing from a BioIO image."""


def read_vsi_overview(
    path: str | Path,
    scene: int | str | None = 0,
    channel: int = 0,
    z: int = 0,
    time: int = 0,
    z_projection: str | None = None,
    source_position_is_center: bool = False,
    fallback_pixel_size_x_um: float | None = None,
    fallback_pixel_size_y_um: float | None = None,
) -> BioioReadResult:
    """Read one overview plane from a VSI file using BioIO.

    The returned metadata uses the first matching OME plane's
    ``position_x/y/z`` and the image ``physical_size_x/y`` fields.
    """

    try:
        from bioio import BioImage
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "BioIO could not be imported. In this environment it may need a "
            "missing dependency such as 'packaging'."
        ) from exc

    image = BioImage(path)
    if scene is not None:
        image.set_scene(scene)

    if z_projection is None:
        image_2d = image.get_image_data("YX", C=channel, Z=z, T=time)
    else:
        stack = image.get_image_data("ZYX", C=channel, T=time)
        projection = z_projection.lower()
        if projection == "max":
            image_2d = np.max(stack, axis=0)
        elif projection == "mean":
            image_2d = np.mean(stack, axis=0)
        else:
            raise ValueError("z_projection must be one of: max, mean")

    pixels = _current_pixels(image)
    plane = _find_plane(pixels, channel=channel, z=z, time=time)
    pixel_size_x_um = _as_float(
        _first_present(
            getattr(pixels, "physical_size_x", None),
            getattr(getattr(image, "physical_pixel_sizes", None), "X", None),
            fallback_pixel_size_x_um,
        ),
        "physical_size_x",
    )
    pixel_size_y_um = _as_float(
        _first_present(
            getattr(pixels, "physical_size_y", None),
            getattr(getattr(image, "physical_pixel_sizes", None), "Y", None),
            fallback_pixel_size_y_um,
            fallback_pixel_size_x_um,
        ),
        "physical_size_y",
    )

    metadata = OverviewImageMetadata(
        stage_x_um=_as_float(getattr(plane, "position_x", None), "position_x"),
        stage_y_um=_as_float(getattr(plane, "position_y", None), "position_y"),
        stage_z_um=_as_float(getattr(plane, "position_z", None), "position_z"),
        pixel_size_x_um=pixel_size_x_um,
        pixel_size_y_um=pixel_size_y_um,
        size_x_px=int(getattr(pixels, "size_x", np.asarray(image_2d).shape[1])),
        size_y_px=int(getattr(pixels, "size_y", np.asarray(image_2d).shape[0])),
        source_position_is_center=source_position_is_center,
    )
    return BioioReadResult(image=np.asarray(image_2d), metadata=metadata)


def read_vsi_zstack(
    path: str | Path,
    scene: int | str | None = 0,
    channel: int = 0,
    time: int = 0,
    source_position_is_center: bool = False,
    fallback_pixel_size_x_um: float | None = None,
    fallback_pixel_size_y_um: float | None = None,
    fallback_z_step_um: float | None = None,
    projection_pixel_size_um: float | tuple[float, float] | None = None,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    write_cache: bool = True,
) -> BioioReadResult:
    """Read one z-stack from a VSI file using BioIO and return a MIP.

    The returned metadata uses the first matching OME plane's
    ``position_x/y/z`` and the image ``physical_size_x/y`` fields. If
    ``projection_pixel_size_um`` is set, each plane is resampled before the
    max projection and argmax-Z cache are built.
    """

    path = Path(path)
    if cache_dir is not None and use_cache:
        cached = _read_z_projection_cache(
            path,
            scene,
            channel,
            time,
            cache_dir,
            projection_pixel_size_um,
            fallback_pixel_size_x_um,
            fallback_pixel_size_y_um,
        )
        if cached is not None:
            return cached

    try:
        from bioio import BioImage
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "BioIO could not be imported. In this environment it may need a "
            "missing dependency such as 'packaging'."
        ) from exc

    image = BioImage(path)
    if scene is not None:
        image.set_scene(scene)

    n_z = int(image.dims.Z)
    pixels = _current_pixels(image)
    metadata_plane = _find_plane(pixels, channel=channel, z=0, time=time)
    pixel_size_x_um = _as_float(
        _first_present(
            getattr(pixels, "physical_size_x", None),
            getattr(getattr(image, "physical_pixel_sizes", None), "X", None),
            fallback_pixel_size_x_um,
        ),
        "physical_size_x",
    )
    pixel_size_y_um = _as_float(
        _first_present(
            getattr(pixels, "physical_size_y", None),
            getattr(getattr(image, "physical_pixel_sizes", None), "Y", None),
            fallback_pixel_size_y_um,
            fallback_pixel_size_x_um,
        ),
        "physical_size_y",
    )

    native_metadata = OverviewImageMetadata(
        stage_x_um=_as_float(
            getattr(metadata_plane, "position_x", None),
            "position_x",
        ),
        stage_y_um=_as_float(
            getattr(metadata_plane, "position_y", None),
            "position_y",
        ),
        stage_z_um=_as_float(
            getattr(metadata_plane, "position_z", None),
            "position_z",
        ),
        pixel_size_x_um=pixel_size_x_um,
        pixel_size_y_um=pixel_size_y_um,
        size_x_px=int(getattr(pixels, "size_x")),
        size_y_px=int(getattr(pixels, "size_y")),
        source_position_is_center=source_position_is_center,
    )
    z_positions_um = _z_positions_um(
        pixels,
        channel=channel,
        time=time,
        n_z=n_z,
        center_z_um=native_metadata.stage_z_um,
        fallback_z_step_um=fallback_z_step_um,
    )

    image_2d = None
    argmax_z = None
    projection_metadata = native_metadata

    for z in range(n_z):
        plane = np.asarray(image.get_image_data("YX", C=channel, Z=z, T=time))
        plane, projection_metadata = resample_to_pixel_size(
            plane,
            native_metadata,
            projection_pixel_size_um,
        )

        if image_2d is None:
            image_2d = plane.copy()
            argmax_z = np.zeros(plane.shape, dtype=np.uint16)
        else:
            update_mask = plane > image_2d
            image_2d[update_mask] = plane[update_mask]
            argmax_z[update_mask] = z

    if image_2d is None or argmax_z is None:
        raise ValueError(f"VSI z-stack does not contain any Z planes: {path}")

    result = BioioReadResult(
        image=np.asarray(image_2d),
        metadata=projection_metadata,
        argmax_z=np.asarray(argmax_z),
        z_positions_um=z_positions_um,
    )
    if cache_dir is not None and write_cache:
        _write_z_projection_cache(
            path,
            scene,
            channel,
            time,
            cache_dir,
            result,
            projection_pixel_size_um,
            fallback_pixel_size_x_um,
            fallback_pixel_size_y_um,
        )
    return result


def _z_positions_um(
    pixels: Any,
    channel: int,
    time: int,
    n_z: int,
    center_z_um: float,
    fallback_z_step_um: float | None,
) -> tuple[float, ...]:
    z_positions: list[float | None] = [None] * n_z
    for plane in list(getattr(pixels, "planes", None) or []):
        if (
            getattr(plane, "the_c", channel) == channel
            and getattr(plane, "the_t", time) == time
        ):
            z_index = getattr(plane, "the_z", None)
            if z_index is None:
                continue
            z_index = int(z_index)
            if 0 <= z_index < n_z:
                position_z = getattr(plane, "position_z", None)
                if position_z is not None:
                    z_positions[z_index] = float(position_z)

    if all(value is not None for value in z_positions):
        return tuple(float(value) for value in z_positions)

    if fallback_z_step_um is None:
        return tuple(float(center_z_um) for _index in range(n_z))

    center_index = (n_z - 1) / 2.0
    return tuple(
        float(center_z_um + (z_index - center_index) * fallback_z_step_um)
        for z_index in range(n_z)
    )


def _cache_prefix(
    path: Path,
    scene: int | str | None,
    channel: int,
    time: int,
    cache_dir: str | Path,
    projection_pixel_size_um: float | tuple[float, float] | None = None,
    source_pixel_size_x_um: float | None = None,
    source_pixel_size_y_um: float | None = None,
) -> Path:
    scene_text = "none" if scene is None else str(scene).replace(" ", "_")
    pixel_text = _pixel_size_cache_text(projection_pixel_size_um)
    source_text = _source_pixel_size_cache_text(
        source_pixel_size_x_um,
        source_pixel_size_y_um,
    )
    return Path(cache_dir) / (
        f"{path.stem}_scene{scene_text}_c{channel}_t{time}_{source_text}_{pixel_text}"
    )


def _pixel_size_cache_text(
    projection_pixel_size_um: float | tuple[float, float] | None,
) -> str:
    if projection_pixel_size_um is None:
        return "pxnative"
    if isinstance(projection_pixel_size_um, tuple):
        if len(projection_pixel_size_um) != 2:
            raise ValueError("projection pixel size tuple must be (y_um, x_um)")
        y_um, x_um = projection_pixel_size_um
    else:
        y_um = projection_pixel_size_um
        x_um = projection_pixel_size_um

    def format_value(value: float) -> str:
        return f"{float(value):.6g}".replace(".", "p").replace("-", "m")

    return f"px_y{format_value(y_um)}_x{format_value(x_um)}"


def _source_pixel_size_cache_text(
    source_pixel_size_x_um: float | None,
    source_pixel_size_y_um: float | None,
) -> str:
    if source_pixel_size_x_um is None and source_pixel_size_y_um is None:
        return "srcpx_auto"
    x_text = "auto" if source_pixel_size_x_um is None else _format_cache_float(
        source_pixel_size_x_um
    )
    y_text = "auto" if source_pixel_size_y_um is None else _format_cache_float(
        source_pixel_size_y_um
    )
    return f"srcpx_y{y_text}_x{x_text}"


def _format_cache_float(value: float) -> str:
    return f"{float(value):.6g}".replace(".", "p").replace("-", "m")


def _read_z_projection_cache(
    path: Path,
    scene: int | str | None,
    channel: int,
    time: int,
    cache_dir: str | Path,
    projection_pixel_size_um: float | tuple[float, float] | None = None,
    source_pixel_size_x_um: float | None = None,
    source_pixel_size_y_um: float | None = None,
) -> BioioReadResult | None:
    prefix = _cache_prefix(
        path,
        scene,
        channel,
        time,
        cache_dir,
        projection_pixel_size_um,
        source_pixel_size_x_um,
        source_pixel_size_y_um,
    )
    mip_path = prefix.with_name(f"{prefix.name}_mip.tif")
    argmax_path = prefix.with_name(f"{prefix.name}_argmax_z.tif")
    metadata_path = prefix.with_name(f"{prefix.name}_zmeta.json")
    if not (mip_path.exists() and argmax_path.exists() and metadata_path.exists()):
        return None

    import tifffile

    with metadata_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    metadata = OverviewImageMetadata(**payload["metadata"])
    return BioioReadResult(
        image=tifffile.imread(mip_path),
        metadata=metadata,
        argmax_z=tifffile.imread(argmax_path),
        z_positions_um=tuple(float(value) for value in payload["z_positions_um"]),
    )


def _write_z_projection_cache(
    path: Path,
    scene: int | str | None,
    channel: int,
    time: int,
    cache_dir: str | Path,
    result: BioioReadResult,
    projection_pixel_size_um: float | tuple[float, float] | None = None,
    source_pixel_size_x_um: float | None = None,
    source_pixel_size_y_um: float | None = None,
) -> None:
    if result.argmax_z is None or result.z_positions_um is None:
        return

    import tifffile

    prefix = _cache_prefix(
        path,
        scene,
        channel,
        time,
        cache_dir,
        projection_pixel_size_um,
        source_pixel_size_x_um,
        source_pixel_size_y_um,
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(prefix.with_name(f"{prefix.name}_mip.tif"), result.image)
    tifffile.imwrite(
        prefix.with_name(f"{prefix.name}_argmax_z.tif"),
        np.asarray(result.argmax_z, dtype=np.uint16),
    )
    payload = {
        "source_vsi": str(path),
        "scene": scene,
        "channel": channel,
        "time": time,
        "projection_pixel_size_um": projection_pixel_size_um,
        "source_pixel_size_x_um": source_pixel_size_x_um,
        "source_pixel_size_y_um": source_pixel_size_y_um,
        "metadata": asdict(result.metadata),
        "z_positions_um": list(result.z_positions_um),
    }
    with prefix.with_name(f"{prefix.name}_zmeta.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(payload, handle, indent=2)


def _current_pixels(image: Any) -> Any:
    metadata = getattr(image, "metadata", None)
    images = getattr(metadata, "images", None)
    if not images:
        raise BioioMetadataError("BioIO metadata does not contain OME images")

    scene_index = int(getattr(image, "current_scene_index", 0) or 0)
    if scene_index >= len(images):
        scene_index = 0
    pixels = getattr(images[scene_index], "pixels", None)
    if pixels is None:
        raise BioioMetadataError("BioIO metadata image does not contain pixels")
    return pixels


def _find_plane(pixels: Any, channel: int, z: int, time: int) -> Any:
    planes = list(getattr(pixels, "planes", None) or [])
    if not planes:
        raise BioioMetadataError("BioIO pixels metadata does not contain planes")

    for plane in planes:
        if (
            getattr(plane, "the_c", channel) == channel
            and getattr(plane, "the_z", z) == z
            and getattr(plane, "the_t", time) == time
        ):
            return plane
    return planes[0]


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _as_float(value: Any, field_name: str) -> float:
    if value is None:
        raise BioioMetadataError(f"Required BioIO metadata field is missing: {field_name}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise BioioMetadataError(
            f"Required BioIO metadata field is not numeric: {field_name}={value!r}"
        ) from exc
