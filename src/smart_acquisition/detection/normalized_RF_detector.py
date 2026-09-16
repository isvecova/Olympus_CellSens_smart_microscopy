"""Random-forest detector for normalized overview images."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class ComponentMeasurement:
    """Measurements used to filter and report one predicted component."""

    label: int
    area_px: int
    elongation: float
    axis_ratio: float
    mean_3nn_distance_um: float
    centroid_x_px: float
    centroid_y_px: float
    mean_positive_probability: float = float("nan")
    min_positive_probability: float = float("nan")
    positive_probability_sum: float = float("nan")
    size_weighted_confidence: float = float("nan")


@dataclass(frozen=True)
class ComponentFilterLimits:
    """Component size and shape limits.

    ``elongation`` is defined as ``1 - minor_axis / major_axis``:
    0 is round and values near 1 are line-like.
    """

    min_area_px: int = 100
    max_area_px: int | None = None
    min_elongation: float = 0.0
    max_elongation: float = 1.0
    max_mean_3nn_distance_um: float | None = None
    min_size_weighted_confidence: float | None = None


@dataclass(frozen=True)
class RandomForestDetectionResult:
    """RF prediction and the mask selected by component filters."""

    normalized_image: np.ndarray
    labels: np.ndarray
    positive_probability: np.ndarray
    mask: np.ndarray
    measurements: tuple[ComponentMeasurement, ...]


@dataclass(frozen=True)
class NapariTilingPreview:
    """Geometry shown in napari for the currently filtered acquisition plan."""

    tile_rectangles_yx: tuple[np.ndarray, ...] = ()
    region_outlines_yx: tuple[np.ndarray, ...] = ()
    point_centers_yx: tuple[tuple[float, float], ...] = ()
    tile_count: int = 0
    target_count: int = 0
    group_count: int = 0


def normalize_to_muscle_peak(
    image: np.ndarray,
    lower_absolute: float = 100,
    upper_percentile: float | None = 99.8,
    bins: int = 500,
    histogram_sigma: float = 3,
    search_min: float | None = None,
    search_max: float | None = None,
    histogram_output_path: str | Path | None = None,
    ignore_zero_pixels: bool = True,
) -> np.ndarray:
    """Normalize intensities to the muscle peak.

    By default, zero-valued pixels are treated as missing/padded image area:
    they are excluded from the histogram fit and kept at zero after
    normalization.
    """

    print("Normalizing")

    image_float = np.asarray(image, dtype=np.float32)
    if image_float.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {image_float.shape}")

    finite_mask = np.isfinite(image_float)
    value_mask = finite_mask
    if ignore_zero_pixels:
        value_mask = value_mask & (image_float != 0)

    finite_values = image_float[value_mask]
    if finite_values.size == 0:
        if ignore_zero_pixels:
            raise ValueError("Image contains no finite nonzero values")
        raise ValueError("Image contains no finite values")

    from scipy.ndimage import gaussian_filter1d

    if ignore_zero_pixels:
        histogram_values = finite_values
        normalization_floor = float(np.min(histogram_values))
        ignored_count = int(np.count_nonzero(finite_mask & (image_float == 0)))
        print(f"Ignoring zero-valued pixels: {ignored_count}")
    else:
        normalization_floor = float(lower_absolute)
        histogram_values = np.maximum(finite_values, normalization_floor)

    mixture_diagnostics = _fit_poisson_gaussian_histogram(
        histogram_values,
        lower_absolute=normalization_floor,
        upper_percentile=upper_percentile,
        bins=bins,
        histogram_sigma=histogram_sigma,
    )

    use_mixture_peak = search_min is None and search_max is None
    if search_min is None:
        search_min = mixture_diagnostics["threshold"]
    if search_max is None:
        search_max = mixture_diagnostics["upper"]
    if search_min >= search_max:
        raise ValueError("search_min must be smaller than search_max")

    print(
        "Normalizing to muscle peak: "
        f"search range {search_min:.2f} to {search_max:.2f}"
    )

    search_bins = _normalization_histogram_bin_count(
        histogram_values,
        lower=float(search_min),
        upper=float(search_max),
        requested_bins=bins,
    )
    histogram, edges = np.histogram(
        histogram_values,
        bins=search_bins,
        range=(search_min, search_max),
    )
    if histogram.sum() == 0:
        raise ValueError(
            "No image values fall inside the normalization search range "
            f"{search_min:.2f} to {search_max:.2f}"
        )
    centers = (edges[:-1] + edges[1:]) / 2
    signal_histogram = _background_subtracted_search_histogram(
        centers,
        histogram,
        mixture_diagnostics=mixture_diagnostics,
        lower_absolute=normalization_floor,
    )
    smoothed_histogram = gaussian_filter1d(
        histogram.astype(np.float32),
        sigma=histogram_sigma,
    )
    smoothed_signal_histogram = gaussian_filter1d(
        signal_histogram.astype(np.float32),
        sigma=histogram_sigma,
    )
    signal_peak_mask = _signal_peak_search_mask(
        centers,
        search_min=float(search_min),
        search_max=float(search_max),
        mixture_diagnostics=mixture_diagnostics,
        use_mixture_peak=use_mixture_peak,
    )
    if np.any(signal_peak_mask) and smoothed_signal_histogram[signal_peak_mask].sum() > 0:
        signal_indices = np.flatnonzero(signal_peak_mask)
        peak_offset = int(np.argmax(smoothed_signal_histogram[signal_peak_mask]))
        muscle_peak = float(centers[signal_indices[peak_offset]])
    elif use_mixture_peak:
        muscle_peak = float(mixture_diagnostics["fit_parameters"][3])
    else:
        muscle_peak = float(centers[np.argmax(smoothed_histogram)])
    if muscle_peak <= normalization_floor:
        raise ValueError(
            f"Detected muscle peak ({muscle_peak:.2f}) is not above "
            f"normalization floor ({normalization_floor:.2f})"
        )

    print(f"Detected muscle peak: {muscle_peak:.2f}")

    if histogram_output_path is not None:
        _write_normalization_histogram(
            histogram_output_path,
            mixture_diagnostics=mixture_diagnostics,
            search_centers=centers,
            search_histogram=histogram,
            search_smoothed_histogram=smoothed_histogram,
            search_signal_histogram=signal_histogram,
            search_smoothed_signal_histogram=smoothed_signal_histogram,
            lower_absolute=normalization_floor,
            search_min=search_min,
            search_max=search_max,
            muscle_peak=muscle_peak,
            histogram_label=(
                "nonzero intensity histogram"
                if ignore_zero_pixels
                else "clipped intensity histogram"
            ),
        )

    if ignore_zero_pixels:
        normalized = (image_float - normalization_floor) / (
            muscle_peak - normalization_floor
        )
        normalized[~value_mask] = 0
        return normalized.astype(np.float32, copy=False)

    clipped = np.maximum(image_float, normalization_floor)
    return ((clipped - normalization_floor) / (muscle_peak - normalization_floor)).astype(
        np.float32,
        copy=False,
    )


def _fit_poisson_gaussian_histogram(
    clipped_values: np.ndarray,
    *,
    lower_absolute: float,
    upper_percentile: float | None,
    bins: int,
    histogram_sigma: float,
) -> dict[str, Any]:
    """Fit a shifted Poisson-like floor and Gaussian signal to the histogram."""

    from scipy.ndimage import gaussian_filter1d
    from scipy.optimize import least_squares
    from scipy.special import gammaln

    values = np.asarray(clipped_values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Image contains no finite values")

    upper = (
        float(np.max(values))
        if upper_percentile is None
        else float(np.percentile(values, upper_percentile))
    )
    if upper <= lower_absolute:
        upper = float(np.max(values))
    if upper <= lower_absolute:
        raise ValueError(
            "Cannot estimate normalization histogram because all finite values "
            f"are <= lower_absolute ({lower_absolute:.2f})"
        )

    histogram, edges = np.histogram(
        values,
        bins=max(32, int(bins)),
        range=(lower_absolute, upper),
    )
    centers = (edges[:-1] + edges[1:]) / 2
    smoothed_histogram = gaussian_filter1d(
        histogram.astype(np.float32),
        sigma=histogram_sigma,
    )
    bin_width = float(edges[1] - edges[0])
    span = float(upper - lower_absolute)
    total_count = float(max(histogram.sum(), 1))

    gaussian_mean_lower = min(
        upper - bin_width / 2.0,
        lower_absolute + 0.15 * span,
    )
    gaussian_mean_initial = float(
        np.clip(
            np.percentile(values, 90),
            gaussian_mean_lower,
            upper - bin_width / 2.0,
        )
    )
    gaussian_sigma_initial = max(span / 12.0, bin_width)

    low_mask = centers <= lower_absolute + 0.35 * span
    if np.any(low_mask) and np.any(histogram[low_mask] > 0):
        shifted_low_centers = centers[low_mask] - lower_absolute
        poisson_lambda_initial = float(
            np.average(
                shifted_low_centers,
                weights=np.maximum(histogram[low_mask], 1),
            )
        )
    else:
        poisson_lambda_initial = span / 20.0
    poisson_lambda_initial = max(poisson_lambda_initial, bin_width / 2.0)

    initial = np.asarray(
        [
            total_count * 0.7,
            poisson_lambda_initial,
            total_count * 0.3,
            gaussian_mean_initial,
            gaussian_sigma_initial,
        ],
        dtype=np.float64,
    )
    lower_bounds = np.asarray(
        [0.0, bin_width / 10.0, 0.0, gaussian_mean_lower, bin_width / 2.0],
        dtype=np.float64,
    )
    upper_bounds = np.asarray(
        [
            np.inf,
            max(span * 0.5, bin_width),
            np.inf,
            upper,
            max(span, bin_width),
        ],
        dtype=np.float64,
    )
    target = smoothed_histogram.astype(np.float64)

    def components(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        poisson_area, poisson_lambda, gaussian_area, gaussian_mean, gaussian_sigma = (
            params
        )
        shifted_centers = np.maximum(centers - lower_absolute, 0.0)
        poisson_log = (
            shifted_centers * np.log(poisson_lambda)
            - poisson_lambda
            - gammaln(shifted_centers + 1.0)
        )
        poisson_component = poisson_area * np.exp(poisson_log)
        gaussian_component = gaussian_area * np.exp(
            -0.5 * ((centers - gaussian_mean) / gaussian_sigma) ** 2
        )
        return poisson_component, gaussian_component, poisson_component + gaussian_component

    def residual(params: np.ndarray) -> np.ndarray:
        _poisson_component, _gaussian_component, combined = components(params)
        return (combined - target) / np.sqrt(target + 1.0)

    try:
        fit = least_squares(
            residual,
            initial,
            bounds=(lower_bounds, upper_bounds),
            max_nfev=300,
        )
        params = fit.x if fit.success else initial
    except ValueError:
        params = initial

    poisson_component, gaussian_component, combined = components(params)
    threshold = _poisson_gaussian_threshold(
        centers=centers,
        smoothed_histogram=smoothed_histogram,
        poisson_component=poisson_component,
        gaussian_component=gaussian_component,
        lower_absolute=lower_absolute,
        upper=upper,
        gaussian_mean=float(params[3]),
    )
    return {
        "histogram": histogram,
        "edges": edges,
        "centers": centers,
        "bin_width": bin_width,
        "smoothed_histogram": smoothed_histogram,
        "poisson_component": poisson_component,
        "gaussian_component": gaussian_component,
        "combined_component": combined,
        "threshold": float(threshold),
        "upper": float(upper),
        "fit_parameters": tuple(float(value) for value in params),
    }


def _normalization_histogram_bin_count(
    values: np.ndarray,
    *,
    lower: float,
    upper: float,
    requested_bins: int,
) -> int:
    span = max(float(upper - lower), np.finfo(float).eps)
    requested_bins = max(8, int(requested_bins))
    finite_values = np.asarray(values)[np.isfinite(values)]
    if finite_values.size == 0:
        return requested_bins

    sample = finite_values[: min(finite_values.size, 10000)]
    is_integer_like = np.all(np.abs(sample - np.round(sample)) < 1e-6)
    if not is_integer_like:
        return requested_bins

    integer_bins = max(8, int(np.ceil(span)))
    return min(requested_bins, integer_bins)


def _signal_peak_search_mask(
    centers: np.ndarray,
    *,
    search_min: float,
    search_max: float,
    mixture_diagnostics: dict[str, Any],
    use_mixture_peak: bool,
) -> np.ndarray:
    if centers.size == 0:
        return np.zeros_like(centers, dtype=bool)
    if not use_mixture_peak:
        return np.ones_like(centers, dtype=bool)

    _poisson_area, _poisson_lambda, _gaussian_area, gaussian_mean, gaussian_sigma = (
        mixture_diagnostics["fit_parameters"]
    )
    lower = max(search_min, float(gaussian_mean) - 0.5 * max(float(gaussian_sigma), 1.0))
    upper = min(search_max, float(gaussian_mean) + 2.0 * max(float(gaussian_sigma), 1.0))
    mask = (centers >= lower) & (centers <= upper)
    if np.any(mask):
        return mask

    nearest_index = int(np.argmin(np.abs(centers - float(gaussian_mean))))
    mask = np.zeros_like(centers, dtype=bool)
    mask[nearest_index] = True
    return mask


def _poisson_gaussian_threshold(
    *,
    centers: np.ndarray,
    smoothed_histogram: np.ndarray,
    poisson_component: np.ndarray,
    gaussian_component: np.ndarray,
    lower_absolute: float,
    upper: float,
    gaussian_mean: float,
) -> float:
    denominator = poisson_component + gaussian_component + np.finfo(float).eps
    gaussian_posterior = gaussian_component / denominator
    candidate_mask = (
        (centers > lower_absolute)
        & (centers < gaussian_mean)
        & (gaussian_posterior >= 0.5)
    )
    if np.any(candidate_mask):
        return float(centers[np.flatnonzero(candidate_mask)[0]])

    poisson_peak = int(np.argmax(poisson_component))
    gaussian_peak = int(np.argmax(gaussian_component))
    if poisson_peak < gaussian_peak:
        valley_slice = slice(poisson_peak, gaussian_peak + 1)
        valley_offset = int(np.argmin(smoothed_histogram[valley_slice]))
        return float(centers[poisson_peak + valley_offset])

    return float(lower_absolute + 0.1 * (upper - lower_absolute))


def _background_subtracted_search_histogram(
    centers: np.ndarray,
    histogram: np.ndarray,
    *,
    mixture_diagnostics: dict[str, Any],
    lower_absolute: float,
) -> np.ndarray:
    if centers.size < 2:
        return histogram.astype(np.float32)

    params = mixture_diagnostics["fit_parameters"]
    source_bin_width = float(mixture_diagnostics["bin_width"])
    target_bin_width = float(centers[1] - centers[0])
    background = _poisson_component(
        centers,
        poisson_area=float(params[0]),
        poisson_lambda=float(params[1]),
        lower_absolute=lower_absolute,
    )
    background *= target_bin_width / source_bin_width
    return np.maximum(histogram.astype(np.float32) - background, 0.0)


def _poisson_component(
    centers: np.ndarray,
    *,
    poisson_area: float,
    poisson_lambda: float,
    lower_absolute: float,
) -> np.ndarray:
    from scipy.special import gammaln

    shifted_centers = np.maximum(centers - lower_absolute, 0.0)
    poisson_log = (
        shifted_centers * np.log(poisson_lambda)
        - poisson_lambda
        - gammaln(shifted_centers + 1.0)
    )
    return poisson_area * np.exp(poisson_log)


def _write_normalization_histogram(
    path: str | Path,
    *,
    mixture_diagnostics: dict[str, Any],
    search_centers: np.ndarray,
    search_histogram: np.ndarray,
    search_smoothed_histogram: np.ndarray,
    search_signal_histogram: np.ndarray,
    search_smoothed_signal_histogram: np.ndarray,
    lower_absolute: float,
    search_min: float,
    search_max: float,
    muscle_peak: float,
    histogram_label: str = "clipped intensity histogram",
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        _write_normalization_histogram_csv(
            output_path.with_suffix(".csv"),
            mixture_diagnostics=mixture_diagnostics,
        )
        return

    centers = mixture_diagnostics["centers"]
    histogram = mixture_diagnostics["histogram"]
    width = float(mixture_diagnostics["edges"][1] - mixture_diagnostics["edges"][0])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        centers,
        np.maximum(histogram, 1),
        width=width,
        color="0.82",
        edgecolor="none",
        label=histogram_label,
    )
    ax.plot(
        centers,
        np.maximum(mixture_diagnostics["smoothed_histogram"], 1),
        color="black",
        linewidth=1.2,
        label="smoothed histogram",
    )
    ax.plot(
        centers,
        np.maximum(mixture_diagnostics["poisson_component"], 1),
        color="#2b8cbe",
        linewidth=1.2,
        label="Poisson-like background",
    )
    ax.plot(
        centers,
        np.maximum(mixture_diagnostics["gaussian_component"], 1),
        color="#e34a33",
        linewidth=1.2,
        label="Gaussian signal",
    )
    ax.plot(
        centers,
        np.maximum(mixture_diagnostics["combined_component"], 1),
        color="#31a354",
        linewidth=1.2,
        label="combined fit",
    )
    ax.plot(
        search_centers,
        np.maximum(search_smoothed_histogram, 1),
        color="#756bb1",
        linewidth=1.0,
        alpha=0.8,
        label="raw peak-search histogram",
    )
    ax.plot(
        search_centers,
        np.maximum(search_smoothed_signal_histogram, 1),
        color="#f16913",
        linewidth=1.5,
        label="background-subtracted signal",
    )
    ax.axvline(lower_absolute, color="0.35", linestyle=":", label="lower absolute")
    ax.axvline(search_min, color="#08519c", linestyle="--", label="search min")
    ax.axvline(search_max, color="#636363", linestyle="--", label="search max")
    ax.axvline(muscle_peak, color="#a50f15", linestyle="-", label="muscle peak")
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Pixel count")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _write_normalization_histogram_csv(
    path: Path,
    *,
    mixture_diagnostics: dict[str, Any],
) -> None:
    centers = mixture_diagnostics["centers"]
    columns = (
        mixture_diagnostics["histogram"],
        mixture_diagnostics["smoothed_histogram"],
        mixture_diagnostics["poisson_component"],
        mixture_diagnostics["gaussian_component"],
        mixture_diagnostics["combined_component"],
    )
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "intensity,count,smoothed_count,poisson_component,"
            "gaussian_component,combined_component\n"
        )
        for row in zip(centers, *columns):
            handle.write(",".join(f"{float(value):.8g}" for value in row))
            handle.write("\n")


# def extract_features(image: np.ndarray, feature_params: dict[str, Any]) -> np.ndarray:
#     """Build the feature stack expected by the trained classifier."""

#     image_2d = np.squeeze(np.asarray(image, dtype=np.float32))
#     if image_2d.ndim != 2:
#         raise ValueError(f"Expected a 2D image, got shape {image_2d.shape}")

#     from scipy.ndimage import gaussian_filter
#     from skimage.feature import structure_tensor, structure_tensor_eigenvalues

#     features: list[np.ndarray] = []
#     for sigma in feature_params["sigmas"]:
#         gaussian = gaussian_filter(image_2d, sigma=sigma, mode="reflect")
#         tensor = structure_tensor(image_2d, sigma=sigma, order="rc")
#         eigenvalues = structure_tensor_eigenvalues(tensor)
#         lambda_max, lambda_min = eigenvalues[0], eigenvalues[1]
#         coherence = (lambda_max - lambda_min) / (lambda_max + lambda_min + 1e-12)
#         features.extend((gaussian, lambda_max, lambda_min, coherence))
#     return np.stack(features, axis=-1)


def extract_features(image: np.ndarray, feature_params: dict[str, Any]) -> np.ndarray:
    """Build the feature stack expected by the trained classifier."""

    image_2d = np.squeeze(np.asarray(image, dtype=np.float32))
    if image_2d.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {image_2d.shape}")

    from scipy.ndimage import gaussian_filter
    from skimage.feature import structure_tensor, structure_tensor_eigenvalues

    features: list[np.ndarray] = []

    for sigma in feature_params["sigmas"]:
        gaussian = gaussian_filter(
            image_2d,
            sigma=sigma,
            mode="reflect",
        )

        tensor = structure_tensor(
            image_2d,
            sigma=sigma,
            order="rc",
        )
        eigenvalues = structure_tensor_eigenvalues(tensor)
        lambda_max, lambda_min = eigenvalues[0], eigenvalues[1]

        local_mean_sq = gaussian_filter(
            image_2d * image_2d,
            sigma=sigma,
            mode="reflect",
        )

        local_variance = local_mean_sq - gaussian * gaussian
        local_variance = np.maximum(local_variance, 0)

        features.extend((
            gaussian,
            lambda_max,
            lambda_min,
            local_variance,
        ))

    return np.stack(features, axis=-1)


def predict_tiled(
    image: np.ndarray,
    model: Any,
    feature_params: dict[str, Any],
    tile_size: int = 1024,
    overlap: int = 12,
    prediction_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Predict a large 2D image tile by tile."""

    labels, _positive_probability = _predict_tiled_impl(
        image,
        model,
        feature_params,
        tile_size=tile_size,
        overlap=overlap,
        prediction_mask=prediction_mask,
        positive_label=None,
    )
    return labels


def predict_tiled_with_probability(
    image: np.ndarray,
    model: Any,
    feature_params: dict[str, Any],
    *,
    positive_label: int,
    tile_size: int = 1024,
    overlap: int = 12,
    prediction_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict labels and positive-class probability for a large 2D image."""

    labels, positive_probability = _predict_tiled_impl(
        image,
        model,
        feature_params,
        tile_size=tile_size,
        overlap=overlap,
        prediction_mask=prediction_mask,
        positive_label=positive_label,
    )
    if positive_probability is None:
        raise RuntimeError("Probability output was not generated")
    return labels, positive_probability


def _predict_tiled_impl(
    image: np.ndarray,
    model: Any,
    feature_params: dict[str, Any],
    *,
    tile_size: int,
    overlap: int,
    prediction_mask: np.ndarray | None,
    positive_label: int | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Predict a large 2D image tile by tile."""

    image_2d = np.asarray(image)
    if image_2d.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {image_2d.shape}")
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("tile_size must be positive and overlap must be smaller")
    if prediction_mask is not None:
        prediction_mask = np.asarray(prediction_mask, dtype=bool)
        if prediction_mask.shape != image_2d.shape:
            raise ValueError(
                f"prediction_mask shape {prediction_mask.shape} does not match "
                f"image shape {image_2d.shape}"
            )

    feature_count = len(feature_params["sigmas"]) * 4
    probe = np.zeros((1, feature_count), dtype=np.float32)
    output_dtype = np.asarray(model.predict(probe)).dtype
    height, width = image_2d.shape
    labels = np.zeros((height, width), dtype=output_dtype)
    positive_probability = (
        np.zeros((height, width), dtype=np.float32)
        if positive_label is not None
        else None
    )

    for y_start in range(0, height, tile_size):
        for x_start in range(0, width, tile_size):
            y_end = min(y_start + tile_size, height)
            x_end = min(x_start + tile_size, width)
            tile_output_mask = None
            if prediction_mask is not None:
                tile_output_mask = prediction_mask[y_start:y_end, x_start:x_end]
                if not np.any(tile_output_mask):
                    continue

            expanded_y_start = max(0, y_start - overlap)
            expanded_x_start = max(0, x_start - overlap)
            expanded_y_end = min(height, y_end + overlap)
            expanded_x_end = min(width, x_end + overlap)

            tile_features = extract_features(
                image_2d[
                    expanded_y_start:expanded_y_end,
                    expanded_x_start:expanded_x_end,
                ],
                feature_params,
            )
            tile_height, tile_width, tile_feature_count = tile_features.shape
            flat_features = tile_features.reshape(-1, tile_feature_count).astype(
                np.float32
            )
            if prediction_mask is None:
                flat_prediction, flat_probability = _predict_flat_labels_and_probability(
                    model,
                    flat_features,
                    positive_label=positive_label,
                )
                tile_prediction = flat_prediction.reshape(tile_height, tile_width)
                tile_probability = (
                    None
                    if flat_probability is None
                    else flat_probability.reshape(tile_height, tile_width)
                )
            else:
                expanded_mask = prediction_mask[
                    expanded_y_start:expanded_y_end,
                    expanded_x_start:expanded_x_end,
                ]
                flat_mask = expanded_mask.reshape(-1)
                tile_prediction = np.zeros(
                    tile_height * tile_width,
                    dtype=output_dtype,
                )
                tile_probability = (
                    np.zeros(tile_height * tile_width, dtype=np.float32)
                    if positive_label is not None
                    else None
                )
                flat_prediction, flat_probability = _predict_flat_labels_and_probability(
                    model,
                    flat_features[flat_mask],
                    positive_label=positive_label,
                )
                tile_prediction[flat_mask] = flat_prediction
                if tile_probability is not None and flat_probability is not None:
                    tile_probability[flat_mask] = flat_probability
                tile_prediction = tile_prediction.reshape(tile_height, tile_width)
                if tile_probability is not None:
                    tile_probability = tile_probability.reshape(tile_height, tile_width)

            crop_y_start = y_start - expanded_y_start
            crop_x_start = x_start - expanded_x_start
            cropped_prediction = tile_prediction[
                crop_y_start : crop_y_start + y_end - y_start,
                crop_x_start : crop_x_start + x_end - x_start,
            ]
            cropped_probability = (
                None
                if tile_probability is None
                else tile_probability[
                    crop_y_start : crop_y_start + y_end - y_start,
                    crop_x_start : crop_x_start + x_end - x_start,
                ]
            )
            if tile_output_mask is None:
                labels[y_start:y_end, x_start:x_end] = cropped_prediction
                if positive_probability is not None and cropped_probability is not None:
                    positive_probability[y_start:y_end, x_start:x_end] = (
                        cropped_probability
                    )
            else:
                label_view = labels[y_start:y_end, x_start:x_end]
                label_view[tile_output_mask] = cropped_prediction[tile_output_mask]
                if positive_probability is not None and cropped_probability is not None:
                    probability_view = positive_probability[y_start:y_end, x_start:x_end]
                    probability_view[tile_output_mask] = cropped_probability[
                        tile_output_mask
                    ]
    return labels, positive_probability


def _predict_flat_labels_and_probability(
    model: Any,
    flat_features: np.ndarray,
    *,
    positive_label: int | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if positive_label is None:
        return np.asarray(model.predict(flat_features)), None

    if not hasattr(model, "predict_proba"):
        raise TypeError("Probability output requires a classifier with predict_proba")

    probabilities = np.asarray(model.predict_proba(flat_features))
    classes = np.asarray(getattr(model, "classes_", ()))
    if probabilities.ndim != 2 or classes.ndim != 1:
        raise TypeError("Probability output requires a single-output classifier")
    if probabilities.shape[1] != classes.shape[0]:
        raise TypeError("predict_proba columns do not match model.classes_")

    labels = classes[np.argmax(probabilities, axis=1)]
    matches = np.flatnonzero(classes == positive_label)
    if len(matches) == 0:
        positive_probability = np.zeros(flat_features.shape[0], dtype=np.float32)
    else:
        positive_probability = probabilities[:, int(matches[0])].astype(
            np.float32,
            copy=False,
        )
    return np.asarray(labels), positive_probability


def measure_positive_components(
    prediction_labels: np.ndarray,
    positive_label: int = 2,
    pixel_size_x_um: float = 1.0,
    pixel_size_y_um: float = 1.0,
    positive_probability: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[ComponentMeasurement, ...]]:
    """Label and measure connected components of the positive RF class."""

    from skimage.measure import label, regionprops

    positive_mask = np.asarray(prediction_labels) == positive_label
    if positive_probability is not None:
        positive_probability = np.asarray(positive_probability, dtype=np.float32)
        if positive_probability.shape != positive_mask.shape:
            raise ValueError(
                f"positive_probability shape {positive_probability.shape} does "
                f"not match labels shape {positive_mask.shape}"
            )
    component_labels = label(positive_mask, connectivity=2)
    measurements: list[ComponentMeasurement] = []

    regions = list(regionprops(component_labels))
    mean_3nn_distances = _mean_nearest_neighbor_distances_um(
        [(float(region.centroid[1]), float(region.centroid[0])) for region in regions],
        pixel_size_x_um=pixel_size_x_um,
        pixel_size_y_um=pixel_size_y_um,
        neighbor_count=3,
    )

    for region, mean_3nn_distance_um in zip(regions, mean_3nn_distances):
        major_axis = float(region.axis_major_length)
        axis_ratio = (
            float(region.axis_minor_length / major_axis) if major_axis > 0 else 0.0
        )
        axis_ratio = min(max(axis_ratio, 0.0), 1.0)
        confidence = _component_confidence_values(region, positive_probability)
        measurements.append(
            ComponentMeasurement(
                label=int(region.label),
                area_px=int(region.area),
                elongation=1.0 - axis_ratio,
                axis_ratio=axis_ratio,
                mean_3nn_distance_um=float(mean_3nn_distance_um),
                centroid_x_px=float(region.centroid[1]),
                centroid_y_px=float(region.centroid[0]),
                mean_positive_probability=confidence[0],
                min_positive_probability=confidence[1],
                positive_probability_sum=confidence[2],
                size_weighted_confidence=confidence[3],
            )
        )
    return component_labels, tuple(measurements)


def _component_confidence_values(
    region: Any,
    positive_probability: np.ndarray | None,
) -> tuple[float, float, float, float]:
    if positive_probability is None:
        nan = float("nan")
        return nan, nan, nan, nan

    coords = region.coords
    values = positive_probability[coords[:, 0], coords[:, 1]]
    if values.size == 0:
        nan = float("nan")
        return nan, nan, nan, nan

    mean_probability = float(np.mean(values))
    min_probability = float(np.min(values))
    probability_sum = float(np.sum(values))
    size_weighted = mean_probability * float(np.sqrt(values.size))
    return mean_probability, min_probability, probability_sum, size_weighted


def filter_component_labels(
    component_labels: np.ndarray,
    measurements: tuple[ComponentMeasurement, ...],
    limits: ComponentFilterLimits,
    pixel_size_x_um: float = 1.0,
    pixel_size_y_um: float = 1.0,
    excluded_labels: set[int] | None = None,
) -> tuple[np.ndarray, tuple[ComponentMeasurement, ...]]:
    """Return a mask containing only measured components within ``limits``."""

    _validate_filter_limits(limits)
    excluded_labels = set() if excluded_labels is None else set(excluded_labels)
    kept_labels = []
    kept_measurements: list[ComponentMeasurement] = []
    for measurement in measurements:
        if measurement.label in excluded_labels:
            continue
        if measurement.area_px < limits.min_area_px:
            continue
        if limits.max_area_px is not None and measurement.area_px > limits.max_area_px:
            continue
        if not limits.min_elongation <= measurement.elongation <= limits.max_elongation:
            continue
        if limits.min_size_weighted_confidence is not None and (
            not np.isfinite(measurement.size_weighted_confidence)
            or measurement.size_weighted_confidence
            < limits.min_size_weighted_confidence
        ):
            continue
        kept_measurements.append(measurement)

    if limits.max_mean_3nn_distance_um is not None:
        kept_measurements = [
            measurement
            for measurement in _measurements_with_neighbor_distances(
                kept_measurements,
                pixel_size_x_um=pixel_size_x_um,
                pixel_size_y_um=pixel_size_y_um,
            )
            if measurement.mean_3nn_distance_um <= limits.max_mean_3nn_distance_um
        ]

    kept_labels = [measurement.label for measurement in kept_measurements]
    if kept_labels:
        mask = np.isin(component_labels, kept_labels)
    else:
        mask = np.zeros_like(component_labels, dtype=bool)
    return mask, tuple(kept_measurements)


def filter_components(
    prediction_labels: np.ndarray,
    positive_label: int = 2,
    min_area_px: int = 100,
    max_area_px: int | None = None,
    min_elongation: float = 0.0,
    max_elongation: float = 1.0,
    max_mean_3nn_distance_um: float | None = None,
    min_size_weighted_confidence: float | None = None,
    pixel_size_x_um: float = 1.0,
    pixel_size_y_um: float = 1.0,
    positive_probability: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[ComponentMeasurement, ...]]:
    """Keep positive components within size and elongation limits."""

    component_labels, measurements = measure_positive_components(
        prediction_labels,
        positive_probability=positive_probability,
        positive_label=positive_label,
        pixel_size_x_um=pixel_size_x_um,
        pixel_size_y_um=pixel_size_y_um,
    )
    return filter_component_labels(
        component_labels,
        measurements,
        ComponentFilterLimits(
            min_area_px=min_area_px,
            max_area_px=max_area_px,
            min_elongation=min_elongation,
            max_elongation=max_elongation,
            max_mean_3nn_distance_um=max_mean_3nn_distance_um,
            min_size_weighted_confidence=min_size_weighted_confidence,
        ),
        pixel_size_x_um=pixel_size_x_um,
        pixel_size_y_um=pixel_size_y_um,
    )


class NormalizedRandomForestDetector:
    """Run the trained normalized-image RF model and apply component filters."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        lower_absolute: float = 100,
        normalization_search_min: float | None = None,
        normalization_search_max: float | None = None,
        normalization_histogram_output_path: str | Path | None = None,
        ignore_zero_pixels: bool = True,
        positive_label: int = 2,
        tile_size: int = 1024,
        overlap: int = 12,
    ):
        self.model_path = Path(model_path)
        self.lower_absolute = lower_absolute
        self.normalization_search_min = normalization_search_min
        self.normalization_search_max = normalization_search_max
        self.normalization_histogram_output_path = normalization_histogram_output_path
        self.ignore_zero_pixels = ignore_zero_pixels
        self.positive_label = positive_label
        self.tile_size = tile_size
        self.overlap = overlap

    def predict(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return normalized image, RF labels, and positive-class probability."""

        import joblib

        bundle = joblib.load(self.model_path)
        prediction_mask = None
        if self.ignore_zero_pixels:
            image_array = np.asarray(image)
            prediction_mask = np.isfinite(image_array) & (image_array != 0)
        normalized_image = normalize_to_muscle_peak(
            image,
            lower_absolute=self.lower_absolute,
            search_min=self.normalization_search_min,
            search_max=self.normalization_search_max,
            histogram_output_path=self.normalization_histogram_output_path,
            ignore_zero_pixels=self.ignore_zero_pixels,
        )
        labels, positive_probability = predict_tiled_with_probability(
            normalized_image,
            bundle["model"],
            bundle["feature_params"],
            positive_label=self.positive_label,
            tile_size=self.tile_size,
            overlap=self.overlap,
            prediction_mask=prediction_mask,
        )
        print(np.unique(labels, return_counts=True))
        return normalized_image, labels, positive_probability

    def detect(
        self,
        image: np.ndarray,
        *,
        min_area_px: int = 100,
        max_area_px: int | None = None,
        min_elongation: float = 0.0,
        max_elongation: float = 1.0,
        max_mean_3nn_distance_um: float | None = None,
        min_size_weighted_confidence: float | None = None,
        max_selected_objects: int | None = None,
        pixel_size_x_um: float = 1.0,
        pixel_size_y_um: float = 1.0,
        interactive: bool = False,
        selection_callback: Callable[
            [tuple[ComponentMeasurement, ...], int, np.ndarray, np.ndarray, np.ndarray],
            tuple[ComponentMeasurement, ...],
        ]
        | None = None,
        preview_callback: Callable[
            [np.ndarray, tuple[ComponentMeasurement, ...], np.ndarray],
            NapariTilingPreview,
        ]
        | None = None,
    ) -> RandomForestDetectionResult:
        """Run prediction, optionally pause in napari, and return selected mask."""

        normalized_image, labels, positive_probability = self.predict(image)
        if interactive:
            mask, measurements = run_napari_filter(
                image,
                labels,
                positive_probability=positive_probability,
                positive_label=self.positive_label,
                min_area_px=min_area_px,
                max_area_px=max_area_px,
                min_elongation=min_elongation,
                max_elongation=max_elongation,
                max_mean_3nn_distance_um=max_mean_3nn_distance_um,
                min_size_weighted_confidence=min_size_weighted_confidence,
                max_selected_objects=max_selected_objects,
                pixel_size_x_um=pixel_size_x_um,
                pixel_size_y_um=pixel_size_y_um,
                normalized_image=normalized_image,
                selection_callback=selection_callback,
                preview_callback=preview_callback,
            )
        else:
            mask, measurements = filter_components(
                labels,
                positive_probability=positive_probability,
                positive_label=self.positive_label,
                min_area_px=min_area_px,
                max_area_px=max_area_px,
                min_elongation=min_elongation,
                max_elongation=max_elongation,
                max_mean_3nn_distance_um=max_mean_3nn_distance_um,
                min_size_weighted_confidence=min_size_weighted_confidence,
                pixel_size_x_um=pixel_size_x_um,
                pixel_size_y_um=pixel_size_y_um,
            )
        return RandomForestDetectionResult(
            normalized_image=normalized_image,
            labels=labels,
            positive_probability=positive_probability,
            mask=mask,
            measurements=measurements,
        )


def run_napari_filter(
    image: np.ndarray,
    prediction_labels: np.ndarray,
    *,
    positive_probability: np.ndarray | None = None,
    positive_label: int = 2,
    min_area_px: int = 100,
    max_area_px: int | None = None,
    min_elongation: float = 0.0,
    max_elongation: float = 1.0,
    max_mean_3nn_distance_um: float | None = None,
    min_size_weighted_confidence: float | None = None,
    max_selected_objects: int | None = None,
    pixel_size_x_um: float = 1.0,
    pixel_size_y_um: float = 1.0,
    normalized_image: np.ndarray | None = None,
    selection_callback: Callable[
        [tuple[ComponentMeasurement, ...], int, np.ndarray, np.ndarray, np.ndarray],
        tuple[ComponentMeasurement, ...],
    ]
    | None = None,
    preview_callback: Callable[
        [np.ndarray, tuple[ComponentMeasurement, ...], np.ndarray],
        NapariTilingPreview,
    ]
    | None = None,
) -> tuple[np.ndarray, tuple[ComponentMeasurement, ...]]:
    """Show predictions in napari and return filters confirmed with ``OK``."""

    try:
        import napari
        from qtpy.QtCore import QTimer
        from qtpy.QtWidgets import (
            QDoubleSpinBox,
            QFormLayout,
            QPushButton,
            QSpinBox,
            QLabel,
            QWidget,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Interactive filtering requires napari and Qt bindings."
        ) from exc

    component_labels, measurements = measure_positive_components(
        prediction_labels,
        positive_probability=positive_probability,
        positive_label=positive_label,
        pixel_size_x_um=pixel_size_x_um,
        pixel_size_y_um=pixel_size_y_um,
    )
    (
        area_map,
        elongation_map,
        mean_3nn_distance_map,
        size_weighted_confidence_map,
    ) = _component_measurement_maps(component_labels, measurements)

    selected: dict[str, Any] = {
        "accepted": False,
        "mask": None,
        "measurements": (),
    }

    viewer = napari.Viewer(title="Confirm smart-acquisition detections")
    viewer.add_image(np.asarray(image), name="Overview image")
    if normalized_image is not None:
        viewer.add_image(
            np.asarray(normalized_image),
            name="Normalized image",
            visible=False,
        )
    viewer.add_labels(
        np.asarray(prediction_labels),
        name="RF prediction labels",
        opacity=0.35,
        visible=False,
    )
    if positive_probability is not None:
        viewer.add_image(
            np.asarray(positive_probability, dtype=np.float32),
            name=f"RF label {positive_label} probability",
            colormap="magma",
            contrast_limits=(0, 1),
            visible=False,
        )
    viewer.add_labels(
        component_labels,
        name="Positive components",
        opacity=0.35,
        visible=False,
    )
    viewer.add_image(
        area_map,
        name="Component area px",
        colormap="viridis",
        visible=False,
    )
    viewer.add_image(
        elongation_map,
        name="Component elongation",
        colormap="turbo",
        contrast_limits=(0, 1),
        visible=False,
    )
    viewer.add_image(
        mean_3nn_distance_map,
        name="Mean 3NN distance um",
        colormap="magma",
        visible=False,
    )
    viewer.add_image(
        size_weighted_confidence_map,
        name="Size-weighted confidence",
        colormap="viridis",
        visible=False,
    )
    tile_shapes_layer = None
    region_shapes_layer = None
    target_points_layer = None
    if preview_callback is not None:
        tile_shapes_layer = viewer.add_shapes(
            [],
            shape_type="polygon",
            name="Planned high-mag tiles",
            edge_color="#1f77b4",
            face_color="#1f77b433",
            edge_width=1,
        )
        region_shapes_layer = viewer.add_shapes(
            [],
            shape_type="polygon",
            name="Planned tile-scan regions",
            edge_color="#ff7f0e",
            face_color="#ff7f0e11",
            edge_width=2,
        )
        target_points_layer = viewer.add_points(
            np.empty((0, 2), dtype=np.float32),
            name="Planned point targets",
            size=16,
            face_color="#00aa66",
        )
    filtered_layer = viewer.add_labels(
        np.zeros_like(component_labels),
        name="Filtered detections",
        opacity=0.6,
    )
    try:
        filtered_layer.selected_label = 0
        filtered_layer.mode = "fill"
    except Exception:
        pass

    deleted_labels: set[int] = set()
    active_labels: set[int] = set()
    tile_selected_labels: set[int] | None = None
    updating_layer = False
    last_preview = NapariTilingPreview()
    tiles_are_current = False
    click_delete_enabled = False

    def update_filters(
        area_min: int,
        area_max: int,
        elongation_min: float,
        elongation_max: float,
        max_mean_3nn_distance: float,
        min_weighted_confidence: float,
        requested_count: int,
        *,
        update_tiles: bool = False,
    ) -> None:
        nonlocal active_labels, tile_selected_labels, updating_layer
        nonlocal last_preview, tiles_are_current
        if not update_tiles:
            tiles_are_current = False
        area_max_value = None if area_max <= 0 else max(area_min, area_max)
        elongation_max_value = max(elongation_min, elongation_max)
        max_mean_3nn_distance_value = (
            None if max_mean_3nn_distance <= 0 else max_mean_3nn_distance
        )
        min_confidence_value = (
            None if min_weighted_confidence <= 0 else min_weighted_confidence
        )
        limits = ComponentFilterLimits(
            min_area_px=area_min,
            max_area_px=area_max_value,
            min_elongation=elongation_min,
            max_elongation=elongation_max_value,
            max_mean_3nn_distance_um=max_mean_3nn_distance_value,
            min_size_weighted_confidence=min_confidence_value,
        )
        candidate_mask, candidate_measurements = filter_component_labels(
            component_labels,
            measurements,
            limits,
            pixel_size_x_um=pixel_size_x_um,
            pixel_size_y_um=pixel_size_y_um,
            excluded_labels=deleted_labels,
        )
        kept_measurements = candidate_measurements
        if requested_count > 0 and len(candidate_measurements) > requested_count:
            if update_tiles and selection_callback is not None:
                kept_measurements = selection_callback(
                    candidate_measurements,
                    requested_count,
                    candidate_mask,
                    component_labels,
                    np.asarray(prediction_labels),
                )
                tile_selected_labels = {
                    measurement.label for measurement in kept_measurements
                }
            elif tile_selected_labels is not None:
                cached_measurements = tuple(
                    measurement
                    for measurement in candidate_measurements
                    if measurement.label in tile_selected_labels
                )
                if len(cached_measurements) >= requested_count:
                    kept_measurements = cached_measurements[:requested_count]
                else:
                    kept_measurements = _fill_count_selection_by_confidence(
                        cached_measurements,
                        candidate_measurements,
                        requested_count,
                    )
            else:
                kept_measurements = tuple(
                    sorted(
                        candidate_measurements,
                        key=lambda item: _finite_score(item.size_weighted_confidence),
                        reverse=True,
                    )[:requested_count]
                )
        elif requested_count <= 0:
            tile_selected_labels = None
        kept_labels = {measurement.label for measurement in kept_measurements}
        mask = (
            np.isin(component_labels, list(kept_labels))
            if kept_labels
            else np.zeros_like(component_labels, dtype=bool)
        )

        if update_tiles and preview_callback is not None:
            last_preview = preview_callback(
                mask,
                tuple(kept_measurements),
                np.asarray(prediction_labels),
            )
            _update_preview_layers(
                last_preview,
                tile_shapes_layer=tile_shapes_layer,
                region_shapes_layer=region_shapes_layer,
                target_points_layer=target_points_layer,
            )
            tiles_are_current = True

        try:
            updating_layer = True
            filtered_layer.data = _filtered_component_data(mask, component_labels)
        finally:
            updating_layer = False
        active_labels = kept_labels
        preview_suffix = _preview_status_suffix(
            last_preview,
            preview_enabled=preview_callback is not None,
            tiles_are_current=tiles_are_current,
        )
        filtered_layer.name = (
            f"Filtered detections ({len(kept_measurements)} objects"
            f"{preview_suffix}, {len(deleted_labels)} deleted)"
        )
        selected["mask"] = mask
        selected["measurements"] = kept_measurements

    initial_max_area = 0 if max_area_px is None else max_area_px
    initial_max_mean_3nn_distance = (
        0.0 if max_mean_3nn_distance_um is None else max_mean_3nn_distance_um
    )
    initial_min_confidence = (
        0.0
        if min_size_weighted_confidence is None
        else min_size_weighted_confidence
    )
    initial_max_selected_objects = (
        0 if max_selected_objects is None else max(0, max_selected_objects)
    )

    filter_widget = QWidget()
    form = QFormLayout(filter_widget)

    min_area_widget = QSpinBox()
    min_area_widget.setRange(0, 100_000_000)
    min_area_widget.setSingleStep(10)
    min_area_widget.setValue(min_area_px)
    form.addRow("Min area px", min_area_widget)

    max_area_widget = QSpinBox()
    max_area_widget.setRange(0, 100_000_000)
    max_area_widget.setSingleStep(10)
    max_area_widget.setSpecialValueText("No maximum")
    max_area_widget.setValue(initial_max_area)
    form.addRow("Max area px", max_area_widget)

    min_elongation_widget = QDoubleSpinBox()
    min_elongation_widget.setRange(0.0, 1.0)
    min_elongation_widget.setSingleStep(0.01)
    min_elongation_widget.setDecimals(2)
    min_elongation_widget.setValue(min_elongation)
    form.addRow("Min elongation", min_elongation_widget)

    max_elongation_widget = QDoubleSpinBox()
    max_elongation_widget.setRange(0.0, 1.0)
    max_elongation_widget.setSingleStep(0.01)
    max_elongation_widget.setDecimals(2)
    max_elongation_widget.setValue(max_elongation)
    form.addRow("Max elongation", max_elongation_widget)

    max_mean_3nn_distance_widget = QDoubleSpinBox()
    max_mean_3nn_distance_widget.setRange(0.0, 1_000_000.0)
    max_mean_3nn_distance_widget.setSingleStep(10.0)
    max_mean_3nn_distance_widget.setDecimals(1)
    max_mean_3nn_distance_widget.setSpecialValueText("No maximum")
    max_mean_3nn_distance_widget.setValue(initial_max_mean_3nn_distance)
    form.addRow("Max mean 3NN distance um", max_mean_3nn_distance_widget)

    min_confidence_widget = QDoubleSpinBox()
    min_confidence_widget.setRange(0.0, 1_000_000_000.0)
    min_confidence_widget.setSingleStep(1.0)
    min_confidence_widget.setDecimals(3)
    min_confidence_widget.setSpecialValueText("No minimum")
    min_confidence_widget.setValue(initial_min_confidence)
    form.addRow("Min weighted confidence", min_confidence_widget)

    object_count_widget = QSpinBox()
    object_count_widget.setRange(0, max(100_000_000, len(measurements)))
    object_count_widget.setSingleStep(1)
    object_count_widget.setSpecialValueText("All filtered")
    object_count_widget.setValue(initial_max_selected_objects)
    form.addRow("Selected object count", object_count_widget)

    deleted_label = QLabel("Manual deletions: 0")
    form.addRow(deleted_label)

    reset_deleted_button = QPushButton("Reset manual deletions")
    form.addRow(reset_deleted_button)

    click_delete_button = QPushButton("Enable click-delete")
    form.addRow(click_delete_button)

    update_tiles_button = QPushButton("Update tiles/count selection")
    form.addRow(update_tiles_button)

    ok_button = QPushButton("OK")
    form.addRow(ok_button)

    def refresh_from_widget_values() -> None:
        deleted_label.setText(f"Manual deletions: {len(deleted_labels)}")
        update_filters(
            min_area_widget.value(),
            max_area_widget.value(),
            min_elongation_widget.value(),
            max_elongation_widget.value(),
            max_mean_3nn_distance_widget.value(),
            min_confidence_widget.value(),
            object_count_widget.value(),
            update_tiles=False,
        )

    def update_tiles_from_widget_values(_checked: bool = False) -> None:
        deleted_label.setText(f"Manual deletions: {len(deleted_labels)}")
        update_filters(
            min_area_widget.value(),
            max_area_widget.value(),
            min_elongation_widget.value(),
            max_elongation_widget.value(),
            max_mean_3nn_distance_widget.value(),
            min_confidence_widget.value(),
            object_count_widget.value(),
            update_tiles=True,
        )

    def reset_manual_deletions(_checked: bool = False) -> None:
        deleted_labels.clear()
        refresh_from_widget_values()

    def toggle_click_delete(_checked: bool = False) -> None:
        nonlocal click_delete_enabled
        click_delete_enabled = not click_delete_enabled
        click_delete_button.setText(
            "Disable click-delete" if click_delete_enabled else "Enable click-delete"
        )

    def record_manual_deletions(_event: Any = None) -> None:
        if updating_layer or not active_labels:
            return
        current_data = np.asarray(filtered_layer.data)
        newly_deleted = {
            label
            for label in active_labels
            if not np.any(current_data[component_labels == label] == label)
        }
        if newly_deleted:
            deleted_labels.update(newly_deleted)
            refresh_from_widget_values()

    def delete_clicked_object(layer: Any, event: Any) -> None:
        if not click_delete_enabled:
            return
        yx = _event_data_yx(layer, event)
        if yx is None:
            return
        y_px, x_px = yx
        current_data = np.asarray(filtered_layer.data)
        if (
            y_px < 0
            or y_px >= current_data.shape[0]
            or x_px < 0
            or x_px >= current_data.shape[1]
        ):
            return
        label = int(current_data[y_px, x_px])
        if label <= 0:
            return
        deleted_labels.add(label)
        refresh_from_widget_values()

    def accept_filters(_checked: bool = False) -> None:
        refresh_from_widget_values()
        selected["accepted"] = True
        QTimer.singleShot(100, viewer.close)

    min_area_widget.valueChanged.connect(refresh_from_widget_values)
    max_area_widget.valueChanged.connect(refresh_from_widget_values)
    min_elongation_widget.valueChanged.connect(refresh_from_widget_values)
    max_elongation_widget.valueChanged.connect(refresh_from_widget_values)
    max_mean_3nn_distance_widget.valueChanged.connect(refresh_from_widget_values)
    min_confidence_widget.valueChanged.connect(refresh_from_widget_values)
    object_count_widget.valueChanged.connect(refresh_from_widget_values)
    reset_deleted_button.pressed.connect(reset_manual_deletions)
    click_delete_button.pressed.connect(toggle_click_delete)
    update_tiles_button.pressed.connect(update_tiles_from_widget_values)
    filtered_layer.events.data.connect(record_manual_deletions)
    filtered_layer.mouse_drag_callbacks.append(delete_clicked_object)
    ok_button.pressed.connect(accept_filters)

    update_filters(
        min_area_px,
        initial_max_area,
        min_elongation,
        max_elongation,
        initial_max_mean_3nn_distance,
        initial_min_confidence,
        initial_max_selected_objects,
        update_tiles=False,
    )
    viewer.window.add_dock_widget(filter_widget, area="right", name="Filters")
    napari.run()

    if not selected["accepted"]:
        raise RuntimeError("Interactive filtering was closed before pressing OK")
    return selected["mask"], selected["measurements"]


def _update_preview_layers(
    preview: NapariTilingPreview,
    *,
    tile_shapes_layer: Any,
    region_shapes_layer: Any,
    target_points_layer: Any,
) -> None:
    if tile_shapes_layer is not None:
        tile_shapes_layer.data = list(preview.tile_rectangles_yx)
    if region_shapes_layer is not None:
        region_shapes_layer.data = list(preview.region_outlines_yx)
    if target_points_layer is not None:
        if preview.point_centers_yx:
            target_points_layer.data = np.asarray(
                preview.point_centers_yx,
                dtype=np.float32,
            )
        else:
            target_points_layer.data = np.empty((0, 2), dtype=np.float32)


def _filtered_component_data(mask: np.ndarray, component_labels: np.ndarray) -> np.ndarray:
    return np.where(mask, component_labels, 0).astype(component_labels.dtype, copy=False)


def _event_data_yx(layer: Any, event: Any) -> tuple[int, int] | None:
    position = getattr(event, "position", None)
    if position is None:
        return None
    try:
        data_position = layer.world_to_data(position)
    except Exception:
        data_position = position
    if len(data_position) < 2:
        return None
    y_px = int(round(float(data_position[-2])))
    x_px = int(round(float(data_position[-1])))
    return y_px, x_px


def _preview_status_suffix(
    preview: NapariTilingPreview,
    *,
    preview_enabled: bool,
    tiles_are_current: bool,
) -> str:
    if not preview_enabled:
        return ""
    if tiles_are_current:
        return f", {preview.tile_count} tiles"
    if preview.tile_count > 0:
        return f", last {preview.tile_count} tiles"
    return ", tiles not updated"


def _validate_filter_limits(limits: ComponentFilterLimits) -> None:
    if limits.min_area_px < 0:
        raise ValueError("min_area_px must be non-negative")
    if limits.max_area_px is not None and limits.max_area_px < limits.min_area_px:
        raise ValueError("max_area_px must be greater than or equal to min_area_px")
    if not 0.0 <= limits.min_elongation <= 1.0:
        raise ValueError("min_elongation must be in [0, 1]")
    if not 0.0 <= limits.max_elongation <= 1.0:
        raise ValueError("max_elongation must be in [0, 1]")
    if limits.max_elongation < limits.min_elongation:
        raise ValueError(
            "max_elongation must be greater than or equal to min_elongation"
        )
    if (
        limits.max_mean_3nn_distance_um is not None
        and limits.max_mean_3nn_distance_um < 0
    ):
        raise ValueError("max_mean_3nn_distance_um must be non-negative")
    if (
        limits.min_size_weighted_confidence is not None
        and limits.min_size_weighted_confidence < 0
    ):
        raise ValueError("min_size_weighted_confidence must be non-negative")


def _mean_nearest_neighbor_distances_um(
    centroids_xy_px: list[tuple[float, float]],
    *,
    pixel_size_x_um: float,
    pixel_size_y_um: float,
    neighbor_count: int,
) -> np.ndarray:
    if pixel_size_x_um <= 0 or pixel_size_y_um <= 0:
        raise ValueError("pixel sizes must be positive")
    if neighbor_count <= 0:
        raise ValueError("neighbor_count must be positive")

    if not centroids_xy_px:
        return np.asarray([], dtype=np.float32)
    if len(centroids_xy_px) == 1:
        return np.asarray([np.inf], dtype=np.float32)

    from scipy.spatial import cKDTree

    centroids = np.asarray(centroids_xy_px, dtype=np.float32)
    centroids_physical = centroids * np.asarray(
        [pixel_size_x_um, pixel_size_y_um],
        dtype=np.float32,
    )
    query_count = min(neighbor_count + 1, len(centroids_xy_px))
    tree = cKDTree(centroids_physical)
    distances, _indices = tree.query(centroids_physical, k=query_count)
    distances = np.asarray(distances, dtype=np.float32)
    if distances.ndim == 1:
        distances = distances[:, np.newaxis]
    nearest_distances = distances[:, 1:]
    return np.mean(nearest_distances, axis=1, dtype=np.float32)


def _measurements_with_neighbor_distances(
    measurements: list[ComponentMeasurement],
    *,
    pixel_size_x_um: float,
    pixel_size_y_um: float,
) -> list[ComponentMeasurement]:
    distances = _mean_nearest_neighbor_distances_um(
        [
            (measurement.centroid_x_px, measurement.centroid_y_px)
            for measurement in measurements
        ],
        pixel_size_x_um=pixel_size_x_um,
        pixel_size_y_um=pixel_size_y_um,
        neighbor_count=3,
    )
    return [
        replace(measurement, mean_3nn_distance_um=float(distance))
        for measurement, distance in zip(measurements, distances)
    ]


def _component_measurement_maps(
    component_labels: np.ndarray,
    measurements: tuple[ComponentMeasurement, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    area_values = np.zeros(int(component_labels.max()) + 1, dtype=np.float32)
    elongation_values = np.zeros_like(area_values)
    mean_3nn_distance_values = np.zeros_like(area_values)
    size_weighted_confidence_values = np.zeros_like(area_values)
    for measurement in measurements:
        area_values[measurement.label] = measurement.area_px
        elongation_values[measurement.label] = measurement.elongation
        mean_3nn_distance_values[measurement.label] = measurement.mean_3nn_distance_um
        size_weighted_confidence_values[measurement.label] = _finite_score(
            measurement.size_weighted_confidence
        )
    return (
        area_values[component_labels],
        elongation_values[component_labels],
        mean_3nn_distance_values[component_labels],
        size_weighted_confidence_values[component_labels],
    )


def _finite_score(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


def _fill_count_selection_by_confidence(
    selected: tuple[ComponentMeasurement, ...],
    candidates: tuple[ComponentMeasurement, ...],
    requested_count: int,
) -> tuple[ComponentMeasurement, ...]:
    selected_labels = {measurement.label for measurement in selected}
    remaining = [
        measurement
        for measurement in candidates
        if measurement.label not in selected_labels
    ]
    remaining.sort(
        key=lambda item: _finite_score(item.size_weighted_confidence),
        reverse=True,
    )
    return tuple((list(selected) + remaining)[:requested_count])
