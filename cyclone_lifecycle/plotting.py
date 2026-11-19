"""Shared plotting helpers for model and cross-model visualizations."""

from typing import Optional, Sequence, Tuple

import numpy as np


VARIABLE_DISPLAY_UNITS = {
    "msl": (lambda values: values / 100.0, "Mean sea level pressure, hPa"),
}


def convert_for_display(variable_name: str, values: np.ndarray) -> Tuple[np.ndarray, str]:
    """Convert physical data to the display units and label used by project plots."""
    if variable_name in VARIABLE_DISPLAY_UNITS:
        convert, label = VARIABLE_DISPLAY_UNITS[variable_name]
        return convert(values), label
    return values, variable_name


def plot_comparison_triplet(
    figure,
    axes: Sequence,
    target: np.ndarray,
    prediction: np.ndarray,
    titles: Sequence[str],
    extent=None,
    origin: str = "upper",
    show_coordinates: bool = False,
    label_fontsize: Optional[int] = None,
):
    """Render target, prediction, and absolute difference with consistent scales."""
    lower = min(np.nanmin(target), np.nanmin(prediction))
    upper = max(np.nanmax(target), np.nanmax(prediction))
    panels = (
        (target, lower, upper, "viridis"),
        (prediction, lower, upper, "viridis"),
        (np.abs(target - prediction), None, None, "inferno"),
    )
    images = []
    for axis, (values, vmin, vmax, cmap), title in zip(axes, panels, titles):
        image = axis.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, origin=origin)
        axis.set_title(title, fontsize=9)
        if show_coordinates:
            axis.set_xlabel("Longitude", fontsize=label_fontsize)
            axis.set_ylabel("Latitude", fontsize=label_fontsize)
        else:
            axis.axis("off")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        images.append(image)
    return images
