"""Compatibility exports; new code should import from :mod:`cyclone_lifecycle.dataset`."""

from .dataset import (
    CyclonePatchDataset,
    MaskedVAEFrameDataset,
    NetCDFSpatioTemporalDataset,
    PatchMeanTransform,
    SampleMetadata,
    collect_window_info,
    compute_normalization_stats,
    denormalize,
    denormalize_channels,
    fit_anomaly_scale,
    load_norm_stats,
    parse_patch_filename,
    read_window_info,
    save_norm_stats,
    stats_to_arrays,
)

__all__ = [
    "CyclonePatchDataset",
    "MaskedVAEFrameDataset",
    "NetCDFSpatioTemporalDataset",
    "PatchMeanTransform",
    "SampleMetadata",
    "collect_window_info",
    "compute_normalization_stats",
    "denormalize",
    "denormalize_channels",
    "fit_anomaly_scale",
    "load_norm_stats",
    "parse_patch_filename",
    "read_window_info",
    "save_norm_stats",
    "stats_to_arrays",
]
