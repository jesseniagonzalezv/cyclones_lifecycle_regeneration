"""
videomae_earth_pipeline.py
===========================
Single PyTorch script implementing the full requested pipeline:

  1. CONFIG        -> data/model geometry (T frames, 128x128, 7 channels).
  2. DATASET       -> reads NetCDF (.nc) files + per-channel Z-score normalization.
  3. PATCH UTILS   -> patchify/unpatchify (inverse reconstruction function), replicating
                       VideoMAE's internal reshape logic EXACTLY.
  4. MASKING       -> Mode A (spatial inpainting), Mode B (missing variable), and
                       Mode C (missing timestep), plus the matching fixed inference
                       masks (Case 1 / Case 2 / Case 3).
  5. MODEL UTILS   -> model construction and a "full-sequence" forward pass for Mode B.
  6. LOSSES        -> masked-patch MSE loss (Mode A / Mode C) and custom MSE loss (Mode B).
  7. TRAIN LOOP    -> native PyTorch loop alternating between Mode A / Mode B / Mode C,
                       so training masking patterns match every inference test case.
  8. INFERENCE     -> loads with .from_pretrained(), Case 1 (3x3 spatial block),
                       Case 2 (full removal of one variable), and Case 3 (full removal
                       of one timestep, all variables), reporting MAE.
  9. VISUALIZATION -> target vs. prediction vs. difference images, both for a single
                       channel and for ALL channels at once (see notes below).
 10. BASELINES     -> non-learned linear interpolation for Case 1 (spatial) and
                       Case 3 (timestep), reported alongside the model's MAE so you
                       can tell whether VideoMAE is actually beating a trivial guess.

IMPORTANT NOTE ON THE MODEL CLASS
----------------------------------
The original spec asked for `VideoMaeForVideoInpainting`. That class does NOT exist
in the `transformers` library (verified against transformers==5.14.1: only
VideoMAEModel, VideoMAEForPreTraining and VideoMAEForVideoClassification exist,
with "MAE" capitalized, not "Mae"). The class that implements exactly the requested
behaviour -- self-supervised masked-patch reconstruction, with
`forward(pixel_values, bool_masked_pos) -> (loss, logits)` -- is
`VideoMAEForPreTraining`. That is the class used throughout this script.

IMPORTANT NOTE ON AXIS ORDER
------------------------------
The project-level tensor layout is [B, C, T, H, W] (channel before time), matching
the original spec. HuggingFace's model internally expects [B, T, C, H, W] (time
before channel). The conversion between the two layouts happens in exactly one
place (`to_model_layout` / `to_project_layout`) so the rest of the code never has
to think about axis order again.

IMPORTANT NOTE ON THE NUMBER OF TIME FRAMES (T) -- READ THIS IF YOU ONLY HAVE A
SINGLE TIMESTEP PER SAMPLE
------------------------------------------------------------------------------------
Nothing in this pipeline hardcodes T=3. The ONLY real constraint from VideoMAE is
`num_frames % tubelet_size == 0`. With `tubelet_size=1` (used throughout this
script), ANY value of `num_frames` works, including `num_frames=1`.

Three days were used in the original example because it was assumed the model
"needed a video". If your actual goal is different -- recovering missing
variables or missing spatial areas for a SINGLE, specific timestep, with no
temporal modelling at all -- then `num_frames=1` is the correct, natural setting:
each sample becomes a "single-frame video" with 7 channels and no time axis to
reason about. Case 1 (spatial block) and Case 2 (missing variable) work exactly
the same way with T=1; the visualization functions below automatically adapt to
whatever `cfg.num_frames` you configure (they never assume 3 rows).

If your raw .nc files still contain multiple days (e.g. T_file=3) but you want
independent single-day training samples, set `cfg.file_num_frames = 3` and
`cfg.num_frames = 1`: the dataset will transparently split each file into 3
independent single-timestep samples (see `flatten_time` in
`NetCDFSpatioTemporalDataset`), which also multiplies your effective training set
size by 3 -- with no need to regenerate your NetCDF files.
"""

import argparse
import glob
import json
import os
import random
import re

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr

matplotlib.use("Agg")
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from torch.utils.data import DataLoader, Dataset
from transformers import VideoMAEConfig, VideoMAEForPreTraining


# =====================================================================================
# 1) CONFIG
# =====================================================================================
@dataclass
class ProjectConfig:
    # ------------------------------------------------------------------ Paths
    MAIN_ROOT  = Path("/scale/project/ifremer-isi-jumeaunumerique/NEXTGEMS_PROCESSED_LAT_LON")
    data_root: str = MAIN_ROOT / "data_128x128_3days"                     # contiene /data/train y /data/eval
    output_dir: str = MAIN_ROOT / "ml_result" / "videomae"       # destino de save_pretrained()
    train_dir: str = field(init=False)
    eval_dir: str = field(init=False)
    test_dir: str = field(init=False)
    norm_stats_path: str = field(init=False)              # per-channel mean/std, cached as JSON


    # ------------------------------------------------------------------ Physical variables (channels)
    # ORDER MATTERS: this fixes the channel axis of every tensor built by the Dataset.
    # Edit these 7 names so they match EXACTLY the variable names inside your .nc files.
    variable_names: List[str] = field(default_factory=lambda: [
        "10u", "10v", "msl", "avg_tos", "avg_zos", "avg_sos", "avg_mlotst125",
    ])

    # ------------------------------------------------------------------ Data geometry
    # `file_num_frames`: how many timesteps/days each RAW .nc file actually stores.
    # `num_frames`: how many timesteps go into ONE training sample fed to VideoMAE (T).
    #
    #   - If your .nc files already store exactly one timestep each:
    #         file_num_frames = 1, num_frames = 1, flatten_time = False
    #   - If your .nc files store, e.g., 3 consecutive days AND you want the model to
    #     reason jointly across those 3 days (temporal modelling):
    #         file_num_frames = 3, num_frames = 3, flatten_time = False
    #   - If your .nc files store 3 days but you want INDEPENDENT single-timestep
    #     samples (no temporal modelling, more training samples):
    #         file_num_frames = 3, num_frames = 1, flatten_time = True
    #
    # See the module docstring above for a full explanation.
    file_num_frames: int = 3   # timesteps physically stored in each .nc file
    num_frames: int = 3        # T: timesteps that make up ONE training sample
    flatten_time: bool = False # if True, split each multi-day file into num_frames=1 samples
    image_size: int = 128      # H = W = 128
    patch_size: int = 8 #16       # spatial patch side
    tubelet_size: int = 1      # temporal patch depth (1 => every frame is its own tubelet)

    # ------------------------------------------------------------------ Training hyper-parameters
    batch_size: int = 32 #8
    num_epochs: int = 100 #100 #0
    learning_rate: float = 1.5e-4
    weight_decay: float = 0.05
    mode_a_probability: float = 0.5   # P(Mode A) per iteration
    mode_c_probability: float = 0.2   # P(Mode C) per iteration; the rest (1 - A - C) is Mode B.
                                       # Automatically forced to 0 when num_frames == 1 (no time axis).
    mode_a_mask_ratio: float = 0.75   # fraction of patches hidden in Mode A
    patch_boundary_loss_weight: float = 0.1
    num_workers: int = 4
    seed: int = 42
    checkpoint_every_n_epochs: int = 10
    device: str = "cuda"  # falls back to "cpu" automatically if CUDA is unavailable

    # ------------------------------------------------------------------ Model dimensions (encoder/decoder)
    hidden_size: int = 384
    num_hidden_layers: int = 6
    num_attention_heads: int = 6
    intermediate_size: int = 1536
    decoder_hidden_size: int = 384 #192
    decoder_num_hidden_layers: int = 6 #3
    decoder_num_attention_heads: int = 6 #3
    decoder_intermediate_size: int = 768

    def __post_init__(self):
        self.train_dir = os.path.join(self.data_root, "train_patches")
        self.eval_dir = os.path.join(self.data_root, "val_patches")
        self.test_dir = os.path.join(self.data_root, "test_patches")

        self.norm_stats_path = os.path.join(self.output_dir, "norm_stats.json")

        if self.flatten_time and self.num_frames != 1:
            raise ValueError("flatten_time=True requires num_frames=1 (one timestep per sample).")
        if not self.flatten_time and self.num_frames != self.file_num_frames:
            raise ValueError(
                "When flatten_time=False, num_frames must equal file_num_frames "
                "(each sample uses the file's timesteps as-is)."
            )

    @property
    def num_channels(self) -> int:
        return len(self.variable_names)

    @property
    def patches_per_frame(self) -> int:
        return (self.image_size // self.patch_size) ** 2  # 8*8 = 64

    @property
    def num_patches(self) -> int:
        return (self.num_frames // self.tubelet_size) * self.patches_per_frame

    @property
    def patch_grid_side(self) -> int:
        return self.image_size // self.patch_size  # 8

    def build_videomae_config(self) -> VideoMAEConfig:
        """
        norm_pix_loss MUST be False for non-RGB data. The original
        VideoMAEForPreTraining implementation only knows how to "un-normalize"
        pixel_values using ImageNet mean/std when num_channels == 3; with 7
        physical channels that does not apply and would raise a ValueError.
        Our data is already Z-score normalized per channel by the Dataset.
        """
        return VideoMAEConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            num_frames=self.num_frames,
            tubelet_size=self.tubelet_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            decoder_hidden_size=self.decoder_hidden_size,
            decoder_num_hidden_layers=self.decoder_num_hidden_layers,
            decoder_num_attention_heads=self.decoder_num_attention_heads,
            decoder_intermediate_size=self.decoder_intermediate_size,
            norm_pix_loss=False,
            use_mean_pooling=False,
        )


# =====================================================================================
# 2) DATASET  (NetCDF -> tensor [C, T, H, W] + per-channel Z-score normalization)
# =====================================================================================
class NetCDFSpatioTemporalDataset(Dataset):
    """
    Internal layout: [C, T, H, W] (channel-major), as requested in the spec.
    Conversion to [T, C, H, W] (HuggingFace's expected layout) only ever happens
    in model_utils / to_model_layout, never here.

    Supports two modes, controlled by `flatten_time`:
      - flatten_time=False: one sample per file, with T = file_num_frames.
      - flatten_time=True:  one sample per (file, day) pair, with T = 1. This lets
        you reuse multi-day .nc files as independent single-timestep samples.
    """

    def __init__(
        self,
        directory: str,
        variable_names: List[str],
        file_num_frames: int,
        image_size: int,
        norm_stats: Optional[Dict[str, Dict[str, float]]] = None,
        flatten_time: bool = False,
    ):
        self.files = sorted(glob.glob(os.path.join(directory, "*.nc")))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .nc files found under {directory}")
        self.variable_names = variable_names
        self.file_num_frames = file_num_frames
        self.image_size = image_size
        self.norm_stats = norm_stats  # {var: {"mean":.., "std":..}}; None -> raw tensor
        self.flatten_time = flatten_time

    def __len__(self) -> int:
        if self.flatten_time:
            return len(self.files) * self.file_num_frames
        return len(self.files)

    def _load_raw_file(self, path: str) -> torch.Tensor:
        """Reads one .nc file and stacks its variables into [C, file_num_frames, H, W]."""
        ds = xr.open_dataset(path)
        channel_arrays = []
        for var in self.variable_names:
            if var not in ds.variables:
                raise KeyError(f"Variable '{var}' not found in {path}. "
                                f"Available: {list(ds.variables.keys())}")
            arr = np.asarray(ds[var].values, dtype=np.float32)  # expected shape (T_file, H, W)
            if arr.shape != (self.file_num_frames, self.image_size, self.image_size):
                raise ValueError(
                    f"Variable '{var}' in {path} has shape {arr.shape}, "
                    f"expected ({self.file_num_frames}, {self.image_size}, {self.image_size})"
                )
            channel_arrays.append(arr)
        ds.close()
        stacked = np.stack(channel_arrays, axis=0)  # [C, file_num_frames, H, W]
        return torch.from_numpy(stacked)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.flatten_time:
            file_idx = idx // self.file_num_frames
            day_idx = idx % self.file_num_frames
            full_tensor = self._load_raw_file(self.files[file_idx])   # [C, file_num_frames, H, W]
            tensor = full_tensor[:, day_idx:day_idx + 1, :, :].clone()  # [C, 1, H, W]
        else:
            tensor = self._load_raw_file(self.files[idx])             # [C, file_num_frames, H, W]

        if self.norm_stats is not None:
            for c, var in enumerate(self.variable_names):
                mean = self.norm_stats[var]["mean"]
                std = self.norm_stats[var]["std"]
                tensor[c] = (tensor[c] - mean) / (std + 1e-8)

        return tensor

    def get_metadata(self, idx: int, frame_idx: Optional[int] = None) -> "SampleMetadata":
        """
        Reads everything needed to plot and name a sample properly: the track
        filename fields (track_id/season/basin/rank), the lat/lon coordinate
        arrays, the cyclone center for the selected timestep, and the
        timestamp string. This is deliberately kept OUT of __getitem__ (which
        must stay a plain tensor for DataLoader batching/collation) and is
        only called directly for inference/visualization, one sample at a time.

        If `flatten_time` is False (one sample per file containing multiple
        timesteps), callers can pass `frame_idx` to select which timestep's
        per-frame metadata (time, cyclone center) to return. If `frame_idx`
        is omitted, a sensible default (middle timestep) is used so plots can
        show different times instead of always the same first timestep.
        """
        if self.flatten_time:
            file_idx = idx // self.file_num_frames
            day_idx = idx % self.file_num_frames
        else:
            file_idx = idx
            # choose frame index for per-timestep metadata: prefer explicit
            # `frame_idx`; if not provided, use middle frame so plots vary
            # across time instead of always using the first timestep.
            if frame_idx is None:
                day_idx = self.file_num_frames // 2
            else:
                day_idx = int(frame_idx) % self.file_num_frames

        path = self.files[file_idx]
        ds = xr.open_dataset(path)

        # file_info = parse_patch_filename(path)
        file_info = parse_patch_filename(ds)

        # lat/lon are expected as 1D coordinate arrays of length H and W
        # respectively (standard for a lat/lon patch grid).
        # here the 3 timesteps has the same, lat, lon double absolute_lat(time, lat) ; double absolute_lon(time, lon) ;
        lat = np.asarray(ds["absolute_lat"][0].values, dtype=np.float64) if "absolute_lat" in ds.coords or "absolute_lat" in ds.variables else None
        lon = np.asarray(ds["absolute_lon"][0].values, dtype=np.float64) if "absolute_lon" in ds.coords or "absolute_lon" in ds.variables else None

        # cyclone_center_lat/lon and time can be stored either as a single
        # scalar (one value for the whole file) or as one value per
        # timestep (shape [file_num_frames]) -- handle both.
        def _select_scalar_or_indexed(var_name: str):
            if var_name not in ds.variables:
                return None
            values = np.asarray(ds[var_name].values)
            if values.ndim == 0 or values.size == 1:
                return values.reshape(-1)[0]
            return values[day_idx]

        # cyclone center can be scalar or per-timestep; collect both
        latc = _select_scalar_or_indexed("cyclone_center_lat")
        lonc = _select_scalar_or_indexed("cyclone_center_lon")

        # Build per-timestep arrays for time and cyclone center when available
        time_val = None
        time_strs = None
        if "time" in ds.variables or "time" in ds.coords:
            time_raw = np.asarray(ds["time"].values)
            if time_raw.ndim == 0:
                time_single = time_raw
                time_val = np.datetime_as_string(time_single, unit="h")
                time_strs = [time_val]
            else:
                # array of times per file frame
                time_strs = [np.datetime_as_string(t, unit="h") for t in time_raw]
                time_val = time_strs[day_idx]

        cyclone_center_lats = None
        cyclone_center_lons = None
        if "cyclone_center_lat" in ds.variables:
            vals = np.asarray(ds["cyclone_center_lat"].values)
            if vals.ndim == 0 or vals.size == 1:
                cyclone_center_lats = np.array([float(vals.reshape(-1)[0])])
            else:
                cyclone_center_lats = np.asarray(vals, dtype=np.float64)
        if "cyclone_center_lon" in ds.variables:
            vals = np.asarray(ds["cyclone_center_lon"].values)
            if vals.ndim == 0 or vals.size == 1:
                cyclone_center_lons = np.array([float(vals.reshape(-1)[0])])
            else:
                cyclone_center_lons = np.asarray(vals, dtype=np.float64)

        ds.close()

        return SampleMetadata(
            source_file=path,
            track_id=file_info.get("track_id") if file_info else None,
            season=file_info.get("season") if file_info else None,
            basin=file_info.get("basin") if file_info else None,
            rank=file_info.get("rank") if file_info else None,
            win_idx=file_info.get("win_idx") if file_info else None,
            lat=lat,
            lon=lon,
            cyclone_center_lat=float(latc) if latc is not None and np.ndim(latc) == 0 else (
                float(cyclone_center_lats[len(cyclone_center_lats) // 2]) if cyclone_center_lats is not None else None
            ),
            cyclone_center_lon=float(lonc) if lonc is not None and np.ndim(lonc) == 0 else (
                float(cyclone_center_lons[len(cyclone_center_lons) // 2]) if cyclone_center_lons is not None else None
            ),
            time_str=time_val,
            time_strs=time_strs,
            cyclone_center_lats=cyclone_center_lats,
            cyclone_center_lons=cyclone_center_lons,
        )


@dataclass
class SampleMetadata:
    """Everything needed to title/name/geolocate a plot for one sample."""
    source_file: str
    track_id: Optional[str]
    season: Optional[str]
    basin: Optional[str]
    rank: Optional[str]
    win_idx: Optional[str]
    lat: Optional[np.ndarray]           # 1D array, length H
    lon: Optional[np.ndarray]           # 1D array, length W
    # Per-sample scalar metadata (keeps backwards compatibility): a
    # representative value (e.g. middle timestep) for convenience.
    cyclone_center_lat: Optional[float]
    cyclone_center_lon: Optional[float]
    time_str: Optional[str]             # e.g. "2020-09-14T06"
    # Per-timestep metadata (when the source file contains multiple
    # frames). These are optional; plotting helpers should prefer them
    # when available and indexed by the displayed frame index.
    time_strs: Optional[List[str]] = None
    cyclone_center_lats: Optional[np.ndarray] = None
    cyclone_center_lons: Optional[np.ndarray] = None


# def parse_patch_filename(path: str) -> Optional[Dict[str, str]]:
#     """
#     Parses filenames following the convention:
#         # old: patch_trackid_<track_id>_<season>_<basin>_<rank>...(.nc)
#     # old:e.g. "patch_trackid_2020123_2020_ATL_01.nc" ->
#             # patch_trackid_<track_id>_<season>_<basin>_<rank>...(.nc)

#     Returns None (and prints a warning) if the filename doesn't match this
#     convention -- callers should treat missing metadata gracefully rather
#     than crash on it.
#     """
#     parts = Path(path).stem.split("_")
#     if len(parts) >= 6 and parts[0] == "patch" and parts[1] == "trackid":
#         return {
#             "track_id": parts[2],
#             "season": parts[3],
#             "basin": parts[4],
#             "rank": parts[5],
#         }
#     print(f"Unexpected filename format (metadata will be blank): {path}")
#     return None

def parse_patch_filename(ds: xr.Dataset):
    """
    Returns None (and prints a warning) if the filename doesn't match this
    convention -- callers should treat missing metadata gracefully rather
    than crash on it.
    """
    if ds.attrs:
        return {
            "track_id": ds.attrs["track_id"],
            "season": ds.attrs["season"],
            "basin": ds.attrs["basin"],
            "rank": ds.attrs["rank"],
            "win_idx": ds.attrs["win_idx"],

        }
    else:
        print("There are no attributes in this dataset.")
        return None


def compute_normalization_stats(
    directory: str, variable_names: List[str], file_num_frames: int, image_size: int,
) -> Dict[str, Dict[str, float]]:
    """Per-channel mean/std computed over the ENTIRE training set (never over eval)."""
    raw_dataset = NetCDFSpatioTemporalDataset(
        directory, variable_names, file_num_frames, image_size, norm_stats=None, flatten_time=False
    )
    num_channels = len(variable_names)
    sum_ = np.zeros(num_channels, dtype=np.float64)
    sum_sq = np.zeros(num_channels, dtype=np.float64)
    count = 0
    for i in range(len(raw_dataset)):
        tensor = raw_dataset[i].numpy().astype(np.float64)  # [C, T, H, W]
        c = tensor.shape[0]
        flat = tensor.reshape(c, -1)
        sum_ += flat.sum(axis=1)
        sum_sq += (flat ** 2).sum(axis=1)
        count += flat.shape[1]
    mean = sum_ / count
    var = sum_sq / count - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-12))
    return {name: {"mean": float(mean[c]), "std": float(std[c])} for c, name in enumerate(variable_names)}


def save_norm_stats(stats: Dict[str, Dict[str, float]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)


def load_norm_stats(path: str) -> Dict[str, Dict[str, float]]:
    with open(path, "r") as f:
        return json.load(f)


def denormalize(
    pixel_values_ctwh: torch.Tensor, norm_stats: Dict[str, Dict[str, float]], variable_names: List[str],
) -> torch.Tensor:
    """
    Inverse of the Dataset's Z-score normalization: x_physical = x_norm * std + mean.
    Used ONLY for plotting -- training/inference always operate in normalized
    space, since that's what the model was trained on. Returns a NEW tensor
    (does not mutate the input), still in [B, C, T, H, W] project layout.
    """
    physical = pixel_values_ctwh.clone()
    for c, var in enumerate(variable_names):
        mean = norm_stats[var]["mean"]
        std = norm_stats[var]["std"]
        physical[:, c] = physical[:, c] * std + mean
    return physical


# Per-variable display conversion: maps the raw physical value (as stored in the
# .nc file, e.g. Pa for sea-level pressure) to the unit you actually want to see
# on a plot, plus a human-readable axis label. Add more entries here as needed;
# any variable not listed is displayed as-is, labeled with its raw name.
VARIABLE_DISPLAY_UNITS: Dict[str, Tuple] = {
    "msl": (lambda x: x / 100.0, "Mean sea level pressure, hPa"),
}


def convert_for_display(variable_name: str, values: np.ndarray) -> Tuple[np.ndarray, str]:
    """Applies the unit conversion above, if one is registered for this variable."""
    if variable_name in VARIABLE_DISPLAY_UNITS:
        convert_fn, label = VARIABLE_DISPLAY_UNITS[variable_name]
        return convert_fn(values), label
    return values, variable_name


# =====================================================================================
# 3) PATCH UTILS  (patchify / unpatchify -- same algebra VideoMAE uses internally)
# =====================================================================================
def patchify(pixel_values: torch.Tensor, tubelet_size: int, patch_size: int) -> torch.Tensor:
    """
    [B, T, C, H, W] -> [B, N, ts*ps*ps*C], N = (T//ts)*(H//ps)*(W//ps).
    Replicates EXACTLY the view+permute+view sequence used internally by
    modeling_videomae.py, so patch index `i` here matches 1-to-1 with patch
    index `i` in bool_masked_pos and in the model's decoder logits.
    """
    B, T, C, H, W = pixel_values.shape
    ts, ps = tubelet_size, patch_size
    # Step 1: split T into (T//ts, ts) and H, W into (H//ps, ps), (W//ps, ps)
    x = pixel_values.view(B, T // ts, ts, C, H // ps, ps, W // ps, ps)
    # Step 2: reorder so all pixels belonging to one spatio-temporal patch are contiguous
    x = x.permute(0, 1, 4, 6, 2, 5, 7, 3).contiguous()
    # Step 3: flatten the patch grid into N, and (ts, ps, ps, C) into the feature dim
    x = x.view(B, (T // ts) * (H // ps) * (W // ps), ts * ps * ps * C)
    return x


def unpatchify(
    patches: torch.Tensor, tubelet_size: int, patch_size: int,
    num_frames: int, image_size: int, num_channels: int,
) -> torch.Tensor:
    """
    Inverse of `patchify`: [B, N, ts*ps*ps*C] -> [B, T, C, H, W].
    This is the "inverse mathematical function" required by the spec to unpack
    VideoMAE's flat patch predictions back into the original tensor shape.
    """
    B, N, feat = patches.shape
    ts, ps, C = tubelet_size, patch_size, num_channels
    T, H, W = num_frames, image_size, image_size
    assert feat == ts * ps * ps * C, f"Unexpected feature dim: {feat} != {ts*ps*ps*C}"
    assert N == (T // ts) * (H // ps) * (W // ps), "Unexpected number of patches"
    # Step 1: undo the final view
    x = patches.view(B, T // ts, H // ps, W // ps, ts, ps, ps, C)
    # Step 2: inverse permutation of (0,1,4,6,2,5,7,3) -> (0,1,4,7,2,5,3,6)
    x = x.permute(0, 1, 4, 7, 2, 5, 3, 6).contiguous()
    # Step 3: merge (T//ts, ts) -> T, (H//ps, ps) -> H, (W//ps, ps) -> W
    x = x.view(B, T, C, H, W)
    return x


# def unpatchify(
#     patches: torch.Tensor, 
#     tubelet_size: int, 
#     patch_size: int,
#     num_frames: int, 
#     image_size: int, 
#     num_channels: int,
# ) -> torch.Tensor:
#     """
#     Correctly unpacks Hugging Face VideoMAE decoder logits:
#     [B, N, ts*ps*ps*C] -> [B, T, C, H, W]
#     """
#     B, N, feat = patches.shape
#     ts, ps, C = tubelet_size, patch_size, num_channels
#     T, H, W = num_frames, image_size, image_size

#     gh = H // ps  # Number of patches along Height (128 // 8 = 16)
#     gw = W // ps  # Number of patches along Width  (128 // 8 = 16)
#     gt = T // ts  # Number of temporal tubelets   (3 // 1 = 3)

#     # 1. Reshape patch vector into grid layout
#     # HF VideoMAE patch order is (B, gt, gh, gw, ts, ps, ps, C)
#     x = patches.view(B, gt, gh, gw, ts, ps, ps, C)

#     # 2. Permute dimensions back to standard video layout
#     # Move channels (C) up, merge temporal (gt * ts), height (gh * ps), width (gw * ps)
#     x = x.permute(0, 4, 7, 1, 2, 5, 3, 6).contiguous()
#     # Shapes after permute: (B, ts, C, gt, gh, ps, gw, ps)

#     # 3. Collapse merged dimensions into (B, T, C, H, W)
#     x = x.view(B, ts * gt, C, gh * ps, gw * ps)
    
#     # 4. Reorder to match PyTorch video shape (B, T, C, H, W) -> if your pipeline uses (B, T, C, H, W)
#     # Note: If your original tensor was (B, T, C, H, W), return x as (B, T, C, H, W):
#     return x


def extract_channel_from_patches(
    patches: torch.Tensor, channel_idx: int, tubelet_size: int, patch_size: int, num_channels: int,
) -> torch.Tensor:
    """Slices a single physical channel out of flat patch tokens -> [B, N, ts*ps*ps]."""
    B, N, feat = patches.shape
    ts, ps, C = tubelet_size, patch_size, num_channels
    x = patches.view(B, N, ts, ps, ps, C)
    x = x[..., channel_idx]
    x = x.reshape(B, N, ts * ps * ps)
    return x


# =====================================================================================
# 4) MASKING  (Mode A / Mode B / inference test cases)
# =====================================================================================
def random_mask_mode_a(batch_size: int, num_patches: int, mask_ratio: float, device: torch.device) -> torch.Tensor:
    """Mode A: same NUMBER of hidden patches per sample (required by the decoder's batching)."""
    num_masked = int(round(num_patches * mask_ratio))
    bool_masked_pos = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
    for b in range(batch_size):
        perm = torch.randperm(num_patches, device=device)
        bool_masked_pos[b, perm[:num_masked]] = True
    return bool_masked_pos


def no_mask_mode_b(batch_size: int, num_patches: int, device: torch.device) -> torch.Tensor:
    """Mode B: nothing is spatially hidden; every patch stays visible to the encoder."""
    return torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)


def random_mask_mode_c(
    batch_size: int, num_patches: int, patches_per_frame: int, num_frames: int,
    tubelet_size: int, device: torch.device,
) -> torch.Tensor:
    """
    Mode C: hides one ENTIRE timestep per sample (all spatial locations, all
    variables, since each patch token already mixes every channel together).
    A different random timestep is chosen per sample in the batch, so the
    model sees every possible "missing day" position during training, not
    just one fixed one. This trains the exact masking pattern used by
    inference Case 3 (full-timestep recovery) -- without it, the model only
    ever saw scattered random patches (Mode A) during training and had never
    been asked to reconstruct a whole frame from pure temporal context.
    Requires num_frames > 1 (the caller must guard this).
    """
    num_tubelets = num_frames // tubelet_size
    bool_masked_pos = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
    for b in range(batch_size):
        t = random.randrange(num_tubelets)
        start = t * patches_per_frame
        end = start + patches_per_frame
        bool_masked_pos[b, start:end] = True
    return bool_masked_pos


def zero_out_channel(pixel_values_ctwh: torch.Tensor, channel_idx: int) -> torch.Tensor:
    """Simulates total absence of one physical variable: overwrites its entire [T,H,W] slice with 0.0."""
    corrupted = pixel_values_ctwh.clone()
    corrupted[:, channel_idx, :, :, :] = 0.0
    return corrupted


def central_block_patch_indices(grid_side: int, num_frames: int, tubelet_size: int, block_size: int = 3) -> torch.Tensor:
    """Flat patch indices of a block_size x block_size block centered in the grid, repeated per tubelet."""
    start = (grid_side - block_size) // 2
    rows = list(range(start, start + block_size))
    cols = list(range(start, start + block_size))
    num_tubelets = num_frames // tubelet_size
    indices = []
    for t in range(num_tubelets):
        for r in rows:
            for c in cols:
                indices.append(t * (grid_side * grid_side) + r * grid_side + c)
    return torch.tensor(sorted(set(indices)), dtype=torch.long)


def build_case1_mask(
    batch_size: int, num_patches: int, grid_side: int, num_frames: int,
    tubelet_size: int, block_size: int, device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bool_masked_pos = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
    idx = central_block_patch_indices(grid_side, num_frames, tubelet_size, block_size).to(device)
    bool_masked_pos[:, idx] = True
    return bool_masked_pos, idx


def build_case3_missing_timestep_mask(
    batch_size: int, num_patches: int, patches_per_frame: int, num_frames: int,
    tubelet_size: int, target_frame_idx: int, device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CASE 3: hides every patch belonging to ONE ENTIRE timestep (all spatial
    locations, i.e. all 7 variables at that timestep, since each patch token
    already mixes every channel together via the patch embedding). The
    encoder only ever sees the OTHER timesteps; the decoder reconstructs the
    whole missing frame purely from temporal context -- no cross-channel
    inference needed, unlike Case 2. Requires cfg.num_frames > 1 (nothing to
    condition on with a single-timestep sample).
    """
    target_tubelet = target_frame_idx // tubelet_size
    start = target_tubelet * patches_per_frame
    end = start + patches_per_frame
    idx = torch.arange(start, end, dtype=torch.long, device=device)

    bool_masked_pos = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
    bool_masked_pos[:, idx] = True
    return bool_masked_pos, idx


# =====================================================================================
# 5) MODEL UTILS
# =====================================================================================
def build_model(cfg: ProjectConfig) -> VideoMAEForPreTraining:
    return VideoMAEForPreTraining(cfg.build_videomae_config())


def to_model_layout(pixel_values_ctwh: torch.Tensor) -> torch.Tensor:
    """[B, C, T, H, W] (project layout) -> [B, T, C, H, W] (HuggingFace layout)."""
    return pixel_values_ctwh.permute(0, 2, 1, 3, 4).contiguous()


def to_project_layout(pixel_values_tchw: torch.Tensor) -> torch.Tensor:
    """[B, T, C, H, W] (HuggingFace layout) -> [B, C, T, H, W] (project layout)."""
    return pixel_values_tchw.permute(0, 2, 1, 3, 4).contiguous()


def forward_full_sequence(model: VideoMAEForPreTraining, pixel_values_ctwh: torch.Tensor, bool_masked_pos: torch.Tensor) -> torch.Tensor:
    """
    Manual replica of VideoMAEForPreTraining.forward() up to `logits`, WITHOUT
    running HuggingFace's internal loss computation.

    This is required for BOTH modes, not only Mode B:

    - Mode B: with bool_masked_pos entirely False, HF's internal loss block builds
      an empty `labels` tensor that doesn't broadcast against `logits`
      (RuntimeError).
    - Mode A: in transformers==5.14.1, the internal loss block -- even with
      `config.norm_pix_loss=False` -- checks `if self.config.num_channels != 3:
      raise ValueError(...)`. In other words, that code path literally does not
      support non-RGB data (like our 7 physical channels) regardless of
      norm_pix_loss. This looks like a library inconsistency/bug.

    Because of this, the script never calls `model(...).loss`. Instead it
    manually replicates the encoder+decoder forward pass (identical to the
    original) and computes the loss ourselves with our own `patchify` (see
    `masked_patch_mse_loss_mode_a` and `mode_b_variable_inference_loss`),
    completely avoiding the problematic internal path.

    With a real, partial `bool_masked_pos` (Mode A) this returns logits ONLY
    for the masked patches (same behaviour as the original decoder). With
    `bool_masked_pos` entirely False (Mode B) it returns logits for ALL
    patches (thanks to `hidden_states[:, -0:]` in VideoMAEDecoder returning
    the full sequence when return_token_num == 0).
    """
    pixel_values_tchw = to_model_layout(pixel_values_ctwh)

    outputs = model.videomae(pixel_values_tchw, bool_masked_pos=bool_masked_pos)
    sequence_output = model.encoder_to_decoder(outputs.last_hidden_state)
    batch_size, _, num_channels = sequence_output.shape

    expanded_pos_emb = model.position_embeddings.expand(batch_size, -1, -1).type_as(pixel_values_tchw)
    expanded_pos_emb = expanded_pos_emb.detach().to(device=pixel_values_tchw.device, copy=True)
    pos_emb_visible = expanded_pos_emb[~bool_masked_pos].reshape(batch_size, -1, num_channels)
    pos_emb_mask = expanded_pos_emb[bool_masked_pos].reshape(batch_size, -1, num_channels)

    # bool_masked_pos entirely False in Mode B -> pos_emb_mask is empty, and
    # x_full is simply the full encoded sequence, in original patch order.
    x_full = torch.cat([sequence_output + pos_emb_visible, model.mask_token + pos_emb_mask], dim=1)

    # `hidden_states[:, -return_token_num:]` in VideoMAEDecoder returns the FULL
    # sequence when return_token_num == 0 (because -0 == 0 in Python slicing),
    # which is exactly the behaviour we rely on here.
    decoder_outputs = model.decoder(x_full, pos_emb_mask.shape[1])
    return decoder_outputs.logits


# =====================================================================================
# 6) LOSSES
# =====================================================================================
def masked_patch_mse_loss_mode_a(
    logits: torch.Tensor, clean_pixel_values_ctwh: torch.Tensor, bool_masked_pos: torch.Tensor,
    tubelet_size: int, patch_size: int,
) -> torch.Tensor:
    """
    Mode A MSE loss (spatial recovery), computed outside the model using our own
    `patchify` (see the note in `forward_full_sequence` on why we do not use
    `model(...).loss`).

    logits: [B, num_masked, ts*ps*ps*C] -- decoder output ONLY for patches marked
        True in bool_masked_pos (same order used internally by HF's forward).
    clean_pixel_values_ctwh: [B, C, T, H, W] the clean, uncorrupted tensor (in
        Mode A the model input is already clean; only patch POSITIONS are hidden,
        pixel values are never altered).
    bool_masked_pos: [B, N] boolean mask used in the forward pass.
    """
    clean_tchw = clean_pixel_values_ctwh.permute(0, 2, 1, 3, 4).contiguous()  # [B,C,T,H,W]->[B,T,C,H,W]
    target_patches_full = patchify(clean_tchw, tubelet_size, patch_size)  # [B, N, feat]

    batch_size, _, feat = target_patches_full.shape
    # Select, per sample, only the masked patches, preserving the same order
    # HuggingFace's own decoder uses internally.
    target_masked = target_patches_full[bool_masked_pos].reshape(batch_size, -1, feat)

    return F.mse_loss(logits, target_masked)


def mode_b_variable_inference_loss(
    logits: torch.Tensor, clean_pixel_values_ctwh: torch.Tensor, channel_idx: int,
    tubelet_size: int, patch_size: int, num_channels: int,
) -> torch.Tensor:
    """
    Mode B custom MSE loss (variable inference). Compares the decoder logits (for
    ALL patches) against the ORIGINAL CLEAN tensor (before zeroing the target
    channel), but ONLY on the target channel. This forces the network to infer
    the hidden variable purely through inter-variable (cross-channel) attention,
    since that channel carried zero information going into the encoder.
    """
    clean_tchw = clean_pixel_values_ctwh.permute(0, 2, 1, 3, 4).contiguous()
    target_patches = patchify(clean_tchw, tubelet_size, patch_size)
    pred_channel = extract_channel_from_patches(logits, channel_idx, tubelet_size, patch_size, num_channels)
    target_channel = extract_channel_from_patches(target_patches, channel_idx, tubelet_size, patch_size, num_channels)
    return F.mse_loss(pred_channel, target_channel)


def patch_boundary_gradient_loss(
    logits: torch.Tensor,
    clean_pixel_values_ctwh: torch.Tensor,
    bool_masked_pos: torch.Tensor,
    tubelet_size: int,
    patch_size: int,
    num_frames: int,
    image_size: int,
    num_channels: int,
    channel_idx: Optional[int] = None,
) -> torch.Tensor:
    """Match target gradients at patch seams for masked and full predictions."""
    target_patches = patchify(to_model_layout(clean_pixel_values_ctwh), tubelet_size, patch_size)
    if bool_masked_pos.any():
        reconstructed_patches = target_patches.clone()
        reconstructed_patches[bool_masked_pos] = logits.reshape(-1, logits.shape[-1])
    else:
        reconstructed_patches = logits

    predicted_tchw = unpatchify(
        reconstructed_patches, tubelet_size, patch_size, num_frames, image_size, num_channels
    )
    predicted = to_project_layout(predicted_tchw)
    target = clean_pixel_values_ctwh
    if channel_idx is not None:
        predicted = predicted[:, channel_idx:channel_idx + 1]
        target = target[:, channel_idx:channel_idx + 1]

    seam_losses = []
    if image_size > patch_size:
        pred_vertical = predicted[..., patch_size::patch_size, :] - predicted[..., patch_size - 1:-1:patch_size, :]
        target_vertical = target[..., patch_size::patch_size, :] - target[..., patch_size - 1:-1:patch_size, :]
        pred_horizontal = predicted[..., :, patch_size::patch_size] - predicted[..., :, patch_size - 1:-1:patch_size]
        target_horizontal = target[..., :, patch_size::patch_size] - target[..., :, patch_size - 1:-1:patch_size]
        seam_losses.extend((
            F.l1_loss(pred_vertical, target_vertical),
            F.l1_loss(pred_horizontal, target_horizontal),
        ))

    if not seam_losses:
        return logits.new_zeros(())
    return torch.stack(seam_losses).mean()


# =====================================================================================
# 7) TRAIN LOOP
# =====================================================================================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(cfg: ProjectConfig) -> torch.device:
    return torch.device("cuda") if (cfg.device == "cuda" and torch.cuda.is_available()) else torch.device("cpu")


def train(cfg: ProjectConfig) -> VideoMAEForPreTraining:
    set_seed(cfg.seed)
    device = get_device(cfg)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # --- 1) Normalization: computed ONCE over train, cached to JSON ---
    if os.path.exists(cfg.norm_stats_path):
        norm_stats = load_norm_stats(cfg.norm_stats_path)
    else:
        print("Computing Z-score normalization stats from the training set...")
        norm_stats = compute_normalization_stats(cfg.train_dir, cfg.variable_names, cfg.file_num_frames, cfg.image_size)
        save_norm_stats(norm_stats, cfg.norm_stats_path)

    # --- 2) Datasets / dataloaders ---
    train_dataset = NetCDFSpatioTemporalDataset(
        cfg.train_dir, cfg.variable_names, cfg.file_num_frames, cfg.image_size, norm_stats, cfg.flatten_time
    )
    eval_dataset = NetCDFSpatioTemporalDataset(
        cfg.eval_dir, cfg.variable_names, cfg.file_num_frames, cfg.image_size, norm_stats, cfg.flatten_time
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)
    eval_loader = DataLoader(eval_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, drop_last=False)

    # --- 3) Model / optimizer ---
    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)

    # Per-iteration mode weights. Mode C (whole-timestep masking) only makes
    # sense with more than one frame per sample -- with num_frames == 1 there
    # is no "other timestep" to reconstruct from, so its weight collapses to
    # 0 and that probability mass is absorbed by Mode B instead.
    mode_a_w = cfg.mode_a_probability
    mode_c_w = cfg.mode_c_probability if cfg.num_frames > 1 else 0.0
    mode_b_w = max(1.0 - mode_a_w - mode_c_w, 0.0)
    print(f"Training mode weights -> A: {mode_a_w:.2f}  B: {mode_b_w:.2f}  C: {mode_c_w:.2f}")

    # --- 4) Training loop ---
    for epoch in range(cfg.num_epochs):
        model.train()
        running_loss = 0.0
        running_a, running_b, running_c = 0.0, 0.0, 0.0
        n_a, n_b, n_c = 0, 0, 0

        for step, pixel_values_ctwh in enumerate(train_loader):
            pixel_values_ctwh = pixel_values_ctwh.to(device)  # [B, C, T, H, W]
            batch_size = pixel_values_ctwh.shape[0]
            optimizer.zero_grad()

            # Random choice per ITERATION (not per sample): Mode A vs B vs C
            mode = random.choices(["A", "B", "C"], weights=[mode_a_w, mode_b_w, mode_c_w], k=1)[0]

            if mode == "A":
                # ===== MODE A: spatial recovery =====
                bool_masked_pos = random_mask_mode_a(batch_size, cfg.num_patches, cfg.mode_a_mask_ratio, device)
                logits = forward_full_sequence(model, pixel_values_ctwh, bool_masked_pos)
                loss = masked_patch_mse_loss_mode_a(
                    logits, pixel_values_ctwh, bool_masked_pos, cfg.tubelet_size, cfg.patch_size
                )
                loss = loss + cfg.patch_boundary_loss_weight * patch_boundary_gradient_loss(
                    logits, pixel_values_ctwh, bool_masked_pos, cfg.tubelet_size, cfg.patch_size,
                    cfg.num_frames, cfg.image_size, cfg.num_channels,
                )
                running_a += loss.item(); n_a += 1
            elif mode == "B":
                # ===== MODE B: variable recovery =====
                channel_idx = random.randrange(cfg.num_channels)
                corrupted = zero_out_channel(pixel_values_ctwh, channel_idx)
                bool_masked_pos = no_mask_mode_b(batch_size, cfg.num_patches, device)
                logits = forward_full_sequence(model, corrupted, bool_masked_pos)
                loss = mode_b_variable_inference_loss(
                    logits, pixel_values_ctwh, channel_idx, cfg.tubelet_size, cfg.patch_size, cfg.num_channels
                )
                loss = loss + cfg.patch_boundary_loss_weight * patch_boundary_gradient_loss(
                    logits, pixel_values_ctwh, bool_masked_pos, cfg.tubelet_size, cfg.patch_size,
                    cfg.num_frames, cfg.image_size, cfg.num_channels, channel_idx,
                )
                running_b += loss.item(); n_b += 1
            else:
                # ===== MODE C: timestep recovery =====
                # Hides one whole (randomly chosen, per-sample) frame -- all
                # variables, all spatial locations -- so the model is trained
                # on exactly the masking pattern used by inference Case 3.
                # Reuses the generic masked-patch loss: it only cares which
                # positions are masked, not how they got that way.
                bool_masked_pos = random_mask_mode_c(
                    batch_size, cfg.num_patches, cfg.patches_per_frame, cfg.num_frames, cfg.tubelet_size, device
                )
                logits = forward_full_sequence(model, pixel_values_ctwh, bool_masked_pos)
                loss = masked_patch_mse_loss_mode_a(
                    logits, pixel_values_ctwh, bool_masked_pos, cfg.tubelet_size, cfg.patch_size
                )
                loss = loss + cfg.patch_boundary_loss_weight * patch_boundary_gradient_loss(
                    logits, pixel_values_ctwh, bool_masked_pos, cfg.tubelet_size, cfg.patch_size,
                    cfg.num_frames, cfg.image_size, cfg.num_channels,
                )
                running_c += loss.item(); n_c += 1

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if step % 10 == 0:
                print(f"epoch {epoch} step {step} mode {mode} loss {loss.item():.5f}")

        scheduler.step()
        print(
            f"[epoch {epoch}] avg_loss={running_loss/max(len(train_loader),1):.5f} "
            f"mode_a_avg={running_a/max(n_a,1):.5f} (n={n_a}) "
            f"mode_b_avg={running_b/max(n_b,1):.5f} (n={n_b}) "
            f"mode_c_avg={running_c/max(n_c,1):.5f} (n={n_c})"
        )

        if (epoch + 1) % max(cfg.checkpoint_every_n_epochs, 1) == 0 or epoch == cfg.num_epochs - 1:
            model.eval()
            eval_loss_total, eval_batches = 0.0, 0
            with torch.no_grad():
                for pixel_values_ctwh in eval_loader:
                    pixel_values_ctwh = pixel_values_ctwh.to(device)
                    batch_size = pixel_values_ctwh.shape[0]
                    bool_masked_pos = random_mask_mode_a(batch_size, cfg.num_patches, cfg.mode_a_mask_ratio, device)
                    logits = forward_full_sequence(model, pixel_values_ctwh, bool_masked_pos)
                    loss = masked_patch_mse_loss_mode_a(
                        logits, pixel_values_ctwh, bool_masked_pos, cfg.tubelet_size, cfg.patch_size
                    )
                    eval_loss_total += loss.item(); eval_batches += 1
            print(f"[epoch {epoch}] eval_mode_a_loss={eval_loss_total/max(eval_batches,1):.5f}")

            ckpt_dir = os.path.join(cfg.output_dir, f"checkpoint-epoch-{epoch+1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            print(f"Checkpoint saved to {ckpt_dir}")

    # --- 5) Final save: weights + config.json via save_pretrained() ---
    model.save_pretrained(cfg.output_dir)
    metadata = {
        "variable_names": cfg.variable_names, "num_frames": cfg.num_frames,
        "file_num_frames": cfg.file_num_frames, "flatten_time": cfg.flatten_time,
        "image_size": cfg.image_size, "patch_size": cfg.patch_size, "tubelet_size": cfg.tubelet_size,
    }
    with open(os.path.join(cfg.output_dir, "project_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Final model saved to {cfg.output_dir}")
    return model


# =====================================================================================
# 8) INFERENCE / TESTING  (.from_pretrained() + Case 1 + Case 2 + MAE)
# =====================================================================================
def load_trained_model(cfg: ProjectConfig, checkpoint_dir: Optional[str] = None) -> VideoMAEForPreTraining:
    path = checkpoint_dir or cfg.output_dir
    return VideoMAEForPreTraining.from_pretrained(path)


def test_case1_spatial_block(
    model: VideoMAEForPreTraining, cfg: ProjectConfig, clean_sample_ctwh: torch.Tensor, device: torch.device,
) -> float:
    """
    CASE 1: blocks a central 3x3 block of patches (in the patch grid) across all
    `cfg.num_frames` timesteps, and measures the reconstruction MAE against the
    original values in that region.

    clean_sample_ctwh: [1, C, T, H, W] a clean validation sample.
    """
    model.eval()
    clean_sample_ctwh = clean_sample_ctwh.to(device)
    batch_size = clean_sample_ctwh.shape[0]

    bool_masked_pos, block_idx = build_case1_mask(
        batch_size, cfg.num_patches, cfg.patch_grid_side, cfg.num_frames, cfg.tubelet_size, block_size=3, device=device
    )
    # NOTE: we use forward_full_sequence (not model(...).loss) for the same
    # reason explained there -- transformers==5.14.1 does not support
    # num_channels != 3 in its internal loss path, regardless of norm_pix_loss.
    with torch.no_grad():
        pred_masked_patches = forward_full_sequence(model, clean_sample_ctwh, bool_masked_pos)

    pixel_values_tchw = to_model_layout(clean_sample_ctwh)
    target_patches_full = patchify(pixel_values_tchw, cfg.tubelet_size, cfg.patch_size)  # [B, N, feat]
    target_masked_patches = target_patches_full[:, block_idx, :]  # [B, num_masked, feat]

    mae = F.l1_loss(pred_masked_patches, target_masked_patches).item()
    print(f"[Case 1 - spatial recovery 3x3] MAE over blocked region: {mae:.6f}")
    return mae


def test_case2_missing_variable(
    model: VideoMAEForPreTraining, cfg: ProjectConfig, clean_sample_ctwh: torch.Tensor,
    channel_idx: int, device: torch.device,
) -> float:
    """
    CASE 2: completely removes one variable (channel) and measures the
    reconstruction MAE for THAT channel, across all patches, against the
    original values.

    clean_sample_ctwh: [1, C, T, H, W] a clean validation sample.
    """
    model.eval()
    clean_sample_ctwh = clean_sample_ctwh.to(device)
    batch_size = clean_sample_ctwh.shape[0]

    corrupted = zero_out_channel(clean_sample_ctwh, channel_idx)
    bool_masked_pos = no_mask_mode_b(batch_size, cfg.num_patches, device)

    with torch.no_grad():
        logits = forward_full_sequence(model, corrupted, bool_masked_pos)  # [B, N, feat] -- all patches

    pixel_values_tchw = to_model_layout(clean_sample_ctwh)
    target_patches = patchify(pixel_values_tchw, cfg.tubelet_size, cfg.patch_size)

    pred_channel = extract_channel_from_patches(logits, channel_idx, cfg.tubelet_size, cfg.patch_size, cfg.num_channels)
    target_channel = extract_channel_from_patches(target_patches, channel_idx, cfg.tubelet_size, cfg.patch_size, cfg.num_channels)

    mae = F.l1_loss(pred_channel, target_channel).item()
    var_name = cfg.variable_names[channel_idx]
    print(f"[Case 2 - recovery of variable '{var_name}'] MAE over removed channel: {mae:.6f}")
    return mae


def test_case3_missing_timestep(
    model: VideoMAEForPreTraining, cfg: ProjectConfig, clean_sample_ctwh: torch.Tensor,
    target_frame_idx: int, device: torch.device,
) -> float:
    """
    CASE 3: hides an ENTIRE timestep (all variables, all spatial locations at
    `target_frame_idx`) and measures the reconstruction MAE for that whole
    frame, using only the other timesteps as context. Requires
    cfg.num_frames > 1.

    clean_sample_ctwh: [1, C, T, H, W] a clean validation sample.
    """
    assert cfg.num_frames > 1, "Reconstructing a whole timestep needs at least one other frame for context."
    model.eval()
    clean_sample_ctwh = clean_sample_ctwh.to(device)
    batch_size = clean_sample_ctwh.shape[0]

    bool_masked_pos, frame_idx = build_case3_missing_timestep_mask(
        batch_size, cfg.num_patches, cfg.patches_per_frame, cfg.num_frames,
        cfg.tubelet_size, target_frame_idx, device,
    )
    with torch.no_grad():
        pred_masked_patches = forward_full_sequence(model, clean_sample_ctwh, bool_masked_pos)

    pixel_values_tchw = to_model_layout(clean_sample_ctwh)
    target_patches_full = patchify(pixel_values_tchw, cfg.tubelet_size, cfg.patch_size)
    target_masked_patches = target_patches_full[:, frame_idx, :]

    mae = F.l1_loss(pred_masked_patches, target_masked_patches).item()
    print(f"[Case 3 - full timestep recovery, t={target_frame_idx}] MAE over missing frame: {mae:.6f}")
    return mae


# =====================================================================================
# 9) VISUALIZATION  (target vs. prediction vs. difference, saved as PNG images)
# =====================================================================================
def _frame_to_numpy(pixel_values_ctwh: torch.Tensor, channel_idx: int, frame_idx: int) -> np.ndarray:
    """Extracts a 2D [H, W] numpy frame from a [1, C, T, H, W] tensor (batch size 1)."""
    return pixel_values_ctwh[0, channel_idx, frame_idx].detach().cpu().numpy()


def _lat_lon_extent_and_origin(lat: np.ndarray, lon: np.ndarray) -> Tuple[list, str]:
    """
    Builds the `extent`/`origin` pair imshow needs to place a [H, W] array
    correctly on real lat/lon axes, regardless of whether `lat` is stored
    north-to-south or south-to-north.
    """
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
    origin = "upper" if lat[0] > lat[-1] else "lower"  # row 0 is north -> plot it at the top
    return extent, origin


def _pixel_box_to_lonlat_box(
    pixel_box: Tuple[int, int, int, int], lat: np.ndarray, lon: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Converts a (row0, row1, col0, col1) pixel box into (lon0, lat0, width, height) for a Rectangle patch."""
    row0, row1, col0, col1 = pixel_box
    row1 = min(row1, len(lat) - 1)
    col1 = min(col1, len(lon) - 1)
    lon0, lon1 = lon[col0], lon[col1]
    lat0, lat1 = lat[row0], lat[row1]
    return lon0, lat0, (lon1 - lon0), (lat1 - lat0)


def _build_panel_title(panel_kind: str, metadata: Optional["SampleMetadata"], var_label: str, frame_idx: Optional[int] = None) -> str:
    """
    Builds a title in the style requested:
        Time: <time>
        Center: (<lat>, <lon>)

        <variable short label>
        <Target / Prediction / Difference>
    """
    lines = []
    # Prefer per-frame metadata when available and a frame index was provided.
    if metadata is not None:
        time_to_use = None
        if frame_idx is not None and metadata.time_strs is not None and frame_idx < len(metadata.time_strs):
            time_to_use = metadata.time_strs[frame_idx]
        else:
            time_to_use = metadata.time_str
        if time_to_use:
            lines.append(f"Time: {time_to_use}")

        latc = None
        lonc = None
        if frame_idx is not None and metadata.cyclone_center_lats is not None and metadata.cyclone_center_lons is not None:
            if frame_idx < len(metadata.cyclone_center_lats) and frame_idx < len(metadata.cyclone_center_lons):
                latc = float(metadata.cyclone_center_lats[frame_idx])
                lonc = float(metadata.cyclone_center_lons[frame_idx])
        if latc is None and metadata.cyclone_center_lat is not None and metadata.cyclone_center_lon is not None:
            latc = metadata.cyclone_center_lat
            lonc = metadata.cyclone_center_lon
        if latc is not None and lonc is not None:
            lines.append(f"Center: ({latc:.2f}°, {lonc:.2f}°)")
    short_label = var_label.split(",")[0].strip()
    header = "\n".join(lines)
    return f"{header}\n\n{short_label}\n{panel_kind}" if header else f"{short_label}\n{panel_kind}"


def build_comparison_filename(metadata: Optional["SampleMetadata"], recon_type: str, images_dir: str, type_model: str="videomae", frame_idx: int=None) -> str:
    """
    Builds a filename following the project's own convention:
        comparison_track_<track_id>_season_<season>_basin_<basin>_rank_<rank>_window_<window>_<time>_<TYPE>.png
    Falls back to "unknown" for any field that couldn't be parsed from the
    source filename, so this never crashes on unexpected inputs.
    """
    track_id = (metadata.track_id if metadata else None) or "unknown"
    season = (metadata.season if metadata else None) or "unknown"
    basin = (metadata.basin if metadata else None) or "unknown"
    rank = (metadata.rank if metadata else None) or "unknown"
    # time_val = (metadata.time_str if metadata else None) or "unknown-time"
    window = (metadata.win_idx if metadata else None) or "unknown-window"
    if frame_idx is not None and metadata and metadata.time_strs and 0 <= frame_idx < len(metadata.time_strs):
        time_val = (metadata.time_strs[frame_idx] if metadata and metadata.time_strs and 0 <= frame_idx < len(metadata.time_strs) else None) or "unknown-time"
        name_file = f"{recon_type}_{type_model}_track{track_id}_{season}_{basin}_rank{rank}_window{window}_{time_val}.png"
    else:
        name_file = f"{recon_type}_{type_model}_track{track_id}_{season}_{basin}_rank{rank}_window{window}.png"    
    return os.path.join(images_dir, name_file)


def plot_target_pred_diff_grid(
    target_ctwh_norm: torch.Tensor,
    prediction_ctwh_norm: torch.Tensor,
    channel_idx: int,
    variable_name: str,
    save_path: str,
    num_frames: int,
    norm_stats: Dict[str, Dict[str, float]],
    variable_names: List[str],
    metadata: Optional["SampleMetadata"] = None,
    highlight_pixel_box: Optional[Tuple[int, int, int, int]] = None,
    title_prefix: str = "",
) -> None:
    """
    Builds and SAVES (does not display on screen) a figure with `num_frames` rows
    x 3 columns: Target (ground truth) | Prediction (model) | Absolute difference
    |target - pred|, for a single physical channel.

    IMPORTANT: `target_ctwh_norm` / `prediction_ctwh_norm` are expected in
    NORMALIZED (Z-score) space -- the space the model actually operates in.
    This function denormalizes them internally (back to raw physical units,
    e.g. Kelvin, Pa, m/s) before plotting, using `norm_stats`, and applies any
    registered unit conversion (see `VARIABLE_DISPLAY_UNITS`, e.g. Pa -> hPa
    for "msl"). Plots are always in physical units, never in Z-score units.

    This function automatically adapts to whatever `num_frames` you pass in --
    it never assumes 3 timesteps. With num_frames=1 it renders a single row.

    If `metadata` provides lat/lon coordinate arrays, the axes show real
    lat/lon instead of pixel indices, and the cyclone center (if present) is
    marked with a black "X" on every panel.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless backend: writes files only, no display
    import matplotlib.pyplot as plt

    target_ctwh = denormalize(target_ctwh_norm, norm_stats, variable_names)
    prediction_ctwh = denormalize(prediction_ctwh_norm, norm_stats, variable_names)

    has_latlon = metadata is not None and metadata.lat is not None and metadata.lon is not None
    if has_latlon:
        extent, origin = _lat_lon_extent_and_origin(metadata.lat, metadata.lon)
    else:
        extent, origin = None, "upper"

    fig, axes = plt.subplots(num_frames, 3, figsize=(12, 4 * num_frames))
    if num_frames == 1:
        axes = axes[None, :]  # normalize shape so axes[t, col] indexing always works

    for t in range(num_frames):
        target_frame_raw = _frame_to_numpy(target_ctwh, channel_idx, t)
        pred_frame_raw = _frame_to_numpy(prediction_ctwh, channel_idx, t)

        target_frame, var_label = convert_for_display(variable_name, target_frame_raw)
        pred_frame, _ = convert_for_display(variable_name, pred_frame_raw)
        diff_frame = np.abs(target_frame - pred_frame)

        # Same color scale for Target and Prediction so they are visually
        # comparable; Difference uses its own scale.
        vmin = min(target_frame.min(), pred_frame.min())
        vmax = max(target_frame.max(), pred_frame.max())

        panels = [
            (target_frame, _build_panel_title("Target", metadata, var_label, frame_idx=t), vmin, vmax, "viridis"),
            (pred_frame, _build_panel_title("Predicted", metadata, var_label, frame_idx=t), vmin, vmax, "viridis"),
            (diff_frame, _build_panel_title("Difference", metadata, var_label, frame_idx=t), None, None, "inferno"),
        ]
        for col, (frame, subtitle, pmin, pmax, cmap) in enumerate(panels):
            ax = axes[t, col]
            im = ax.imshow(frame, cmap=cmap, vmin=pmin, vmax=pmax, extent=extent, origin=origin)
            ax.set_title(subtitle, fontsize=9)
            if has_latlon:
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
            else:
                ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


            if highlight_pixel_box is not None and col < 2:
                if has_latlon:
                    x0, y0, w, h = _pixel_box_to_lonlat_box(highlight_pixel_box, metadata.lat, metadata.lon)
                else:
                    r0, r1, c0, c1 = highlight_pixel_box
                    x0, y0, w, h = c0, r0, c1 - c0, r1 - r0
                rect = plt.Rectangle((x0, y0), w, h, linewidth=2, edgecolor="red", facecolor="none")
                ax.add_patch(rect)

            # Prefer per-frame cyclone center when available; fall back to the
            # scalar representative center. Only plot if values exist and are
            # finite (not NaN).
            latc = None
            lonc = None
            if metadata is not None:
                if getattr(metadata, "cyclone_center_lats", None) is not None and getattr(metadata, "cyclone_center_lons", None) is not None:
                    if t < len(metadata.cyclone_center_lats) and t < len(metadata.cyclone_center_lons):
                        latc = metadata.cyclone_center_lats[t]
                        lonc = metadata.cyclone_center_lons[t]
                if latc is None and metadata.cyclone_center_lat is not None and metadata.cyclone_center_lon is not None:
                    latc = metadata.cyclone_center_lat
                    lonc = metadata.cyclone_center_lon
            if has_latlon and latc is not None and lonc is not None and not (np.isnan(latc) or np.isnan(lonc)):
                ax.plot(lonc, latc, "kX", ms=10, mew=1.5, zorder=5)

    fig.suptitle(f"{title_prefix}Variable: {variable_name}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison image saved to: {save_path}")


def plot_all_variables_grid(
    target_ctwh_norm: torch.Tensor,
    prediction_ctwh_norm: torch.Tensor,
    variable_names: List[str],
    save_path: str,
    norm_stats: Dict[str, Dict[str, float]],
    frame_idx: int = 0,
    metadata: Optional["SampleMetadata"] = None,
    highlight_pixel_box: Optional[Tuple[int, int, int, int]] = None,
    title_prefix: str = "",
) -> None:
    """
    Builds and SAVES a figure with one row PER VARIABLE (all channels at once)
    and 3 columns: Target | Prediction | Absolute difference, for a single
    timestep `frame_idx`. This is the plot for the "all 7 variables, some
    spatially missing area, then recovered" use case: instead of looking at one
    channel across time, you look at every channel at one specific timestep.

    Same as `plot_target_pred_diff_grid`: inputs are expected in NORMALIZED
    space and are denormalized to physical units (with per-variable unit
    conversion) internally before plotting.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target_ctwh = denormalize(target_ctwh_norm, norm_stats, variable_names)
    prediction_ctwh = denormalize(prediction_ctwh_norm, norm_stats, variable_names)

    has_latlon = metadata is not None and metadata.lat is not None and metadata.lon is not None
    if has_latlon:
        extent, origin = _lat_lon_extent_and_origin(metadata.lat, metadata.lon)
    else:
        extent, origin = None, "upper"

    num_channels = len(variable_names)
    fig, axes = plt.subplots(num_channels, 3, figsize=(12, 3.4 * num_channels))
    if num_channels == 1:
        axes = axes[None, :]

    for c in range(num_channels):
        target_frame_raw = _frame_to_numpy(target_ctwh, c, frame_idx)
        pred_frame_raw = _frame_to_numpy(prediction_ctwh, c, frame_idx)

        target_frame, var_label = convert_for_display(variable_names[c], target_frame_raw)
        pred_frame, _ = convert_for_display(variable_names[c], pred_frame_raw)
        diff_frame = np.abs(target_frame - pred_frame)

        vmin = min(target_frame.min(), pred_frame.min())
        vmax = max(target_frame.max(), pred_frame.max())

        panels = [
            (target_frame, f"{var_label} - Target", vmin, vmax, "viridis"),
            (pred_frame, f"{var_label} - Predicted", vmin, vmax, "viridis"),
            (diff_frame, f"{var_label} - Difference", None, None, "inferno"),
        ]
        for col, (frame, subtitle, pmin, pmax, cmap) in enumerate(panels):
            ax = axes[c, col]
            im = ax.imshow(frame, cmap=cmap, vmin=pmin, vmax=pmax, extent=extent, origin=origin)
            ax.set_title(subtitle, fontsize=9)
            if has_latlon:
                ax.set_xlabel("Longitude", fontsize=8)
                ax.set_ylabel("Latitude", fontsize=8)
            else:
                ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            if highlight_pixel_box is not None and col < 2:
                if has_latlon:
                    x0, y0, w, h = _pixel_box_to_lonlat_box(highlight_pixel_box, metadata.lat, metadata.lon)
                else:
                    r0, r1, c0, c1 = highlight_pixel_box
                    x0, y0, w, h = c0, r0, c1 - c0, r1 - r0
                rect = plt.Rectangle((x0, y0), w, h, linewidth=2, edgecolor="red", facecolor="none")
                ax.add_patch(rect)

            # Prefer per-frame cyclone center when available; fall back to the
            # scalar representative center. Only plot if values exist and are
            # finite (not NaN).
            latc = None
            lonc = None
            if metadata is not None:
                if getattr(metadata, "cyclone_center_lats", None) is not None and getattr(metadata, "cyclone_center_lons", None) is not None:
                    if frame_idx is not None and frame_idx < len(metadata.cyclone_center_lats) and frame_idx < len(metadata.cyclone_center_lons):
                        latc = metadata.cyclone_center_lats[frame_idx]
                        lonc = metadata.cyclone_center_lons[frame_idx]
                if latc is None and metadata.cyclone_center_lat is not None and metadata.cyclone_center_lon is not None:
                    latc = metadata.cyclone_center_lat
                    lonc = metadata.cyclone_center_lon
            if has_latlon and latc is not None and lonc is not None and not (np.isnan(latc) or np.isnan(lonc)):
                ax.plot(lonc, latc, "kX", ms=9, mew=1.5, zorder=5)

    if metadata:
        time_str = None
        if metadata.time_strs is not None and frame_idx is not None and frame_idx < len(metadata.time_strs):
            time_str = metadata.time_strs[frame_idx]
        else:
            time_str = metadata.time_str
    else:
        time_str = None
    header = f" | Time: {time_str}" if time_str else ""
    fig.suptitle(f"{title_prefix}All variables at t={frame_idx}{header}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison image saved to: {save_path}")


# def plot_prediction_comparison(
#     target_ctwh_norm: torch.Tensor,
#     predictions: Dict[str, torch.Tensor],
#     rows: List[Tuple[int, int, str]],
#     variable_names: List[str],
#     norm_stats: Dict[str, Dict[str, float]],
#     save_path: str,
#     metadata: Optional["SampleMetadata"] = None,
#     highlight_pixel_box: Optional[Tuple[int, int, int, int]] = None,
#     title_prefix: str = "",
# ) -> None:
#     """
#     Generic, model-agnostic comparison plot: any number of full reconstructions
#     (VideoMAE, the linear-interpolation baseline, a future diffusion model,
#     etc.) side by side against the same target, with their error maps grouped
#     together so you can immediately see WHERE one method beats another
#     (a sharp front, a smooth region, the edge of the missing area...) rather
#     than just comparing a single averaged MAE number.

#     Column layout: Target | pred_1 | pred_2 | ... | |diff_1| | |diff_2| | ...
#     -- the two (or more) error maps end up next to each other, which is the
#     comparison that actually answers "in which structure does model A have
#     an advantage over model B".

#     Parameters
#     ----------
#     target_ctwh_norm : [1, C, T, H, W], Z-score normalized (project layout).
#     predictions : dict mapping a display label -> a FULL reconstruction
#         tensor, same shape and same normalized space as `target_ctwh_norm`
#         (i.e. identical to the target everywhere except the recovered
#         region/timestep). Insertion order controls column order. This is
#         intentionally generic: pass in as many entries as you like --
#         {"VideoMAE": ..., "Linear interpolation": ..., "Diffusion model": ...}
#         -- to compare any number of models/baselines with the exact same plot.
#     rows : list of (channel_idx, frame_idx, row_label) tuples, one per row of
#         the figure. This is what makes the function reusable for both Case 1
#         style plots (fixed channel, one row per timestep) and Case 3 style
#         plots (fixed timestep, one row per variable) -- the caller decides
#         what a "row" means; see `case_comparison_rows` / `case3_comparison_rows`
#         below for the two ready-made row builders.
#     highlight_pixel_box : optional (row0, row1, col0, col1) in pixels, drawn
#         on every panel (values AND error maps) -- e.g. the blocked 3x3 patch
#         region in Case 1. Leave as None for Case 3 (the whole frame is
#         "missing", not a sub-region, so there is nothing partial to outline).

#     Color scale note: the error maps share ONE color scale WITHIN each row
#     (so the different predictions for that row are fairly comparable to each
#     other) but NOT across rows -- different rows can be different physical
#     variables with very different units/magnitudes, and a single global
#     scale would let a large-magnitude variable (e.g. pressure in hPa) swamp
#     every other row and make them look artificially well-predicted.
#     """
#     import matplotlib
#     matplotlib.use("Agg")
#     import matplotlib.pyplot as plt

#     pred_labels = list(predictions.keys())
#     n_preds = len(pred_labels)
#     n_cols = 1 + 2 * n_preds  # Target + each prediction + each |diff|
#     n_rows = len(rows)

#     target_ctwh = denormalize(target_ctwh_norm, norm_stats, variable_names)
#     predictions_physical = {
#         label: denormalize(pred, norm_stats, variable_names) for label, pred in predictions.items()
#     }


UNIT_SCALE_FOR_STD = {
    "msl": 1.0 / 100.0,  # Pa -> hPa
    # 10u, 10v, avg_tos, avg_zos: adjust if their raw storage units differ from display units
}

def _box_slice(frame: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    """Slice a [H, W] frame down to the highlighted region only.
    box is (r0, r1, c0, c1) in pixel space — same convention used
    elsewhere in this function for the rectangle overlay, and valid
    regardless of whether we're displaying in lat/lon or pixel coords,
    since the underlying array is always indexed by pixel."""
    r0, r1, c0, c1 = box
    return frame[r0:r1, c0:c1]

def _compute_row_metrics(
    target_frame: np.ndarray,
    pred_frame: np.ndarray,
    box: Optional[Tuple[int, int, int, int]] = None,
    dataset_std: Optional[float] = None,
) -> Dict[str, float]:
    """
    Error metrics for one (variable, model) pair, restricted to `box` if given.
    Uses the SIGNED error (pred - target)
    dataset_std: per-channel std from training-set normalization
    stats (in the SAME physical units as target_frame/pred_frame, i.e.
    already converted the same way convert_for_display() msl need to divide /100 scale factor before passing in here) .
    """
    if box is not None:
        t = _box_slice(target_frame, box)
        p = _box_slice(pred_frame, box)
    else:
        t = target_frame
        p = pred_frame

    err = p - t

    metrics = {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)),
    }
    # nrmse = rmse / std  # in "std-units" -> comparable across variables/rows
    if dataset_std is not None and dataset_std > 1e-8:
        metrics["nrmse"] = metrics["rmse"] / dataset_std
    else:
        metrics["nrmse"] = None
    return metrics

# def _get_dataset_std(var_name: str, norm_stats, unit_scale: float = 1.0) -> Optional[float]:
#     """
#     Pulls the training-set std for this variable from norm_stats and applies
#     unit_scale to convert it into the same units convert_for_display() uses
#     (e.g. 1/100 for Pa->hPa). unit_scale defaults to 1.0 (no conversion needed).
#     """
#     target_stats = norm_stats.get("target", norm_stats) if isinstance(norm_stats, dict) else None
#     if target_stats is None or var_name not in target_stats:
#         return None
#     std = target_stats[var_name].get("std")
#     return std * unit_scale if std is not None else None

def _get_dataset_std(
    var_name: str, 
    norm_stats: dict, 
    label: str = "target", 
    unit_scale: float = 1.0
) -> Optional[float]:
    """
    Pulls std for `var_name` from `norm_stats`.
    Handles nested dicts formatted as {label: {var_name: {"std": ...}}}
    or flat dicts formatted as {var_name: {"std": ...}}.
    """
    if not isinstance(norm_stats, dict):
        return None

    # Determine stats dictionary for the specified label
    if label in norm_stats and isinstance(norm_stats[label], dict):
        stats_dict = norm_stats[label]
    elif "target" in norm_stats and isinstance(norm_stats["target"], dict):
        stats_dict = norm_stats["target"]
    else:
        stats_dict = norm_stats

    # Extract variable-specific stats
    var_stats = stats_dict.get(var_name)
    if isinstance(var_stats, dict):
        std = var_stats.get("std")
        return std * unit_scale if std is not None else None
        
    return None



def nounits_plot_prediction_comparison(
    target_ctwh_norm: torch.Tensor,
    predictions: Dict[str, torch.Tensor],
    rows: List[Tuple[int, int, str]],
    variable_names: List[str],
    norm_stats: Union[Dict[str, Dict[str, float]], Dict[str, Dict[str, Dict[str, float]]]],
    save_path: str,
    metadata: Optional["SampleMetadata"] = None,
    highlight_pixel_box: Optional[Tuple[int, int, int, int]] = None,
    title_prefix: str = "",
)-> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Returns metrics as {row_label: {pred_label: {"mae":..., "rmse":..., "bias":..., "nrmse":...}}}
    """
    pred_labels = list(predictions.keys())
    n_preds = len(pred_labels)
    n_cols = 1 + 2 * n_preds  # Target + cada predicción + cada |diff|
    n_rows = len(rows)

    def get_stats(label: str) -> Dict[str, Dict[str, float]]:
        if isinstance(norm_stats, dict) and label in norm_stats:
            return norm_stats[label]
        return norm_stats  # if it is a only one norma_stats 

    target_stats = get_stats("target") # i use the same std, mean because here the dataset was the same for the VAE
    target_ctwh = denormalize(target_ctwh_norm, target_stats, variable_names)

    predictions_physical = {
        label: denormalize(pred, get_stats(label), variable_names)
        for label, pred in predictions.items()
    }


    has_latlon = metadata is not None and metadata.lat is not None and metadata.lon is not None
    if has_latlon:
        extent, origin = _lat_lon_extent_and_origin(metadata.lat, metadata.lon)
    else:
        extent, origin = None, "upper"

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 3.0 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]
    if n_cols == 1:
        axes = axes[:, None]

    # Pre-compute every row's frames and diffs up front so we can share a
    # color scale for the diff maps WITHIN each row (i.e. between the N
    # predictions being compared for that specific variable/timestep). The
    # scale is NOT shared ACROSS rows: different rows can be different
    # physical variables with very different units and magnitudes (e.g.
    # pressure in hPa vs. temperature in K), so a single global scale would
    # make the large-magnitude variable's row swamp every other row and make
    # them look artificially well-predicted.
    row_data = []
    all_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}


    for (channel_idx, frame_idx, row_label) in rows:
        target_frame_raw = _frame_to_numpy(target_ctwh, channel_idx, frame_idx)
        target_frame, var_label = convert_for_display(variable_names[channel_idx], target_frame_raw)

        pred_frames = {}
        diff_frames = {}
        row_metrics = {}
        row_diff_absmax = 0.0

        for label in pred_labels:
            raw = _frame_to_numpy(predictions_physical[label], channel_idx, frame_idx)
            converted, _ = convert_for_display(variable_names[channel_idx], raw)
            pred_frames[label] = converted
            # diff = np.abs(target_frame - converted)
            diff = converted - target_frame
            diff_frames[label] = diff
            # row_diff_max = max(row_diff_max, float(diff.max()))
            row_diff_absmax = max(row_diff_absmax, float(np.abs(diff).max()))

            # Metric restricted to the highlighted box if one was given,
            # else computed over the whole frame.
            var_name = variable_names[channel_idx]
            std_for_var = _get_dataset_std(
                var_name=var_name, 
                norm_stats=norm_stats, 
                label=label, #"target",  
                unit_scale=UNIT_SCALE_FOR_STD.get(var_name, 1.0)
            )
                    
            row_metrics[label] = _compute_row_metrics(
                target_frame=target_frame, 
                pred_frame=converted, 
                box=highlight_pixel_box, 
                dataset_std=std_for_var
            )

        all_metrics[row_label] = row_metrics
        row_data.append((row_label, target_frame, pred_frames, diff_frames, row_diff_absmax, row_metrics))

    for r, (row_label, target_frame, pred_frames, diff_frames, row_diff_absmax, row_metrics) in enumerate(row_data):
        # Shared color scale across Target + all predictions for THIS row (so
        # they are visually comparable to each other); diff maps share a
        # scale with each other WITHIN this row only (see note above).
        all_values = np.concatenate([target_frame.ravel()] + [f.ravel() for f in pred_frames.values()])
        vmin, vmax = all_values.min(), all_values.max()

        panels = [(target_frame, f"Target\n{row_label}", vmin, vmax, "viridis", False, None)]
        for label in pred_labels:
            panels.append((pred_frames[label], f"{label}\n{row_label}", vmin, vmax, "viridis", False, None))
        for label in pred_labels:
            panels.append((diff_frames[label], f"{label} error\n{row_label}",
                           -row_diff_absmax, row_diff_absmax,   # symmetric around 0 --0.0, row_diff_absmax, \
                            "RdBu_r", True, row_metrics[label])) #inferno

        for col, (frame, subtitle, pmin, pmax, cmap, is_diff, metrics) in enumerate(panels):
            ax = axes[r, col]
            im = ax.imshow(frame, cmap=cmap, vmin=pmin, vmax=pmax, extent=extent, origin=origin)
            ax.set_title(subtitle, fontsize=9)
            if has_latlon and r == n_rows - 1:
                ax.set_xlabel("Longitude", fontsize=8)
            if has_latlon and col == 0:
                ax.set_ylabel("Latitude", fontsize=8)
            if not has_latlon:
                ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Annotate diff panels with the box-restricted (or full-frame) metrics.
            if is_diff and metrics is not None:
                box_tag = "Box metrics" if highlight_pixel_box is not None else "" #"Frame metrics"
                nrmse_str = f"\nNRMSE={metrics['nrmse']:.2f}" if metrics.get("nrmse") is not None else ""
                ax.text(
                    0.02, 0.98,
                    # f"RMSE={metrics['rmse']:.2f}\nBias={metrics['bias']:+.2f}\nNRMSE={metrics['nrmse']:.2f}{box_tag}",
                    f"{box_tag}\nRMSE={metrics['rmse']:.2f}\nMAE={metrics['mae']:.2f}\nBias={metrics['bias']:+.2f}", #{nrmse_str}",
                    transform=ax.transAxes, fontsize=7, va="top", ha="left",
                    color="white",
                    bbox=dict(facecolor="black", alpha=0.55, pad=2, edgecolor="none"),
                )


            if highlight_pixel_box is not None:
                if has_latlon:
                    x0, y0, w, h = _pixel_box_to_lonlat_box(highlight_pixel_box, metadata.lat, metadata.lon)
                else:
                    r0, r1, c0, c1 = highlight_pixel_box
                    x0, y0, w, h = c0, r0, c1 - c0, r1 - r0
                edge_color = "red" if not is_diff else "black"
                rect = plt.Rectangle((x0, y0), w, h, linewidth=1.5, edgecolor=edge_color, facecolor="none")
                ax.add_patch(rect)

            # Prefer per-frame cyclone center when available for this row;
            # fall back to scalar representative center. Use row's frame_idx
            # from `rows` and only plot when values are finite.
            latc = None
            lonc = None
            if metadata is not None:
                # rows[r] is (channel_idx, frame_idx, row_label)
                _, row_frame_idx, _ = rows[r]
                if getattr(metadata, "cyclone_center_lats", None) is not None and getattr(metadata, "cyclone_center_lons", None) is not None:
                    if row_frame_idx is not None and row_frame_idx < len(metadata.cyclone_center_lats) and row_frame_idx < len(metadata.cyclone_center_lons):
                        latc = metadata.cyclone_center_lats[row_frame_idx]
                        lonc = metadata.cyclone_center_lons[row_frame_idx]
                if latc is None and metadata.cyclone_center_lat is not None and metadata.cyclone_center_lon is not None:
                    latc = metadata.cyclone_center_lat
                    lonc = metadata.cyclone_center_lon
            if has_latlon and latc is not None and lonc is not None and not (np.isnan(latc) or np.isnan(lonc)):
                ax.plot(lonc, latc, "kX", ms=8, mew=1.3, zorder=5)

    # If all rows share the same frame_idx and metadata has per-frame times,
    # show that time in the header; otherwise fall back to scalar time_str.
    # if metadata:
    #     frame_idxs = {frame_idx for (_, frame_idx, _) in rows}
    #     if len(frame_idxs) == 1:
    #         only_idx = next(iter(frame_idxs))
    #         if metadata.time_strs is not None and only_idx < len(metadata.time_strs):
    #             time_str = metadata.time_strs[only_idx]
    #         else:
    #             time_str = metadata.time_str
    #     else:
    #         time_str = metadata.time_str
    # else:
    #     time_str = None
    # header = f" | Time: {time_str}" if time_str else ""
    # fig.suptitle(f"{title_prefix}{' vs. '.join(pred_labels)}{header}", fontsize=13)
    fig.suptitle(f"{title_prefix}{' vs. '.join(pred_labels)}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison image saved to: {save_path}")
    return all_metrics


# VAR_LABELS = {
#     "10u": "Zonal wind (10u), m/s",
#     "10v": "Meridional wind (10v), m/s",
#     "msl": "Mean sea level pressure (msl), hPa",
#     "avg_tos": "Sea surface temperature (avg_tos), °C",
#     "avg_zos": "Sea surface height (avg_zos), m",
#     "avg_sos": "Sea surface salinity (avg_sos), psu",
#     "avg_mlotst125": "Mixed layer depth (avg_mlotst125), m",
# }



VAR_LABELS = {
    "10u": "Zonal wind 10u, m/s",
    "10v": "Meridional wind 10v, m/s",
    "msl": "Mean sea level pressure, hPa",
    "avg_tos": "Sea surface temperature, °C",
    "avg_zos": "Sea surface height, m",
    "avg_sos": "Sea surface salinity, psu",
    "avg_mlotst125": "Mixed layer depth, m",
}



def _get_var_display_info(var_name: str, row_label: str) -> Tuple[str, str, str]:
    """
    Returns clean (long_name, short_name, unit_str).
    Prevents duplicate trailing short-names in long_name.
    """
    if var_name in VAR_LABELS:
        full_desc = VAR_LABELS[var_name]
        short_name = var_name
    elif row_label in VAR_LABELS:
        full_desc = VAR_LABELS[row_label]
        short_name = row_label
    else:
        full_desc = row_label
        short_name = var_name

    # Extract long name and unit string
    if "," in full_desc:
        long_name, unit_str = full_desc.split(",", 1)
        long_name = long_name.strip()
        unit_str = unit_str.strip()
    else:
        long_name = full_desc.strip()
        unit_str = ""

    # Clean short_name repetitions out of long_name if present (e.g. "Zonal wind 10u" -> "Zonal wind")
    pattern = re.compile(re.escape(short_name), re.IGNORECASE)
    long_name = pattern.sub("", long_name).strip()

    return long_name, short_name, unit_str



def plot_prediction_comparison(
    target_ctwh_norm: torch.Tensor,
    predictions: Dict[str, torch.Tensor],
    rows: List[Tuple[int, int, str]],
    variable_names: List[str],
    norm_stats: Union[Dict[str, Dict[str, float]], Dict[str, Dict[str, Dict[str, float]]]],
    save_path: str,
    metadata: Optional["SampleMetadata"] = None,
    highlight_pixel_box: Optional[Tuple[int, int, int, int]] = None,
    title_prefix: str = "",
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Returns metrics as {row_label: {pred_label: {"mae":..., "rmse":..., "bias":..., "nrmse":...}}}
    Plots target, predictions, and error maps with long variable titles printed ONLY on the middle column.
    """
    pred_labels = list(predictions.keys())
    n_preds = len(pred_labels)
    n_cols = 1 + 2 * n_preds  # Target + N predictions + N error maps
    n_rows = len(rows)

    middle_col_idx = n_cols // 2  # Column index where variable title will be printed

    def get_stats(label: str) -> Dict[str, Dict[str, float]]:
        if isinstance(norm_stats, dict) and label in norm_stats:
            return norm_stats[label]
        return norm_stats

    target_stats = get_stats("target")
    target_ctwh = denormalize(target_ctwh_norm, target_stats, variable_names)

    predictions_physical = {
        label: denormalize(pred, get_stats(label), variable_names)
        for label, pred in predictions.items()
    }

    has_latlon = metadata is not None and metadata.lat is not None and metadata.lon is not None
    if has_latlon:
        extent, origin = _lat_lon_extent_and_origin(metadata.lat, metadata.lon)
    else:
        extent, origin = None, "upper"

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.3 * n_cols, 3.2 * n_rows), squeeze=False)

    row_data = []
    all_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}

    for (channel_idx, frame_idx, row_label) in rows:
        var_name = variable_names[channel_idx]
        target_frame_raw = _frame_to_numpy(target_ctwh, channel_idx, frame_idx)
        target_frame, _ = convert_for_display(var_name, target_frame_raw)

        pred_frames = {}
        diff_frames = {}
        row_metrics = {}
        row_diff_absmax = 0.0

        for label in pred_labels:
            raw = _frame_to_numpy(predictions_physical[label], channel_idx, frame_idx)
            converted, _ = convert_for_display(var_name, raw)
            pred_frames[label] = converted
            diff = converted - target_frame
            diff_frames[label] = diff
            row_diff_absmax = max(row_diff_absmax, float(np.abs(diff).max()))

            std_for_var = _get_dataset_std(
                var_name=var_name,
                norm_stats=norm_stats,
                label=label,
                unit_scale=UNIT_SCALE_FOR_STD.get(var_name, 1.0)
            )

            row_metrics[label] = _compute_row_metrics(
                target_frame=target_frame,
                pred_frame=converted,
                box=highlight_pixel_box,
                dataset_std=std_for_var
            )

        all_metrics[row_label] = row_metrics

        long_name, short_name, unit_str = _get_var_display_info(var_name, row_label)

        row_data.append((
            row_label, var_name, long_name, short_name, unit_str,
            target_frame, pred_frames, diff_frames, row_diff_absmax, row_metrics
        ))

    for r, (
        row_label, var_name, long_name, short_name, unit_str,
        target_frame, pred_frames, diff_frames, row_diff_absmax, row_metrics
    ) in enumerate(row_data):

        all_values = np.concatenate([target_frame.ravel()] + [f.ravel() for f in pred_frames.values()])
        vmin, vmax = all_values.min(), all_values.max()

        panels = [(target_frame, vmin, vmax, "viridis", False, None)]
        for label in pred_labels:
            panels.append((pred_frames[label], vmin, vmax, "viridis", False, None))
        for label in pred_labels:
            panels.append((diff_frames[label], -row_diff_absmax, row_diff_absmax, "RdBu_r", True, row_metrics[label]))

        for col, (frame, pmin, pmax, cmap, is_diff, metrics) in enumerate(panels):
            ax = axes[r, col]
            im = ax.imshow(frame, cmap=cmap, vmin=pmin, vmax=pmax, extent=extent, origin=origin)

            # 1. Base model role label
            if col == 0:
                base_title = "Target"
            elif col <= n_preds:
                base_title = pred_labels[col - 1]
            else:
                base_title = f"{pred_labels[col - 1 - n_preds]} Error"

            # 2. Append time string if row_label is a timestamp
            time_tag = f" [{row_label}]" if (row_label != var_name and row_label not in VAR_LABELS) else ""

            # 3. Add Variable Name ONLY on the middle column
            if col == middle_col_idx:
                full_title = f"{long_name} ({short_name}){time_tag}\n{base_title}"
            else:
                full_title = f"{base_title}" #{time_tag}"

            ax.set_title(full_title, fontsize=9, fontweight="normal", pad=8)

            if has_latlon and r == n_rows - 1:
                ax.set_xlabel("Longitude", fontsize=8)
            if has_latlon and col == 0:
                ax.set_ylabel("Latitude", fontsize=8)
            if not has_latlon:
                ax.axis("off")

            # Colorbars
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=7)
            # if unit_str and (col == n_preds or col == n_cols - 1):
            #     cbar.set_label(unit_str, fontsize=8)
            if unit_str and (col == n_preds):
                cbar.set_label(unit_str, fontsize=8)
            elif col == n_cols - 1:
                cbar.set_label(f"Δ {unit_str}", fontsize=8)



            # Metrics overlay
            if is_diff and metrics is not None:
                # metric_subplots = f"RMSE: {metrics['rmse']:.2f}\nMAE: {metrics['mae']:.2f}\nBias: {metrics['bias']:+.2f}",
                # subplots_titles = f"Box\n{metric_subplots}" if highlight_pixel_box is not None else metric_subplots #Full"
                metric_subplots = (
                    f"RMSE: {metrics['rmse']:.2f}\n"
                    f"MAE: {metrics['mae']:.2f}\n"
                    f"Bias: {metrics['bias']:+.2f}"
                )

                if highlight_pixel_box is not None:
                    subplots_titles = f"Box\n{metric_subplots}"
                else:
                    subplots_titles = metric_subplots
                ax.text(
                    0.02, 0.98,
                    subplots_titles,
                    # f"{box_tag}\nRMSE: {metrics['rmse']:.2f}\nMAE: {metrics['mae']:.2f}\nBias: {metrics['bias']:+.2f}",
                    transform=ax.transAxes, fontsize=7, va="top", ha="left",
                    color="white",
                    bbox=dict(facecolor="black", alpha=0.55, pad=2, edgecolor="none"),
                )

            # Bounding box
            if highlight_pixel_box is not None:
                if has_latlon:
                    x0, y0, w, h = _pixel_box_to_lonlat_box(highlight_pixel_box, metadata.lat, metadata.lon)
                else:
                    r0, r1, c0, c1 = highlight_pixel_box
                    x0, y0, w, h = c0, r0, c1 - c0, r1 - r0
                edge_color = "red" if not is_diff else "black"
                rect = plt.Rectangle((x0, y0), w, h, linewidth=1.5, edgecolor=edge_color, facecolor="none")
                ax.add_patch(rect)

            # Cyclone center marker
            latc, lonc = None, None
            if metadata is not None:
                _, row_frame_idx, _ = rows[r]
                if getattr(metadata, "cyclone_center_lats", None) is not None and getattr(metadata, "cyclone_center_lons", None) is not None:
                    if row_frame_idx is not None and row_frame_idx < len(metadata.cyclone_center_lats):
                        latc = metadata.cyclone_center_lats[row_frame_idx]
                        lonc = metadata.cyclone_center_lons[row_frame_idx]
                if latc is None and metadata.cyclone_center_lat is not None:
                    latc = metadata.cyclone_center_lat
                    lonc = metadata.cyclone_center_lon

            if has_latlon and latc is not None and lonc is not None and not (np.isnan(latc) or np.isnan(lonc)):
                ax.plot(lonc, latc, "kX", ms=8, mew=1.3, zorder=5)

    if title_prefix:
        fig.suptitle(f"{title_prefix}{' vs. '.join(pred_labels)}", fontsize=14, y=0.995)

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    if save_path:
        # plt.savefig(save_path, bbox_inches="tight", dpi=300)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Comparison image saved to: {save_path}")

    return all_metrics





def case_comparison_rows(cfg: ProjectConfig, channel_idx: int, metadata: Optional["SampleMetadata"]) -> List[Tuple[int, int, str]]:
    """Row builder for a Case 1 style plot: fixed variable, one row per timestep."""

    metadata_time_strs = metadata.time_strs if metadata is not None and metadata.time_strs is not None else [f"t={t}" for t in range(cfg.num_frames)]
    return [(channel_idx, t, f"t={metadata_time_strs[t]}") for t in range(cfg.num_frames)]
    # return [(cfg.variable_names[channel_idx], t, f"t={t}") for t in range(cfg.num_frames)]


def case2_comparison_rows(cfg: ProjectConfig, channel_idx: int, metadata: Optional["SampleMetadata"]) -> List[Tuple[int, int, str]]:
    """Row builder for a Case 2 style plot: fixed variable, one row per timestep."""

    metadata_time_strs = metadata.time_strs if metadata is not None and metadata.time_strs is not None else [f"t={t}" for t in range(cfg.num_frames)]
    return [(channel_idx, t, f"t={metadata_time_strs[t]}") for t in range(cfg.num_frames)]


def case3_comparison_rows(cfg: ProjectConfig, frame_idx: int) -> List[Tuple[int, int, str]]:
    """Row builder for a Case 3 style plot: fixed timestep, one row per variable.
    frame_idx = timesteo 0 or 1 or 2 or the 3 timesteps
    channel change"""
    return [(channel, frame_idx, cfg.variable_names[channel]) for channel in range(cfg.num_channels)]


def reconstruct_case1_model_prediction(
    model: VideoMAEForPreTraining, cfg: ProjectConfig, clean_sample_ctwh: torch.Tensor, device: torch.device,
) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """
    Runs the model's Case 1 spatial recovery and returns the FULL
    reconstructed tensor (same shape as the input, identical outside the
    blocked region) plus the pixel box that was blocked -- exposed as a
    standalone function so it can feed `plot_prediction_comparison` alongside
    the baseline or any other model's prediction.
    """
    model.eval()
    clean_sample_ctwh = clean_sample_ctwh.to(device)
    batch_size = clean_sample_ctwh.shape[0]

    bool_masked_pos, block_idx = build_case1_mask(
        batch_size, cfg.num_patches, cfg.patch_grid_side, cfg.num_frames, cfg.tubelet_size, block_size=3, device=device
    )
    with torch.no_grad():
        pred_masked_patches = forward_full_sequence(model, clean_sample_ctwh, bool_masked_pos)

    pixel_values_tchw = to_model_layout(clean_sample_ctwh)
    target_patches_full = patchify(pixel_values_tchw, cfg.tubelet_size, cfg.patch_size)
    reconstructed_patches = target_patches_full.clone()
    reconstructed_patches[:, block_idx, :] = pred_masked_patches
    reconstructed_tchw = unpatchify(
        reconstructed_patches, cfg.tubelet_size, cfg.patch_size, cfg.num_frames, cfg.image_size, cfg.num_channels
    )
    reconstructed_ctwh = to_project_layout(reconstructed_tchw)

    grid_side, ps = cfg.patch_grid_side, cfg.patch_size
    start = (grid_side - 3) // 2
    pixel_box = (start * ps, (start + 3) * ps, start * ps, (start + 3) * ps)
    return reconstructed_ctwh, pixel_box

def reconstruct_case2_model_prediction(
    model: VideoMAEForPreTraining, cfg: ProjectConfig, clean_sample_ctwh: torch.Tensor,
    channel_idx: int, device: torch.device
) -> None:
    """
    CASE 2 (visual): removes `channel_idx` entirely, runs the model
    across all `cfg.num_frames` timesteps for that sample
    """
    model.eval()
    clean_sample_ctwh = clean_sample_ctwh.to(device)
    batch_size = clean_sample_ctwh.shape[0]

    corrupted = zero_out_channel(clean_sample_ctwh, channel_idx)
    bool_masked_pos = no_mask_mode_b(batch_size, cfg.num_patches, device)
    with torch.no_grad():
        logits = forward_full_sequence(model, corrupted, bool_masked_pos)  # [B, N, feat] -- all patches

    reconstructed_tchw = unpatchify(
        logits, cfg.tubelet_size, cfg.patch_size, cfg.num_frames, cfg.image_size, cfg.num_channels
    )
    # reconstructed_ctwh = to_project_layout(reconstructed_tchw)  # [B, C, T, H, W]

    return to_project_layout(reconstructed_tchw)

def reconstruct_case3_model_prediction(
    model: VideoMAEForPreTraining, cfg: ProjectConfig, clean_sample_ctwh: torch.Tensor,
    target_frame_idx: int, device: torch.device,
) -> torch.Tensor:
    """Same idea as `reconstruct_case1_model_prediction`, for Case 3 (full timestep recovery)."""
    model.eval()
    clean_sample_ctwh = clean_sample_ctwh.to(device)
    batch_size = clean_sample_ctwh.shape[0]

    bool_masked_pos, frame_idx = build_case3_missing_timestep_mask(
        batch_size, cfg.num_patches, cfg.patches_per_frame, cfg.num_frames,
        cfg.tubelet_size, target_frame_idx, device,
    )
    with torch.no_grad():
        pred_masked_patches = forward_full_sequence(model, clean_sample_ctwh, bool_masked_pos)

    pixel_values_tchw = to_model_layout(clean_sample_ctwh)
    target_patches_full = patchify(pixel_values_tchw, cfg.tubelet_size, cfg.patch_size)
    reconstructed_patches = target_patches_full.clone()
    # it is not need to have the whole sequence, just the frame that was masked, 
    # change that if it is needed in the future
    reconstructed_patches[:, frame_idx, :] = pred_masked_patches
    reconstructed_tchw = unpatchify(
        reconstructed_patches, cfg.tubelet_size, cfg.patch_size, cfg.num_frames, cfg.image_size, cfg.num_channels
    )
    return to_project_layout(reconstructed_tchw)


def visualize_case1_spatial_block(
    model: VideoMAEForPreTraining, cfg: ProjectConfig, clean_sample_ctwh: torch.Tensor,
    device: torch.device, norm_stats: Dict[str, Dict[str, float]], images_dir: str,
    channel_idx: int = 0, metadata: Optional["SampleMetadata"] = None,
) -> None:
    """
    CASE 1 (single-variable view): reconstructs the full tensor by inserting the
    model's prediction ONLY in the blocked 3x3 patch region, keeping everything
    else identical to the original (those patches were never touched). Plots
    Target vs. Prediction vs. Difference for channel `channel_idx`, in physical
    units, with a red rectangle marking the blocked area.
    """
    model.eval()
    clean_sample_ctwh = clean_sample_ctwh.to(device)
    batch_size = clean_sample_ctwh.shape[0]

    bool_masked_pos, block_idx = build_case1_mask(
        batch_size, cfg.num_patches, cfg.patch_grid_side, cfg.num_frames, cfg.tubelet_size, block_size=3, device=device
    )
    with torch.no_grad():
        pred_masked_patches = forward_full_sequence(model, clean_sample_ctwh, bool_masked_pos)  # [B, num_masked, feat]

    pixel_values_tchw = to_model_layout(clean_sample_ctwh)
    target_patches_full = patchify(pixel_values_tchw, cfg.tubelet_size, cfg.patch_size)  # [B, N, feat]

    # Full reconstruction: start from the REAL (target) patches, and only replace
    # the blocked positions with the model's prediction.
    reconstructed_patches = target_patches_full.clone()
    reconstructed_patches[:, block_idx, :] = pred_masked_patches

    reconstructed_tchw = unpatchify(
        reconstructed_patches, cfg.tubelet_size, cfg.patch_size, cfg.num_frames, cfg.image_size, cfg.num_channels
    )
    reconstructed_ctwh = to_project_layout(reconstructed_tchw)  # [B, C, T, H, W]

    grid_side = cfg.patch_grid_side
    ps = cfg.patch_size
    start = (grid_side - 3) // 2
    pixel_box = (start * ps, (start + 3) * ps, start * ps, (start + 3) * ps)  # (row0,row1,col0,col1)

    save_path = build_comparison_filename(metadata, f"case1_{cfg.variable_names[channel_idx]}", images_dir)
    plot_target_pred_diff_grid(
        target_ctwh_norm=clean_sample_ctwh,
        prediction_ctwh_norm=reconstructed_ctwh,
        channel_idx=channel_idx,
        variable_name=cfg.variable_names[channel_idx],
        save_path=save_path,
        num_frames=cfg.num_frames,
        norm_stats=norm_stats,
        variable_names=cfg.variable_names,
        metadata=metadata,
        highlight_pixel_box=pixel_box,
        title_prefix="Case 1 (3x3 spatial recovery) - ",
    )


def visualize_case1_all_variables(
    model: VideoMAEForPreTraining, cfg: ProjectConfig, clean_sample_ctwh: torch.Tensor,
    device: torch.device, norm_stats: Dict[str, Dict[str, float]], images_dir: str,
    frame_idx: int = 0, metadata: Optional["SampleMetadata"] = None,
) -> None:
    """
    CASE 1 (all-variables view): same reconstruction as
    `visualize_case1_spatial_block`, but plots EVERY physical variable at once
    (one row per variable) for a single timestep `frame_idx`, in physical
    units, with a red rectangle marking the blocked spatial region. Use this
    when you want to see, for one specific timestep, all 7 variables with a
    missing area and how the model filled it in.
    """
    model.eval()
    clean_sample_ctwh = clean_sample_ctwh.to(device)
    batch_size = clean_sample_ctwh.shape[0]

    bool_masked_pos, block_idx = build_case1_mask(
        batch_size, cfg.num_patches, cfg.patch_grid_side, cfg.num_frames, cfg.tubelet_size, block_size=3, device=device
    )
    with torch.no_grad():
        pred_masked_patches = forward_full_sequence(model, clean_sample_ctwh, bool_masked_pos)

    pixel_values_tchw = to_model_layout(clean_sample_ctwh)
    target_patches_full = patchify(pixel_values_tchw, cfg.tubelet_size, cfg.patch_size)

    reconstructed_patches = target_patches_full.clone()
    reconstructed_patches[:, block_idx, :] = pred_masked_patches

    reconstructed_tchw = unpatchify(
        reconstructed_patches, cfg.tubelet_size, cfg.patch_size, cfg.num_frames, cfg.image_size, cfg.num_channels
    )
    reconstructed_ctwh = to_project_layout(reconstructed_tchw)

    grid_side = cfg.patch_grid_side
    ps = cfg.patch_size
    start = (grid_side - 3) // 2
    pixel_box = (start * ps, (start + 3) * ps, start * ps, (start + 3) * ps)

    save_path = build_comparison_filename(metadata, "case1", images_dir, frame_idx=frame_idx)
    plot_all_variables_grid(
        target_ctwh_norm=clean_sample_ctwh,
        prediction_ctwh_norm=reconstructed_ctwh,
        variable_names=cfg.variable_names,
        save_path=save_path,
        norm_stats=norm_stats,
        frame_idx=frame_idx,
        metadata=metadata,
        highlight_pixel_box=pixel_box,
        title_prefix="Case 1 (3x3 spatial recovery) - ",
    )


def visualize_case2_missing_variable(
    model: VideoMAEForPreTraining, cfg: ProjectConfig, clean_sample_ctwh: torch.Tensor,
    channel_idx: int, device: torch.device, norm_stats: Dict[str, Dict[str, float]], images_dir: str,
    metadata: Optional["SampleMetadata"] = None,
) -> None:
    """
    CASE 2 (visual): removes `channel_idx` entirely, runs the model, and plots
    Target vs. Prediction vs. Difference for THAT channel (in physical units)
    across all `cfg.num_frames` timesteps. No rectangle here (the whole
    variable was absent, not a spatial region).
    """
    model.eval()
    clean_sample_ctwh = clean_sample_ctwh.to(device)
    batch_size = clean_sample_ctwh.shape[0]

    corrupted = zero_out_channel(clean_sample_ctwh, channel_idx)
    bool_masked_pos = no_mask_mode_b(batch_size, cfg.num_patches, device)
    with torch.no_grad():
        logits = forward_full_sequence(model, corrupted, bool_masked_pos)  # [B, N, feat] -- all patches

    reconstructed_tchw = unpatchify(
        logits, cfg.tubelet_size, cfg.patch_size, cfg.num_frames, cfg.image_size, cfg.num_channels
    )
    reconstructed_ctwh = to_project_layout(reconstructed_tchw)  # [B, C, T, H, W]

    save_path = build_comparison_filename(metadata, f"case2_{cfg.variable_names[channel_idx]}", images_dir)
    plot_target_pred_diff_grid(
        target_ctwh_norm=clean_sample_ctwh,
        prediction_ctwh_norm=reconstructed_ctwh,
        channel_idx=channel_idx,
        variable_name=cfg.variable_names[channel_idx],
        save_path=save_path,
        num_frames=cfg.num_frames,
        norm_stats=norm_stats,
        variable_names=cfg.variable_names,
        metadata=metadata,
        highlight_pixel_box=None,
        title_prefix="Case 2 (missing variable) - ",
    )


def visualize_case3_missing_timestep(
    model: VideoMAEForPreTraining, cfg: ProjectConfig, clean_sample_ctwh: torch.Tensor,
    target_frame_idx: int, device: torch.device, norm_stats: Dict[str, Dict[str, float]], images_dir: str,
    metadata: Optional["SampleMetadata"] = None,
) -> None:
    """
    CASE 3 (visual): hides an ENTIRE timestep (all variables at
    `target_frame_idx`) and plots every physical variable at once (one row
    per variable) for that timestep, Target vs. Prediction vs. Difference, in
    physical units. No red rectangle (the whole frame was hidden, not a
    partial region). Requires cfg.num_frames > 1.
    """
    assert cfg.num_frames > 1, "Reconstructing a whole timestep needs at least one other frame for context."
    model.eval()
    clean_sample_ctwh = clean_sample_ctwh.to(device)
    batch_size = clean_sample_ctwh.shape[0]
    print(f" clean_sample_ctwh.shape: {clean_sample_ctwh.shape}")

    bool_masked_pos, frame_idx = build_case3_missing_timestep_mask(
        batch_size, cfg.num_patches, cfg.patches_per_frame, cfg.num_frames,
        cfg.tubelet_size, target_frame_idx, device,
    )
    with torch.no_grad():
        pred_masked_patches = forward_full_sequence(model, clean_sample_ctwh, bool_masked_pos)

    pixel_values_tchw = to_model_layout(clean_sample_ctwh)
    target_patches_full = patchify(pixel_values_tchw, cfg.tubelet_size, cfg.patch_size)

    # Full reconstruction: real patches everywhere except the hidden frame,
    # which is replaced by the model's prediction.
    reconstructed_patches = target_patches_full.clone()
    reconstructed_patches[:, frame_idx, :] = pred_masked_patches
    reconstructed_tchw = unpatchify(
        reconstructed_patches, cfg.tubelet_size, cfg.patch_size, cfg.num_frames, cfg.image_size, cfg.num_channels
    )
    reconstructed_ctwh = to_project_layout(reconstructed_tchw)

    save_path = build_comparison_filename(metadata, f"case3_time{target_frame_idx}", images_dir, frame_idx=target_frame_idx)
    plot_all_variables_grid(
        target_ctwh_norm=clean_sample_ctwh,
        prediction_ctwh_norm=reconstructed_ctwh,
        variable_names=cfg.variable_names,
        save_path=save_path,
        norm_stats=norm_stats,
        frame_idx=target_frame_idx,
        metadata=metadata,
        highlight_pixel_box=None,
        title_prefix=f"Case 3 (full timestep recovery, t={target_frame_idx}) - ",
    )


# =====================================================================================
# 10) BASELINES  (non-learned linear interpolation, for comparison against VideoMAE)
# =====================================================================================
def temporal_linear_interpolation_baseline(
    clean_sample_ctwh: torch.Tensor, target_frame_idx: int
) -> np.ndarray:
    """
    Predicts the missing timestep `target_frame_idx` by fitting a least-
    squares line through the OTHER available timesteps, independently for
    every pixel and every channel, then evaluating that line at
    `target_frame_idx`. This is the Case 3 baseline.

    clean_sample_ctwh: [1, C, T, H, W], same normalized layout used
        elsewhere. Returns a numpy array [1, C, H, W].

    With exactly 2 other timesteps (the common T=3 case) the least-squares
    line passes exactly through both, so this is plain linear interpolation
    (target strictly between them) or linear extrapolation (target at an
    edge, e.g. day 0 predicted from days 1-2's trend).
    """
    x = clean_sample_ctwh.detach().cpu().numpy()
    B, C, T, H, W = x.shape
    available = [t for t in range(T) if t != target_frame_idx]

    if len(available) == 0:
        raise ValueError("Need at least one other timestep to interpolate/extrapolate from.")
    if len(available) == 1:
        # Degenerate case (T == 2): can't fit a line from a single point,
        # fall back to persistence (repeat the one available frame).
        return x[:, :, available[0], :, :].copy()

    t_avail = np.asarray(available, dtype=np.float64)
    y_avail = x[:, :, available, :, :]  # [B, C, n, H, W]
    n = len(t_avail)

    t_b = t_avail.reshape(1, 1, n, 1, 1)
    sum_t, sum_t2 = t_avail.sum(), (t_avail ** 2).sum()
    sum_y = y_avail.sum(axis=2)
    sum_ty = (t_b * y_avail).sum(axis=2)

    denom = n * sum_t2 - sum_t ** 2
    slope = (n * sum_ty - sum_t * sum_y) / denom
    intercept = (sum_y - slope * sum_t) / n
    return intercept + slope * target_frame_idx


def test_case3_baseline(clean_sample_ctwh: torch.Tensor, target_frame_idx: int) -> float:
    """Baseline counterpart to test_case3_missing_timestep: linear interpolation, same MAE definition."""
    pred = temporal_linear_interpolation_baseline(clean_sample_ctwh, target_frame_idx)
    true = clean_sample_ctwh[:, :, target_frame_idx, :, :].detach().cpu().numpy()
    mae = float(np.abs(pred - true).mean())
    print(f"[Case 3 BASELINE - linear interpolation, t={target_frame_idx}] MAE: {mae:.6f}")
    return mae


def _fill_2d_field_linear(field_2d: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
    """
    Fills `missing_mask` positions in a single [H, W] field via Delaunay-
    based linear interpolation (scipy griddata), falling back to
    nearest-neighbor only for points outside the convex hull of the known
    points, where linear interpolation is mathematically undefined.
    """
    H, W = field_2d.shape
    rows, cols = np.mgrid[0:H, 0:W]
    known_mask = ~missing_mask

    known_points = np.column_stack([rows[known_mask], cols[known_mask]])
    known_values = field_2d[known_mask]
    missing_points = np.column_stack([rows[missing_mask], cols[missing_mask]])

    filled = field_2d.copy()
    interpolated = griddata(known_points, known_values, missing_points, method="linear")

    nan_mask = np.isnan(interpolated)
    if nan_mask.any():
        interpolated[nan_mask] = griddata(known_points, known_values, missing_points[nan_mask], method="nearest")

    filled[missing_mask] = interpolated
    return filled


def spatial_linear_interpolation_baseline(
    clean_sample_ctwh: torch.Tensor, pixel_box: Tuple[int, int, int, int]
) -> np.ndarray:
    """
    Predicts a missing rectangular spatial region using linear interpolation
    from the surrounding pixels, independently for every channel and every
    timestep -- no cross-channel or cross-time information used at all,
    unlike VideoMAE's Case 1 which can draw on both. This is the Case 1
    baseline. clean_sample_ctwh: [1, C, T, H, W]. Returns numpy [1, C, T, H, W].
    """
    x = clean_sample_ctwh.detach().cpu().numpy()
    B, C, T, H, W = x.shape
    r0, r1, c0, c1 = pixel_box

    missing_mask = np.zeros((H, W), dtype=bool)
    missing_mask[r0:r1, c0:c1] = True

    reconstructed = x.copy()
    for b in range(B):
        for c in range(C):
            for t in range(T):
                reconstructed[b, c, t] = _fill_2d_field_linear(x[b, c, t], missing_mask)
    return reconstructed


def test_case1_baseline(clean_sample_ctwh: torch.Tensor, pixel_box: Tuple[int, int, int, int]) -> float:
    """Baseline counterpart to test_case1_spatial_block: spatial linear interpolation, same MAE definition."""
    r0, r1, c0, c1 = pixel_box
    reconstructed = spatial_linear_interpolation_baseline(clean_sample_ctwh, pixel_box)
    original = clean_sample_ctwh.detach().cpu().numpy()
    mae = float(np.abs(original[:, :, :, r0:r1, c0:c1] - reconstructed[:, :, :, r0:r1, c0:c1]).mean())
    print(f"[Case 1 BASELINE - spatial linear interpolation] MAE over blocked region: {mae:.6f}")
    return mae


def embed_case1_baseline_prediction(
    clean_sample_ctwh: torch.Tensor, pixel_box: Tuple[int, int, int, int]
) -> torch.Tensor:
    """
    Wraps `spatial_linear_interpolation_baseline`'s numpy output back into a
    torch tensor with the same shape/dtype convention as the model's
    reconstructed tensors, so it can be dropped directly into the
    `predictions` dict of `plot_prediction_comparison`.
    """
    reconstructed_np = spatial_linear_interpolation_baseline(clean_sample_ctwh, pixel_box)
    return torch.from_numpy(reconstructed_np).to(clean_sample_ctwh.dtype)


def embed_case3_baseline_prediction(clean_sample_ctwh: torch.Tensor, target_frame_idx: int) -> torch.Tensor:
    """
    Builds a FULL [1, C, T, H, W] tensor equal to `clean_sample_ctwh`
    everywhere except `target_frame_idx`, which is replaced by
    `temporal_linear_interpolation_baseline`'s prediction -- matching the
    same full-tensor convention used by the model's reconstructions, so it
    can be dropped directly into `plot_prediction_comparison`.
    """
    pred_frame = temporal_linear_interpolation_baseline(clean_sample_ctwh, target_frame_idx)  # [B, C, H, W] numpy
    reconstructed = clean_sample_ctwh.clone()
    reconstructed[:, :, target_frame_idx, :, :] = torch.from_numpy(pred_frame).to(clean_sample_ctwh.dtype)
    return reconstructed


def visualize_case1_model_vs_baseline(
    model: VideoMAEForPreTraining, cfg: ProjectConfig, clean_sample_ctwh: torch.Tensor,
    device: torch.device, norm_stats: Dict[str, Dict[str, float]], save_path: str,
    channel_idx: int = 0, metadata: Optional["SampleMetadata"] = None,
    model_label: str = "VideoMAE", baseline_label: str = "Linear interpolation",
) -> None:
    """
    Ready-to-use Case 1 comparison: model vs. the spatial linear-interpolation
    baseline, one row per timestep for `channel_idx`. Under the hood this is
    just two calls (`reconstruct_case1_model_prediction` +
    `embed_case1_baseline_prediction`) feeding the generic
    `plot_prediction_comparison` -- if you later want to add a third source
    (e.g. a diffusion model), build its own full-tensor prediction the same
    way and call `plot_prediction_comparison` directly with a 3-entry dict
    instead of using this convenience wrapper.
    """
    reconstructed_ctwh, pixel_box = reconstruct_case1_model_prediction(model, cfg, clean_sample_ctwh, device)
    baseline_ctwh = embed_case1_baseline_prediction(clean_sample_ctwh, pixel_box)

    save_path = build_comparison_filename(metadata, f"case1_vs_baseline_{cfg.variable_names[channel_idx]}", save_path)
    print(f"clean_sample_ctwh shape: {clean_sample_ctwh.shape}")
    print(f"reconstructed_ctwh shape: {reconstructed_ctwh.shape}")

    
    plot_prediction_comparison(
        target_ctwh_norm=clean_sample_ctwh,
        predictions={model_label: reconstructed_ctwh, baseline_label: baseline_ctwh},
        rows=case_comparison_rows(cfg, channel_idx, metadata),
        variable_names=cfg.variable_names,
        norm_stats=norm_stats,
        save_path=save_path,
        metadata=metadata,
        highlight_pixel_box=pixel_box,
        title_prefix=f"Case 1 (3x3 spatial recovery) - Variable {cfg.variable_names[channel_idx]} - ",
    )


def visualize_case3_model_vs_baseline(
    model: VideoMAEForPreTraining, cfg: ProjectConfig, clean_sample_ctwh: torch.Tensor,
    target_frame_idx: int, device: torch.device, norm_stats: Dict[str, Dict[str, float]], save_path: str,
    metadata: Optional["SampleMetadata"] = None,
    model_label: str = "VideoMAE", baseline_label: str = "Linear interpolation",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same idea as `visualize_case1_model_vs_baseline`, for Case 3 (full timestep recovery)."""
    reconstructed_ctwh = reconstruct_case3_model_prediction(model, cfg, clean_sample_ctwh, target_frame_idx, device)
    baseline_ctwh = embed_case3_baseline_prediction(clean_sample_ctwh, target_frame_idx)

    save_path = build_comparison_filename(metadata, f"case3_time{target_frame_idx}_vs_baseline", save_path, frame_idx=target_frame_idx)

    print(f"clean_sample_ctwh shape: {clean_sample_ctwh.shape}")
    print(f"reconstructed_ctwh shape: {reconstructed_ctwh.shape}")

    plot_prediction_comparison(
        target_ctwh_norm=clean_sample_ctwh,
        predictions={model_label: reconstructed_ctwh, baseline_label: baseline_ctwh},
        rows=case3_comparison_rows(cfg, target_frame_idx),
        variable_names=cfg.variable_names,
        norm_stats=norm_stats,
        save_path=save_path,
        metadata=metadata,
        highlight_pixel_box=None,
        title_prefix=f"Case 3 (full timestep recovery, t={target_frame_idx}) - Time: {metadata.time_strs[target_frame_idx]} - ",
    )
    return clean_sample_ctwh, reconstructed_ctwh, baseline_ctwh

def run_inference_tests(
    cfg: ProjectConfig, checkpoint_dir: Optional[str] = None, num_samples: int = 3,
    save_images: bool = True, images_dir: Optional[str] = None, run_baselines: bool = True,
):
    device = get_device(cfg)
    model = load_trained_model(cfg, checkpoint_dir).to(device)

    norm_stats = load_norm_stats(cfg.norm_stats_path)
    eval_dataset = NetCDFSpatioTemporalDataset(
        cfg.test_dir, cfg.variable_names, cfg.file_num_frames, cfg.image_size, norm_stats, cfg.flatten_time
    )

    images_dir = images_dir or os.path.join(cfg.output_dir, "inference_images")
    run_case3 = cfg.num_frames > 1  # full-timestep recovery only makes sense with >1 frame per sample

    case1_maes, case2_maes, case3_maes = [], [], []
    case1_baseline_maes, case3_baseline_maes = [], []
    for i in range(min(num_samples, len(eval_dataset))): # for now only plotting max 3 samples, to avoid too many images !!!! change to len(eval_dataset) to plot all samples 
        sample = eval_dataset[i].unsqueeze(0)  # [1, C, T, H, W], still Z-score normalized ...  [C, H, W] unsqueeze(0) >[1, C, H, W]
        print(f"Processing file {i+1}/{len(eval_dataset)}, value i: {i}")

        # choose which frame's per-timestep metadata to display in plots
        if run_case3:
            target_frame_idx = i % cfg.num_frames  # rotate the tested timestep across samples
        else:
            target_frame_idx = None

        metadata = eval_dataset.get_metadata(i, frame_idx=target_frame_idx)

        case1_maes.append(test_case1_spatial_block(model, cfg, sample, device))
        channel_idx = i % cfg.num_channels  # rotate the tested channel across samples the variable for case 1, 2!!!! put as input
        case2_maes.append(test_case2_missing_variable(model, cfg, sample, channel_idx, device))
        if run_case3:
            case3_maes.append(test_case3_missing_timestep(model, cfg, sample, target_frame_idx, device))

        if run_baselines:
            # Case 1 baseline needs the same pixel box the model's Case 1 used.
            grid_side, ps = cfg.patch_grid_side, cfg.patch_size
            start = (grid_side - 3) // 2
            pixel_box = (start * ps, (start + 3) * ps, start * ps, (start + 3) * ps)
            case1_baseline_maes.append(test_case1_baseline(sample, pixel_box))
            # name_file = f"{images_dir}/case1_model_vs_baseline_{cfg.variable_names[channel_idx]}_sample{i}.png"
            visualize_case1_model_vs_baseline(
                model, cfg, sample, device, norm_stats, images_dir,
                channel_idx=0, metadata=metadata,
            )
            if run_case3:
                case3_baseline_maes.append(test_case3_baseline(sample, target_frame_idx))
                clean_sample_ctwh, reconstructed_ctwh, baseline_ctwh = visualize_case3_model_vs_baseline(
                    model, cfg, sample, target_frame_idx, device, norm_stats, images_dir,
                    metadata=metadata,
                )




        if save_images:
            visualize_case1_spatial_block(
                model, cfg, sample, device, norm_stats, images_dir,
                channel_idx=channel_idx, metadata=metadata,
            )
            visualize_case1_all_variables(
                model, cfg, sample, device, norm_stats, images_dir,
                frame_idx=0, metadata=metadata,
            )
            visualize_case2_missing_variable(
                model, cfg, sample, channel_idx, device, norm_stats, images_dir,
                metadata=metadata,
            )
            if run_case3:
                visualize_case3_missing_timestep(
                    model, cfg, sample, target_frame_idx, device, norm_stats, images_dir,
                    metadata=metadata,
                )

    print(f"\nCase 1 - average MAE over {len(case1_maes)} samples: {np.mean(case1_maes):.6f}")
    if run_baselines:
        print(f"Case 1 baseline (linear interp.) - average MAE: {np.mean(case1_baseline_maes):.6f}")
    print(f"Case 2 - average MAE over {len(case2_maes)} samples: {np.mean(case2_maes):.6f}")
    if run_case3:
        print(f"Case 3 - average MAE over {len(case3_maes)} samples: {np.mean(case3_maes):.6f}")
        if run_baselines:
            print(f"Case 3 baseline (linear interp.) - average MAE: {np.mean(case3_baseline_maes):.6f}")
    else:
        print("Case 3 (full timestep recovery) skipped: requires cfg.num_frames > 1.")

    # return clean_sample_ctwh, reconstructed_ctwh, baseline_ctwh

# =====================================================================================
# ENTRY POINT
# =====================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VideoMAE self-supervised pipeline for NetCDF climate data")
    parser.add_argument("--mode", choices=["train", "infer"], default="train",
                         help="'train' trains and saves the model; 'infer' loads a trained model and runs Case 1 / Case 2")
    parser.add_argument("--checkpoint_dir", default=None, help="Checkpoint directory to load (only used with --mode infer)")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of eval samples to test in --mode infer")
    parser.add_argument("--no_images", action="store_true", help="Disable saving target/prediction/difference images in --mode infer")
    # parser.add_argument("--images_dir", default=None, help="Folder to save comparison images to (only used with --mode infer)")
    # parser.add_argument("--images_dir", default="/scale/project/ifremer-isi-jumeaunumerique/NEXTGEMS_PROCESSED_LAT_LON/videomae_missing_timestep/comparison_plots", help="Carpeta donde guardar las imágenes de comparación (solo --mode infer)")
    parser.add_argument("--no_baselines", action="store_true", help="Disable the linear-interpolation baselines in --mode infer")
    args = parser.parse_args()

    cfg = ProjectConfig()

    images_dir = f"{cfg.output_dir}/videomae_comparison_plots"

    if args.mode == "train":
        train(cfg)
    else:
        run_inference_tests(
            cfg, checkpoint_dir=args.checkpoint_dir, num_samples=args.num_samples,
            save_images=not args.no_images, images_dir=images_dir,
            run_baselines=not args.no_baselines,
        )
