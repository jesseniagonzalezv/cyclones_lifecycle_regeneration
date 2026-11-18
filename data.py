"""
data.py - dataset of 3-day windows + normalisation helpers.
CyclonePatchDataset and get_metadata are your code, unchanged. Removed: load_raw_test_window (the evaluation now
compares against the normalised test tensors already in memory, which is the same data and much faster).
"""
import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

import config as cfg


def parse_patch_filename(ds: xr.Dataset):
    if ds.attrs:
        return {
            "track_id": ds.attrs.get("track_id"),
            "season": ds.attrs.get("season"),
            "basin": ds.attrs.get("basin"),
            "rank": ds.attrs.get("rank"),
            "win_idx": ds.attrs.get("win_idx"),
        }
    else:
        return None


@dataclass
class SampleMetadata:
    source_file: str
    track_id: Optional[str]
    season: Optional[str]
    basin: Optional[str]
    rank: Optional[str]
    win_idx: Optional[str]
    lat: Optional[np.ndarray]
    lon: Optional[np.ndarray]
    cyclone_center_lat: Optional[float]
    cyclone_center_lon: Optional[float]
    time_str: Optional[str]
    time_strs: Optional[List[str]] = None
    cyclone_center_lats: Optional[np.ndarray] = None
    cyclone_center_lons: Optional[np.ndarray] = None


# ------------------------------------------------------------------
# normalisation helpers
# ------------------------------------------------------------------
def denormalize(data: np.ndarray, var: str, stats: dict) -> np.ndarray:
    mean = stats[var]['mean']
    std = stats[var]['std']
    return data * (std if std > 1e-6 else 1.0) + mean


def stats_to_arrays(stats: dict, vars_list):
    """-> (mean, std) numpy arrays (C,), std already floored exactly like in the dataset."""
    mean = np.array([stats[v]["mean"] for v in vars_list], dtype=np.float64)
    std = np.array([stats[v]["std"] if stats[v]["std"] > 1e-6 else 1.0 for v in vars_list], dtype=np.float64)
    return mean, std


class PatchMeanTransform:
    """x (..., C, H, W) in globally z-scored units  <->  (anomaly / scale, patch mean). Exactly invertible.
    patch mean = spatial mean of each window, day and variable; scale = rms of the anomalies over the TRAIN set."""
    def __init__(self, scale):
        self.scale = torch.as_tensor(scale, dtype=torch.float32).view(-1)

    def to(self, device):
        self.scale = self.scale.to(device)
        return self

    def split(self, x):
        pm = x.mean(dim=(-2, -1))                                           # (..., C)
        return (x - pm[..., None, None]) / self.scale.to(x.device).view(-1, 1, 1), pm

    def merge(self, a, pm):
        return a * self.scale.to(a.device).view(-1, 1, 1) + pm[..., None, None]


def fit_anomaly_scale(data: np.ndarray, vars_list, chunk=256) -> torch.Tensor:
    """rms of the patch-mean-removed field per variable over a (N, F, C, H, W) array (chunked, float64)."""
    C = data.shape[2]
    sq, n = np.zeros(C), 0
    for i in range(0, len(data), chunk):
        x = data[i:i + chunk].astype(np.float64)
        a = x - x.mean(axis=(3, 4), keepdims=True)
        sq += (a ** 2).sum(axis=(0, 1, 3, 4))
        n += x.shape[0] * x.shape[1] * x.shape[3] * x.shape[4]
    scale = np.sqrt(sq / n)
    print("[Data] anomaly rms (fraction of the global std left once the patch mean is removed): "
          + ", ".join(f"{v}={s:.2f}" for v, s in zip(vars_list, scale)))
    return torch.tensor(scale, dtype=torch.float32)


def read_window_info(path) -> dict:
    """Light-weight metadata of one window file (attributes + time axis only, no big arrays)."""
    with xr.open_dataset(path) as ds:
        info = parse_patch_filename(ds) or {}
        times = None
        if "time" in ds.variables or "time" in ds.coords:
            raw = np.asarray(ds["time"].values)
            times = [str(np.datetime_as_string(t, unit="h")) for t in np.atleast_1d(raw)]
    info = dict(info)
    info["source_file"] = Path(path).name
    info["time_strs"] = times
    return info


def collect_window_info(files, num_workers=None) -> List[dict]:
    with ThreadPoolExecutor(max_workers=num_workers or cfg.NUM_WORKERS) as ex:
        return list(ex.map(read_window_info, files))


# ------------------------------------------------------------------
# dataset (unchanged)
# ------------------------------------------------------------------
class CyclonePatchDataset(Dataset):
    def __init__(self, folder_path, variables, stats_dict=None, num_workers=None):
        num_workers = num_workers or cfg.NUM_WORKERS
        self.folder_path = Path(folder_path)
        self.variables = variables

        self.files = sorted(glob.glob(os.path.join(folder_path, "*.nc")))
        if not self.files:
            raise FileNotFoundError(f"No NetCDF files found in {folder_path}")

        start_time = time.perf_counter()

        def _load_file(fpath):
            with xr.open_dataset(fpath) as ds:
                file_data = np.stack([ds[var].values for var in self.variables], axis=1).astype(np.float32)
                filename = Path(fpath).name
            return file_data, filename

        print(f"[Dataset] Loading {len(self.files)} NetCDF window files in parallel...")
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            results = list(ex.map(_load_file, self.files))

        all_data = []
        self.file_to_filename = []
        for file_data, filename in results:
            all_data.append(file_data)
            self.file_to_filename.append(filename)

        self.data = np.stack(all_data, axis=0)
        del all_data

        if stats_dict is None:
            self.stats = {}
            for i, var in enumerate(self.variables):
                var_slice = self.data[:, :, i, :, :]
                self.stats[var] = {
                    'mean': float(var_slice.mean()),
                    'std': float(var_slice.std())
                }
        else:
            self.stats = stats_dict

        for i, var in enumerate(self.variables):
            mean = self.stats[var]['mean']
            std = self.stats[var]['std']
            self.data[:, :, i, :, :] = (self.data[:, :, i, :, :] - mean) / (std if std > 1e-6 else 1.0)

        print(f"[Dataset] Initialization complete in {time.perf_counter() - start_time:.3f}s")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        tensor_sample = torch.from_numpy(self.data[idx])
        return tensor_sample, tensor_sample, idx

    def get_metadata(self, idx: int, frame_idx: Optional[int] = None) -> SampleMetadata:
        file_idx = idx
        file_num_frames = 3
        day_idx = file_num_frames // 2 if frame_idx is None else int(frame_idx) % file_num_frames

        path = self.files[file_idx]
        ds = xr.open_dataset(path)
        file_info = parse_patch_filename(ds)

        lat = np.asarray(ds["absolute_lat"][0].values, dtype=np.float64) if "absolute_lat" in ds.coords or "absolute_lat" in ds.variables else None
        lon = np.asarray(ds["absolute_lon"][0].values, dtype=np.float64) if "absolute_lon" in ds.coords or "absolute_lon" in ds.variables else None

        def _select_scalar_or_indexed(var_name: str):
            if var_name not in ds.variables:
                return None
            values = np.asarray(ds[var_name].values)
            if values.ndim == 0 or values.size == 1:
                return values.reshape(-1)[0]
            return values[day_idx]

        latc = _select_scalar_or_indexed("cyclone_center_lat")
        lonc = _select_scalar_or_indexed("cyclone_center_lon")

        time_val = None
        time_strs = None
        if "time" in ds.variables or "time" in ds.coords:
            time_raw = np.asarray(ds["time"].values)
            if time_raw.ndim == 0:
                time_single = time_raw
                time_val = np.datetime_as_string(time_single, unit="h")
                time_strs = [time_val]
            else:
                time_strs = [np.datetime_as_string(t, unit="h") for t in time_raw]
                time_val = time_strs[day_idx]

        cyclone_center_lats = np.asarray(ds["cyclone_center_lat"].values, dtype=np.float64) if "cyclone_center_lat" in ds.variables else None
        cyclone_center_lons = np.asarray(ds["cyclone_center_lon"].values, dtype=np.float64) if "cyclone_center_lon" in ds.variables else None
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
            cyclone_center_lat=float(latc) if latc is not None and np.ndim(latc) == 0 else None,
            cyclone_center_lon=float(lonc) if lonc is not None and np.ndim(lonc) == 0 else None,
            time_str=time_val,
            time_strs=time_strs,
            cyclone_center_lats=cyclone_center_lats,
            cyclone_center_lons=cyclone_center_lons,
        )
