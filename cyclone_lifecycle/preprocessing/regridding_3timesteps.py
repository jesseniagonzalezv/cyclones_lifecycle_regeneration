
"""Drop-in replacements for regrid_track_fixed_bbox / regrid_create_3step_patches.

Key ideas
---------
* HEALPix nside=512 has ~0.1145 deg pixels, so asking for 0.05 deg oversamples
  (no new information, Delaunay slivers, repeated values). Use res >= ~0.1.
* Instead of scipy.griddata (Delaunay + 'nearest' fallback that copies edge
  values into rows that have no data), look up the closest source cells with a
  KD-tree on 3-D unit vectors. This is immune to lon wrap (0/360, +-180) and
  poles, and anything farther than ~1.5 native pixels from a source cell is set
  to NaN instead of being filled with duplicated values.
* Time matching: each day D uses the atmosphere at exactly D 12:00 UTC (centre of
  the daily-mean window of the ocean) and the track position interpolated to that
  instant. The ocean daily record is selected with an explicit tolerance, so a
  missing day raises instead of silently picking a neighbouring day.
"""

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

NSIDE = 512
NATIVE_RES = float(np.degrees(np.sqrt(np.pi / 3.0) / NSIDE))  # ~0.1145 deg
# --- Configuration Constants ---
DATA_ROOT = Path(
    "/scale/project/ifremer-isi-jumeaunumerique/NEXTGEMS/NEXTGEMS_DATA_EXTRACTION"
)
ATM_ZARR = "2D_hourly_healpix512.zarr"
OCEAN_ZARR = "2D_daily_healpix512_ocean.zarr"
N_TRAIN_TRACKS = 20
N_TEST_TRACKS = 1

BATCH = 1
N_TRACKS = 5 #1 

# TRACKS_CSV = Path("/ontap/user/jgonzale/nextgems_tc/data/tracks.csv")
TRACKS_CSV = Path("/ontap/user/jgonzale/TC_diagnostic/data/tracks.csv")
PROJECT_ROOT = Path("/ontap/user/jgonzale/cyclone_atm")

TRAIN_CSV = PROJECT_ROOT / "data/train.csv"  #6534 samples ? check false
TEST_CSV = PROJECT_ROOT / "data/test.csv" #1587
VAL_CSV = PROJECT_ROOT / "data/val.csv" #1587

# houldout 1566

SAVE_DATA_ROOT = Path(
    "/scale/project/ifremer-isi-jumeaunumerique/NEXTGEMS_PROCESSED_LAT_LON/data_128x128_3days"
)
SAVE_TRAIN_DIR = SAVE_DATA_ROOT / "train_patches"
SAVE_TEST_DIR = SAVE_DATA_ROOT / "test_patches"
SAVE_VAL_DIR = SAVE_DATA_ROOT / "val_patches"


# ------------------------------------------------------------------ spatial

def open_zarr_pair(data_root, season, basin, rank):
    """Open the atmospheric and ocean Zarr datasets for one cyclone track.

    Parameters
    ----------
    data_root : Path
        The root directory containing the raw datasets.
    season : int or str
        The season/year identifier.
    basin : str
        The basin identifier.
    rank : int
        The rank identifier.

    Returns
    -------
    tuple of xr.Dataset
        A tuple containing (ds_atm, ds_ocean).
    """
    base = data_root / f"data/{season}/{basin}/rank{rank}"
    ds_atm   = xr.open_zarr(base / ATM_ZARR,   consolidated=True)
    ds_ocean = xr.open_zarr(base / OCEAN_ZARR, consolidated=True)
    return ds_atm, ds_ocean


def obtain_dataframes_batches(eval_csv, tracks_csv, n_tracks=20, start=0):
    """Load cyclone tracking data frames in batches from split configuration files.

    Parameters
    ----------
    eval_csv : Path
        Path to the evaluation or split tracker CSV file.
    tracks_csv : Path
        Path to the global tracks tracking dataset CSV file.
    n_tracks : int, optional
        Number of tracks to process in this batch, by default 20.
    start : int, optional
        Starting row index for slicing the evaluation split, by default 0.

    Returns
    -------
    tuple of pd.DataFrame
        A tuple containing the sliced of the track_ids to be used (eval_split) and the full
        tracks data frame with generated in each timestamp (tracks_df).
    """

    eval_split = pd.read_csv(eval_csv)
    eval_split.columns = eval_split.columns.str.strip()

    # Skip previous tracks and take the next n_tracks
    eval_split = eval_split.iloc[start:start+n_tracks].reset_index(drop=True)

    tracks_df = pd.read_csv(tracks_csv)
    tracks_df.columns = tracks_df.columns.str.strip()

    tracks_df["time"] = tracks_df.apply(
        lambda r: dt.datetime(
            int(r["year"]), int(r["month"]), int(r["day"]), int(r["hour"])
        ), axis=1
    )

    return eval_split, tracks_df



# ------------------------------------------------------------------ spatial
def _xyz(lat_deg, lon_deg):
    la, lo = np.radians(lat_deg), np.radians(lon_deg)
    return np.column_stack((np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)))


def build_tree(ds):
    """KD-tree over the cell centres of a (possibly subsetted) HEALPix dataset.
    Build once per dataset and reuse for every time step / variable."""
    lat, lon = ds["lat"], ds["lon"]
    if "time" in lat.dims:
        lat, lon = lat.isel(time=0), lon.isel(time=0)
    return cKDTree(_xyz(lat.values, lon.values))


def _select_time(ds, variables, t, tol):
    """t is a Timestamp (nearest within tol) or a (start, end) tuple (mean)."""
    if isinstance(t, tuple):
        sel = ds[variables].sel(time=slice(*t))
        if sel.sizes["time"] == 0:
            raise KeyError(f"no data in {t}")
        return sel.mean("time"), sel["time"].values[sel.sizes["time"] // 2]
    sel = ds[variables].sel(time=t, method="nearest", tolerance=tol)
    return sel, sel["time"].values


def regrid_patch(ds, tree, variables, center_lat, center_lon, track_lat, track_lon,
                 time_target, res=0.1, target_pixels=128, k=4,
                 max_dist_factor=1.5, tol=pd.Timedelta("1h")):
    """Regrid HEALPix cells onto a regular lat/lon patch of exactly `res` degrees.

    * Pixel spacing is exactly `res` (the old linspace(c-box/2, c+box/2, n) gave
      box/(n-1), i.e. slightly different from `res`).
    * Points farther than max_dist_factor native pixels from any source cell
      (outside the subset) are NaN, never filled with copies.
    * If the nearest cell is NaN (land in ocean fields) the pixel is NaN, so
      coastlines are not pushed seaward by the interpolation.
    """
    half = (target_pixels - 1) / 2.0 * res
    # keep the box inside [-90, 90]
    c_lat = float(np.clip(center_lat, -90 + half, 90 - half))
    offs = (np.arange(target_pixels) - (target_pixels - 1) / 2.0) * res
    abs_lat, abs_lon = c_lat + offs, center_lon + offs
    grid_lat, grid_lon = np.meshgrid(abs_lat, abs_lon, indexing="ij")

    dist, idx = tree.query(_xyz(grid_lat.ravel(), grid_lon.ravel()), k=k)
    max_dist = np.radians(NATIVE_RES * max_dist_factor)  # chord ~ arc for small angles
    outside = dist[:, 0] > max_dist

    ds_t, resolved_time = _select_time(ds, variables, time_target, tol)

    w = 1.0 / (dist + 1e-9) ** 2
    out = {}
    for var in variables:
        vals = ds_t[var].values[idx]                    # (N, k)
        fin = np.isfinite(vals)
        ww = w * fin
        wsum = ww.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            res_1d = (np.where(fin, vals, 0.0) * ww).sum(1) / wsum
        res_1d[~fin[:, 0]] = np.nan                     # nearest cell is land/NaN
        res_1d[outside] = np.nan                        # outside the subset
        out[var] = res_1d.reshape(target_pixels, target_pixels)

    ds_out = xr.Dataset(
        {v: (["lat", "lon"], out[v]) for v in variables},
        coords={"lat": offs, "lon": offs},
    ).expand_dims(time=[pd.Timestamp(time_target[0] if isinstance(time_target, tuple) else time_target)])
    return ds_out.assign_coords(
        absolute_lat=(["time", "lat"], abs_lat[None, :]),
        absolute_lon=(["time", "lon"], abs_lon[None, :]),
        cyclone_center_lat=(["time"], [track_lat]),
        cyclone_center_lon=(["time"], [track_lon]),
        file_resolved_time=(["time"], [resolved_time]),
    )


# ----------------------------------------------------------------- temporal
def track_position_at(track, t, max_gap=pd.Timedelta("12h")):
    """Position of the storm at time t from irregular TempestExtremes detections.

    Linear interpolation (longitude unwrapped) between the two detections that
    bracket t, but ONLY if they are at most `max_gap` apart; otherwise (nan, nan),
    because the storm was not detected in between and the position is a guess.
    An exact detection time is always returned as is.
    """
    tt = track["time"].values.astype("datetime64[s]").astype(np.int64)
    t0 = np.datetime64(pd.Timestamp(t), "s").astype(np.int64)
    if t0 < tt[0] or t0 > tt[-1]:
        return np.nan, np.nan
    lon = np.degrees(np.unwrap(np.radians(track["lon"].values)))
    lat = track["lat"].values
    j = np.searchsorted(tt, t0, side="right")           # tt[j-1] <= t0 < tt[j]
    if j == len(tt):                                    # t0 == last detection
        return float(lat[-1]), float(lon[-1])
    lo, hi = j - 1, j
    if tt[lo] == t0:
        return float(lat[lo]), float(lon[lo])
    if (tt[hi] - tt[lo]) > max_gap.total_seconds():
        return np.nan, np.nan
    f = (t0 - tt[lo]) / (tt[hi] - tt[lo])
    return float(lat[lo] + f * (lat[hi] - lat[lo])), float(lon[lo] + f * (lon[hi] - lon[lo]))


def window_centers(track, centers="daily", target_hour=12, tol=pd.Timedelta("6h")):
    """Choose the DETECTIONS that become the middle frame of a window.

    centers='daily' : per calendar day, the real detection closest to `target_hour` UTC,
                      kept only if within +-tol. One sample per day (ocean is daily).
    centers='all'   : every detection (up to 4 per day). More samples, but samples of the
                      same day share the same ocean field.
    Returns a DataFrame: day (calendar day of the detection), t_atm (exact detection
    time), lat, lon.
    """
    if centers == "all":
        return pd.DataFrame({"day": track["time"].dt.normalize(), "t_atm": track["time"],
                             "lat": track["lat"], "lon": track["lon"]}).reset_index(drop=True)
    rows = []
    for d in pd.date_range(track["time"].iloc[0].normalize(), track["time"].iloc[-1].normalize(), freq="D"):
        target = d + pd.Timedelta(hours=target_hour)
        dist = (track["time"] - target).abs()
        k = dist.idxmin()
        if dist[k] <= tol:
            rows.append((d, track.at[k, "time"], track.at[k, "lat"], track.at[k, "lon"]))
    return pd.DataFrame(rows, columns=["day", "t_atm", "lat", "lon"])


def ocean_label_offset(ds_ocean):
    """Time-of-day at which the daily ocean record is stamped (e.g. 0h, 12h)."""
    t0 = pd.Timestamp(ds_ocean["time"].values[0])
    return t0 - t0.normalize()


def create_3step_patches(tracks_df, track_id, ds_atm, ds_ocean, res=0.1, target_pixels=128,
                         centers="daily", target_hour=12, tol=pd.Timedelta("6h"),
                         atm_mode="snapshot", ocean_day_shift=pd.Timedelta(0),
                         max_gap=pd.Timedelta("12h"),
                         atm_vars=("10u", "10v", "msl"),
                         ocn_vars=("avg_tos", "avg_zos", "avg_sos", "avg_mlotst125")):
    """One window per detected centre: frames at t-24h, t, t+24h on ONE fixed box.

    * Middle frame : atmosphere at the exact detection time t, ocean daily record of
                     the detection's calendar day D. Box centred on the detection.
    * Outer frames : atmosphere at t -/+ 24 h, ocean days D -/+ 1. NO detection needed;
                     the box stays where it is so the three frames are aligned.
                     `cyclone_center_lat/lon` of an outer frame is the track position
                     at that time if the storm was detected within `max_gap`, else NaN.
    atm_mode='daily_mean' replaces the atmosphere snapshots by the mean of the 24
    hourly steps of each day (matches the ocean daily mean, smears the storm).
    ocean_day_shift : pd.Timedelta('-1D') if ocean stamps are END-of-day labels.
    """
    track = tracks_df[tracks_df["track_id"] == track_id].sort_values("time").reset_index(drop=True)
    cen = window_centers(track, centers, target_hour, tol)
    if cen.empty:
        print(f"[WARN] Track {track_id}: no usable centre detections. Skipping.")
        return []

    tree_atm, tree_ocn = build_tree(ds_atm), build_tree(ds_ocean)
    ocn_off = ocean_label_offset(ds_ocean)
    windows = []

    for i, c in cen.iterrows():
        D, t_c, c_lat, c_lon = c["day"], c["t_atm"], c["lat"], c["lon"]
        try:
            atm_l, ocn_l = [], []
            for k in (-1, 0, 1):
                t_k = t_c + k * pd.Timedelta("24h")
                d_k = D + k * pd.Timedelta("1D")
                tr_lat, tr_lon = (c_lat, c_lon) if k == 0 else track_position_at(track, t_k, max_gap)
                t_atm = t_k if atm_mode == "snapshot" else (d_k, d_k + pd.Timedelta("23h"))
                # print(ds_atm)
                a = regrid_patch(ds_atm, tree_atm, list(atm_vars), c_lat, c_lon, tr_lat, tr_lon,
                                 t_atm, res, target_pixels, tol=pd.Timedelta("1h"))
                o = regrid_patch(ds_ocean, tree_ocn, list(ocn_vars), c_lat, c_lon, tr_lat, tr_lon,
                                 d_k + ocn_off + ocean_day_shift, res, target_pixels,
                                 tol=pd.Timedelta("1h"))
                atm_l.append(a.assign_coords(time=[t_k]))    # same label on both -> merge(join='exact')
                ocn_l.append(o.assign_coords(time=[t_k]))
        except KeyError as e:
            print(f"    [SKIP] window centred {t_c}: missing time step ({e})")
            continue

        with xr.set_options(use_new_combine_kwarg_defaults=True):
            windows.append({"atm": xr.concat(atm_l, dim="time", join="exact"),
                            "ocean": xr.concat(ocn_l, dim="time", join="exact"),
                            "win_idx": int(i), "center_time": t_c})
    return windows


# ------------------------------------------------------------- NaN handling
def fill_land(ds, variables=("avg_tos", "avg_zos", "avg_sos", "avg_mlotst125")):
    """Keep windows that touch land: add a binary `ocean_mask` (1 = ocean, 0 = land/NaN) and
    replace NaN by the mean of the valid ocean pixels of that frame (0 if none), so the
    arrays are NaN-free and the network can still be told where the land is."""
    ds = ds.copy()
    ref = variables[0]
    ds["ocean_mask"] = ds[ref].notnull().astype("float32")
    for v in variables:
        fill = ds[v].mean(dim=("lat", "lon"), skipna=True).fillna(0.0)
        ds[v] = ds[v].fillna(fill)
    return ds


# ==========================================
# SAVE WINDOW NETCDF WITH METADATA
# ==========================================
def save_3step_xarray(dataset_final_atm_ds, dataset_final_ocean_ds, save_dir, 
                      output_filename, track_meta, max_nan_percentage=0.0):
    """
    Synchronizes, filters land NaNs, attaches track metadata, and saves a 3-step window file.
    """
    ocean_sync = dataset_final_ocean_ds.copy()
    ocean_sync['time'] = dataset_final_atm_ds['time'].values

    atm_ready = dataset_final_atm_ds.rename({'file_resolved_time': 'atm_original_file_time'})
    ocean_ready = ocean_sync.rename({'file_resolved_time': 'ocean_original_file_time'})

    combined_ds = xr.merge([atm_ready, ocean_ready], join='exact', compat='no_conflicts')

    # Land filter check on 'avg_tos' across all 3 steps
    if max_nan_percentage is not None and 'avg_tos' in combined_ds:
        pixels_per_map = combined_ds.sizes["lat"] * combined_ds.sizes["lon"]
        nan_counts_per_day = combined_ds['avg_tos'].isnull().sum(dim=["lat", "lon"])
        nan_pct_per_day = nan_counts_per_day / pixels_per_map

        # Discard the ENTIRE window if any of the 3 days exceeds max_nan_percentage
        if (nan_pct_per_day > max_nan_percentage).any():
            print(f"    [DISCARD] Window failed land mask filter (> {max_nan_percentage * 100}% NaNs)")
            return None

    # Attach track attributes to NetCDF header for downstream dataset loaders
    combined_ds.attrs["track_id"] = int(track_meta["track_id"])
    combined_ds.attrs["season"]   = int(track_meta["season"])
    combined_ds.attrs["basin"]    = str(track_meta["basin"])
    combined_ds.attrs["rank"]     = int(track_meta["rank"])
    combined_ds.attrs["win_idx"]  = int(track_meta["win_idx"])

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        output_nc = save_dir / output_filename

        if output_nc.exists():
            output_nc.unlink()

        # Save NetCDF (existing behavior)
        combined_ds.to_netcdf(output_nc, mode='w')
        print(f"    [SAVED] {output_nc.name}")

        # Also save as Zarr for faster subsequent reads and preserved xarray metadata
        # try:
        #     zarr_dir = output_nc.with_suffix('.zarr')
        #     # remove existing zarr store if present
        #     if zarr_dir.exists():
        #         import shutil
        #         shutil.rmtree(zarr_dir)

        #     combined_ds.to_zarr(str(zarr_dir), mode='w', consolidated=True)
        #     print(f"    [SAVED] {zarr_dir.name}")
        # except Exception as _e:
        #     print(f"    [WARN] Failed to write Zarr store for {output_nc.name}: {_e}")
        # ------------------------------ end zarr
        
        combined_ds.close()

    return combined_ds


# ==========================================
# MAIN EXECUTION ROUTINE
# ==========================================
def run_create_patches(batch=1, eval_csv=TEST_CSV, save_dir=SAVE_TEST_DIR, 
                       tracks_csv=TRACKS_CSV, data_root=DATA_ROOT, n_tracks=20):
    start_idx = (batch - 1) * n_tracks

    print(f"\n==========================================")
    print(f"Running Batch {batch} | Tracks {start_idx} to {start_idx + n_tracks}")
    print(f"==========================================\n")

    eval_split, tracks_df = obtain_dataframes_batches(eval_csv, tracks_csv, n_tracks=n_tracks, start=start_idx)

    for _, ev in eval_split.iterrows():
        tid    = ev["track_id"]
        season = int(ev["season"])
        basin  = str(ev["basin"]).strip()
        rank   = int(ev["rank"])

        print(f"\nProcessing Track ID {int(tid)} — {season}/{basin}/rank{rank}")

        try:
            ds_atm, ds_ocean = open_zarr_pair(data_root, season, basin, rank)
        except Exception as e:
            print(f"Error loading Zarr data for track {tid}: {e}")
            continue

        # Extract all valid 3-step continuous windows
        windows = create_3step_patches(tracks_df, tid, ds_atm, ds_ocean)

        for win in windows:
            win_idx = win['win_idx']
            # output_filename = f"patch_trackid_{int(tid)}_win{win_idx}_{season}_{basin}_{rank}.nc"
            output_filename = f"patch_track{int(tid)}_{season}_{basin}_rank{rank}_win{win_idx}.nc"
            
            track_meta = {
                "track_id": tid,
                "season": season,
                "basin": basin,
                "rank": rank,
                "win_idx": win_idx
            }

            save_3step_xarray(
                dataset_final_atm_ds=win['atm'],
                dataset_final_ocean_ds=win['ocean'],
                save_dir=save_dir,
                output_filename=output_filename,
                track_meta=track_meta,
                max_nan_percentage=0.0  # Ocean-only constraint
            )


if __name__ == "__main__":

    # Parse command-line arguments passed from Slurm
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1, help="Batch number (1-indexed)")
    parser.add_argument("--n_tracks", type=int, default=5, help="Number of tracks per batch")
    parser.add_argument("--set", type=str, default="test", help="train or test")
    args = parser.parse_args()

    batch = args.batch
    n_tracks = args.n_tracks
    
    if args.set == "test":
        run_create_patches(batch=batch,
                        n_tracks=n_tracks, 
                        eval_csv=TEST_CSV, 
                        save_dir=SAVE_TEST_DIR,
                        tracks_csv=TRACKS_CSV, 
                        data_root=DATA_ROOT
                        )

    elif args.set == "val":
        run_create_patches(batch=batch,
                        n_tracks=n_tracks, 
                        eval_csv=VAL_CSV, 
                        save_dir=SAVE_VAL_DIR,
                        tracks_csv=TRACKS_CSV, 
                        data_root=DATA_ROOT
                        )

    elif args.set == "train":
        run_create_patches(batch=batch,
                        n_tracks=n_tracks,
                        eval_csv=TRAIN_CSV,
                        save_dir=SAVE_TRAIN_DIR,
                        tracks_csv=TRACKS_CSV,
                        data_root=DATA_ROOT
                        )     
    
    else:
        print("Select the correct dataset")

    
        
    # batch = 1 
    # n_tracks = 1000 
