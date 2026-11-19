"""
config.py - every path and hyper-parameter in one place.
Nothing here runs anything; the other modules import from it.
Paths can be redirected with the environment variables CYCLONE_DATA_ROOT / CYCLONE_OUT_DIR (used by the tests).
"""
import os
from pathlib import Path

import torch

# ==========================================
# DATA
# ==========================================
VARS_LIST = ["10u", "10v", "msl", "avg_tos", "avg_zos", "avg_sos", "avg_mlotst125"]

DATA_ROOT = Path(os.environ.get(
    "CYCLONE_DATA_ROOT", "/scale/project/ifremer-isi-jumeaunumerique/NEXTGEMS_PROCESSED_LAT_LON"))
TRAIN_DIR = DATA_ROOT / "data_128x128_3days/train_patches"
VAL_DIR = DATA_ROOT / "data_128x128_3days/val_patches"     # used for monitoring / choosing; TEST is only for the final report
TEST_DIR = DATA_ROOT / "data_128x128_3days/test_patches"
N_FRAMES = 3                      # timesteps per sample (3 days)
NUM_WORKERS = int(os.environ.get("CYCLONE_NUM_WORKERS", 16))   # threads used to read NetCDF files (set 1 if HDF5 crashes)

# ==========================================
# OUTPUTS  (one folder per experiment TYPE)
# ==========================================
CHECKPOINT_DIR = Path(os.environ.get("CYCLONE_OUT_DIR", DATA_ROOT / "result_ocean_model/diffusion"))
# CHECKPOINT_DIR = Path(os.environ.get("CYCLONE_OUT_DIR", DATA_ROOT / "result_vf/vae_diffusion"))
TYPE = "vae_diffusion_no_masked"
SAVE_DIR = CHECKPOINT_DIR / "plots" / TYPE

VAE_CHECKPOINT = CHECKPOINT_DIR / f"checkpoint_model_{TYPE}.pt"
DIFFUSION_TYPE = f"unet_{TYPE}"
DIFFUSION_CHECKPOINT = CHECKPOINT_DIR / f"checkpoint_{DIFFUSION_TYPE}.pt"

LATENTS_DIR = CHECKPOINT_DIR / "latents" / TYPE                      # train windows, one .pt each (+ latent_normalizer.pt)
LATENTS_DIR_TEST = CHECKPOINT_DIR / "latents" / f"test_{TYPE}"       # test windows, one .pt each
LATENTS_DIR_VAL = CHECKPOINT_DIR / "latents" / f"val_{TYPE}"         # validation windows, one .pt each
LATENTS_TRAIN = LATENTS_DIR
LATENTS_VAL = LATENTS_DIR_VAL
LATENTS_TEST = LATENTS_DIR_TEST
EVAL_METRICS_JSON = CHECKPOINT_DIR / f"diffusion_eval_metrics_{DIFFUSION_TYPE}.json"      # test split
EVAL_METRICS_CSV = CHECKPOINT_DIR / f"diffusion_eval_metrics_{DIFFUSION_TYPE}.csv"


def metrics_paths(split: str = "test"):
    """metrics files of an evaluation split; the test files keep their original names, val gets its own (no overwriting)"""
    if split == "test":
        return EVAL_METRICS_JSON, EVAL_METRICS_CSV
    return (CHECKPOINT_DIR / f"diffusion_eval_metrics_{DIFFUSION_TYPE}_{split}.json",
            CHECKPOINT_DIR / f"diffusion_eval_metrics_{DIFFUSION_TYPE}_{split}.csv")

# ==========================================
# VAE
# ==========================================
LATENT_CHANNELS = 128
VAE_EPOCHS = 1                    # <- raise this for real runs
VAE_BATCH_SIZE = 32
VAE_LR = 1e-3
VAE_BETA = 1e-4
# True: the VAE reconstructs the ANOMALY (patch mean removed, rescaled to unit variance per variable); the patch mean is
# carried by extra latent channels. Needs the VAE (and everything downstream) to be retrained; old checkpoints still load.
ANOMALY_MODE = bool(int(os.environ.get("CYCLONE_ANOMALY_MODE", 1)))

# ==========================================
# LATENT CACHE / NORMALISATION
# ==========================================
STORE_LOGVAR = False              # True: also cache logvar so the diffusion can train on sampled z (not only mu)
LATENT_STD_FLOOR_REL = 0.1        # per-channel std is never below 0.1 * median std (avoids blowing up dead channels)

# ==========================================
# DIFFUSION
# ==========================================
DIFFUSION_EPOCHS = 1              # <- raise this for real runs
DIFFUSION_BATCH_SIZE = 64
DIFFUSION_STEPS = 1000
DIFFUSION_LR = 2e-4
DIFFUSION_BASE_CHANNELS = 128
EMA_DECAY = 0.999
DIFFUSION_PREDICTION = "v"       # "v" (recommended: well-conditioned at high noise) or "eps"
SAMPLE_LATENTS_TRAIN = False      # needs STORE_LOGVAR=True

# Which frames are missing. Names -> indices of the MISSING frames (0 = first day, 2 = last day)
SCENARIOS = {
    "mid":    (1,),       # interpolation        : know t0,t2 -> predict t1
    "last":   (2,),       # forecast             : know t0,t1 -> predict t2
    "first":  (0,),       # backcast             : know t1,t2 -> predict t0
    "last2":  (1, 2),     # 2-step forecast      : know t0    -> predict t1,t2
    "first2": (0, 1),     # 2-step backcast      : know t2    -> predict t0,t1
    "ends":   (0, 2),     # centre only          : know t1    -> predict t0,t2
}
# how often each pattern is drawn while TRAINING (normalised automatically)
TRAIN_SCENARIO_PROBS = {"mid": 0.30, "last": 0.20, "first": 0.20, "last2": 0.10, "first2": 0.10, "ends": 0.10}

# ==========================================
# EVALUATION
# ==========================================
EVAL_BATCH_SIZE = 64
DDIM_STEPS = 100
DDIM_ETA = 0.0                    # 0 = deterministic DDIM, 1 = DDPM-like stochastic
N_ENSEMBLE = 1                    # >1: also report the ensemble-mean score
N_PLOT = 10                       # figures per scenario (metrics always use the WHOLE test set)
PLOT_SCENARIOS = ("mid", "last", "first")
EVAL_SCENARIOS = tuple(SCENARIOS)
IMSHOW_ORIGIN = "upper"           # matplotlib default; set "lower" if latitude is stored south->north

PHYS_SCALE = {"msl": 100.0}       # Pa -> hPa for metrics and plots
VAR_LABELS = {
    "10u": "Zonal wind 10u, m/s",
    "10v": "Meridional wind 10v, m/s",
    "msl": "Mean sea level pressure, hPa",
    "avg_tos": "Sea surface temperature, °C",
    "avg_zos": "Sea surface height, m",
    "avg_sos": "Sea surface salinity, psu",
    "avg_mlotst125": "Mixed layer depth, m",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dirs():
    for d in (CHECKPOINT_DIR, LATENTS_DIR, LATENTS_DIR_VAL, LATENTS_DIR_TEST, SAVE_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)
