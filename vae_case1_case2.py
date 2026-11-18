
"""
Adapted sections for the cyclone VAE (masked reconstruction in space and variable).

REPLACE in your file:  apply_masking, CycloneVAE, VAELoss, MetricsManager,
                       train_epoch, evaluate_model, run_training
KEEP as they are:      parse_patch_filename, CyclonePatchDataset, ResBlock,
                       LatentStatsCollector and your constants
                       (TRAIN_DIR, TEST_DIR, BATCH_SIZE, EPOCHS, CHECKPOINT, ...)
"""
import gc
import glob
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import DataLoader, Dataset

# ==========================================
# CONFIGURATION & GLOBAL SETUP
# ==========================================

DATA_ROOT  = Path("/scale/project/ifremer-isi-jumeaunumerique/NEXTGEMS_PROCESSED_LAT_LON")
TRAIN_DIR  = DATA_ROOT / "data_128x128_3days/train_patches"
TEST_DIR   = DATA_ROOT / "data_128x128_3days/test_patches"

BATCH_SIZE = 32
EPOCHS     = 200 #00 # 200# 00 #100 #00 #0

CHECKPOINT_DIR = DATA_ROOT / "ml_result"/"vae_case1_case2"
TYPE = "vae_case1_case2"
CHECKPOINT = CHECKPOINT_DIR / f"checkpoint_model_{TYPE}.pt"
SAVE_DIR   = CHECKPOINT_DIR / "plots/eval"


if SAVE_DIR is not None:
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

# Plotting aesthetics
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'legend.frameon': True,
    'figure.titlesize': 22,
    'lines.linewidth': 2.5,
})

VAR_LABELS = {
    "10u":             "Zonal wind 10u, m/s",
    "10v":             "Meridional wind 10v, m/s",
    "msl":             "Mean sea level pressure, Pa",
    "avg_tos":         "Sea surface temperature, °C",
    "avg_zos":         "Sea surface height, m",
    "avg_sos":         "Sea surface salinity, psu",
    "avg_mlotst125":   "Mixed layer depth, m",
}


VARS_LIST = ["10u", "10v", "msl", "avg_tos", "avg_zos", "avg_sos", "avg_mlotst125"]



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
    
class CyclonePatchDataset(Dataset):
    def __init__(self, folder_path, variables, stats_dict=None, num_workers=16):
        self.folder_path = Path(folder_path)
        self.variables = variables
        
        self.files = sorted(glob.glob(os.path.join(folder_path, "*.nc")))
        if not self.files:
            raise FileNotFoundError(f"No NetCDF files found in {folder_path}")

        start_time = time.perf_counter()

        # # 1. Store lazy xarray Dataset reference so .ds.isel(time=idx) works for evaluation
        # self.ds = xr.open_mfdataset(self.files, combine='nested', concat_dim='time')

        # 2. Fast Parallel NetCDF IO into RAM for PyTorch training loop
        def _load_file(fpath):
            with xr.open_dataset(fpath) as ds:
                file_data = np.stack([ds[var].values for var in self.variables], axis=1).astype(np.float32)
                filename = Path(fpath).name
                timesteps = file_data.shape[0]
            return file_data, filename, timesteps

        print(f"[dataset] Loading {len(self.files)} NetCDF files in parallel...")
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            results = list(ex.map(_load_file, self.files))

        # 3. Reconstruct contiguous data array & time_to_filename
        all_data = []
        self.time_to_filename = []

        for file_data, filename, timesteps in results:
            all_data.append(file_data)
            self.time_to_filename.extend([filename] * timesteps)

        self.data = np.concatenate(all_data, axis=0)
        del all_data

        # 4. Compute / apply normalization statistics
        if stats_dict is None:
            self.stats = {}
            for i, var in enumerate(self.variables):
                self.stats[var] = {
                    'mean': float(self.data[:, i, :, :].mean()),
                    'std': float(self.data[:, i, :, :].std())
                }
        else:
            self.stats = stats_dict

        for i, var in enumerate(self.variables):
            mean = self.stats[var]['mean']
            std = self.stats[var]['std']
            self.data[:, i, :, :] = (self.data[:, i, :, :] - mean) / (std if std > 1e-6 else 1.0)

        total_elapsed = time.perf_counter() - start_time
        print(f"[dataset] Total initialization time: {total_elapsed:.3f} s")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        tensor_sample = torch.from_numpy(self.data[idx])
        return tensor_sample, tensor_sample, idx

    def get_metadata(self, idx: int) -> "SampleMetadata":
        """
        Reads everything needed to plot and name a sample properly: the track
        filename fields (track_id/season/basin/rank), the lat/lon coordinate
        arrays, the cyclone center for the selected timestep, and the
        timestamp string. This is deliberately kept OUT of __getitem__ (which
        must stay a plain tensor for DataLoader batching/collation) and is
        only called directly for inference/visualization, one sample at a time.
        """

        file_idx = idx //3 # self.file_num_frames each file has 3 timesteps, so integer division gives the file index
        day_idx = idx % 3 #self.file_num_frames each file has 3 timesteps, so modulo gives the day index within the file
        
        path = self.files[file_idx]
        ds = xr.open_dataset(path)

        # file_info = parse_patch_filename(path)
        file_info = parse_patch_filename(ds)

        lat = np.asarray(ds["absolute_lat"][0].values, dtype=np.float64) if "absolute_lat" in ds.coords or "absolute_lat" in ds.variables else None
        lon = np.asarray(ds["absolute_lon"][0].values, dtype=np.float64) if "absolute_lon" in ds.coords or "absolute_lon" in ds.variables else None

        # timestep (shape [file_num_frames]) -- handle both.
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
        if "time" in ds.variables or "time" in ds.coords:
            time_raw = np.asarray(ds["time"].values)
            time_single = time_raw if time_raw.ndim == 0 else time_raw[day_idx]
            time_val = np.datetime_as_string(time_single, unit="h")

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
            cyclone_center_lat=float(latc) if latc is not None else None,
            cyclone_center_lon=float(lonc) if lonc is not None else None,
            time_str=time_val,
        )



# ==========================================
# MASKING CONFIG
# ==========================================
# Variables that go missing together ("units"). 10u/10v are physically one
# vector field (same product/sensor) and are so correlated that masking only one
# is an easy task, so they are masked as ONE unit. Split them into
# ["10u"], ["10v"] here if you also want to test the single-component case.
VAR_GROUPS = [
    ["10u", "10v"],
    ["msl"],
    ["avg_tos"],
    ["avg_zos"],
    ["avg_sos"],
    ["avg_mlotst125"],
]

MODE_NAMES = ["none", "block_all", "block_group", "group", "patches"]
MODE_PROBS = [0.10, 0.25, 0.20, 0.25, 0.20]   # per-sample probability of each mode
P_TWO_GROUPS = 0.15        # prob. that a 2nd unit is also masked (never more than 2)
BLOCK_FRAC = (0.25, 0.50)  # block height/width as a fraction of H/W
CENTER_BLOCK_PROB = 0.30   # fraction of blocks centred near the patch centre (the eye),
                           # assumes your patches are roughly cyclone-centred
PATCH_SIZE = 8             # MAE-style scattered patches
PATCH_RATIO = (0.3, 0.7)   # fraction of patches dropped, sampled per sample


def build_group_matrix(vars_list, groups, device):
    """(G, C) float matrix: entry [g, c] = 1 if channel c belongs to group g."""
    gm = torch.zeros(len(groups), len(vars_list), dtype=torch.float32)
    for g, names in enumerate(groups):
        for n in names:
            gm[g, vars_list.index(n)] = 1.0
    assert (gm.sum(0) == 1).all(), "every variable must belong to exactly one group"
    return gm.to(device)


def _rand(shape, device, gen):
    return torch.rand(shape, device=device, generator=gen)


def _random_blocks(B, H, W, device, gen, frac=BLOCK_FRAC, center_prob=CENTER_BLOCK_PROB):
    """Vectorised rectangular blocks, (B, H, W) bool. Size 25-50% of H/W (inclusive)."""
    lo, hi = frac
    h = ((lo + (hi - lo) * _rand((B,), device, gen)) * H).long().clamp(min=1)
    w = ((lo + (hi - lo) * _rand((B,), device, gen)) * W).long().clamp(min=1)

    # uniform position so that the block always fits (top in [0, H-h])
    top = (_rand((B,), device, gen) * (H - h + 1).float()).long()
    left = (_rand((B,), device, gen) * (W - w + 1).float()).long()

    # a fraction of the blocks is centred (+-10% jitter) on the patch centre
    centred = _rand((B,), device, gen) < center_prob
    jit_y = ((_rand((B,), device, gen) - 0.5) * 0.2 * H).long()
    jit_x = ((_rand((B,), device, gen) - 0.5) * 0.2 * W).long()
    top = torch.where(centred, (H - h) // 2 + jit_y, top)
    left = torch.where(centred, (W - w) // 2 + jit_x, left)
    top = torch.minimum(top.clamp(min=0), H - h)
    left = torch.minimum(left.clamp(min=0), W - w)

    ys = torch.arange(H, device=device).view(1, H, 1)
    xs = torch.arange(W, device=device).view(1, 1, W)
    t, l = top.view(B, 1, 1), left.view(B, 1, 1)
    return (ys >= t) & (ys < t + h.view(B, 1, 1)) & (xs >= l) & (xs < l + w.view(B, 1, 1))


def _select_groups(B, G, device, gen, p_two=P_TWO_GROUPS):
    """(B, G) bool: one random group per sample, plus a 2nd one with prob p_two."""
    order = _rand((B, G), device, gen).argsort(dim=1)
    sel = F.one_hot(order[:, 0], G).bool()
    if G > 1:
        second = F.one_hot(order[:, 1], G).bool()
        use_second = (_rand((B,), device, gen) < p_two).view(B, 1)
        sel = sel | (second & use_second)
    return sel


def apply_masking(batch_tensor, group_matrix, mode_probs=MODE_PROBS, generator=None):
    """
    Per-sample masking, fully vectorised. One mode is drawn independently for
    each sample (probabilities `mode_probs`, order = MODE_NAMES):

    - none         : nothing masked.
    - block_all    : one rectangular block (25-50% of H and W) masked in ALL variables
                     (missing region in space).
    - block_group  : the same kind of block, but only in 1 unit (rarely 2) of
                     variables, e.g. a hole in the ocean fields only.
    - group        : 1 unit (rarely 2) of variables completely missing
                     (missing in variable). 10u/10v always go together.
    - patches      : scattered MAE-style patches, drawn independently per unit.

    Missing values are set to 0.0 (the mean after z-score), so the model MUST
    also receive the mask (see build_model_input).

    Parameters
    ----------
    batch_tensor : (B, C, H, W) float tensor (normalised)
    group_matrix : (G, C) float tensor from build_group_matrix
    generator    : optional torch.Generator on the same device (for fixed eval masks)

    Returns
    -------
    masked_tensor : (B, C, H, W) with missing entries set to 0.0
    mask          : (B, C, H, W) bool, True = missing (hidden from the model)
    modes         : (B,) long, index into MODE_NAMES (for per-mode metrics)
    """
    B, C, H, W = batch_tensor.shape
    G = group_matrix.shape[0]
    dev = batch_tensor.device

    probs = torch.tensor(mode_probs, device=dev, dtype=torch.float32)
    modes = torch.multinomial(probs, B, replacement=True, generator=generator)  # (B,)

    block = _random_blocks(B, H, W, dev, generator)          # (B, H, W)
    sel = _select_groups(B, G, dev, generator)               # (B, G)

    nH, nW = -(-H // PATCH_SIZE), -(-W // PATCH_SIZE)
    ratio = PATCH_RATIO[0] + (PATCH_RATIO[1] - PATCH_RATIO[0]) * _rand((B, 1, 1, 1), dev, generator)
    pm = (_rand((B, G, nH, nW), dev, generator) < ratio).float()
    pm = F.interpolate(pm, size=(H, W), mode="nearest") > 0.5   # (B, G, H, W)

    unit = torch.zeros(B, G, H, W, dtype=torch.bool, device=dev)
    unit |= (modes == 1).view(B, 1, 1, 1) & block.unsqueeze(1)
    unit |= (modes == 2).view(B, 1, 1, 1) & sel.view(B, G, 1, 1) & block.unsqueeze(1)
    unit |= (modes == 3).view(B, 1, 1, 1) & sel.view(B, G, 1, 1)
    unit |= (modes == 4).view(B, 1, 1, 1) & pm

    # unit (B,G,H,W) -> channel mask (B,C,H,W)
    mask = torch.einsum("bghw,gc->bchw", unit.float(), group_matrix) > 0.5
    masked_tensor = batch_tensor.masked_fill(mask, 0.0)
    return masked_tensor, mask, modes


def build_model_input(masked_tensor, mask):
    """Concatenate the masked data and the observed-mask: (B, 2C, H, W)."""
    return torch.cat([masked_tensor, (~mask).float()], dim=1)


# ==========================================
# MODEL DEFINITION
# ==========================================
class ResBlock(nn.Module):  # unchanged
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + x)


class CycloneVAE(nn.Module):
    """
    in_channels  = 2 * n_vars  (masked data + observed-mask)
    out_channels = n_vars      (full reconstruction)
    """
    def __init__(self, in_channels=14, out_channels=7, latent_channels=128):
        super().__init__()

        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            ResBlock(32),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            ResBlock(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            ResBlock(128),
        )

        self.to_mu = nn.Conv2d(128, latent_channels, kernel_size=1)
        self.to_logvar = nn.Conv2d(128, latent_channels, kernel_size=1)
        self.from_latent = nn.Conv2d(latent_channels, 128, kernel_size=1)

        self.dec = nn.Sequential(
            ResBlock(128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),

            ResBlock(64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),

            ResBlock(32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1, padding_mode='replicate'),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.to_mu(h), self.to_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        return self.dec(self.from_latent(z))

    def forward(self, x, sample=None):
        """sample=None -> sample z in train mode, use the mean mu in eval mode."""
        if sample is None:
            sample = self.training
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if sample else mu
        return self.decode(z), mu, logvar


class MaskedVAELoss(nn.Module):
    """
    loss = w_obs * MSE(observed) + w_miss * MSE(missing) + beta * KL

    The two regions are disjoint (no double counting), each normalised by its
    own element count. w_obs > 0 keeps the output faithful on visible pixels
    without letting the model learn a pure identity; w_miss drives inpainting.
    """
    def __init__(self, beta=1e-5, w_obs=0.2, w_miss=1.0):
        super().__init__()
        self.beta = beta          # updated from the training loop (KL warm-up)
        self.w_obs = w_obs
        self.w_miss = w_miss

    def forward(self, x_hat, x, mu, logvar, mask=None):
        sq = (x_hat - x) ** 2
        if mask is None:
            mask = torch.zeros_like(x, dtype=torch.bool)
        obs = ~mask

        mse_obs = (sq * obs).sum() / obs.sum().clamp(min=1)
        mse_miss = (sq * mask).sum() / mask.sum().clamp(min=1)   # 0 if nothing masked
        recon = self.w_obs * mse_obs + self.w_miss * mse_miss

        # KL: sum over latent dims per sample, then mean over batch
        kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=(1, 2, 3))
        kl = torch.mean(kl_per_sample)

        return recon + self.beta * kl, recon.detach(), kl.detach()


# ==========================================
# METRICS (missing region only, streaming: nothing is stored in RAM)
# ==========================================
class MaskedMetrics:
    """MSE / MAE / bias computed ONLY on masked entries (normalised units)."""
    def __init__(self, variables):
        self.variables = variables
        self.reset()

    def reset(self):
        C, M = len(self.variables), len(MODE_NAMES)
        z = lambda n: torch.zeros(n, dtype=torch.float64)
        self.sq, self.ab, self.bi, self.n = z(C), z(C), z(C), z(C)
        self.m_sq, self.m_n = z(M), z(M)

    @torch.no_grad()
    def update(self, preds, targets, mask, modes):
        err = (preds - targets).double()
        m = mask.double()
        self.sq += ((err ** 2) * m).sum(dim=(0, 2, 3)).cpu()
        self.ab += (err.abs() * m).sum(dim=(0, 2, 3)).cpu()
        self.bi += (err * m).sum(dim=(0, 2, 3)).cpu()
        self.n += m.sum(dim=(0, 2, 3)).cpu()
        modes = modes.cpu()
        self.m_sq.index_add_(0, modes, ((err ** 2) * m).sum(dim=(1, 2, 3)).cpu())
        self.m_n.index_add_(0, modes, m.sum(dim=(1, 2, 3)).cpu())

    def calculate_metrics(self):
        nan = float("nan")
        n_tot = self.n.sum().clamp(min=1)
        summary = {
            "overall": {
                "mse": float(self.sq.sum() / n_tot),
                "mae": float(self.ab.sum() / n_tot),
                "bias": float(self.bi.sum() / n_tot),
            },
            "per_variable": {},
            "per_mode": {},
        }
        for i, var in enumerate(self.variables):
            n = float(self.n[i])
            summary["per_variable"][var] = {
                "mse": float(self.sq[i]) / n if n > 0 else nan,
                "mae": float(self.ab[i]) / n if n > 0 else nan,
                "bias": float(self.bi[i]) / n if n > 0 else nan,
            }
        for k, name in enumerate(MODE_NAMES):
            n = float(self.m_n[k])
            summary["per_mode"][name] = float(self.m_sq[k]) / n if n > 0 else nan
        return summary


# ==========================================
# TRAIN / EVAL
# ==========================================
def train_epoch(model, dataloader, criterion, optimizer, device, metrics_mgr,
                group_matrix, latent_collector=None):
    model.train()
    metrics_mgr.reset()
    epoch_loss = recon_total = kl_total = 0.0

    for inputs, targets, _ in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        masked_inputs, mask, modes = apply_masking(inputs, group_matrix)
        model_in = build_model_input(masked_inputs, mask)

        optimizer.zero_grad()
        x_hat, mu, logvar = model(model_in)
        loss, recon, kl = criterion(x_hat, targets, mu, logvar, mask)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = inputs.size(0)
        epoch_loss += loss.item() * bs
        recon_total += recon.item() * bs
        kl_total += kl.item() * bs
        metrics_mgr.update(x_hat.detach(), targets, mask, modes)
        if latent_collector is not None:
            try:
                latent_collector.update(mu, logvar)
            except Exception:
                pass

    n = len(dataloader.dataset)
    return epoch_loss / n, metrics_mgr.calculate_metrics(), recon_total / n, kl_total / n


def evaluate_model(model, dataloader, criterion, device, metrics_mgr, group_matrix,
                   seed=1234, latent_collector=None):
    """Same masking distribution as training, but with FIXED seeded masks
    (identical every epoch, so epochs are comparable) and z = mu (deterministic)."""
    model.eval()
    metrics_mgr.reset()
    test_loss = recon_total = kl_total = 0.0

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    with torch.no_grad():
        for inputs, targets, _ in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            masked_inputs, mask, modes = apply_masking(inputs, group_matrix, generator=gen)
            x_hat, mu, logvar = model(build_model_input(masked_inputs, mask), sample=False)
            loss, recon, kl = criterion(x_hat, targets, mu, logvar, mask)

            bs = inputs.size(0)
            test_loss += loss.item() * bs
            recon_total += recon.item() * bs
            kl_total += kl.item() * bs
            metrics_mgr.update(x_hat, targets, mask, modes)
            if latent_collector is not None:
                try:
                    latent_collector.update(mu, logvar)
                except Exception:
                    pass

    n = len(dataloader.dataset)
    return test_loss / n, metrics_mgr.calculate_metrics(), recon_total / n, kl_total / n


# ==========================================
# EXECUTION PIPELINE
# ==========================================
BETA_KL = 1e-5
KL_WARMUP_EPOCHS = 10


def run_training(vars_list=VARS_LIST, train_dir=None, test_dir=None, batch_size=None, n_epochs=None):
    # TRAIN_DIR, TEST_DIR, BATCH_SIZE, EPOCHS, CHECKPOINT_DIR, CHECKPOINT,
    # CyclonePatchDataset and LatentStatsCollector come from your own file.
    train_dir = train_dir or TRAIN_DIR
    test_dir = test_dir or TEST_DIR
    batch_size = batch_size or BATCH_SIZE
    n_epochs = n_epochs or EPOCHS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = CyclonePatchDataset(train_dir, variables=vars_list)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("train_dataset len:", len(train_dataset))
    print("train_loader len:", len(train_loader))

    test_dataset = CyclonePatchDataset(test_dir, variables=vars_list, stats_dict=train_dataset.stats)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # keep False: fixed masks

    C = len(vars_list)
    group_matrix = build_group_matrix(vars_list, VAR_GROUPS, device)

    model = CycloneVAE(in_channels=2 * C, out_channels=C, latent_channels=128).to(device)
    criterion = MaskedVAELoss(beta=0.0, w_obs=0.2, w_miss=1.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    train_metrics_mgr = MaskedMetrics(vars_list)
    test_metrics_mgr = MaskedMetrics(vars_list)
    train_losses, test_losses = [], []
    train_recons, test_recons = [], []
    train_kls, test_kls = [], []

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    collector = LatentStatsCollector()
    best_miss_mse, best_epoch = float('inf'), 0

    for epoch in range(n_epochs):
        # KL warm-up: beta ramps 0 -> BETA_KL
        criterion.beta = BETA_KL * min(1.0, epoch / KL_WARMUP_EPOCHS)

        collector.reset()
        train_loss, train_m, train_recon, train_kl = train_epoch(
            model, train_loader, criterion, optimizer, device, train_metrics_mgr,
            group_matrix, latent_collector=collector)
        scheduler.step()

        test_loss, test_m, test_recon, test_kl = evaluate_model(
            model, test_loader, criterion, device, test_metrics_mgr, group_matrix)

        train_losses.append(train_loss); test_losses.append(test_loss)
        train_recons.append(train_recon); test_recons.append(test_recon)
        train_kls.append(train_kl); test_kls.append(test_kl)

        miss_mse = test_m["overall"]["mse"]   # MSE on missing entries only
        print(f"Epoch {epoch+1}/{n_epochs} | beta {criterion.beta:.1e} | "
              f"Train Loss: {train_loss:.5f} | Test Loss: {test_loss:.5f} | "
              f"Train miss-MSE: {train_m['overall']['mse']:.4f} | Test miss-MSE: {miss_mse:.4f}")
        print("   test miss-MSE per mode:", {k: round(v, 4) for k, v in test_m["per_mode"].items()})

        # Select the checkpoint on the missing-region MSE: it does not depend on
        # beta (the total loss changes during KL warm-up, so it is not comparable).
        if miss_mse < best_miss_mse:
            best_miss_mse, best_epoch = miss_mse, epoch + 1
            collector.save_csv(CHECKPOINT_DIR / "latent_stats_train.csv")
            torch.save({
                "model_state": model.state_dict(),
                "stats": train_dataset.stats,
                "vars_list": vars_list,
                "var_groups": VAR_GROUPS,
                "in_channels": 2 * C,
                "out_channels": C,
                "epoch": best_epoch,
                "test_miss_mse": miss_mse,
                "test_metrics": test_m,
            }, CHECKPOINT)
            print(f"  -> New best model (test miss-MSE={miss_mse:.4f}), checkpoint + latent stats updated")

    print(f"\nBest checkpoint saved to {CHECKPOINT} "
          f"(test miss-MSE={best_miss_mse:.4f}, epoch {best_epoch})")
    
    # --- Plot per-epoch reconstruction and KL losses for analysis ---
    try:
        epochs = list(range(1, n_epochs + 1))
        Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_recons, label='train_recon')
        plt.plot(epochs, test_recons, label='test_recon')
        plt.xlabel('Epoch')
        plt.ylabel('Reconstruction MSE')
        plt.title('Reconstruction Loss per Epoch')
        plt.legend()
        fname_rec = Path(SAVE_DIR) / 'reconstruction_loss_per_epoch.png'
        plt.savefig(fname_rec, dpi=150, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_kls, label='train_kl')
        plt.plot(epochs, test_kls, label='test_kl')
        plt.xlabel('Epoch')
        plt.ylabel('KL Divergence')
        plt.title('KL Loss per Epoch')
        plt.legend()
        fname_kl = Path(SAVE_DIR) / 'kl_loss_per_epoch.png'
        plt.savefig(fname_kl, dpi=150, bbox_inches='tight')
        plt.close()

        # Combined plot (log scale for KL to see dynamics)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_losses, label='train_total')
        plt.plot(epochs, test_losses, label='test_total')
        plt.xlabel('Epoch')
        plt.ylabel('Total Loss')
        plt.title('Total Loss per Epoch')
        plt.legend()
        fname_tot = Path(SAVE_DIR) / 'total_loss_per_epoch.png'
        plt.savefig(fname_tot, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved loss plots to {SAVE_DIR}")
    except Exception as e:
        print(f"Could not save loss plots: {e}")





class LatentStatsCollector:
    """Accumulates per-channel latent `mu` and `logvar` statistics across batches.

    Usage:
      collector = LatentStatsCollector()
      collector.update(mu, logvar)
      collector.save_csv(path)
    """
    def __init__(self):
        self.mu_sum = None
        self.mu_sq_sum = None
        self.logvar_sum = None
        self.logvar_sq_sum = None
        self.count = 0
        self.mu_min = None
        self.mu_max = None
        self.logvar_min = None
        self.logvar_max = None

    def reset(self):
        self.__init__()

    def update(self, mu: torch.Tensor, logvar: torch.Tensor):
        # Expect shapes: [B, C, H, W]
        m = mu.detach().cpu().double()
        lv = logvar.detach().cpu().double()
        B, C, H, W = m.shape
        n = B * H * W

        mu_sum_batch = m.sum(dim=(0, 2, 3))  # (C,)
        mu_sq_sum_batch = (m * m).sum(dim=(0, 2, 3))
        lv_sum_batch = lv.sum(dim=(0, 2, 3))
        lv_sq_sum_batch = (lv * lv).sum(dim=(0, 2, 3))

        if self.mu_sum is None:
            self.mu_sum = mu_sum_batch.clone()
            self.mu_sq_sum = mu_sq_sum_batch.clone()
            self.logvar_sum = lv_sum_batch.clone()
            self.logvar_sq_sum = lv_sq_sum_batch.clone()
            # initialize min/max
            try:
                self.mu_min = m.amin(dim=(0, 2, 3)).clone()
                self.mu_max = m.amax(dim=(0, 2, 3)).clone()
                self.logvar_min = lv.amin(dim=(0, 2, 3)).clone()
                self.logvar_max = lv.amax(dim=(0, 2, 3)).clone()
            except Exception:
                # fallback for older PyTorch: flatten then reduce
                mflat = m.view(B, C, -1)
                lvflat = lv.view(B, C, -1)
                self.mu_min = mflat.min(dim=0)[0].min(dim=1)[0].clone()
                self.mu_max = mflat.max(dim=0)[0].max(dim=1)[0].clone()
                self.logvar_min = lvflat.min(dim=0)[0].min(dim=1)[0].clone()
                self.logvar_max = lvflat.max(dim=0)[0].max(dim=1)[0].clone()
        else:
            self.mu_sum += mu_sum_batch
            self.mu_sq_sum += mu_sq_sum_batch
            self.logvar_sum += lv_sum_batch
            self.logvar_sq_sum += lv_sq_sum_batch
            # update min/max
            try:
                self.mu_min = torch.minimum(self.mu_min, m.amin(dim=(0, 2, 3)))
                self.mu_max = torch.maximum(self.mu_max, m.amax(dim=(0, 2, 3)))
                self.logvar_min = torch.minimum(self.logvar_min, lv.amin(dim=(0, 2, 3)))
                self.logvar_max = torch.maximum(self.logvar_max, lv.amax(dim=(0, 2, 3)))
            except Exception:
                mflat = m.view(B, C, -1)
                lvflat = lv.view(B, C, -1)
                batch_mu_min = mflat.min(dim=0)[0].min(dim=1)[0]
                batch_mu_max = mflat.max(dim=0)[0].max(dim=1)[0]
                batch_lv_min = lvflat.min(dim=0)[0].min(dim=1)[0]
                batch_lv_max = lvflat.max(dim=0)[0].max(dim=1)[0]
                self.mu_min = torch.minimum(self.mu_min, batch_mu_min)
                self.mu_max = torch.maximum(self.mu_max, batch_mu_max)
                self.logvar_min = torch.minimum(self.logvar_min, batch_lv_min)
                self.logvar_max = torch.maximum(self.logvar_max, batch_lv_max)

        self.count += n

    def finalize(self):
        if self.count == 0:
            raise RuntimeError("No latent stats collected")
        mu_mean = (self.mu_sum / self.count).numpy()
        mu_var = (self.mu_sq_sum / self.count - (self.mu_sum / self.count) ** 2).numpy()
        mu_std = np.sqrt(np.maximum(mu_var, 1e-12))

        lv_mean = (self.logvar_sum / self.count).numpy()
        lv_var = (self.logvar_sq_sum / self.count - (self.logvar_sum / self.count) ** 2).numpy()
        lv_std = np.sqrt(np.maximum(lv_var, 1e-12))

        mu_min = self.mu_min.cpu().numpy() if self.mu_min is not None else None
        mu_max = self.mu_max.cpu().numpy() if self.mu_max is not None else None
        lv_min = self.logvar_min.cpu().numpy() if self.logvar_min is not None else None
        lv_max = self.logvar_max.cpu().numpy() if self.logvar_max is not None else None

        return mu_mean, mu_std, mu_min, mu_max, lv_mean, lv_std, lv_min, lv_max

    def save_csv(self, path: str):
        mu_mean, mu_std, mu_min, mu_max, lv_mean, lv_std, lv_min, lv_max = self.finalize()
        import csv
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['channel', 'mu_mean', 'mu_std', 'mu_min', 'mu_max', 'logvar_mean', 'logvar_std', 'logvar_min', 'logvar_max'])
            for i in range(len(mu_mean)):
                writer.writerow([i,
                                 float(mu_mean[i]), float(mu_std[i]), float(mu_min[i]) if mu_min is not None else '', float(mu_max[i]) if mu_max is not None else '',
                                 float(lv_mean[i]), float(lv_std[i]), float(lv_min[i]) if lv_min is not None else '', float(lv_max[i]) if lv_max is not None else ''])

        # Print a concise summary across channels
        agg = {
            'mu_mean_mean': float(np.mean(mu_mean)),
            'mu_min_over_channels': float(np.min(mu_min)) if mu_min is not None else None,
            'mu_max_over_channels': float(np.max(mu_max)) if mu_max is not None else None,
            'logvar_mean_mean': float(np.mean(lv_mean)),
            'logvar_min_over_channels': float(np.min(lv_min)) if lv_min is not None else None,
            'logvar_max_over_channels': float(np.max(lv_max)) if lv_max is not None else None,
        }
        print(f"Saved latent stats CSV to {path}")
        print("Latent summary:", agg)















# ==========================================
# 4. PLOTTING FUNCTIONS
# ==========================================
def all_area_prediction_plot_spatial_recovery_all_variables(vars_list, target_ds, masked_ds, pred_ds, save_dir=None):
    """
    Plots Case 1 across ALL variables simultaneously.
    Grid layout: N rows (one for each variable) x 4 columns (Ground Truth | Masked Input | Inpainted Prediction | Error)
    """
    source_file = target_ds.attrs.get("source_file", "unknown_cyclone.nc")
    
    base_name = Path(source_file).stem
    parts = base_name.split("_")
    track_id = parts[2] if (len(parts) >= 6 and parts[0] == "patch") else "unk"
    time_val = np.datetime_as_string(target_ds['time'].values, unit='h')
    
    latc = float(target_ds['cyclone_center_lat'].values)
    lonc = float(target_ds['cyclone_center_lon'].values)

    num_vars = len(vars_list)
    fig, axes = plt.subplots(num_vars, 4, figsize=(22, 4.5 * num_vars), squeeze=False)
    
    for row_idx, var in enumerate(vars_list):
        label = VAR_LABELS.get(var, var)
        
        true_da = target_ds[var]
        masked_da = masked_ds[var]
        pred_da = pred_ds[var]
        
        # Scale MSL to hPa for plotting readability
        if var == "msl":
            true_vals = true_da.values / 100.0
            masked_vals = masked_da.values / 100.0
            pred_vals = pred_da.values / 100.0
        else:
            true_vals = true_da.values
            masked_vals = masked_da.values
            pred_vals = pred_da.values
            
        diff_vals = pred_vals - true_vals

        clim_floor = 1e-6
        vmin_shared = min(np.nanpercentile(true_vals, 2), np.nanpercentile(pred_vals, 2))
        vmax_shared = max(np.nanpercentile(true_vals, 98), np.nanpercentile(pred_vals, 98))

        if vmin_shared < 0 and vmax_shared > 0:
            clim_abs = max(abs(vmin_shared), abs(vmax_shared))
            clim_abs = clim_abs if clim_abs > 0 else clim_floor
            vmin, vmax = -clim_abs, clim_abs
            current_cmap = "RdBu_r"
        else:
            vmin, vmax = vmin_shared, vmax_shared
            current_cmap = "viridis"

        clim_diff = np.nanpercentile(np.abs(diff_vals), 98)
        clim_diff = clim_diff if clim_diff > 0 else clim_floor

        # 1. Ground Truth
        ax_t = axes[row_idx, 0]
        sc_t = ax_t.pcolormesh(target_ds['absolute_lon'].values, target_ds['absolute_lat'].values, true_vals, cmap=current_cmap, vmin=vmin, vmax=vmax, shading='auto')
        plt.colorbar(sc_t, ax=ax_t, shrink=0.85, label=label)
        ax_t.plot(lonc, latc, "kX", ms=9, zorder=5)
        ax_t.set(ylabel="Latitude", title=f"Ground Truth ({var})" if row_idx == 0 else f"{var} - Ground Truth")

        # 2. Masked Input
        ax_m = axes[row_idx, 1]
        sc_m = ax_m.pcolormesh(target_ds['absolute_lon'].values, target_ds['absolute_lat'].values, masked_vals, cmap=current_cmap, vmin=vmin, vmax=vmax, shading='auto')
        plt.colorbar(sc_m, ax=ax_m, shrink=0.85, label=label)
        ax_m.plot(lonc, latc, "kX", ms=9, zorder=5)
        ax_m.set(title=f"Spatially Masked Input ({var})" if row_idx == 0 else f"{var} - Masked")

        # 3. Model Prediction
        ax_p = axes[row_idx, 2]
        sc_p = ax_p.pcolormesh(pred_ds['absolute_lon'].values, pred_ds['absolute_lat'].values, pred_vals, cmap=current_cmap, vmin=vmin, vmax=vmax, shading='auto')
        plt.colorbar(sc_p, ax=ax_p, shrink=0.85, label=label)
        ax_p.plot(lonc, latc, "kX", ms=9, zorder=5)
        ax_p.set(title=f"Inpainted Prediction ({var})" if row_idx == 0 else f"{var} - Inpainted")

        # 4. Difference Map
        ax_d = axes[row_idx, 3]
        sc_d = ax_d.pcolormesh(target_ds['absolute_lon'].values, target_ds['absolute_lat'].values, diff_vals, cmap="RdBu_r", vmin=-clim_diff, vmax=clim_diff, shading='auto')
        plt.colorbar(sc_d, ax=ax_d, shrink=0.85, label="Error")
        ax_d.plot(lonc, latc, "kX", ms=9, zorder=5)
        ax_d.set(title=f"Difference ({var})" if row_idx == 0 else f"{var} - Error")

        rmse = np.sqrt(np.mean(diff_vals ** 2))
        ax_d.text(0.02, 0.04, f"RMSE = {rmse:.4f}", transform=ax_d.transAxes, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="gray"))

        if row_idx == num_vars - 1:
            for ax in axes[row_idx, :]:
                ax.set_xlabel("Longitude")

    fig.suptitle(f"Case 1: Spatial Block Recovery Across All Variables | Track: {track_id} ({time_val})", y=1.01, fontsize=16)
    plt.tight_layout()

    if save_dir is not None:
        fname = Path(save_dir) / f"all_area_infer_case1_spatial_recovery_all_vars_track_{track_id}_{time_val}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"--> Saved Case 1 (All Variables) plot: {fname}")
        plt.close(fig)
    # else:
    #     plt.show()


def plot_variable_recovery(target_var_name, target_ds, pred_ds, save_dir=None, metadata: Optional["SampleMetadata"] = None):
    """
    Plots diagnostic panel for CASE 2: MISSING VARIABLE recovery test
    """
    source_file = target_ds.attrs.get("source_file", "unknown_cyclone.nc")
    base_name = Path(source_file).stem
    parts = base_name.split("_")
    track_id = parts[2] if (len(parts) >= 6 and parts[0] == "patch") else "unk"
    time_val = np.datetime_as_string(target_ds['time'].values, unit='h')
    label = VAR_LABELS.get(target_var_name, target_var_name)
    
    true_da = target_ds[target_var_name]
    pred_da = pred_ds[target_var_name]
    
    if target_var_name == "msl":
        true_vals = true_da.values / 100.0
        pred_vals = pred_da.values / 100.0
        label = "Mean sea level pressure, hPa"

    else:
        true_vals = true_da.values
        pred_vals = pred_da.values
        
    diff_vals = pred_vals - true_vals

    latc = float(target_ds['cyclone_center_lat'].values)
    lonc = float(target_ds['cyclone_center_lon'].values)

    fig, axes = plt.subplots(1, 3, figsize=(22, 5))
    clim_floor = 1e-6
    vmin_shared = min(np.nanpercentile(true_vals, 2), np.nanpercentile(pred_vals, 2))
    vmax_shared = max(np.nanpercentile(true_vals, 98), np.nanpercentile(pred_vals, 98))

    if vmin_shared < 0 and vmax_shared > 0:
        clim_abs = max(abs(vmin_shared), abs(vmax_shared))
        clim_abs = clim_abs if clim_abs > 0 else clim_floor
        vmin, vmax = -clim_abs, clim_abs
        current_cmap = "RdBu_r"
    else:
        vmin, vmax = vmin_shared, vmax_shared
        current_cmap = "viridis"

    clim_diff = np.nanpercentile(np.abs(diff_vals), 98)
    clim_diff = clim_diff if clim_diff > 0 else clim_floor

    # 1. Target
    ax_t = axes[0]
    sc_t = ax_t.pcolormesh(target_ds['absolute_lon'].values, target_ds['absolute_lat'].values, true_vals, cmap=current_cmap, vmin=vmin, vmax=vmax, shading='auto')
    # plt.colorbar(sc_t, ax=ax_t, shrink=0.85, label=label)
    plt.colorbar(sc_t, ax=ax_t, shrink=0.85, label=label.split(',')[1].strip() if ',' in label else '')

    ax_t.plot(lonc, latc, "kX", ms=10, zorder=5)
    # ax_t.set(xlabel="Longitude", ylabel="Latitude", title=f"True Target: {target_var_name}")
    ax_t.set(xlabel="Longitude", ylabel="Latitude", title="True Target") 

    # 2. Inferred Prediction
    ax_p = axes[1]
    sc_p = ax_p.pcolormesh(pred_ds['absolute_lon'].values, pred_ds['absolute_lat'].values, pred_vals, cmap=current_cmap, vmin=vmin, vmax=vmax, shading='auto')
    # plt.colorbar(sc_p, ax=ax_p, shrink=0.85, label=label)
    plt.colorbar(sc_p, ax=ax_p, shrink=0.85, label=label.split(',')[1].strip() if ',' in label else '')
    ax_p.plot(lonc, latc, "kX", ms=10, zorder=5)
    # ax_p.set(xlabel="Longitude", ylabel="Latitude", title=f"Inferred {target_var_name}")

    ax_p.set_title(
                    f"Case 2: Missing Variable Recovery [{target_var_name}] | Track: {track_id} ({time_val}) \n {label.split(',')[0].strip()} \n Inferred",
                    # f"Predicted: {vname}"
                )

    # 3. Difference
    ax_d = axes[2]
    sc_d = ax_d.pcolormesh(target_ds['absolute_lon'].values, target_ds['absolute_lat'].values, diff_vals, cmap="RdBu_r", vmin=-clim_diff, vmax=clim_diff, shading='auto')
    # plt.colorbar(sc_d, ax=ax_d, shrink=0.85, label="Error")
    diff_label = f"Δ {label.split(',')[1].strip()}" if ',' in label else "Δ"
    plt.colorbar(sc_d, ax=ax_d, shrink=0.85, label=diff_label)
    ax_d.plot(lonc, latc, "kX", ms=10, zorder=5)
    ax_d.set(xlabel="Longitude", ylabel="Latitude", title="Difference (Pred - Target)")

    rmse = np.sqrt(np.mean(diff_vals ** 2))
    ax_d.text(0.02, 0.04, f"RMSE = {rmse:.4f}", transform=ax_d.transAxes, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="gray"))

    
    if save_dir is not None:
        fname = build_comparison_filename(metadata, f"case2_{target_var_name}", save_dir)

        # fname = Path(save_dir) / f"case2_missing_var_{target_var_name}_track_{track_id}_{time_val}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"--> Saved Case 2 plot: {fname}")
        plt.close(fig)
    # else:
    #     plt.show()


# ==========================================
# 5. EVALUATION PIPELINES
# ==========================================
def all_area_prediction_evaluate_spatial_masking_pipeline(model, dataloader, vars_list, dataset, device, save_dir=None):
    """
    CASE 1 PIPELINE: Zeroes out a central spatial patch across ALL channels 
    and generates full multi-variable panel plots.
    """
    model.eval()
    stats = dataset.stats

    print(f"\n---> [Case 1] Evaluating Spatial Masking Pipeline across ALL variables")
    
    pred_ds_list = []
    masked_ds_list = []
    target_ds_list = []

    with torch.no_grad():
        for inputs, targets, indices in dataloader:
            inputs = inputs.to(device)
            B, C, H, W = inputs.shape
            
            # Mask central region across ALL channel variables
            masked_inputs = inputs.clone()
            mask_h, mask_w = int(H * 0.35), int(W * 0.35)
            top, left = (H - mask_h) // 2, (W - mask_w) // 2
            masked_inputs[:, :, top:top+mask_h, left:left+mask_w] = 0.0

            x_hat, _, _ = model(masked_inputs)
            
            outputs_np = x_hat.cpu().numpy()
            masked_np = masked_inputs.cpu().numpy()
            targets_np = targets.numpy()
            indices_np = indices.numpy()

            for b in range(outputs_np.shape[0]):
                idx = int(indices_np[b])
                source_filename = dataset.time_to_filename[idx]
                orig_slice = dataset.ds.isel(time=idx)

                pred_dict, masked_dict, target_dict = {}, {}, {}
                
                for i, var in enumerate(vars_list):
                    mean = stats[var]['mean']
                    std = stats[var]['std']
                    
                    var_pred_unnorm = (outputs_np[b, i] * std) + mean
                    var_mask_unnorm = (masked_np[b, i] * std) + mean
                    var_true_unnorm = (targets_np[b, i] * std) + mean
                    
                    pred_dict[var] = (['lat', 'lon'], var_pred_unnorm)
                    masked_dict[var] = (['lat', 'lon'], var_mask_unnorm)
                    target_dict[var] = (['lat', 'lon'], var_true_unnorm)
                
                coords_to_keep = {c: orig_slice[c] for c in orig_slice.coords}
                pred_ds = xr.Dataset(data_vars=pred_dict, coords=coords_to_keep)
                masked_ds = xr.Dataset(data_vars=masked_dict, coords=coords_to_keep)
                target_ds = xr.Dataset(data_vars=target_dict, coords=coords_to_keep)
                
                pred_ds.attrs["source_file"] = source_filename
                target_ds.attrs["source_file"] = source_filename
                
                pred_ds_list.append(pred_ds)
                masked_ds_list.append(masked_ds)
                target_ds_list.append(target_ds)

    # Plot multi-variable panel for the first 3-5 test samples
    num_plots = min(5, len(pred_ds_list))
    for k in range(num_plots):
        all_area_prediction_plot_spatial_recovery_all_variables(vars_list, target_ds_list[k], masked_ds_list[k], pred_ds_list[k], save_dir=save_dir, metadata=metadata)


def evaluate_spatial_masking_pipeline(model, dataloader, vars_list, dataset, device, save_dir=None):
    """
    CASE 1 PIPELINE: Zeroes out a central spatial patch across ALL channels,
    creates composite predictions for plotting, and computes Patch-Only metrics (R2, RMSE, MAE).
    """
    model.eval()
    stats = dataset.stats

    print(f"\n---> [Case 1] Evaluating Spatial Masking Pipeline across ALL variables")
    
    pred_ds_list = []
    masked_ds_list = []
    target_ds_list = []

    # Dictionary to store metrics ONLY inside the masked spatial patch
    patch_metrics = {var: {'mae': [], 'rmse': [], 'r2': []} for var in vars_list}

    with torch.no_grad():
        for inputs, targets, indices in dataloader:
            inputs = inputs.to(device)
            B, C, H, W = inputs.shape
            
            # 1. Mask central region across ALL channel variables
            masked_inputs = inputs.clone()
            mask_h, mask_w = int(H * 0.35), int(W * 0.35)
            top, left = (H - mask_h) // 2, (W - mask_w) // 2
            
            # Apply zero mask
            masked_inputs[:, :, top:top+mask_h, left:left+mask_w] = 0.0

            # 2. Forward pass
            x_hat, _, _ = model(masked_inputs)
            
            outputs_np = x_hat.cpu().numpy()
            masked_np = masked_inputs.cpu().numpy()
            targets_np = targets.numpy()
            indices_np = indices.numpy()

            for b in range(outputs_np.shape[0]):
                idx = int(indices_np[b])
                source_filename = dataset.time_to_filename[idx]
                orig_slice = dataset.ds.isel(time=idx)

                pred_dict, masked_dict, target_dict = {}, {}, {}
                
                for i, var in enumerate(vars_list):
                    mean = stats[var]['mean']
                    std = stats[var]['std']
                    
                    var_pred_unnorm = (outputs_np[b, i] * std) + mean
                    var_mask_unnorm = (masked_np[b, i] * std) + mean
                    var_true_unnorm = (targets_np[b, i] * std) + mean
                    
                    # --- COMPUTE METRICS STRICTLY INSIDE THE PATCH ---
                    y_true_patch = var_true_unnorm[top:top+mask_h, left:left+mask_w]
                    y_pred_patch = var_pred_unnorm[top:top+mask_h, left:left+mask_w]
                    
                    # MAE and RMSE inside the patch
                    patch_mae = np.mean(np.abs(y_true_patch - y_pred_patch))
                    patch_rmse = np.sqrt(np.mean((y_true_patch - y_pred_patch) ** 2))
                    
                    # R² inside the patch
                    ss_res = np.sum((y_true_patch - y_pred_patch) ** 2)
                    ss_tot = np.sum((y_true_patch - np.mean(y_true_patch)) ** 2)
                    patch_r2 = 1.0 - (ss_res / (ss_tot + 1e-8)) # epsilon to prevent zero division
                    
                    patch_metrics[var]['mae'].append(patch_mae)
                    patch_metrics[var]['rmse'].append(patch_rmse)
                    patch_metrics[var]['r2'].append(patch_r2)
                    
                    # --- COMPOSITE PREDICTION FOR PLOTTING ---
                    var_composite = var_true_unnorm.copy()
                    var_composite[top:top+mask_h, left:left+mask_w] = var_pred_unnorm[top:top+mask_h, left:left+mask_w]
                    
                    pred_dict[var] = (['lat', 'lon'], var_composite)
                    masked_dict[var] = (['lat', 'lon'], var_mask_unnorm)
                    target_dict[var] = (['lat', 'lon'], var_true_unnorm)
                
                coords_to_keep = {c: orig_slice[c] for c in orig_slice.coords}
                pred_ds = xr.Dataset(data_vars=pred_dict, coords=coords_to_keep)
                masked_ds = xr.Dataset(data_vars=masked_dict, coords=coords_to_keep)
                target_ds = xr.Dataset(data_vars=target_dict, coords=coords_to_keep)
                
                pred_ds.attrs["source_file"] = source_filename
                target_ds.attrs["source_file"] = source_filename
                
                pred_ds_list.append(pred_ds)
                masked_ds_list.append(masked_ds)
                target_ds_list.append(target_ds)

    # --- PRINT SUMMARY OF PATCH-ONLY METRICS ---
    print("\n" + "="*50)
    print("      PATCH-ONLY INPAINTING METRICS (EVALUATION)")
    print("="*50)
    for var in vars_list:
        mean_r2 = np.mean(patch_metrics[var]['r2'])
        mean_rmse = np.mean(patch_metrics[var]['rmse'])
        mean_mae = np.mean(patch_metrics[var]['mae'])
        print(f"[{var.upper():<6}] Patch R²: {mean_r2:.4f} | Patch RMSE: {mean_rmse:.4f} | Patch MAE: {mean_mae:.4f}")
    print("="*50 + "\n")

    # Plot multi-variable panel for the first 3-5 test samples
    # num_plots = min(5, len(pred_ds_list))
    num_plots = len(pred_ds_list)
    for k in range(num_plots):

        metadata = dataset.get_metadata(k) # n files (with 3 timesteps each) in the dataset, so k is the index of the sample

        plot_spatial_recovery_all_variables(vars_list, target_ds_list[k], masked_ds_list[k], pred_ds_list[k], save_dir=save_dir, metadata=metadata)


def expand_to_group(channel_idx, vars_list, groups=VAR_GROUPS):
    """channel_idx (int or list) -> all channels of the same unit (10u -> [10u, 10v])."""
    idxs = [channel_idx] if isinstance(channel_idx, int) else list(channel_idx)
    names = {vars_list[i] for i in idxs}
    out = set()
    for g in groups:
        if names & set(g):
            out |= {vars_list.index(n) for n in g}
    return sorted(out)


def _predict_with_mask(model, inputs, mask):
    """inputs (T,C,H,W) clean, mask (T,C,H,W) bool True = missing."""
    masked_inputs = inputs.masked_fill(mask, 0.0)
    model_in = build_model_input(masked_inputs, mask)      # (T, 2C, H, W)
    pred_hat, _, _ = model(model_in, sample=False)         # z = mu, deterministic
    return pred_hat


def inferencia_by_sample_case1(model, clean_samples_cthw, pixel_box, device, save_dir=None):
    """Spatial block missing in ALL variables. clean_samples_cthw: [1, C, T, H, W]"""
    model.eval()
    print("\n---> [Case 1] Evaluating Spatial Masking Pipeline across ALL variables")

    with torch.no_grad():
        inputs = clean_samples_cthw.squeeze(0).permute(1, 0, 2, 3).to(device)  # T,C,H,W

        mask = torch.zeros_like(inputs, dtype=torch.bool)
        mask[:, :, pixel_box[0]:pixel_box[1], pixel_box[2]:pixel_box[3]] = True

        pred_hat = _predict_with_mask(model, inputs, mask)

        # observed pixels from the real data, missing pixels from the model
        pred_box = torch.where(mask, pred_hat, inputs)

    return pred_hat, pred_box


def inferencia_by_sample_case2(model, clean_samples_cthw, channel_idx, device,
                               vars_list=VARS_LIST, expand_group=True, save_dir=None):
    """Variable(s) missing everywhere. clean_samples_cthw: [1, C, T, H, W]"""
    model.eval()
    print("\n---> [Case 2] missing variable across different timesteps")

    with torch.no_grad():
        inputs = clean_samples_cthw.squeeze(0).permute(1, 0, 2, 3).to(device)  # T,C,H,W

        chans = expand_to_group(channel_idx, vars_list) if expand_group else \
                ([channel_idx] if isinstance(channel_idx, int) else list(channel_idx))
        print("masked channels:", [vars_list[c] for c in chans])

        mask = torch.zeros_like(inputs, dtype=torch.bool)
        mask[:, chans, :, :] = True

        pred_hat = _predict_with_mask(model, inputs, mask)

    return pred_hat  # T, C, H, W


# def inferencia_by_sample_case1(model, clean_samples_cthw, pixel_box, device, save_dir=None):
#     """
#     it receive 3 timesteps tensor 
#     clean_samples_cthw [1, C, T, H, W]
#     """
#     model.eval()

#     print(f"\n---> [Case 1] Evaluating Spatial Masking Pipeline across ALL variables")

#     with torch.no_grad():
#         clean_samples_tchw = clean_samples_cthw.squeeze(0).permute(1, 0, 2, 3) #T, C, H, W  here T is like a batch


#         inputs = clean_samples_tchw

#         inputs = inputs.to(device)
#         # B, C, H, W = inputs.shape # here batch_size is the 3 timesteps
        
#         # Apply zero mask
#         pred_box = inputs.clone()
#         masked_inputs = inputs.clone()
#         print(f"masked_inputs shape: {masked_inputs.shape}") 
#         masked_inputs[:, :, pixel_box[0]:pixel_box[1], pixel_box[2]:pixel_box[3]] = 0.0

#         # 2. Forward pass
#         pred_hat, _, _ = model(masked_inputs)
        

#         pred_box[:, :, pixel_box[0]:pixel_box[1], pixel_box[2]:pixel_box[3]] = pred_hat[:, :, pixel_box[0]:pixel_box[1], pixel_box[2]:pixel_box[3]]

#     return pred_hat, pred_box #inputs #T, C, H, W 


# def inferencia_by_sample_case2(model, clean_samples_cthw, channel_idx, device, save_dir=None):
#     """
#     it receive 3 timesteps tensor 
#     clean_samples_cthw [1, C, T, H, W]
#     """
#     model.eval()

#     print(f"\n---> [Case 2] missing variable accross different timesteps")

#     with torch.no_grad():
#         clean_samples_tchw = clean_samples_cthw.squeeze(0).permute(1, 0, 2, 3) #T, C, H, W  here T is like a batch


#         inputs = clean_samples_tchw

#         inputs = inputs.to(device)
#         # B, C, H, W = inputs.shape # here batch_size is the 3 timesteps
        
#         # Apply zero mask
#         masked_inputs = inputs.clone()
#         print(f"masked_inputs shape: {masked_inputs.shape}") 
#         masked_inputs[:, channel_idx, :, :] = 0.0

#         # 2. Forward pass
#         pred_hat, _, _ = model(masked_inputs)
        

#     return pred_hat  #T, C, H, W 



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
    cyclone_center_lat: Optional[float]
    cyclone_center_lon: Optional[float]
    time_str: Optional[str]             # e.g. "2020-09-14T06"


def build_comparison_filename(metadata: Optional["SampleMetadata"], recon_type: str, images_dir: str) -> str:
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
    time_val = (metadata.time_str if metadata else None) or "unknown-time"
    window = (metadata.win_idx if metadata else None) or "unknown-window"
    name_file = f"{recon_type}_VAE_track{track_id}_{season}_{basin}_rank{rank}_window{window}_{time_val}.png"
    return os.path.join(images_dir, name_file)


def plot_spatial_recovery_all_variables(vars_list, target_ds, masked_ds, pred_ds, save_dir=None, metadata: Optional["SampleMetadata"] = None):
    """
    Plots Case 1 across ALL variables simultaneously.
    Metrics (R2, RMSE) displayed on the plot are calculated strictly inside the masked patch.
    unnormalize input?
    """
    source_file = target_ds.attrs.get("source_file", "unknown_cyclone.nc")
    base_name = Path(source_file).stem
    parts = base_name.split("_")
    track_id = parts[2] if (len(parts) >= 6 and parts[0] == "patch") else "unk"
    time_val = np.datetime_as_string(target_ds['time'].values, unit='h')
    
    latc = float(target_ds['cyclone_center_lat'].values)
    lonc = float(target_ds['cyclone_center_lon'].values)

    num_vars = len(vars_list)
    fig, axes = plt.subplots(num_vars, 4, figsize=(22, 4.5 * num_vars), squeeze=False)
    
    # --- GET MASKED PATCH SLICE BOUNDS (Central 35% Box) ---
    H, W = target_ds[vars_list[0]].shape
    mask_h, mask_w = int(H * 0.35), int(W * 0.35)
    top, left = (H - mask_h) // 2, (W - mask_w) // 2
    bottom, right = top + mask_h, left + mask_w

    for row_idx, var in enumerate(vars_list):
        label = VAR_LABELS.get(var, var)
        
        true_da = target_ds[var]
        masked_da = masked_ds[var]
        pred_da = pred_ds[var]
        
        # Scale MSL to hPa for plotting readability
        if var == "msl":
            true_vals = true_da.values / 100.0
            masked_vals = masked_da.values / 100.0
            pred_vals = pred_da.values / 100.0
        else:
            true_vals = true_da.values
            masked_vals = masked_da.values
            pred_vals = pred_da.values
            
        diff_vals = pred_vals - true_vals

        # =========================================================
        # --- CALCULATE METRICS STRICTLY INSIDE THE PATCH ONLY ---
        # =========================================================
        patch_true = true_vals[top:bottom, left:right]
        patch_pred = pred_vals[top:bottom, left:right]
        patch_diff = diff_vals[top:bottom, left:right]

        # Patch RMSE
        patch_rmse = np.sqrt(np.mean(patch_diff ** 2))

        # Patch R²
        ss_res = np.sum(patch_diff ** 2)
        ss_tot = np.sum((patch_true - np.mean(patch_true)) ** 2)
        patch_r2 = 1.0 - (ss_res / (ss_tot + 1e-8))
        # =========================================================

        clim_floor = 1e-6
        vmin_shared = min(np.nanpercentile(true_vals, 2), np.nanpercentile(pred_vals, 2))
        vmax_shared = max(np.nanpercentile(true_vals, 98), np.nanpercentile(pred_vals, 98))

        if vmin_shared < 0 and vmax_shared > 0:
            clim_abs = max(abs(vmin_shared), abs(vmax_shared))
            clim_abs = clim_abs if clim_abs > 0 else clim_floor
            vmin, vmax = -clim_abs, clim_abs
            current_cmap = "RdBu_r"
        else:
            vmin, vmax = vmin_shared, vmax_shared
            current_cmap = "viridis"

        clim_diff = np.nanpercentile(np.abs(diff_vals), 98)
        clim_diff = clim_diff if clim_diff > 0 else clim_floor

        # 1. Ground Truth
        ax_t = axes[row_idx, 0]
        sc_t = ax_t.pcolormesh(target_ds['absolute_lon'].values, target_ds['absolute_lat'].values, true_vals, cmap=current_cmap, vmin=vmin, vmax=vmax, shading='auto')
        plt.colorbar(sc_t, ax=ax_t, shrink=0.85, label=label)
        ax_t.plot(lonc, latc, "kX", ms=9, zorder=5)
        ax_t.set(ylabel="Latitude", title=f"Ground Truth ({var})" if row_idx == 0 else f"{var} - Ground Truth")

        # 2. Masked Input
        ax_m = axes[row_idx, 1]
        sc_m = ax_m.pcolormesh(target_ds['absolute_lon'].values, target_ds['absolute_lat'].values, masked_vals, cmap=current_cmap, vmin=vmin, vmax=vmax, shading='auto')
        plt.colorbar(sc_m, ax=ax_m, shrink=0.85, label=label)
        ax_m.plot(lonc, latc, "kX", ms=9, zorder=5)
        ax_m.set(title=f"Spatially Masked Input ({var})" if row_idx == 0 else f"{var} - Masked")

        # 3. Model Prediction (Composite)
        ax_p = axes[row_idx, 2]
        sc_p = ax_p.pcolormesh(pred_ds['absolute_lon'].values, pred_ds['absolute_lat'].values, pred_vals, cmap=current_cmap, vmin=vmin, vmax=vmax, shading='auto')
        plt.colorbar(sc_p, ax=ax_p, shrink=0.85, label=label)
        ax_p.plot(lonc, latc, "kX", ms=9, zorder=5)
        ax_p.set(title=f"Inpainted Prediction ({var})" if row_idx == 0 else f"{var} - Inpainted")

        # 4. Difference Map
        ax_d = axes[row_idx, 3]
        sc_d = ax_d.pcolormesh(target_ds['absolute_lon'].values, target_ds['absolute_lat'].values, diff_vals, cmap="RdBu_r", vmin=-clim_diff, vmax=clim_diff, shading='auto')
        plt.colorbar(sc_d, ax=ax_d, shrink=0.85, label="Error")
        ax_d.plot(lonc, latc, "kX", ms=9, zorder=5)
        ax_d.set(title=f"Difference ({var})" if row_idx == 0 else f"{var} - Error")

        # --- DISPLAY PATCH-ONLY METRICS ON THE PLOT ---
        ax_d.text(
            0.02, 0.04, 
            f"Patch RMSE = {patch_rmse:.4f}\nPatch R² = {patch_r2:.4f}", 
            transform=ax_d.transAxes, 
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="gray")
        )

        if row_idx == num_vars - 1:
            for ax in axes[row_idx, :]:
                ax.set_xlabel("Longitude")

    fig.suptitle(f"Case 1: Spatial Block Recovery Across All Variables | Track: {track_id} ({time_val})", y=1.01, fontsize=16)
    plt.tight_layout()

    if save_dir is not None:
        fname = build_comparison_filename(metadata, "case1", save_dir)

        # fname = Path(save_dir) / f"case1_spatial_recovery_all_vars_track_{track_id}_{time_val}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"--> Saved Case 1 (All Variables) plot: {fname}")
        plt.close(fig)


def evaluate_missing_variable_pipeline(model, dataloader, target_var_name, vars_list, dataset, device, save_dir=None):
    """
    CASE 2 PIPELINE: Zeroes out target_var_name completely across spatial dimensions.
    """
    model.eval()
    target_idx = vars_list.index(target_var_name)
    stats = dataset.stats

    print(f"\n---> [Case 2] Evaluating Missing Variable Pipeline for: [{target_var_name}]", flush=True)
    
    pred_ds_list = []
    target_ds_list = []

    batch_count = 0
    with torch.no_grad():
        for inputs, targets, indices in dataloader:
            batch_count += 1
            if batch_count % 10 == 0:
                print(f"--> Case2 {target_var_name}: processed {batch_count} batches", flush=True)
            inputs = inputs.to(device)
            
            masked_inputs = inputs.clone()
            masked_inputs[:, target_idx, :, :] = 0.0
            
            x_hat, _, _ = model(masked_inputs)
            
            outputs_np = x_hat.cpu().numpy()
            targets_np = targets.numpy()
            indices_np = indices.numpy()

            for b in range(outputs_np.shape[0]):
                idx = int(indices_np[b])
                source_filename = dataset.time_to_filename[idx]
                orig_slice = dataset.ds.isel(time=idx)

                pred_dict, target_dict = {}, {}
                
                for i, var in enumerate(vars_list):
                    mean = stats[var]['mean']
                    std = stats[var]['std']
                    
                    var_pred_unnorm = (outputs_np[b, i] * std) + mean
                    var_true_unnorm = (targets_np[b, i] * std) + mean
                    
                    pred_dict[var] = (['lat', 'lon'], var_pred_unnorm)
                    target_dict[var] = (['lat', 'lon'], var_true_unnorm)
                
                coords_to_keep = {c: orig_slice[c] for c in orig_slice.coords}
                # pred_ds = xr.Dataset(data_vars=pred_dict, coords=coords_to_keep)
                # target_ds = xr.Dataset(data_vars=target_dict, coords=coords_to_keep)

                pred_ds = xr.Dataset(
                                    data_vars=pred_dict,
                                    coords=coords_to_keep,
                                    attrs=orig_slice.attrs
                                )

                target_ds = xr.Dataset(
                    data_vars=target_dict,
                    coords=coords_to_keep,
                    attrs=orig_slice.attrs
                )
                pred_ds.attrs["source_file"] = source_filename
                target_ds.attrs["source_file"] = source_filename
                
                pred_ds_list.append(pred_ds)
                target_ds_list.append(target_ds)

    print(f"--> Case2 {target_var_name}: finished dataloader loop; collected {len(pred_ds_list)} samples", flush=True)
    num_plots = min(5, len(pred_ds_list))
    print(f"ploting only first 5/{len(pred_ds_list)} samples for Case2", flush=True)
    for k in range(num_plots):
        metadata = dataset.get_metadata(k)
        # metadata =  parse_patch_filename(dataset[k])
        # print(target_ds_list[k])

        plot_variable_recovery(target_var_name, target_ds_list[k], pred_ds_list[k], save_dir=save_dir, metadata=metadata)
        print(f"--> Case2 {target_var_name}: plotted {k+1}/{num_plots}", flush=True)


# ==========================================
# 6. STANDALONE INFERENCE & TESTING
# ==========================================
def load_model_and_test(checkpoint_path=CHECKPOINT, test_dir=TEST_DIR, save_dir=SAVE_DIR):
    """
    Loads saved checkpoint and evaluates both Case 1 (All Variables Spatial Masking) 
    and Case 2 (Missing Variable Recovery).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==========================================")
    print(f"LOADING CHECKPOINT & TESTING MODEL")
    print(f"==========================================")
    
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    vars_list = checkpoint["vars_list"]

    checkpoint = torch.load(checkpoint_path, map_location=device)
    C = len(VARS_LIST)

    # channel order must match training, otherwise the variables get swapped silently
    assert list(checkpoint["vars_list"]) == list(VARS_LIST)

    saved_stats = checkpoint["stats"]   # mean/std used for normalisation


    test_dataset = CyclonePatchDataset(test_dir, variables=vars_list, stats_dict=saved_stats)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # model = CycloneVAE(in_channels=len(vars_list), latent_channels=128).to(device)
    # model.load_state_dict(checkpoint["model_state"])

    vae_model = CycloneVAE(
        in_channels=checkpoint.get("in_channels", 2 * C),
        out_channels=checkpoint.get("out_channels", C),
        latent_channels=128,
    ).to(device)
    vae_model.load_state_dict(checkpoint["model_state"])
    vae_model.eval()


    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 1. Run Case 1: Spatial Masking Plots (Generates row-per-variable layout)
    evaluate_spatial_masking_pipeline(vae_model, test_loader, vars_list, test_dataset, device, save_dir=save_dir)

    # all_area_prediction_evaluate_spatial_masking_pipeline(model, test_loader, vars_list, test_dataset, device, save_dir=save_dir)
    # Cleanup between evaluation runs to avoid worker/plotting/resource conflicts
    print("--> Finished Case 1. Cleaning up before Case 2 evaluations...")
    plt.close('all')

    # 2a. Run Case 2: Missing Variable Plots for 'msl'
    print("--> Starting Case 2 for '10u'")
    evaluate_missing_variable_pipeline(vae_model, test_loader, "10u", vars_list, test_dataset, device, save_dir=save_dir)

    # 2a. Run Case 2: Missing Variable Plots for 'msl'
    print("--> Starting Case 2 for 'msl'")
    evaluate_missing_variable_pipeline(vae_model, test_loader, "msl", vars_list, test_dataset, device, save_dir=save_dir)

    # 2b. Run Case 2: Missing Variable Plots for 'avg_tos'
    print("--> Starting Case 2 for 'avg_tos'")
    evaluate_missing_variable_pipeline(vae_model, test_loader, "avg_tos", vars_list, test_dataset, device, save_dir=save_dir)


if __name__ == "__main__":
    # Option A: Train and execute evaluations
    run_training()
    
    # Option B: Run standalone inference on trained checkpoint directly:
    load_model_and_test(checkpoint_path=CHECKPOINT, test_dir=TEST_DIR, save_dir=SAVE_DIR)

