
import gc
import glob
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
# 0. CONFIGURATION & HYPERPARAMETERS
# ==========================================
VARS_LIST = ["10u", "10v", "msl", "avg_tos", "avg_zos", "avg_sos", "avg_mlotst125"]

DATA_ROOT = Path("/scale/project/ifremer-isi-jumeaunumerique/NEXTGEMS_PROCESSED_LAT_LON")
TRAIN_DIR = DATA_ROOT / "data_128x128_3days/train_patches"
TEST_DIR = DATA_ROOT / "data_128x128_3days/test_patches"
CHECKPOINT_DIR = DATA_ROOT / "ml_result/vae_diffusion"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

TYPE = "vae_diffusion_no_masked"
SAVE_DIR = CHECKPOINT_DIR / "plots"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# VAE Configuration
VAE_CHECKPOINT = CHECKPOINT_DIR / f"checkpoint_model_{TYPE}.pt"
VAE_EPOCHS = 100
VAE_BATCH_SIZE = 32
LATENT_CHANNELS = 128  # Matched with evaluation script expectation

# Diffusion Configuration
DIFFUSION_TYPE = f"unet_{TYPE}"
DIFFUSION_CHECKPOINT = CHECKPOINT_DIR / f"checkpoint_{DIFFUSION_TYPE}.pt"
DIFFUSION_EPOCHS = 100
DIFFUSION_BATCH_SIZE = 64
DIFFUSION_STEPS = 1000
DIFFUSION_LR = 2e-4

# Latent Caching Directories
LATENTS_DIR = CHECKPOINT_DIR / "latents" / TYPE
LATENTS_DIR_TEST = CHECKPOINT_DIR / "latents" / f"test_{TYPE}" 
LATENTS_DIR.mkdir(parents=True, exist_ok=True)
LATENTS_DIR_TEST.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VAR_LABELS = {
    "10u": "Zonal wind 10u, m/s",
    "10v": "Meridional wind 10v, m/s",
    "msl": "Mean sea level pressure, Pa",
    "avg_tos": "Sea surface temperature, °C",
    "avg_zos": "Sea surface height, m",
    "avg_sos": "Sea surface salinity, psu",
    "avg_mlotst125": "Mixed layer depth, m",
}


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


# ==========================================
# 1. DATASET & NORMALIZATION UTILITIES
# ==========================================
def denormalize(data: np.ndarray, var: str, stats: dict) -> np.ndarray:
    mean = stats[var]['mean']
    std = stats[var]['std']
    return data * (std if std > 1e-6 else 1.0) + mean


def load_raw_test_window(track_id, season, basin, rank, time_str, data_dir, vars_list):
    """Loads matching raw test NetCDF window for precise evaluation."""
    pattern = f"patch_trackid_{track_id}_{season}_{basin}_rank_{rank}_*.nc"
    files = glob.glob(os.path.join(data_dir, pattern))
    if not files:
        # Fallback search
        files = glob.glob(os.path.join(data_dir, "*.nc"))
    
    for fpath in files:
        with xr.open_dataset(fpath) as ds:
            file_data = np.stack([ds[var].values for var in vars_list], axis=1).astype(np.float32)
            # Find matching time index if possible
            if "time" in ds.coords or "time" in ds.variables:
                times = [np.datetime_as_string(t, unit="h") for t in ds["time"].values]
                if time_str in times:
                    idx = times.index(time_str)
                    # Return the 3-step window containing this timestep
                    return torch.from_numpy(file_data), idx
            ds.close()
    
    # Default fallback to first file if exact match fails
    with xr.open_dataset(files[0]) as ds:
        file_data = np.stack([ds[var].values for var in vars_list], axis=1).astype(np.float32)
    return torch.from_numpy(file_data), 1


class CyclonePatchDataset(Dataset):
    def __init__(self, folder_path, variables, stats_dict=None, num_workers=16):
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


# ==========================================
# 2. VAE MODEL & LOSS
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(ch)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + x)


class CycloneVAE(nn.Module):
    def __init__(self, in_channels: int = len(VARS_LIST), latent_channels: int = LATENT_CHANNELS):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            ResidualBlock(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            ResidualBlock(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            ResidualBlock(128)
        )
        self.to_mu = nn.Conv2d(128, latent_channels, kernel_size=1)
        self.to_logvar = nn.Conv2d(128, latent_channels, kernel_size=1)

        self.from_latent = nn.Conv2d(latent_channels, 128, kernel_size=1)
        self.dec = nn.Sequential(
            ResidualBlock(128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            ResidualBlock(64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            ResidualBlock(32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, in_channels, kernel_size=3, padding=1, padding_mode='replicate')
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        return self.to_mu(h), self.to_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(self.from_latent(z))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


class VAELoss(nn.Module):
    def __init__(self, beta: float = 1e-4):
        super().__init__()
        self.beta = beta
        self.mse = nn.MSELoss()

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
        recon = self.mse(x_hat, x)
        kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=(1, 2, 3))
        kl = torch.mean(kl_per_sample)
        return recon + self.beta * kl, recon.detach(), kl.detach()


def train_vae(model: nn.Module, dataloader: DataLoader, epochs: int) -> nn.Module:
    print("\n[VAE Training] Starting...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = VAELoss(beta=1e-4)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for window, _, _ in dataloader:
            B, T, C, H, W = window.shape
            x = window.view(B * T, C, H, W).to(DEVICE)

            optimizer.zero_grad()
            x_hat, mu, logvar = model(x)
            loss, recon, kl = criterion(x_hat, x, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * (B * T)

        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Loss: {epoch_loss / len(dataloader.dataset) / 3:.5f}")

    torch.save({"model_state": model.state_dict(), "stats": dataloader.dataset.stats}, VAE_CHECKPOINT)
    print(f"[VAE Training] Saved checkpoint to {VAE_CHECKPOINT}")
    return model


# ==========================================
# 3. OFFLINE LATENT EXTRACTION & CACHING
# ==========================================
@torch.no_grad()
def cache_latents_by_track(model, dataset, device, out_dir=LATENTS_DIR, batch_size=32):
    model.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(f"[Caching] Encoding window dataset into latent representations...")
    count = 0
    for windows, _, indices in loader:
        B, T_win, C, H, W = windows.shape
        x_flat = windows.view(B * T_win, C, H, W).to(device)

        mu, _ = model.encode(x_flat)
        latent_dim_c, latent_h, latent_w = mu.shape[1], mu.shape[2], mu.shape[3]
        z_windows = mu.view(B, T_win, latent_dim_c, latent_h, latent_w).cpu()

        for b in range(B):
            idx = indices[b].item()
            metadata = dataset.get_metadata(idx)

            payload = {
                "source_file": metadata.source_file,
                "z": z_windows[b].clone(),
                "track_id": metadata.track_id,
                "season": metadata.season,
                "basin": metadata.basin,
                "rank": metadata.rank,
                "win_idx": metadata.win_idx,
                "lat": metadata.lat,
                "lon": metadata.lon,
                "time_strs": metadata.time_strs,
                "cyclone_center_lats": metadata.cyclone_center_lats,
                "cyclone_center_lons": metadata.cyclone_center_lons,
            }
            
            track_str = metadata.track_id if metadata.track_id else "NA"
            season_str = metadata.season if metadata.season else "NA"
            basin_str = metadata.basin if metadata.basin else "NA"
            rank_str = metadata.rank if metadata.rank else "NA"
            win_str = metadata.win_idx if metadata.win_idx else "0"

            save_filename = f"latent_track{track_str}_{season_str}_{basin_str}_rank{rank_str}_win{win_str}.pt"
            torch.save(payload, out_dir / save_filename)
            count += 1

    print(f"[Caching] Successfully saved {count} latent window payloads to {out_dir}")


class CachedLatentDataset(Dataset):
    def __init__(self, latent_dir: Path):
        self.files = sorted(glob.glob(os.path.join(latent_dir, "*.pt")))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # print(self.files[idx])
        payload = torch.load(self.files[idx], weights_only=False)
        latent_seq = payload["z"]  # Shape: (3, C_latent, H_lat, W_lat)
        cond = torch.cat([latent_seq[0], latent_seq[2]], dim=0)  # [prev, next] -> (2*C, H, W)
        target = latent_seq[1]  # target frame -> (C, H, W)
        return cond, target


# ==========================================
# 4. CONDITIONAL DIFFUSION MODEL & SCHEDULE
# ==========================================
class DiffusionSchedule:
    def __init__(self, T=DIFFUSION_STEPS, device=DEVICE):
        self.T = T
        self.device = device
        self.beta = torch.linspace(1e-4, 0.02, T, device=device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)


class LatentUNetDenoiser(nn.Module):
    def __init__(self, in_channels: int = LATENT_CHANNELS, cond_channels: int = LATENT_CHANNELS * 2):
        super().__init__()
        self.time_embed = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64))
        self.inc = nn.Conv2d(in_channels + cond_channels + 64, 64, kernel_size=3, padding=1)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1), nn.ReLU())
        self.down2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(128, 256, 3, padding=1), nn.ReLU())
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up1 = nn.Conv2d(256 + 128, 128, 3, padding=1)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up2 = nn.Conv2d(128 + 64, 64, 3, padding=1)
        
        self.outc = nn.Conv2d(64, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        t_emb = self.time_embed(t.unsqueeze(-1).float()).view(B, 64, 1, 1).expand(-1, -1, H, W)
        h_in = torch.cat([x, cond, t_emb], dim=1)
        h1 = self.inc(h_in)
        h2 = self.down1(h1)
        h3 = self.down2(h2)

        u1 = self.up1(h3)
        u1 = torch.cat([u1, h2], dim=1)
        u1 = self.conv_up1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, h1], dim=1)
        u2 = self.conv_up2(u2)
        return self.outc(u2)


def impute_missing_timestep(denoiser, schedule, context, device, scale_factor=1.0, ddim=True, ddim_steps=200):
    denoiser.eval()
    B, _, H, W = context.shape
    shape = (B, LATENT_CHANNELS, H, W)
    x = torch.randn(shape, device=device) * scale_factor

    timesteps = torch.linspace(schedule.T - 1, 0, ddim_steps, dtype=torch.long, device=device) if ddim else range(schedule.T - 1, -1, -1)

    for i in timesteps:
        t = torch.full((B,), i, device=device, dtype=torch.long)
        t_norm = t.float() / schedule.T
        pred_noise = denoiser(x, t_norm, context)

        alpha_bar = schedule.alpha_bar[i]
        alpha = schedule.alpha[i]

        if ddim:
            # DDIM step update
            pred_x0 = (x - torch.sqrt(1.0 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)
            if i > 0:
                alpha_bar_prev = schedule.alpha_bar[timesteps[torch.where(timesteps == i)[0] + 1]] if (timesteps == i).nonzero()[0] < len(timesteps) - 1 else torch.tensor(1.0, device=device)
                dir_xt = torch.sqrt(1.0 - alpha_bar_prev) * pred_noise
                x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt
            else:
                x = pred_x0
        else:
            # DDPM step update
            beta = schedule.beta[i]
            noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
            x = 1.0 / torch.sqrt(alpha) * (x - (beta / torch.sqrt(1.0 - alpha_bar)) * pred_noise) + torch.sqrt(beta) * noise
            
    return x


def train_diffusion(denoiser, schedule, dataloader, epochs):
    print("\n[Diffusion Training] Starting...")
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=DIFFUSION_LR)

    denoiser.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for cond, target in dataloader:
            cond, target = cond.to(DEVICE), target.to(DEVICE)
            B = target.shape[0]
            t = torch.randint(0, schedule.T, (B,), device=DEVICE).long()
            noise = torch.randn_like(target)
            
            sqrt_alpha_bar = torch.sqrt(schedule.alpha_bar[t])[:, None, None, None]
            sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - schedule.alpha_bar[t])[:, None, None, None]
            x_noisy = sqrt_alpha_bar * target + sqrt_one_minus_alpha_bar * noise

            optimizer.zero_grad()
            pred_noise = denoiser(x_noisy, t.float() / schedule.T, cond)
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * B

        print(f"Epoch {epoch+1:03d}/{epochs:03d} | Loss: {epoch_loss / len(dataloader.dataset):.5f}")

    torch.save({"model_state": denoiser.state_dict(), "T": schedule.T, "scale_factor": 1.0}, DIFFUSION_CHECKPOINT)
    print(f"[Diffusion Training] Saved checkpoint to {DIFFUSION_CHECKPOINT}")


# ==========================================
# 5. EVALUATION & VISUALIZATION
# ==========================================
def plot_side_by_side(x_true, x_hat, vars_list, stats, title_prefix, save_path):
    n_vars = len(vars_list)
    fig, axes = plt.subplots(n_vars, 2, figsize=(10, 4 * n_vars))
    if n_vars == 1:
        axes = np.array([axes])

    for i, var in enumerate(vars_list):
        true_val = denormalize(x_true[i], var, stats)
        hat_val = denormalize(x_hat[i], var, stats)
        if var == "msl":
            true_val, hat_val = true_val / 100.0, hat_val / 100.0
        diff_vals = hat_val - true_val

        im0 = axes[i, 0].imshow(true_val, cmap='viridis')
        axes[i, 0].set_title(f"{var} - True")
        fig.colorbar(im0, ax=axes[i, 0], shrink=0.8)

        rmse = np.sqrt(np.mean(diff_vals ** 2))
        im1 = axes[i, 1].imshow(hat_val, cmap='viridis')
        axes[i, 1].set_title(f"{var} - Diffusion Pred (RMSE={rmse:.3f})")
        fig.colorbar(im1, ax=axes[i, 1], shrink=0.8)

    fig.suptitle(title_prefix)
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def evaluate_diffusion_imputation(vae, denoiser, schedule, latents_dir, stats, vars_list,
                                   device, scale_factor=1.0, n_eval_tracks=20, ddim_steps=200, save_dir=None):
    vae.eval()
    denoiser.eval()
    latents_dir = Path(latents_dir)
    files = sorted(latents_dir.glob("*.pt"))[:n_eval_tracks]

    per_var_errors = {v: [] for v in vars_list}
    results = []

    for fpath in files:
        payload = torch.load(fpath, map_location=device, weights_only=False)
        z = payload["z"]  # (3, 128, 16, 16)

        z_prev, z_true, z_next = z[0], z[1], z[2]
        context = torch.cat([z_prev, z_next], dim=0).unsqueeze(0).to(device)

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        z_hat = impute_missing_timestep(
            denoiser, schedule, context, device, 
            scale_factor=scale_factor, ddim=True, ddim_steps=ddim_steps
        )

        x_hat = vae.decode(z_hat).detach().cpu().numpy()[0]

        raw_window, _ = load_raw_test_window(
            payload.get("track_id"), payload.get("season"), payload.get("basin"), payload.get("rank"),
            payload.get("time_strs")[1], data_dir=TEST_DIR, vars_list=vars_list,
        )
        x_true = raw_window[1].numpy()

        for i, var in enumerate(vars_list):
            var_pred_unnorm = denormalize(x_hat[i], var, stats)
            var_true_unnorm = x_true[i]
            scale = 100.0 if var == "msl" else 1.0
            per_var_errors[var].append(((var_pred_unnorm - var_true_unnorm) / scale).ravel())

        results.append({
            "track": payload.get("track_id", fpath.stem),
            "season": payload.get("season", None),
            "basin": payload.get("basin", None),
            "rank": payload.get("rank", None),
            "win_idx": payload.get("win_idx", None),
            "time": payload.get("time_strs")[1],
            "x_hat": x_hat,
            "x_true": x_true
        })

    summary = {}
    for var in vars_list:
        if per_var_errors[var]:
            errs = np.concatenate(per_var_errors[var])
            summary[var] = {
                "rmse": float(np.sqrt(np.mean(errs ** 2))),
                "mae": float(np.mean(np.abs(errs))),
                "bias": float(np.mean(errs)),
            }
    
    print("[Diffusion Eval] Per-variable RMSE/MAE/bias:")
    for var, m in summary.items():
        print(f"  {var}: rmse={m['rmse']:.4f} mae={m['mae']:.4f} bias={m['bias']:.4f}")

    if save_dir is not None and results:
        for example in results:
            plot_side_by_side(
                example["x_true"], example["x_hat"], vars_list, stats,
                title_prefix=f"Diffusion imputation — track {example['track']}, time {example['time']}",
                save_path=Path(save_dir) / f"diffusion_time_{TYPE}_track{example['track']}_{example['basin']}_rank{example['rank']}_wind{example['win_idx']}_{example['time']}.png"
            )

    return summary, results


def load_trained_diffusion(checkpoint_path=DIFFUSION_CHECKPOINT, device=DEVICE):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Diffusion checkpoint not found at {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    T_steps = ckpt.get("T", DIFFUSION_STEPS)
    scale_factor = ckpt.get("scale_factor", 1.0)
    if torch.is_tensor(scale_factor):
        scale_factor = scale_factor.to(device)

    schedule = DiffusionSchedule(T=T_steps, device=device)
    denoiser = LatentUNetDenoiser().to(device)
    denoiser.load_state_dict(ckpt["model_state"])
    denoiser.eval()

    print(f"[Diffusion] Loaded pre-trained checkpoint from {checkpoint_path}")
    return denoiser, schedule, scale_factor


@torch.no_grad()
def evaluate_latent_diffusion(vars_list=VARS_LIST, checkpoint=VAE_CHECKPOINT, 
                              diffusion_checkpoint=DIFFUSION_CHECKPOINT, 
                              latents_dir=LATENTS_DIR_TEST, save_dir=SAVE_DIR):
    device = DEVICE
    print(f"[Eval] Running diffusion evaluation on device: {device}")

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    stats = ckpt["stats"]

    vae = CycloneVAE(in_channels=len(vars_list), latent_channels=LATENT_CHANNELS).to(device)
    vae.load_state_dict(ckpt["model_state"])
    vae.eval()

    denoiser, schedule, scale_factor = load_trained_diffusion(diffusion_checkpoint, device=device)

    diffusion_summary, results = evaluate_diffusion_imputation(
        vae, denoiser, schedule, latents_dir, stats, vars_list, device, save_dir=save_dir, scale_factor=scale_factor
    )

    metrics_path = CHECKPOINT_DIR / f"diffusion_eval_metrics_{DIFFUSION_TYPE}.json"
    with open(metrics_path, "w") as f:
        json.dump(diffusion_summary, f, indent=4)
    print(f"--> Saved evaluation metrics to {metrics_path}")

    return denoiser, schedule, diffusion_summary, results


# ==========================================
# 6. MAIN EXECUTION PIPELINE
# ==========================================
def main():
    # train_dataset = CyclonePatchDataset(TRAIN_DIR, variables=VARS_LIST)
    # train_loader = DataLoader(train_dataset, batch_size=VAE_BATCH_SIZE, shuffle=True)

    # test_dataset = CyclonePatchDataset(TEST_DIR, variables=VARS_LIST, stats_dict=train_dataset.stats)
    # test_loader = DataLoader(test_dataset, batch_size=VAE_BATCH_SIZE, shuffle=False)

    # 1. VAE Training / Loading
    vae = CycloneVAE(in_channels=len(VARS_LIST), latent_channels=LATENT_CHANNELS).to(DEVICE)
    if VAE_CHECKPOINT.exists():
        ckpt = torch.load(VAE_CHECKPOINT, map_location=DEVICE, weights_only=False)
        vae.load_state_dict(ckpt["model_state"])
        print("[Pipeline] Loaded pre-trained VAE.")
    else:
        vae = train_vae(vae, train_loader, epochs=VAE_EPOCHS)

    # 2. Cache Latents Offline
    # cache_latents_by_track(vae, train_dataset, DEVICE, LATENTS_DIR)
    # cache_latents_by_track(vae, test_dataset, DEVICE, LATENTS_DIR_TEST)

    # 3. Diffusion Training / Loading
    latent_train_ds = CachedLatentDataset(LATENTS_DIR)
    latent_train_loader = DataLoader(latent_train_ds, batch_size=DIFFUSION_BATCH_SIZE, shuffle=True)

    schedule = DiffusionSchedule(T=DIFFUSION_STEPS, device=DEVICE)
    denoiser = LatentUNetDenoiser().to(DEVICE)

    if DIFFUSION_CHECKPOINT.exists():
        print("[Pipeline] Loaded pre-trained Diffusion model.")
    else:
        train_diffusion(denoiser, schedule, latent_train_loader, epochs=DIFFUSION_EPOCHS)

    # 4. Run Evaluation Pipeline
    evaluate_latent_diffusion(
        vars_list=VARS_LIST,
        checkpoint=VAE_CHECKPOINT,
        diffusion_checkpoint=DIFFUSION_CHECKPOINT,
        latents_dir=LATENTS_DIR_TEST,
        save_dir=SAVE_DIR
    )

    print("[Pipeline] Execution completed successfully!")


if __name__ == "__main__":
    main()

