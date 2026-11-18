"""
latents.py - encode windows once and cache ONE FILE PER WINDOW (same layout as your cache_latents_by_track).

latent_track{track}_{season}_{basin}_rank{rank}_win{win}.pt  contains
    z (F,C,h,w) = posterior mean mu, raw / un-normalised (decode it directly with vae.decode, as before)
    source_file, track_id, season, basin, rank, win_idx, lat, lon, time_strs, cyclone_center_lats/lons
    window_idx (position in the dataset), vae_stamp, and logvar if STORE_LOGVAR
Besides the windows, the folder holds
    latent_normalizer.pt   per-channel mean/std fitted on TRAIN latents (train folder only)
    _cache_info.json       VAE stamp + number of windows, used to detect a stale / interrupted cache
The normalisation is applied only inside CachedLatentDataset, never written into the window files.
"""
import json
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

import config as cfg
from vae import file_stamp

WINDOW_GLOB = "latent_track*.pt"


# ------------------------------------------------------------------
# normalisation
# ------------------------------------------------------------------
class LatentNormalizer:
    """z_norm = (z - mean_c) / std_c, per latent channel. Works on any tensor shaped (..., C, h, w)."""
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean.float().view(-1, 1, 1)
        self.std = std.float().view(-1, 1, 1)

    def to(self, device):
        self.mean, self.std = self.mean.to(device), self.std.to(device)
        return self

    def normalize(self, z):
        return (z - self.mean.to(z.device)) / self.std.to(z.device)

    def denormalize(self, z):
        return z * self.std.to(z.device) + self.mean.to(z.device)

    def state_dict(self):
        return {"mean": self.mean.flatten().cpu(), "std": self.std.flatten().cpu()}

    @classmethod
    def from_state_dict(cls, d):
        return cls(d["mean"], d["std"])


class RunningLatentStats:
    """Streaming per-channel statistics (float64), so the whole train set never has to sit in memory."""
    def __init__(self, C):
        self.s1 = torch.zeros(C, dtype=torch.float64)
        self.s2 = torch.zeros(C, dtype=torch.float64)
        self.sv = torch.zeros(C, dtype=torch.float64)       # sum of exp(logvar) = posterior variance
        self.n = 0
        self.p1 = self.p2 = None                            # patch-mean channels (optional)
        self.np = 0

    def update(self, mu: torch.Tensor, logvar: torch.Tensor, pm: Optional[torch.Tensor] = None):   # (B,F,C,h,w), pm (B,F,Cv)
        if pm is not None:
            p = pm.double()
            if self.p1 is None:
                self.p1 = torch.zeros(p.shape[-1], dtype=torch.float64)
                self.p2 = torch.zeros(p.shape[-1], dtype=torch.float64)
            self.p1 += p.sum(dim=(0, 1))
            self.p2 += (p ** 2).sum(dim=(0, 1))
            self.np += p.shape[0] * p.shape[1]
        m = mu.double()
        self.s1 += m.sum(dim=(0, 1, 3, 4)).cpu()
        self.s2 += (m ** 2).sum(dim=(0, 1, 3, 4)).cpu()
        self.sv += logvar.double().exp().sum(dim=(0, 1, 3, 4)).cpu()
        self.n += m.shape[0] * m.shape[1] * m.shape[3] * m.shape[4]

    def finalize(self, with_logvar: bool, floor_rel: float = cfg.LATENT_STD_FLOOR_REL) -> LatentNormalizer:
        mean = self.s1 / self.n
        var_mu = (self.s2 / self.n - mean ** 2).clamp(min=0)
        sig2 = self.sv / self.n
        snr = var_mu / sig2
        print(f"[Latents] latent SNR = Var(mu) / mean(sigma^2) per channel: median {snr.median():.1f} | min {snr.min():.2f} | "
              f"channels with SNR > 10: {int((snr > 10).sum())}/{len(snr)}   "
              "(>> 1: z ~ mu, training on mu is safe; <~ 1: the decoder was trained on noisy z, decode(mu) is off-distribution)")
        # if the diffusion trains on sampled z the variance of z is Var(mu) + sigma^2
        std = (var_mu + (sig2 if with_logvar else 0.0)).sqrt()
        floor = floor_rel * std.median()
        n_dead = int((std < 0.05 * std.median()).sum())
        std = std.clamp(min=float(floor))
        print(f"[Latents] per-channel std: min {std.min():.3f} | median {std.median():.3f} | max {std.max():.3f} "
              f"| channels below 5% of the median (near-dead): {n_dead}/{len(std)}")
        if self.p1 is not None:        # append the patch-mean channels (they are constant maps, std over windows/days)
            pmean = self.p1 / self.np
            pstd = (self.p2 / self.np - pmean ** 2).clamp(min=1e-12).sqrt()
            mean, std = torch.cat([mean, pmean]), torch.cat([std, pstd])
            print(f"[Latents] + {len(pmean)} patch-mean channels, std {pstd.min():.3f}..{pstd.max():.3f}")
        return LatentNormalizer(mean, std)


def augment_with_patch_mean(z: torch.Tensor, pm: Optional[torch.Tensor]) -> torch.Tensor:
    """z (..., F, Lc, h, w), pm (..., F, C) -> (..., F, Lc + C, h, w): the patch mean appended as constant maps.
    pm=None returns z unchanged (VAE trained without anomaly mode)."""
    if pm is None:
        return z
    return torch.cat([z, pm[..., None, None].expand(*pm.shape, z.shape[-2], z.shape[-1])], dim=-3)


# ------------------------------------------------------------------
# cache build / load
# ------------------------------------------------------------------
def _window_filename(md, used: set, idx: int) -> str:
    s = lambda v, d="NA": str(v) if v else d
    name = f"latent_track{s(md.track_id)}_{s(md.season)}_{s(md.basin)}_rank{s(md.rank)}_win{s(md.win_idx, '0')}.pt"
    if name in used:                                  # two windows with identical labels must not overwrite each other
        name = name.replace(".pt", f"_idx{idx}.pt")
    used.add(name)
    return name


@torch.no_grad()
def build_latent_cache(vae, dataset, out_dir, device=cfg.DEVICE, batch_size=32, store_logvar=cfg.STORE_LOGVAR,
                       fit_normalizer=False):
    """Encode every window with the VAE -> one .pt per window in out_dir. fit_normalizer=True only for the train split."""
    vae.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in list(out_dir.glob(WINDOW_GLOB)) + list(out_dir.glob("_cache_info.json")):
        old.unlink()                                  # files of a previous VAE must not survive
    stamp = file_stamp(cfg.VAE_CHECKPOINT)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(f"[Latents] Encoding {len(dataset)} windows -> {out_dir}")
    stats, used, count = None, set(), 0
    for windows, _, indices in loader:
        B, T_win, C, H, W = windows.shape
        x, pm = windows.to(device), None
        if vae.patch_transform is not None:              # VAE works on anomalies; the patch mean is stored separately
            x, pm = vae.patch_transform.split(x)
            pm = pm.cpu()
        mu, logvar = vae.encode(x.reshape(B * T_win, C, H, W))
        z_w = mu.view(B, T_win, *mu.shape[1:]).cpu()
        lv_w = logvar.view(B, T_win, *logvar.shape[1:]).cpu()
        if stats is None:
            stats = RunningLatentStats(z_w.shape[2])
        stats.update(z_w, lv_w, pm)

        for b in range(B):
            idx = int(indices[b])
            md = dataset.get_metadata(idx)
            payload = {
                "source_file": md.source_file,
                "z": z_w[b].clone(),
                "track_id": md.track_id, "season": md.season, "basin": md.basin, "rank": md.rank,
                "win_idx": md.win_idx, "lat": md.lat, "lon": md.lon, "time_strs": md.time_strs,
                "cyclone_center_lats": md.cyclone_center_lats, "cyclone_center_lons": md.cyclone_center_lons,
                "window_idx": idx, "vae_stamp": stamp,
                "patch_mean": None if pm is None else pm[b].clone(),          # (F, C) or None
            }
            if store_logvar:
                payload["logvar"] = lv_w[b].clone()
            torch.save(payload, out_dir / _window_filename(md, used, idx))
            count += 1

    if fit_normalizer:
        torch.save(stats.finalize(store_logvar).state_dict(), out_dir / "latent_normalizer.pt")
    with open(out_dir / "_cache_info.json", "w") as f:
        json.dump({"vae_stamp": stamp, "n_windows": count}, f)
    print(f"[Latents] Successfully saved {count} latent window payloads to {out_dir}")


def load_latent_cache(path) -> dict:
    """Read every window file of a folder (ordered like the dataset) -> {'mu','logvar','meta','vae_stamp'}."""
    path = Path(path)
    files = sorted(path.glob(WINDOW_GLOB))
    if not files:
        raise FileNotFoundError(f"No latent window files in {path}")
    items = [(f, torch.load(f, map_location="cpu", weights_only=False)) for f in files]
    items.sort(key=lambda fp: fp[1]["window_idx"])
    mu = torch.stack([p["z"] for _, p in items]).half()
    logvar = torch.stack([p["logvar"] for _, p in items]).half() if "logvar" in items[0][1] else None
    meta = [{**{k: p.get(k) for k in ("track_id", "season", "basin", "rank", "win_idx", "time_strs")},
             "source_file": Path(p["source_file"]).name, "latent_file": f.name} for f, p in items]
    pm = items[0][1].get("patch_mean")
    patch_mean = None if pm is None else torch.stack([p["patch_mean"] for _, p in items])
    return {"mu": mu, "logvar": logvar, "patch_mean": patch_mean, "meta": meta, "vae_stamp": items[0][1].get("vae_stamp")}


def latent_cache_exists(path) -> bool:
    return (Path(path) / "_cache_info.json").exists()


def latent_cache_is_current(path, vae_checkpoint=cfg.VAE_CHECKPOINT) -> bool:
    """True if the folder is complete and was produced by the current VAE checkpoint."""
    info_path = Path(path) / "_cache_info.json"
    if not info_path.exists():
        return False
    info = json.load(open(info_path))
    n_files = len(list(Path(path).glob(WINDOW_GLOB)))
    return info["vae_stamp"] == file_stamp(vae_checkpoint) and info["n_windows"] == n_files


class CachedLatentDataset(Dataset):
    """Returns normalised latents (F, C, h, w). With sample_latents=True and a cached logvar, z ~ N(mu, sigma^2)."""
    def __init__(self, path, normalizer: Optional[LatentNormalizer] = None, sample_latents=False):
        payload = load_latent_cache(path)
        self.mu = payload["mu"]
        self.logvar = payload.get("logvar")
        self.pm = payload.get("patch_mean")              # (N, F, C) or None
        self.meta: List[dict] = payload["meta"]
        self.vae_stamp = payload.get("vae_stamp")
        if normalizer is None:
            normalizer = LatentNormalizer.from_state_dict(torch.load(Path(path) / "latent_normalizer.pt"))
        self.normalizer = normalizer
        self.sample_latents = sample_latents and self.logvar is not None
        if sample_latents and self.logvar is None:
            print("[Latents] sample_latents requested but no logvar cached -> using mu")

    def __len__(self):
        return self.mu.shape[0]

    def __getitem__(self, idx):
        z = self.mu[idx].float()
        if self.sample_latents:
            z = z + torch.randn_like(z) * torch.exp(0.5 * self.logvar[idx].float())
        z = augment_with_patch_mean(z, None if self.pm is None else self.pm[idx].float())
        return self.normalizer.normalize(z)


def check_split_leakage(train_meta: List[dict], test_meta: List[dict]) -> float:
    """Fraction of test windows whose cyclone (track/season/basin) also appears in train. Should be 0."""
    key = lambda m: (str(m.get("track_id")), str(m.get("season")), str(m.get("basin")))
    train_keys = {key(m) for m in train_meta}
    shared = sum(key(m) in train_keys for m in test_meta)
    frac = shared / max(len(test_meta), 1)
    flag = "OK" if shared == 0 else "WARNING: the same cyclone is in train and test -> optimistic test scores"
    print(f"[Latents] test windows whose track also appears in train: {shared}/{len(test_meta)} ({100*frac:.1f}%) {flag}")
    return frac
