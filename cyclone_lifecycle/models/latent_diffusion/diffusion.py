"""
diffusion.py - latent diffusion over the whole 3-day window with an arbitrary set of missing days.

How "any timestep" works
------------------------
The denoiser sees all F=3 latent frames stacked on the channel axis (F*C channels) plus a 0/1 "observed" flag per frame.
  * observed frames are fed CLEAN,
  * missing frames are fed noisy (noise level t),
  * the network predicts the noise of every frame, but the loss only counts the MISSING frames.
During training a random pattern of missing days is drawn per sample (config.TRAIN_SCENARIO_PROBS):
t1 | t2 | t0 | t1+t2 | t0+t1 | t0+t2. One model therefore does interpolation (t1), forecast (t2), backcast (t0)
and their multi-day versions. At sampling the observed frames stay fixed and only the missing ones are denoised.
"""
import copy
import math
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ... import config as cfg
from .latents import LatentNormalizer


# ------------------------------------------------------------------
# schedule
# ------------------------------------------------------------------
class DiffusionSchedule:
    def __init__(self, T=cfg.DIFFUSION_STEPS, device=cfg.DEVICE):
        self.T = T
        self.device = device
        self.beta = torch.linspace(1e-4, 0.02, T, device=device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)


# ------------------------------------------------------------------
# denoiser
# ------------------------------------------------------------------
def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlockT(nn.Module):
    def __init__(self, cin, cout, emb_dim):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb_dim, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = self.c1(F.silu(self.n1(x))) + self.emb(F.silu(emb))[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, self.heads, C // self.heads, H * W).unbind(1)
        o = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        return x + self.proj(o.transpose(-1, -2).reshape(B, C, H, W))


class LatentUNetDenoiser(nn.Module):
    """
    forward(x, obs, t)
        x   (B, F*C, h, w)  frames stacked on channels: observed = clean, missing = noisy
        obs (B, F)          1 = observed, 0 = missing
        t   (B,)            diffusion step in [0, T-1]
        -> predicted noise (B, F*C, h, w)
    h and w must be divisible by 4 (128x128 pixels -> 16x16 latents is fine).
    """
    def __init__(self, latent_channels=cfg.LATENT_CHANNELS, n_frames=cfg.N_FRAMES, base=cfg.DIFFUSION_BASE_CHANNELS):
        super().__init__()
        self.latent_channels, self.n_frames, self.base = latent_channels, n_frames, base
        emb = base * 4
        self.time_mlp = nn.Sequential(nn.Linear(base, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.inc = nn.Conv2d(n_frames * latent_channels + n_frames, base, 3, padding=1)

        self.e1 = ResBlockT(base, base, emb)
        self.d1 = nn.Conv2d(base, base, 3, stride=2, padding=1)
        self.e2 = ResBlockT(base, 2 * base, emb)
        self.d2 = nn.Conv2d(2 * base, 2 * base, 3, stride=2, padding=1)

        self.m1 = ResBlockT(2 * base, 4 * base, emb)
        self.attn = AttnBlock(4 * base)
        self.m2 = ResBlockT(4 * base, 4 * base, emb)

        self.u2 = nn.Conv2d(4 * base, 4 * base, 3, padding=1)
        self.r2 = ResBlockT(4 * base + 2 * base, 2 * base, emb)
        self.u1 = nn.Conv2d(2 * base, 2 * base, 3, padding=1)
        self.r1 = ResBlockT(2 * base + base, base, emb)

        self.out_norm = nn.GroupNorm(8, base)
        self.out = nn.Conv2d(base, n_frames * latent_channels, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, obs, t):
        B, _, H, W = x.shape
        assert H % 4 == 0 and W % 4 == 0, "latent height/width must be divisible by 4"
        emb = self.time_mlp(timestep_embedding(t, self.base))
        obs_map = obs.float()[:, :, None, None].expand(-1, -1, H, W)

        h1 = self.e1(self.inc(torch.cat([x, obs_map], dim=1)), emb)          # H
        h2 = self.e2(self.d1(h1), emb)                                       # H/2
        h = self.m1(self.d2(h2), emb)                                        # H/4
        h = self.m2(self.attn(h), emb)

        h = self.u2(F.interpolate(h, scale_factor=2, mode="nearest"))
        h = self.r2(torch.cat([h, h2], dim=1), emb)
        h = self.u1(F.interpolate(h, scale_factor=2, mode="nearest"))
        h = self.r1(torch.cat([h, h1], dim=1), emb)
        return self.out(F.silu(self.out_norm(h)))


# ------------------------------------------------------------------
# missing-day patterns
# ------------------------------------------------------------------
def scenario_table(names: Sequence[str], n_frames=cfg.N_FRAMES, device=None) -> torch.Tensor:
    """(S, F) bool, True = frame is MISSING in that scenario."""
    tab = torch.zeros(len(names), n_frames, dtype=torch.bool)
    for i, n in enumerate(names):
        for f in cfg.SCENARIOS[n]:
            tab[i, f] = True
    return tab.to(device) if device is not None else tab


def sample_missing_mask(B, names, probs, n_frames=cfg.N_FRAMES, device=cfg.DEVICE, generator=None) -> torch.Tensor:
    """(B, F) bool missing-frame mask, one scenario drawn per sample."""
    p = torch.tensor([probs[n] for n in names], dtype=torch.float32, device=device)
    k = torch.multinomial(p / p.sum(), B, replacement=True, generator=generator)
    return scenario_table(names, n_frames, device)[k]


# ------------------------------------------------------------------
# loss, sampler, EMA
# ------------------------------------------------------------------
def diffusion_loss(denoiser, schedule, z0, miss, t=None, noise=None, generator=None, prediction=cfg.DIFFUSION_PREDICTION):
    """
    z0   (B, F, C, h, w) normalised clean latents,  miss (B, F) bool
    Noise is added to the missing frames only; the loss is averaged over missing frames only.
    prediction="eps": the network predicts the noise.   prediction="v": it predicts v = sqrt(ab)*eps - sqrt(1-ab)*x0.
    """
    B, Fr, C, h, w = z0.shape
    if t is None:
        t = torch.randint(0, schedule.T, (B,), device=z0.device, generator=generator)
    if noise is None:
        noise = torch.randn(z0.shape, device=z0.device, generator=generator)
    m5 = miss.view(B, Fr, 1, 1, 1)
    ab = schedule.alpha_bar[t].view(B, 1, 1, 1, 1)
    x_t = torch.where(m5, ab.sqrt() * z0 + (1 - ab).sqrt() * noise, z0)
    out = denoiser(x_t.reshape(B, Fr * C, h, w), (~miss).float(), t).view(B, Fr, C, h, w)
    target = noise if prediction == "eps" else ab.sqrt() * noise - (1 - ab).sqrt() * z0
    sq = (out - target) ** 2 * m5
    return sq.sum() / (m5.sum() * C * h * w)


@torch.no_grad()
def ddim_sample(denoiser, schedule, z_obs, miss, steps=cfg.DDIM_STEPS, eta=cfg.DDIM_ETA, generator=None,
                x0_clip: Optional[float] = None, prediction=cfg.DIFFUSION_PREDICTION):
    """
    Fill the missing frames. z_obs (B,F,C,h,w) normalised latents (values at missing frames are ignored),
    miss (B,F) bool. Observed frames are returned bit-for-bit unchanged.
    """
    B, Fr, C, h, w = z_obs.shape
    dev = z_obs.device
    m5 = miss.view(B, Fr, 1, 1, 1)
    obs = (~miss).float()

    x = torch.where(m5, torch.randn(z_obs.shape, device=dev, generator=generator), z_obs)
    ts = torch.linspace(schedule.T - 1, 0, steps).round().long().unique_consecutive().tolist()

    for i, t in enumerate(ts):
        ab = schedule.alpha_bar[t]
        ab_prev = schedule.alpha_bar[ts[i + 1]] if i + 1 < len(ts) else torch.ones((), device=dev)
        out = denoiser(x.reshape(B, Fr * C, h, w), obs, torch.full((B,), t, device=dev)).view(B, Fr, C, h, w)

        if prediction == "eps":
            eps = out
            x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
        else:                                                   # v-prediction
            x0 = ab.sqrt() * x - (1 - ab).sqrt() * out
            eps = (1 - ab).sqrt() * x + ab.sqrt() * out
        if x0_clip is not None:
            x0 = x0.clamp(-x0_clip, x0_clip)
            eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()
        sigma = eta * ((1 - ab_prev) / (1 - ab)).sqrt() * (1 - ab / ab_prev).clamp(min=0).sqrt()
        direction = (1 - ab_prev - sigma ** 2).clamp(min=0).sqrt() * eps
        x_new = ab_prev.sqrt() * x0 + direction
        if eta > 0:
            x_new = x_new + sigma * torch.randn(x.shape, device=dev, generator=generator)
        x = torch.where(m5, x_new, z_obs)
    return x


class EMA:
    """Exponential moving average of the weights; sampling uses the EMA copy (much better samples)."""
    def __init__(self, model, decay=cfg.EMA_DECAY):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay, self.n = decay, 0

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))      # warm-up so short runs are not stuck near init
        for pe, p in zip(self.module.parameters(), model.parameters()):
            pe.mul_(d).add_(p.detach(), alpha=1 - d)


# ------------------------------------------------------------------
# training / loading
# ------------------------------------------------------------------
@torch.no_grad()
def diffusion_val_loss(denoiser, schedule, dataset, names, probs, batch_size, device, seed=123,
                       prediction=cfg.DIFFUSION_PREDICTION):
    """Seeded (same masks, t and noise every epoch) so the curve is comparable across epochs."""
    denoiser.eval()
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    tot, n = 0.0, 0
    for z0 in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        z0 = z0.to(device)
        miss = sample_missing_mask(z0.shape[0], names, probs, z0.shape[1], device, gen)
        tot += diffusion_loss(denoiser, schedule, z0, miss, generator=gen, prediction=prediction).item() * z0.shape[0]
        n += z0.shape[0]
    return tot / n


def train_diffusion(denoiser, schedule, train_ds: Dataset, normalizer: LatentNormalizer, val_ds: Optional[Dataset] = None,
                    epochs=cfg.DIFFUSION_EPOCHS, batch_size=cfg.DIFFUSION_BATCH_SIZE, lr=cfg.DIFFUSION_LR,
                    ema_decay=cfg.EMA_DECAY, scenario_probs: Optional[Dict[str, float]] = None, device=cfg.DEVICE,
                    checkpoint_path=cfg.DIFFUSION_CHECKPOINT, extra_meta: Optional[dict] = None,
                    prediction=cfg.DIFFUSION_PREDICTION):
    scenario_probs = scenario_probs or cfg.TRAIN_SCENARIO_PROBS
    names = list(scenario_probs)
    print(f"\n[Diffusion Training] {len(train_ds)} windows, {prediction}-prediction, scenarios: "
          + ", ".join(f"{n}={p:.2f}" for n, p in scenario_probs.items()))
    opt = torch.optim.AdamW(denoiser.parameters(), lr=lr, weight_decay=1e-4)
    ema = EMA(denoiser, ema_decay)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=len(train_ds) > batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs * len(loader), 1), eta_min=lr * 0.05)
    train_hist, val_hist = [], []

    for epoch in range(epochs):
        denoiser.train()
        tot, n = 0.0, 0
        for z0 in loader:
            z0 = z0.to(device)
            miss = sample_missing_mask(z0.shape[0], names, scenario_probs, z0.shape[1], device)
            loss = diffusion_loss(denoiser, schedule, z0, miss, prediction=prediction)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
            opt.step()
            sched.step()
            ema.update(denoiser)
            tot += loss.item() * z0.shape[0]
            n += z0.shape[0]
        train_hist.append(tot / n)
        msg = f"Epoch {epoch+1:03d}/{epochs:03d} | train loss {train_hist[-1]:.5f}"
        if val_ds is not None:
            val_hist.append(diffusion_val_loss(ema.module, schedule, val_ds, names, scenario_probs, batch_size, device,
                                                prediction=prediction))
            msg += f" | test loss (EMA, fixed masks) {val_hist[-1]:.5f}"
        print(msg)
        save_diffusion(ema.module, schedule, normalizer, checkpoint_path, epoch + 1, train_hist, val_hist,
                       {"prediction": prediction, **(extra_meta or {})})
    print(f"[Diffusion Training] Saved checkpoint to {checkpoint_path}")
    return ema.module


def save_diffusion(denoiser, schedule, normalizer, path, epoch, train_hist, val_hist, extra_meta=None):
    torch.save({
        "model_state": denoiser.state_dict(),            # EMA weights
        "T": schedule.T,
        "latent_channels": denoiser.latent_channels, "n_frames": denoiser.n_frames, "base": denoiser.base,
        "normalizer": normalizer.state_dict(),           # latent mean/std fitted on TRAIN latents
        "epoch": epoch, "train_loss": train_hist, "val_loss": val_hist,
        **(extra_meta or {}),
    }, path)


def load_trained_diffusion(checkpoint_path=cfg.DIFFUSION_CHECKPOINT, device=cfg.DEVICE):
    """-> denoiser (eval), schedule, normalizer, ckpt dict"""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Diffusion checkpoint not found at {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    denoiser = LatentUNetDenoiser(ckpt["latent_channels"], ckpt["n_frames"], ckpt["base"]).to(device)
    denoiser.load_state_dict(ckpt["model_state"])
    denoiser.eval()
    schedule = DiffusionSchedule(T=ckpt["T"], device=device)
    normalizer = LatentNormalizer.from_state_dict(ckpt["normalizer"]).to(device)
    print(f"[Diffusion] Loaded checkpoint from {checkpoint_path} (epoch {ckpt['epoch']})")
    return denoiser, schedule, normalizer, ckpt
