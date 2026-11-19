"""
ocean_model.py - predict the OCEAN fields of any missing day of the 3-day window, directly in pixel space.
ocean model abcd "ef" 
Why this and not the VAE + latent diffusion: the day-to-day change of the ocean is small (interpolating t0 and t2 is
already within ~5% of the field's spread), and a lossy VAE latent cannot keep that precision. So the network here:
  * works on the pixels (no VAE, no latent, no diffusion),
  * starts from the temporal baseline (linear interpolation between observed days, persistence at the edges) and only
    learns the CORRECTION to it (zero-initialised output = it starts exactly at the baseline),
  * sees the atmosphere (10u, 10v, msl) of all days, which is the forcing the baseline cannot know,
  * sees a "distance to the storm centre" map (the box is centred on the storm at the middle day).
The loss is the MSE of the correction divided by the baseline error of each variable, so it is ~ (1 - skill) and
every ocean variable counts the same (zos and mlotst have very different units).

    python ocean_model.py --stage train            # needs the train + test NetCDF (stats come from the train set)
    python ocean_model.py --stage eval             # needs only the checkpoint + test NetCDF; all test windows + figures
    python ocean_model.py --stage train,eval --epochs 60

    # compare with other models / plots, as before (the window is the dataset item, no latent files needed)
    imputer = OceanImputer()                                              # load once, outside any loop
    x_hat = evaluate_sample(imputer, eval_dataset[i][0], target_frame_idx=1)   # (3, 7, H, W) normalised, all days
    x_phys = imputer.to_physical(x_hat)                                   # physical units (msl in hPa)
    # x_hat: atmosphere = the input, observed ocean days = the input, missing ocean days = the prediction
"""
import argparse
import copy
import csv
import json
import time
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

from .. import config as cfg
from ..dataset import denormalize

ATM = ["10u", "10v", "msl"]
OCN = ["avg_tos", "avg_zos", "avg_sos", "avg_mlotst125"]
N_FRAMES = 3
OUT_DIR = Path(cfg.CHECKPOINT_DIR) / "ocean_residual"
CKPT = OUT_DIR / "checkpoint_ocean_unet.pt"
SCENARIOS = {"mid": (1,), "last": (2,), "first": (0,), "last2": (1, 2), "first2": (0, 1), "ends": (0, 2)}
TRAIN_PROBS = {"mid": 0.30, "last": 0.20, "first": 0.20, "last2": 0.10, "first2": 0.10, "ends": 0.10}
PLOT_SCENARIOS = ["mid", "last", "first"]


# ------------------------------------------------------------------------------------------------------------------
# baseline
# ------------------------------------------------------------------------------------------------------------------
def temporal_fill(o: torch.Tensor, missing: Sequence[int]) -> torch.Tensor:
    """o (B, T, K, H, W). Missing days <- linear interpolation between the nearest observed days before and after,
    or persistence of the nearest observed day when there is only one side. Observed days are returned untouched."""
    T = o.shape[1]
    obs = [t for t in range(T) if t not in missing]
    out = o.clone()
    for t in missing:
        before = [s for s in obs if s < t]
        after = [s for s in obs if s > t]
        if before and after:
            a, b = before[-1], after[0]
            w = (t - a) / (b - a)
            out[:, t] = (1 - w) * o[:, a] + w * o[:, b]
        else:
            out[:, t] = o[:, before[-1] if before else after[0]]
    return out


# ------------------------------------------------------------------------------------------------------------------
# network
# ------------------------------------------------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.n1, self.n2 = nn.GroupNorm(8, ci), nn.GroupNorm(8, co)
        self.c1 = nn.Conv2d(ci, co, 3, padding=1, padding_mode="replicate")
        self.c2 = nn.Conv2d(co, co, 3, padding=1, padding_mode="replicate")
        self.skip = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.c1(self.act(self.n1(x)))
        h = self.c2(self.act(self.n2(h)))
        return h + self.skip(x)


class OceanUNet(nn.Module):
    """4-level U-Net, H and W divisible by 8. Output starts at exactly 0.
    residual   : True  -> the output is a correction added to the temporal baseline (default)
                 False -> the output IS the field (ablation: same inputs, no baseline added)
    bottleneck : 0 -> normal U-Net with skip connections
                 K -> ablation: NO skip connections and the 8x-downsampled features are squeezed to K channels
                      (an autoencoder-like information bottleneck, like the VAE but without the KL term)"""
    def __init__(self, c_in: int, c_out: int, base: int = 48, bottleneck: int = 0, residual: bool = True,
                 mask_atm: bool = False, use_atm: bool = True):
        super().__init__()
        b = base
        self.bottleneck, self.residual, self.mask_atm = int(bottleneck), bool(residual), bool(mask_atm)
        self.use_atm = bool(use_atm)               # False -> ocean-only model: the atmosphere is not an input at all
        self.inp = nn.Conv2d(c_in, b, 3, padding=1, padding_mode="replicate")
        self.e1, self.e2 = ResBlock(b, b), ResBlock(b, 2 * b)
        self.e3, self.e4 = ResBlock(2 * b, 4 * b), ResBlock(4 * b, 8 * b)
        self.mid = ResBlock(8 * b, 8 * b)
        if self.bottleneck:
            self.squeeze, self.expand = nn.Conv2d(8 * b, self.bottleneck, 1), nn.Conv2d(self.bottleneck, 8 * b, 1)
        self.pool = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        k = 1 if self.bottleneck else 1.5          # decoder input = upsampled features (+ skip if not bottleneck)
        self.d3 = ResBlock(int(8 * b * k) if not self.bottleneck else 8 * b, 4 * b)
        self.d2 = ResBlock(int(4 * b * k), 2 * b)
        self.d1 = ResBlock(int(2 * b * k), b)
        self.out = nn.Sequential(nn.GroupNorm(8, b), nn.SiLU(), nn.Conv2d(b, c_out, 1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x):
        e1 = self.e1(self.inp(x))
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.mid(self.e4(self.pool(e3)))
        if self.bottleneck:
            e4 = self.expand(self.squeeze(e4))
            d3 = self.d3(self.up(e4))
            d2 = self.d2(self.up(d3))
            d1 = self.d1(self.up(d2))
        else:
            d3 = self.d3(torch.cat([self.up(e4), e3], 1))
            d2 = self.d2(torch.cat([self.up(d3), e2], 1))
            d1 = self.d1(torch.cat([self.up(d2), e1], 1))
        return self.out(d1)


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay, self.n = decay, 0
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))        # warm-up: short runs are not stuck at the initial weights
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(d).add_(p.detach(), alpha=1 - d)
        for s, b in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(b)


# ------------------------------------------------------------------------------------------------------------------
# inputs / forward (shared by training, evaluation and inference)
# ------------------------------------------------------------------------------------------------------------------
def radius_map(H, W, device):
    yy = (torch.arange(H, device=device, dtype=torch.float32) - (H - 1) / 2) / (H / 2)
    xx = (torch.arange(W, device=device, dtype=torch.float32) - (W - 1) / 2) / (W / 2)
    return torch.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2)[None, None]


def build_input(win: torch.Tensor, missing: Sequence[int], vars_list, mask_atm: bool = False, use_atm: bool = True):
    """mask_atm=True: the atmosphere of the missing days is hidden too (strict forecast / hindcast: the wind of the day
    to predict is not known). Default False: the atmosphere of all days is given (gap filling).
    win (B, T, C, H, W) normalised dataset window ->
       net input (B, T*8 + 1, H, W), filled ocean (B, T, 4, H, W) = observed days + baseline at the missing days,
       ocean (B, T, 4, H, W) = the truth (only meaningful at observed days at inference)."""
    B, T, C, H, W = win.shape
    io = [vars_list.index(v) for v in OCN]
    ia = [vars_list.index(v) for v in ATM]
    ocean, atm = win[:, :, io], win[:, :, ia]
    if mask_atm:
        atm = atm.clone()
        atm[:, list(missing)] = 0.0
    filled = temporal_fill(ocean, missing)
    flag = torch.zeros(B, T, 1, H, W, device=win.device, dtype=win.dtype)
    flag[:, list(missing)] = 1.0
    x = torch.cat([filled, atm, flag] if use_atm else [filled, flag], 2).flatten(1, 2)
    x = torch.cat([x, radius_map(H, W, win.device).expand(B, 1, H, W).to(win.dtype)], 1)
    return x, filled, ocean


def predict_ocean(net, win, missing, scale, vars_list, amp=False):
    """-> (prediction at the missing days (B, K, 4, H, W), baseline at the missing days, truth at the missing days)"""
    missing = list(missing)
    x, filled, ocean = build_input(win, missing, vars_list, getattr(net, "mask_atm", False), getattr(net, "use_atm", True))
    B, T, _, H, W = filled.shape
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=amp):
        out = net(x)
    out = out.float().view(B, T, len(OCN), H, W)[:, missing]
    base = filled[:, missing]
    pred = base + out * scale.view(1, 1, -1, 1, 1) if getattr(net, "residual", True) else out
    return pred, base, ocean[:, missing]


# ------------------------------------------------------------------------------------------------------------------
# training
# ------------------------------------------------------------------------------------------------------------------
def fit_scales(data: np.ndarray, vars_list, n_max=2000, bs=250):
    """Per scenario and ocean variable: rms of (truth - baseline) on train windows (normalised units)."""
    io = [vars_list.index(v) for v in OCN]
    sel = np.linspace(0, len(data) - 1, min(n_max, len(data))).astype(int)
    scales = {}
    for name, miss in SCENARIOS.items():
        se, n = np.zeros(len(OCN)), 0
        for i in range(0, len(sel), bs):
            o = torch.from_numpy(data[sel[i:i + bs]][:, :, io])
            e = (temporal_fill(o, miss) - o)[:, list(miss)]
            se += (e.double() ** 2).sum(dim=(0, 1, 3, 4)).numpy()
            n += e.shape[0] * e.shape[1] * e.shape[3] * e.shape[4]
        scales[name] = np.sqrt(se / n).tolist()
    return scales


def skill_summary(res):
    return " ".join(f"{v[4:9]}={res[v]['all']['skill']:+.3f}" for v in OCN)


def set_tag(tag: str):
    """each variant (ablation) writes to its own folder: <out>/ocean_residual<tag>/"""
    global OUT_DIR, CKPT
    OUT_DIR = Path(cfg.CHECKPOINT_DIR) / f"ocean_residual{tag}"
    CKPT = OUT_DIR / "checkpoint_ocean_unet.pt"


def train(train_data, val_data, stats, vars_list, epochs, bs, lr, base, device, core, bottleneck=0, residual=True,
          mask_atm=False, use_atm=True):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scales = fit_scales(train_data, vars_list)
    print("[Ocean] residual scale (rms of the baseline error, normalised units):")
    for k, v in scales.items():
        print(f"    {k:6s} " + " ".join(f"{o}={x:.4f}" for o, x in zip(OCN, v)))
    c_in = N_FRAMES * (len(OCN) + (len(ATM) if use_atm else 0) + 1) + 1
    net = OceanUNet(c_in, N_FRAMES * len(OCN), base, bottleneck, residual, mask_atm, use_atm).to(device)
    ema = EMA(net)
    print(f"[Ocean] U-Net {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M parameters, {len(train_data)} train / "
          f"{len(val_data)} val windows | residual={residual} bottleneck={bottleneck or 'off'} mask_atm={mask_atm} "
          f"use_atm={use_atm} | {c_in} input channels")
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps_ep = (len(train_data) + bs - 1) // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_ep, pct_start=0.05)
    names = list(TRAIN_PROBS)
    probs = np.array([TRAIN_PROBS[n] for n in names]) / sum(TRAIN_PROBS.values())
    amp = device.type == "cuda"
    stds = std_vector(stats)
    sc_t = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in scales.items()}
    rng = np.random.default_rng(0)
    t0 = time.time()
    net.train()
    for ep in range(epochs):
        perm = rng.permutation(len(train_data))
        tot, nb = 0.0, 0
        for i in range(0, len(perm), bs):
            name = names[rng.choice(len(names), p=probs)]          # one missing-pattern per batch
            win = torch.from_numpy(train_data[np.sort(perm[i:i + bs])]).to(device)
            pred, _, truth = predict_ocean(net, win, SCENARIOS[name], sc_t[name], vars_list, amp)
            loss = (((pred - truth) / sc_t[name].view(1, 1, -1, 1, 1)) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            ema.update(net)
            tot, nb = tot + loss.item(), nb + 1
        msg = f"Epoch {ep + 1:03d}/{epochs} | train loss {tot / nb:.4f} (1.0 = baseline)"
        if (ep + 1) % max(1, epochs // 10) == 0 or ep + 1 == epochs:
            sub = val_data[np.linspace(0, len(val_data) - 1, min(300, len(val_data))).astype(int)]
            r = evaluate(ema.shadow, sub, vars_list, ["mid"], scales, stds, core, device, bs=64)["mid"]
            msg += " | val skill mid " + skill_summary(r)
        print(msg + f" | {time.time() - t0:.0f}s")
        # the VAL curve is only monitored; the checkpoint is always the latest epoch (EMA weights); test is never touched here
        torch.save({"model_state": ema.shadow.state_dict(), "stats": stats, "vars_list": list(vars_list), "scales": scales,
                    "base": base, "c_in": c_in, "epoch": ep + 1, "bottleneck": bottleneck, "residual": residual,
                    "mask_atm": mask_atm, "use_atm": use_atm}, CKPT)
    print(f"[Ocean] saved {CKPT}")


# ------------------------------------------------------------------------------------------------------------------
# evaluation
# ------------------------------------------------------------------------------------------------------------------
def std_vector(stats):
    return np.array([stats[v]["std"] if stats[v]["std"] > 1e-6 else 1.0 for v in OCN])


def grad_mag(x):
    """mean |gradient| over the image, (..., H, W) -> (...)"""
    gx = (x[..., :, 1:] - x[..., :, :-1]).abs().mean(dim=(-2, -1))
    gy = (x[..., 1:, :] - x[..., :-1, :]).abs().mean(dim=(-2, -1))
    return 0.5 * (gx + gy)


@torch.no_grad()
def evaluate(net, data, vars_list, scenarios, scales, stds, core, device, bs=64, plot_idx=None):
    """Streaming metrics over ALL windows in `data` (numpy (N,T,C,H,W)). -> {scenario: {variable: {scope: metrics}}}.
    RMSE in physical units = rmse(normalised) * std (the mean cancels in an error)."""
    net.eval()
    H = data.shape[-1]
    lo, hi = (H - core) // 2, (H + core) // 2
    out = {}
    for name in scenarios:
        miss = SCENARIOS[name]
        sc = torch.tensor(scales[name], dtype=torch.float32, device=device)
        z = lambda: torch.zeros(len(OCN), dtype=torch.float64)
        acc = {k: z() for k in ("se_m", "se_b", "se_m_c", "se_b_c", "bias_m", "g_m", "g_b", "g_t")}
        n = 0
        for i in range(0, len(data), bs):
            win = torch.from_numpy(data[i:i + bs]).to(device)
            pred, base, truth = predict_ocean(net, win, miss, sc, vars_list)
            em, eb = pred - truth, base - truth
            for k, e in (("se_m", em), ("se_b", eb)):
                acc[k] += (e.double() ** 2).sum(dim=(0, 1, 3, 4)).cpu()
                acc[k + "_c"] += (e[..., lo:hi, lo:hi].double() ** 2).sum(dim=(0, 1, 3, 4)).cpu()
            acc["bias_m"] += em.double().sum(dim=(0, 1, 3, 4)).cpu()
            acc["g_m"] += grad_mag(pred).double().sum(dim=(0, 1)).cpu()
            acc["g_b"] += grad_mag(base).double().sum(dim=(0, 1)).cpu()
            acc["g_t"] += grad_mag(truth).double().sum(dim=(0, 1)).cpu()
            n += pred.shape[0] * pred.shape[1]
        px, pxc = n * H * H, n * core * core
        res = {}
        for j, v in enumerate(OCN):
            r = {}
            for scope, sfx, cnt in (("all", "", px), ("core", "_c", pxc)):
                m = float(np.sqrt(acc["se_m" + sfx][j] / cnt)) * stds[j]
                b = float(np.sqrt(acc["se_b" + sfx][j] / cnt)) * stds[j]
                r[scope] = {"rmse_baseline": b, "rmse_model": m, "skill": 1 - (m / b) ** 2 if b > 0 else float("nan")}
            r["all"]["bias_model"] = float(acc["bias_m"][j] / px) * stds[j]
            r["all"]["sharp_model"] = float(acc["g_m"][j] / acc["g_t"][j])
            r["all"]["sharp_baseline"] = float(acc["g_b"][j] / acc["g_t"][j])
            res[v] = r
        out[name] = res
    net.train()
    return out


def print_tables(results):
    for name, res in results.items():
        print(f"\n=== {name}: missing day(s) {list(SCENARIOS[name])} | ocean RMSE in physical units ===")
        for scope in ("all", "core"):
            print(f"--- {scope} ---")
            print(f"{'variable':16s} {'baseline':>11s} {'model':>11s} {'skill':>8s}" +
                  (f" {'bias':>10s} {'sharp(model)':>13s} {'sharp(base)':>12s}" if scope == "all" else ""))
            for v in OCN:
                m = res[v][scope]
                row = f"{v:16s} {m['rmse_baseline']:11.4g} {m['rmse_model']:11.4g} {m['skill']:+8.3f}"
                if scope == "all":
                    row += f" {m['bias_model']:10.3g} {m['sharp_model']:13.3f} {m['sharp_baseline']:12.3f}"
                print(row)
    print("\nskill = 1 - MSE_model / MSE_baseline (baseline = temporal interpolation / persistence). sharpness 1 = as sharp as the truth.")


def write_metrics(results, path_json, path_csv):
    with open(path_json, "w") as f:
        json.dump(results, f, indent=2)
    with open(path_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "variable", "scope", "rmse_baseline", "rmse_model", "skill"])
        for s, res in results.items():
            for v in OCN:
                for scope in ("all", "core"):
                    m = res[v][scope]
                    w.writerow([s, v, scope, m["rmse_baseline"], m["rmse_model"], m["skill"]])


@torch.no_grad()
def plot_examples(net, data, vars_list, stats, scales, scenarios, n_plot, device, out_dir, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(data), size=min(n_plot, len(data)), replace=False))
    net.eval()
    n = 0
    for name in scenarios:
        miss = SCENARIOS[name]
        sc = torch.tensor(scales[name], dtype=torch.float32, device=device)
        t = miss[0]                                            # first missing day is the one drawn
        for i in idx:
            win = torch.from_numpy(data[i:i + 1]).to(device)
            pred, base, truth = predict_ocean(net, win, miss, sc, vars_list)
            k = 0                                              # index of day t inside the missing days
            fig, axes = plt.subplots(len(OCN), 5, figsize=(17, 3.1 * len(OCN)))
            for r, v in enumerate(OCN):
                tr, pr, ba = (denormalize(a[0, k, r].cpu().numpy().astype(np.float64), v, stats) for a in (truth, pred, base))
                vmin, vmax = np.percentile(tr, 1), np.percentile(tr, 99)
                emax = max(np.abs(pr - tr).max(), np.abs(ba - tr).max(), 1e-12)
                panels = [(tr, "truth", "viridis", vmin, vmax), (pr, "model", "viridis", vmin, vmax),
                          (ba, "baseline", "viridis", vmin, vmax), (pr - tr, "model - truth", "RdBu_r", -emax, emax),
                          (ba - tr, "baseline - truth", "RdBu_r", -emax, emax)]
                for c, (a, title, cmap, lo, hi) in enumerate(panels):
                    im = axes[r, c].imshow(a, origin="lower", cmap=cmap, vmin=lo, vmax=hi)
                    axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
                    if r == 0:
                        axes[r, c].set_title(title)
                    if c == 0:
                        axes[r, c].set_ylabel(v)
                    plt.colorbar(im, ax=axes[r, c], fraction=0.046, pad=0.02)
            fig.suptitle(f"{name} | missing day {t} | window {i}")
            fig.tight_layout()
            fig.savefig(out_dir / f"ocean_{name}_win{i}.png", dpi=110)
            plt.close(fig)
            n += 1
    net.train()
    return n


# ------------------------------------------------------------------------------------------------------------------
# inference (drop-in for the old evaluate_sample)
# ------------------------------------------------------------------------------------------------------------------
def load_ocean_model(path=None, device=cfg.DEVICE):
    ck = torch.load(path or CKPT, map_location=device, weights_only=False)
    net = OceanUNet(ck["c_in"], N_FRAMES * len(OCN), ck["base"], ck.get("bottleneck", 0), ck.get("residual", True),
                    ck.get("mask_atm", False), ck.get("use_atm", True)).to(device)
    net.load_state_dict(ck["model_state"])
    return net.eval(), ck


class OceanImputer:
    def __init__(self, checkpoint=None, device=cfg.DEVICE):
        """checkpoint: path to checkpoint_ocean_unet.pt (default: the one of the current --tag)"""
        self.device = device
        self.net, self.ck = load_ocean_model(checkpoint, device)
        self.stats, self.vars_list, self.scales = self.ck["stats"], self.ck["vars_list"], self.ck["scales"]

    @torch.no_grad()
    def predict(self, window, missing: Union[int, Sequence[int]] = (1,)) -> torch.Tensor:
        """window: one dataset window (T, C, H, W), normalised (e.g. eval_dataset[i][0]); missing: day index/indices of the
        OCEAN to predict. -> (T, C, H, W) normalised: input everywhere, except the ocean channels of the missing days."""
        missing = tuple(sorted({int(missing)} if isinstance(missing, int) else {int(m) for m in missing}))
        if missing not in SCENARIOS.values():
            raise ValueError(f"missing={missing} not supported for {N_FRAMES} days; use one of {list(SCENARIOS.values())}")
        name = next(k for k, v in SCENARIOS.items() if v == missing)
        win = torch.as_tensor(np.asarray(window), dtype=torch.float32).unsqueeze(0).to(self.device)
        sc = torch.tensor(self.scales[name], dtype=torch.float32, device=self.device)
        pred, _, _ = predict_ocean(self.net, win, missing, sc, self.vars_list)
        out = win.clone()
        io = [self.vars_list.index(v) for v in OCN]
        for k, t in enumerate(missing):
            out[0, t, io] = pred[0, k]
        return out[0].cpu()

    def to_physical(self, x: torch.Tensor, msl_hpa: bool = True) -> np.ndarray:
        a = x.detach().cpu().numpy().astype(np.float64)
        out = np.empty_like(a)
        for i, v in enumerate(self.vars_list):
            out[..., i, :, :] = denormalize(a[..., i, :, :], v, self.stats)
            if v == "msl" and msl_hpa:
                out[..., i, :, :] /= 100.0
        return out


def evaluate_sample(imputer: OceanImputer, window, target_frame_idx: Optional[int] = 1,
                    missing: Optional[Union[int, Sequence[int]]] = None) -> torch.Tensor:
    """Drop-in replacement of the old evaluate_sample: -> (T, C, H, W) normalised, all days."""
    return imputer.predict(window, missing if missing is not None else target_frame_idx)


@torch.no_grad()
def fill_ocean_of_prediction(imputer: OceanImputer, true_tchw, pred_tchw,
                             missing: Union[int, Sequence[int]] = 1) -> torch.Tensor:
    """Hybrid for the case where the WHOLE day is missing (all 7 variables), e.g. predicted by the latent diffusion.
    true_tchw : (T, C, H, W) normalised true window. Only its OBSERVED days are used.
    pred_tchw : (T, C, H, W) normalised full prediction of another model (atmosphere + ocean).
    -> copy of pred_tchw where the ocean channels of the missing day(s) come from the ocean model.
    No leak: the true ocean AND the true atmosphere of the missing days are replaced before calling the model
    (atmosphere <- the other model's prediction, ocean <- 0, which the model ignores). Other days/channels of pred_tchw
    are left untouched."""
    missing = [int(missing)] if isinstance(missing, int) else sorted(int(m) for m in missing)
    vl = imputer.vars_list
    io, ia = [vl.index(v) for v in OCN], [vl.index(v) for v in ATM]
    to_cpu = lambda a: (a.detach().cpu() if torch.is_tensor(a) else torch.as_tensor(np.asarray(a))).float()
    win, pred = to_cpu(true_tchw).clone(), to_cpu(pred_tchw)
    for t in missing:
        win[t, ia] = pred[t, ia]
        win[t, io] = 0.0
    ocean = imputer.predict(win, missing)
    out = pred.clone()
    for t in missing:
        out[t, io] = ocean[t, io]
    return out


# ------------------------------------------------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="train,eval", help="comma list of: train, eval")
    p.add_argument("--force", action="store_true", help="retrain even if a checkpoint exists")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base", type=int, default=48, help="U-Net width")
    p.add_argument("--no-residual", action="store_true", help="ablation: predict the full field, not the correction")
    p.add_argument("--bottleneck", type=int, default=0, help="ablation: no skips, squeeze to K channels at 1/8 resolution")
    p.add_argument("--no-atm", action="store_true",
                   help="ocean-only model: the wind and msl are NOT inputs (the baseline is still built from the ocean days)")
    p.add_argument("--mask-atm-missing", action="store_true",
                   help="strict forecast/hindcast: hide the atmosphere of the missing days too (default: all days' atmosphere is given)")
    p.add_argument("--tag", default="", help="suffix of the output folder, e.g. _fullfield (one folder per variant)")
    p.add_argument("--core", type=int, default=32, help="side (pixels) of the central storm-core box")
    p.add_argument("--split", default="val", choices=["val", "test"],
                   help="eval split. Compare variants / decide on VAL; run --split test once at the end for the report")
    p.add_argument("--scenarios", help="eval scenarios (default all): " + ",".join(SCENARIOS))
    p.add_argument("--n-plot", type=int, default=10, help="figures per plotted scenario (metrics use ALL windows of the split)")
    args = p.parse_args(argv)
    stages = [s.strip() for s in args.stage.split(",")]
    from ..dataset import CyclonePatchDataset
    device = cfg.DEVICE
    set_tag(args.tag)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[Ocean] device {device} | stages {stages} | output {OUT_DIR}")

    train_ds = None
    if "train" in stages:
        if CKPT.exists() and not args.force:
            print(f"[Ocean] checkpoint exists -> training skipped ({CKPT}); use --force to retrain")
        else:
            train_ds = CyclonePatchDataset(cfg.TRAIN_DIR, variables=cfg.VARS_LIST)
            val_ds = CyclonePatchDataset(cfg.VAL_DIR, variables=cfg.VARS_LIST, stats_dict=train_ds.stats)
            train(train_ds.data, val_ds.data, train_ds.stats, list(cfg.VARS_LIST), args.epochs, args.batch_size,
                  args.lr, args.base, device, args.core, args.bottleneck, not args.no_residual, args.mask_atm_missing,
                  not args.no_atm)
    if "eval" in stages:
        net, ck = load_ocean_model(CKPT, device)
        print(f"[Ocean] model: residual={ck.get('residual', True)} bottleneck={ck.get('bottleneck', 0) or 'off'} "
              f"mask_atm={ck.get('mask_atm', False)} use_atm={ck.get('use_atm', True)} (epoch {ck['epoch']}) | "
              f"evaluating on the {args.split.upper()} set")
        split_dir = cfg.VAL_DIR if args.split == "val" else cfg.TEST_DIR
        ev_ds = CyclonePatchDataset(split_dir, variables=cfg.VARS_LIST, stats_dict=ck["stats"])
        scen = args.scenarios.split(",") if args.scenarios else list(SCENARIOS)
        res = evaluate(net, ev_ds.data, ck["vars_list"], scen, ck["scales"], std_vector(ck["stats"]), args.core, device)
        print_tables(res)
        write_metrics(res, OUT_DIR / f"ocean_eval_metrics_{args.split}.json", OUT_DIR / f"ocean_eval_metrics_{args.split}.csv")
        if args.n_plot > 0:
            k = plot_examples(net, ev_ds.data, ck["vars_list"], ck["stats"], ck["scales"],
                              [s for s in PLOT_SCENARIOS if s in scen], args.n_plot, device, OUT_DIR / f"plots_{args.split}")
            print(f"[Ocean] {k} figures -> {OUT_DIR / f'plots_{args.split}'}")
        print(f"[Ocean] metrics -> {OUT_DIR / f'ocean_eval_metrics_{args.split}.json'}")
    print("\n[Ocean] done")


if __name__ == "__main__":
    main()
