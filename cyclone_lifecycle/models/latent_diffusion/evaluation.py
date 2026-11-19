"""
evaluation.py - metrics over the WHOLE test set (streaming, nothing stored) + a few example figures.

For every scenario (which day(s) are missing) the diffusion is compared with
    vae_recon    : decode(encode(truth))   -> the best the latent space can possibly do (oracle floor)
    temporal     : linear interpolation between the observed days (t1) / persistence of the nearest observed day
    climatology  : the train mean (0 in normalised space)  -> no-skill reference
All errors are accumulated on the missing frames only, in normalised units, and converted to physical units
(msl in hPa) at the end.
"""
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ... import config as cfg
from ...dataset import CyclonePatchDataset, denormalize, stats_to_arrays
from ...metrics import DiffusionErrorAccumulator
from .diffusion import ddim_sample, load_trained_diffusion
from .latents import augment_with_patch_mean, load_latent_cache
from .vae import file_stamp, load_vae


# ------------------------------------------------------------------
# baselines / accumulators
# ------------------------------------------------------------------
def temporal_baseline(x: torch.Tensor, obs_idx: Sequence[int], miss_idx: Sequence[int]) -> torch.Tensor:
    """x (B,F,C,H,W). For each missing frame: linear interpolation if observed days exist on both sides,
    otherwise persistence of the nearest observed day. -> (B, len(miss_idx), C, H, W)"""
    out = []
    for k in miss_idx:
        lower = [o for o in obs_idx if o < k]
        upper = [o for o in obs_idx if o > k]
        if lower and upper:
            a, b = max(lower), min(upper)
            w = (k - a) / (b - a)
            out.append((1 - w) * x[:, a] + w * x[:, b])
        else:
            out.append(x[:, min(obs_idx, key=lambda o: (abs(o - k), o))])
    return torch.stack(out, dim=1)


# ------------------------------------------------------------------
# plotting
# ------------------------------------------------------------------
def _phys(field, var, stats):
    out = denormalize(field, var, stats)
    return out / cfg.PHYS_SCALE.get(var, 1.0)


def plot_example(truth, preds: Dict[str, np.ndarray], vars_list, stats, title, save_path):
    """truth (C,H,W) and preds[method] (C,H,W), all normalised. Columns: truth | diffusion | temporal | errors."""
    cols = [("diffusion", "Diffusion"), ("temporal", "Temporal baseline")]
    n = len(vars_list)
    fig, axes = plt.subplots(n, 3 + len(cols), figsize=(3.1 * (3 + len(cols)), 2.5 * n), layout="constrained")
    axes = np.atleast_2d(axes)
    for r, var in enumerate(vars_list):
        t = _phys(truth[r], var, stats)
        fields = [t] + [_phys(preds[m][r], var, stats) for m, _ in cols]
        lo, hi = min(f.min() for f in fields), max(f.max() for f in fields)
        errs = [f - t for f in fields[1:]]
        lim = max(max(np.abs(e).max() for e in errs), 1e-12)
        ims = []
        for c, f in enumerate(fields):
            ims.append(axes[r, c].imshow(f, cmap="viridis", vmin=lo, vmax=hi, origin=cfg.IMSHOW_ORIGIN))
        eims = [axes[r, len(fields) + j].imshow(e, cmap="RdBu_r", vmin=-lim, vmax=lim, origin=cfg.IMSHOW_ORIGIN)
                for j, e in enumerate(errs)]
        fig.colorbar(ims[0], ax=axes[r, :len(fields)].tolist(), fraction=0.025, pad=0.01)
        fig.colorbar(eims[0], ax=axes[r, len(fields):].tolist(), fraction=0.04, pad=0.01)
        axes[r, 0].set_ylabel(var, fontsize=10)
        heads = ["Truth"] + [name for _, name in cols] + [f"{name} - truth" for _, name in cols]
        for c in range(axes.shape[1]):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(heads[c], fontsize=10)
        # the RMSE of each method on this sample, in the error panels
        for j, e in enumerate(errs):
            axes[r, len(fields) + j].set_xlabel(f"RMSE {np.sqrt(np.mean(e ** 2)):.3g}", fontsize=8)
    fig.suptitle(title, fontsize=11)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------
# main evaluation
# ------------------------------------------------------------------
def _fmt(x):
    return "nan" if x != x else f"{x:.4g}"


def print_tables(results, scenarios, vars_list):
    for sc in scenarios:
        methods = list(results[sc])
        print(f"\n=== {sc}: missing day(s) {list(cfg.SCENARIOS[sc])} | RMSE in physical units (msl in hPa) ===")
        print(f"{'variable':<16}" + "".join(f"{m:>15}" for m in methods))
        for v in vars_list:
            print(f"{v:<16}" + "".join(f"{_fmt(results[sc][m]['per_variable'][v]['rmse']):>15}" for m in methods))
        print(f"{'overall (std units)':<16}" + "".join(f"{_fmt(results[sc][m]['overall_rmse_norm']):>15}" for m in methods))
        shown = [m for m in methods if m != "climatology"]
        print(f"--- sharpness = mean|gradient| of prediction / of truth (1 = as sharp as truth, <1 = blurred) ---")
        print(f"{'variable':<16}" + "".join(f"{m:>15}" for m in shown))
        for v in vars_list:
            print(f"{v:<16}" + "".join(f"{_fmt(results[sc][m]['per_variable'][v]['sharpness']):>15}" for m in shown))


@torch.no_grad()
def evaluate_latent_diffusion(vars_list=None, vae_checkpoint=cfg.VAE_CHECKPOINT,
                              diffusion_checkpoint=cfg.DIFFUSION_CHECKPOINT, test_dir=cfg.TEST_DIR,
                              latents_test=cfg.LATENTS_TEST, scenarios: Sequence[str] = cfg.EVAL_SCENARIOS,
                              ddim_steps=cfg.DDIM_STEPS, eta=cfg.DDIM_ETA, n_ensemble=cfg.N_ENSEMBLE,
                              batch_size=cfg.EVAL_BATCH_SIZE, n_plot=cfg.N_PLOT,
                              plot_scenarios: Sequence[str] = cfg.PLOT_SCENARIOS, save_dir=cfg.SAVE_DIR,
                              device=cfg.DEVICE, seed=0, test_dataset: Optional[CyclonePatchDataset] = None,
                              split: str = "test"):
    """split only labels the outputs ('val' -> *_val.json/csv and plots/<split>/); pass the matching dir / latents."""
    scenarios = list(scenarios)
    if split != "test":
        save_dir = Path(save_dir) / split
    print(f"[Eval] split '{split}' | device {device} | scenarios {scenarios} | DDIM steps {ddim_steps} | ensemble {n_ensemble}")

    vae, stats, ck_vars = load_vae(vae_checkpoint, device)
    vars_list = list(vars_list or ck_vars)
    assert vars_list == list(ck_vars), "variable order differs from the VAE checkpoint"
    denoiser, schedule, normalizer, dckpt = load_trained_diffusion(diffusion_checkpoint, device)
    stamp = file_stamp(vae_checkpoint)
    if dckpt.get("vae_stamp") not in (None, stamp):
        print("[Eval] WARNING: the diffusion was trained on latents of a DIFFERENT VAE checkpoint -> retrain it")

    ds = test_dataset or CyclonePatchDataset(test_dir, variables=vars_list, stats_dict=stats)
    cache = load_latent_cache(latents_test)
    if cache.get("vae_stamp") != stamp:
        print("[Eval] WARNING: the test latent cache is older than the VAE checkpoint -> run the 'latents' stage")
    meta: List[dict] = cache["meta"]
    assert len(meta) == len(ds), f"latent cache has {len(meta)} windows, dataset {len(ds)}"
    assert [m["source_file"] for m in meta] == [Path(f).name for f in ds.files], "latent cache / dataset order mismatch"
    mu, pm_cache = cache["mu"], cache.get("patch_mean")
    if (vae.patch_transform is None) != (pm_cache is None):
        raise RuntimeError("VAE anomaly mode and the latent cache disagree: re-run the 'latents' stage")
    print(f"[Eval] anomaly mode: {vae.patch_transform is not None}")
    N, Fr = mu.shape[:2]
    C_vars = len(vars_list)

    _, std_np = stats_to_arrays(stats, vars_list)
    phys_factor = std_np / np.array([cfg.PHYS_SCALE.get(v, 1.0) for v in vars_list])

    rng = np.random.default_rng(seed)
    plot_idx = set(rng.choice(N, size=min(n_plot, N), replace=False).tolist()) if n_plot > 0 else set()
    method_names = ["diffusion"] + (["diffusion_mean"] if n_ensemble > 1 else []) + ["vae_recon", "temporal", "climatology"]
    accs = {sc: {m: DiffusionErrorAccumulator(C_vars) for m in method_names} for sc in scenarios}
    store: Dict[str, Dict[int, dict]] = {sc: {} for sc in plot_scenarios if sc in scenarios}
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    def decode_frames(z_norm, mi):              # (B,F,Lc[+Cv],h,w) normalised latents -> (B, len(mi), Cv, H, W)
        z = normalizer.denormalize(z_norm[:, mi])
        B, nm = z.shape[:2]
        pm = None
        if vae.patch_transform is not None:     # last Cv channels = patch mean maps -> spatial average
            z, pm = z[:, :, :-C_vars], z[:, :, -C_vars:].mean(dim=(-2, -1))
        x = vae.decode(z.reshape(B * nm, *z.shape[2:]))
        x = x.view(B, nm, *x.shape[1:])
        return x if pm is None else vae.patch_transform.merge(x, pm)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B = end - start
        x_true = torch.from_numpy(ds.data[start:end]).to(device)                 # (B,F,Cv,H,W) normalised
        z_true = normalizer.normalize(augment_with_patch_mean(
            mu[start:end].float(), None if pm_cache is None else pm_cache[start:end].float()).to(device))   # (B,F,C',h,w)
        for sc in scenarios:
            mi = list(cfg.SCENARIOS[sc])
            obs_idx = [f for f in range(Fr) if f not in mi]
            miss = torch.zeros(B, Fr, dtype=torch.bool, device=device)
            miss[:, mi] = True
            z_obs = z_true.masked_fill(miss.view(B, Fr, 1, 1, 1), 0.0)

            decoded = [decode_frames(ddim_sample(denoiser, schedule, z_obs, miss, ddim_steps, eta, gen,
                                                 prediction=dckpt.get("prediction", "eps")), mi)
                       for _ in range(n_ensemble)]
            truth = x_true[:, mi]
            preds = {"diffusion": decoded[0],
                     "vae_recon": decode_frames(z_true, mi),
                     "temporal": temporal_baseline(x_true, obs_idx, mi),
                     "climatology": torch.zeros_like(truth)}
            if n_ensemble > 1:
                preds["diffusion_mean"] = torch.stack(decoded).mean(0)
            for m in method_names:
                accs[sc][m].add(preds[m] - truth, preds[m], truth)

            if sc in store:
                for b in range(B):
                    if start + b in plot_idx:
                        store[sc][start + b] = {"truth": truth[b, 0].cpu().numpy(),
                                                "preds": {m: preds[m][b, 0].cpu().numpy() for m in ("diffusion", "temporal")}}
        print(f"[Eval] {end}/{N} windows")

    results = {sc: {m: accs[sc][m].summary(vars_list, phys_factor) for m in method_names} for sc in scenarios}
    for sc in scenarios:
        for m in method_names:
            results[sc][m]["n_windows"] = N
            results[sc][m]["missing_days"] = list(cfg.SCENARIOS[sc])
    print_tables(results, scenarios, vars_list)
    write_metrics(results, vars_list, split)

    for sc, items in store.items():
        mi = cfg.SCENARIOS[sc]
        for idx, d in sorted(items.items()):
            m = meta[idx]
            tstr = (m.get("time_strs") or [None] * Fr)[mi[0]]
            name = f"{cfg.TYPE}_{sc}_idx{idx:05d}_track{m.get('track_id')}_{m.get('basin')}_rank{m.get('rank')}_win{m.get('win_idx')}.png"
            plot_example(d["truth"], d["preds"], vars_list, stats,
                         f"{sc} (missing day {mi[0]}) | window {idx} | track {m.get('track_id')} {m.get('basin')} | {tstr}",
                         Path(save_dir) / name)
    n_fig = sum(len(v) for v in store.values())
    print(f"\n[Eval] {n_fig} figures -> {save_dir}\n[Eval] metrics -> {cfg.metrics_paths(split)[0]}")
    return results


def write_metrics(results, vars_list, split: str = "test"):
    json_path, csv_path = cfg.metrics_paths(split)
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "missing_days", "method", "variable", "rmse", "mae", "bias", "rmse_norm", "sharpness"])
        for sc, d in results.items():
            for m, r in d.items():
                for v in vars_list:
                    s = r["per_variable"][v]
                    w.writerow([sc, list(cfg.SCENARIOS[sc]), m, v, s["rmse"], s["mae"], s["bias"], s["rmse_norm"], s["sharpness"]])
