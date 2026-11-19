"""
evaluate_masked_vae.py
Evaluate the masked cyclone VAE on the test set and compare it with simple baselines.

Scenarios (all masks are seeded, so every method sees exactly the same missing data):
  * block_random : rectangular block (25-50% of H/W) missing in ALL variables, random position
  * block_center : same, but centred on the patch centre (where the eye should be)
  * var:<group>  : one variable unit missing everywhere ("var:10u+10v", "var:msl", ...)
                   10u/10v are masked together, as in training (see VAR_GROUPS)

Methods:
  * VAE     : the model
  * Interp  : Laplace (harmonic) interpolation of the hole from its surroundings. Spatial scenarios only:
              a variable that is missing in the WHOLE patch cannot be interpolated spatially
  * Regr    : OPTIONAL (only if train_ds is passed): linear regression from the other variables, per pixel,
              fitted on the TRAIN set. Variable scenarios only. Not an interpolation.
  * Clim    : climatology = the train mean (0 in normalised space), the no-skill reference

Metrics are computed ONLY on the missing entries: RMSE / MAE / bias (normalised and physical units),
skill = 1 - MSE/MSE_climatology (1 = perfect, 0 = no better than the mean), and the derived 10 m wind
speed when both 10u and 10v are missing.

Outputs in out_dir: metrics.json, metrics.csv, summary.png, examples_spatial.png, examples_variable.png
"""
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ..metrics import MaskedRecoveryAccumulator, summarize_masked_recovery
from ..models.vae_case1_case2 import VAR_GROUPS, CycloneVAE, _random_blocks, build_model_input

IMSHOW_ORIGIN = "lower"   # use "upper" if your latitude axis is stored north -> south

# ---- plot style -----------------------------------------------------------------------------------
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
COL = {"VAE": "#2a78d6", "Interp": "#eb6834", "Regr": "#eb6834", "Clim": "#8a8984"}
LABEL = {"VAE": "VAE", "Interp": "Laplace interpolation",
         "Regr": "Linear regression (other variables)", "Clim": "Climatology (train mean)"}
SPATIAL_SCENARIOS = {"block_random": 0.0, "block_center": 1.0}   # name -> prob. of a centred block


# =====================================================================================================
# BASELINES
# =====================================================================================================
@torch.no_grad()
def laplace_fill(x, mask, n_iter=600, omega=1.9, tol=1e-4):
    """
    Harmonic (Laplace) interpolation: masked pixels are replaced by the solution of
    laplacian(y) = 0 with the observed pixels as boundary condition (smooth interpolation of the
    surroundings, the classic simple inpainting baseline). Red-black SOR, vectorised over (B, C).
    x: (B,C,H,W) data (values under the mask are ignored), mask: (B,C,H,W) bool, True = missing.
    """
    B, C, H, W = x.shape
    obs = ~mask
    cnt = obs.sum(dim=(2, 3), keepdim=True).clamp(min=1)
    fill0 = (x * obs).sum(dim=(2, 3), keepdim=True) / cnt           # start from the observed mean
    y = torch.where(mask, fill0.expand_as(x), x)

    ii = torch.arange(H, device=x.device).view(H, 1)
    jj = torch.arange(W, device=x.device).view(1, W)
    red = ((ii + jj) % 2) == 0
    colours = [mask & red, mask & ~red]

    snap = y.clone()
    for it in range(1, n_iter + 1):
        for upd in colours:
            p = F.pad(y, (1, 1, 1, 1), mode="replicate")
            nb = 0.25 * (p[:, :, :-2, 1:-1] + p[:, :, 2:, 1:-1] + p[:, :, 1:-1, :-2] + p[:, :, 1:-1, 2:])
            y = torch.where(upd, y + omega * (nb - y), y)
        if it % 25 == 0:
            if ((y - snap).abs() * mask).max() < tol:
                break
            snap = y.clone()
    return y


def fit_linear_baselines(train_ds, vars_list, groups, max_samples=1500, pix_per_sample=300, seed=0):
    """For each group g: least squares  x_g(pixel) = W . x_others(pixel) + b  on random TRAIN pixels."""
    rng = np.random.default_rng(seed)
    N = len(train_ds)
    idx = rng.choice(N, size=min(N, max_samples), replace=False)
    cols = []
    for i in idx:
        x = train_ds[int(i)][0].numpy()                      # (C,H,W)
        _, H, W = x.shape
        ys = rng.integers(0, H, pix_per_sample)
        xs = rng.integers(0, W, pix_per_sample)
        cols.append(x[:, ys, xs].T)                          # (pix, C)
    X = np.concatenate(cols).astype(np.float64)

    C = len(vars_list)
    models = {}
    for gi, g in enumerate(groups):
        M = [vars_list.index(n) for n in g]
        O = [c for c in range(C) if c not in M]
        A = np.hstack([X[:, O], np.ones((len(X), 1))])
        coef, *_ = np.linalg.lstsq(A, X[:, M], rcond=None)   # (|O|+1, |M|)
        models[gi] = (O, M, coef)
    return models


def predict_spatial(model, x, mask, interp_iters):
    xm = x.masked_fill(mask, 0.0)
    vae, _, _ = model(build_model_input(xm, mask), sample=False)
    return {"VAE": vae, "Interp": laplace_fill(xm, mask, n_iter=interp_iters), "Clim": torch.zeros_like(x)}


def predict_variable(model, x, mask, reg):
    xm = x.masked_fill(mask, 0.0)
    vae, _, _ = model(build_model_input(xm, mask), sample=False)
    out = {"VAE": vae, "Clim": torch.zeros_like(x)}
    if reg is not None:
        O, M, coef = reg
        Wm = torch.tensor(coef[:-1], dtype=x.dtype, device=x.device)          # (|O|, |M|)
        bm = torch.tensor(coef[-1], dtype=x.dtype, device=x.device)           # (|M|,)
        pr = torch.zeros_like(x)
        pr[:, M] = torch.einsum("bohw,om->bmhw", x[:, O], Wm) + bm.view(1, -1, 1, 1)
        out["Regr"] = pr
    return out


def print_table(results, scen, methods, vars_list):
    print(f"\n=== {scen} ===")
    head = f"{'variable':<16}" + "".join(f"{'RMSE ' + m:>13}" for m in methods)
    head += f"{'VAE RMSE phys':>15}{'VAE skill':>11}{'VAE bias':>10}"
    print(head)
    for v in vars_list:
        if v not in results[scen]["VAE"]["per_variable"]:
            continue
        row = f"{v:<16}"
        for m in methods:
            row += f"{results[scen][m]['per_variable'][v]['rmse_norm']:>13.4f}"
        s = results[scen]["VAE"]["per_variable"][v]
        row += f"{s['rmse_phys']:>15.4g}{s['skill']:>11.3f}{s['bias_norm']:>10.3f}"
        print(row)
    ws = {m: results[scen][m]["wind_speed"] for m in methods}
    if ws["VAE"] is not None:
        print("wind speed (derived, physical) RMSE: " + "  ".join(f"{m}={ws[m]['rmse_phys']:.4g}" for m in methods))


# =====================================================================================================
# PLOTS
# =====================================================================================================
def _style_axes(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_summary(results, vars_list, groups, has_regr, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), layout="constrained", facecolor=SURF)
    x = np.arange(len(vars_list))

    def bars(ax, series, title):
        k = len(series)
        w = 0.8 / k
        for j, (m, vals) in enumerate(series):
            ax.bar(x + (j - (k - 1) / 2) * w, vals, width=w, color=COL[m], edgecolor=SURF,
                   linewidth=2, label=LABEL[m])
        ax.set_xticks(x)
        ax.set_xticklabels(vars_list)
        ax.set_ylabel("RMSE, normalised units\n(lower is better)", color=INK2, fontsize=9)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False, fontsize=9,
                  labelcolor=INK2)
        _style_axes(ax)

    get = lambda sc, m, v: results[sc][m]["per_variable"].get(v, {}).get("rmse_norm", np.nan)
    bars(axes[0], [(m, [get("block_random", m, v) for v in vars_list]) for m in ("VAE", "Interp", "Clim")],
         "Missing block, random position, all variables")
    bars(axes[1], [(m, [get("block_center", m, v) for v in vars_list]) for m in ("VAE", "Interp", "Clim")],
         "Missing block centred on the eye, all variables")

    var_scen = {}
    for g in groups:
        for n in g:
            var_scen["var:" + "+".join(g)] = None
            var_scen[n] = "var:" + "+".join(g)
    meths = ("VAE", "Regr", "Clim") if has_regr else ("VAE", "Clim")
    bars(axes[2], [(m, [get(var_scen[v], m, v) for v in vars_list]) for m in meths],
         "Variable missing everywhere (u/v together)")
    fig.savefig(out_path, dpi=150, facecolor=SURF)
    plt.close(fig)


def _row_fields(arrs):
    """shared colour limits for a row of fields"""
    lo = min(np.nanmin(a) for a in arrs)
    hi = max(np.nanmax(a) for a in arrs)
    return lo, hi


def _err_lim(errs):
    return max(max(np.nanpercentile(np.abs(e), 99) for e in errs), 1e-12)


def plot_example_spatial(model, x1, mean, std, vars_list, interp_iters, seed, out_path):
    """x1: (1,C,H,W) normalised. Block centred on the eye, all variables."""
    _, C, H, W = x1.shape
    gen = torch.Generator(device=x1.device)
    gen.manual_seed(seed)
    block = _random_blocks(1, H, W, x1.device, gen, center_prob=1.0)
    mask = block.unsqueeze(1).expand(1, C, H, W).contiguous()
    preds = predict_spatial(model, x1, mask, interp_iters)

    phys = lambda t: (t[0].double() * std.view(C, 1, 1) + mean.view(C, 1, 1)).cpu().numpy()
    truth, vae, itp = phys(x1), phys(preds["VAE"]), phys(preds["Interp"])
    m = mask[0].cpu().numpy()

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d0cfca")
    titles = ["Truth", "Input (missing = grey)", "VAE", "Laplace interpolation", "VAE - truth", "Interp - truth"]
    fig, axes = plt.subplots(C, 6, figsize=(15, 2.2 * C), layout="constrained", facecolor=SURF)
    for r, v in enumerate(vars_list):
        inp = np.where(m[r], np.nan, truth[r])
        lo, hi = _row_fields([truth[r], vae[r], itp[r]])
        e1, e2 = vae[r] - truth[r], itp[r] - truth[r]
        lim = _err_lim([e1[m[r]], e2[m[r]]])
        ims = []
        for c, (a, kw) in enumerate([(truth[r], {}), (inp, {}), (vae[r], {}), (itp[r], {})]):
            ims.append(axes[r, c].imshow(a, origin=IMSHOW_ORIGIN, cmap=cmap, vmin=lo, vmax=hi))
        eims = [axes[r, 4].imshow(e1, origin=IMSHOW_ORIGIN, cmap="RdBu_r", vmin=-lim, vmax=lim),
                axes[r, 5].imshow(e2, origin=IMSHOW_ORIGIN, cmap="RdBu_r", vmin=-lim, vmax=lim)]
        fig.colorbar(ims[0], ax=axes[r, :4].tolist(), fraction=0.02, pad=0.01)
        fig.colorbar(eims[0], ax=axes[r, 4:].tolist(), fraction=0.03, pad=0.01)
        axes[r, 0].set_ylabel(v, color=INK, fontsize=10)
        for c in range(6):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=10, color=INK)
    fig.savefig(out_path, dpi=130, facecolor=SURF)
    plt.close(fig)


def plot_example_variable(model, x1, regress, groups, vars_list, mean, std, out_path, focus=("10u", "msl")):
    """Whole group missing. Rows: every variable of the focus groups (+ derived wind speed)."""
    _, C, H, W = x1.shape
    phys = lambda t: (t[0].double() * std.view(C, 1, 1) + mean.view(C, 1, 1)).cpu().numpy()
    truth = phys(x1)
    rows = []   # (label, truth, {method: field})
    for f in focus:
        if f not in vars_list:
            continue
        gi = next(i for i, g in enumerate(groups) if f in g)
        chans = [vars_list.index(n) for n in groups[gi]]
        mask = torch.zeros_like(x1, dtype=torch.bool)
        mask[:, chans] = True
        preds = predict_variable(model, x1, mask, None if regress is None else regress[gi])
        meths = [m for m in ("VAE", "Regr") if m in preds]
        pp = {m: phys(preds[m]) for m in meths}
        for c in chans:
            rows.append((vars_list[c] + "  (missing)", truth[c], {m: pp[m][c] for m in meths}))
        if "10u" in groups[gi] and "10v" in groups[gi]:
            iu, iv = vars_list.index("10u"), vars_list.index("10v")
            rows.append(("wind speed (derived)", np.hypot(truth[iu], truth[iv]),
                         {m: np.hypot(pp[m][iu], pp[m][iv]) for m in meths}))
    if not rows:
        return
    meths = list(rows[0][2].keys())
    ncol = 1 + 2 * len(meths)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(3.0 * ncol, 2.3 * len(rows)), layout="constrained",
                             facecolor=SURF, squeeze=False)
    names = {"VAE": "VAE", "Regr": "Linear regression"}
    for r, (lab, tr, d) in enumerate(rows):
        lo, hi = _row_fields([tr] + list(d.values()))
        im = axes[r, 0].imshow(tr, origin=IMSHOW_ORIGIN, cmap="viridis", vmin=lo, vmax=hi)
        for j, m in enumerate(meths):
            axes[r, 1 + j].imshow(d[m], origin=IMSHOW_ORIGIN, cmap="viridis", vmin=lo, vmax=hi)
        lim = _err_lim([d[m] - tr for m in meths])
        for j, m in enumerate(meths):
            eim = axes[r, 1 + len(meths) + j].imshow(d[m] - tr, origin=IMSHOW_ORIGIN, cmap="RdBu_r",
                                                      vmin=-lim, vmax=lim)
        fig.colorbar(im, ax=axes[r, :1 + len(meths)].tolist(), fraction=0.03, pad=0.01)
        fig.colorbar(eim, ax=axes[r, 1 + len(meths):].tolist(), fraction=0.04, pad=0.01)
        axes[r, 0].set_ylabel(lab, color=INK, fontsize=10)
        heads = ["Truth"] + [names[m] for m in meths] + [names[m] + " - truth" for m in meths]
        for c in range(ncol):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(heads[c], fontsize=10, color=INK)
    fig.savefig(out_path, dpi=130, facecolor=SURF)
    plt.close(fig)


# =====================================================================================================
# MAIN ENTRY
# =====================================================================================================
@torch.no_grad()
def pick_example_idx(ds, mean, std, vars_list, batch_size, device):
    """index of the test sample with the strongest 10 m wind (physical units)."""
    if "10u" not in vars_list or "10v" not in vars_list:
        return 0
    iu, iv = vars_list.index("10u"), vars_list.index("10v")
    best, best_idx, off = -1.0, 0, 0
    for x, _, _ in DataLoader(ds, batch_size=batch_size, shuffle=False):
        x = x.to(device).double()
        sp = torch.hypot(x[:, iu] * std[iu] + mean[iu], x[:, iv] * std[iv] + mean[iv]).amax(dim=(1, 2))
        k = int(sp.argmax())
        if float(sp[k]) > best:
            best, best_idx = float(sp[k]), off + k
        off += x.shape[0]
    return best_idx


@torch.no_grad()
def run_evaluation(checkpoint_path, test_ds, train_ds=None, out_dir="eval_out", batch_size=64,
                   device=None, seed=1234, interp_iters=600, example_idx=None):
    """
    test_ds / train_ds must be built with stats_dict=ckpt["stats"] so they are normalised exactly like in
    training. train_ds is only used to fit the regression baseline (skipped if None).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vars_list = list(ckpt["vars_list"])
    C = len(vars_list)
    if hasattr(test_ds, "variables"):
        assert list(test_ds.variables) == vars_list, "dataset variable order != checkpoint variable order"
    groups = ckpt.get("var_groups", VAR_GROUPS)
    latent = ckpt["model_state"]["to_mu.weight"].shape[0]
    model = CycloneVAE(in_channels=ckpt.get("in_channels", 2 * C), out_channels=ckpt.get("out_channels", C),
                       latent_channels=latent).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    std_np = np.array([ckpt["stats"][v]["std"] if ckpt["stats"][v]["std"] > 1e-6 else 1.0 for v in vars_list])
    mean_np = np.array([ckpt["stats"][v]["mean"] for v in vars_list])
    std_t = torch.tensor(std_np, dtype=torch.float64, device=device)
    mean_t = torch.tensor(mean_np, dtype=torch.float64, device=device)
    uv = (vars_list.index("10u"), vars_list.index("10v")) if "10u" in vars_list and "10v" in vars_list else None

    regress = fit_linear_baselines(train_ds, vars_list, groups) if train_ds is not None else None
    has_regr = regress is not None

    var_scen = {"var:" + "+".join(g): gi for gi, g in enumerate(groups)}
    all_scen = list(SPATIAL_SCENARIOS) + list(var_scen)
    methods_of = {sc: (["VAE", "Interp", "Clim"] if sc in SPATIAL_SCENARIOS
                       else ["VAE"] + (["Regr"] if has_regr else []) + ["Clim"]) for sc in all_scen}
    accs = {sc: {m: MaskedRecoveryAccumulator(C) for m in methods_of[sc]} for sc in all_scen}
    gens = {}
    for sc in SPATIAL_SCENARIOS:
        gens[sc] = torch.Generator(device=device)
        gens[sc].manual_seed(seed)

    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    for x, _, _ in loader:
        x = x.to(device)
        B, _, H, W = x.shape
        for sc, cp in SPATIAL_SCENARIOS.items():
            block = _random_blocks(B, H, W, device, gens[sc], center_prob=cp)
            mask = block.unsqueeze(1).expand(B, C, H, W).contiguous()
            for m, p in predict_spatial(model, x, mask, interp_iters).items():
                accs[sc][m].add(p, x, mask, mean_t, std_t, uv)
        for sc, gi in var_scen.items():
            mask = torch.zeros_like(x, dtype=torch.bool)
            mask[:, [vars_list.index(n) for n in groups[gi]]] = True
            for m, p in predict_variable(model, x, mask, regress[gi] if has_regr else None).items():
                accs[sc][m].add(p, x, mask, mean_t, std_t, uv)

    results = {
        sc: {method: summarize_masked_recovery(acc, vars_list, std_np)
             for method, acc in accs[sc].items()}
        for sc in all_scen
    }
    for sc in all_scen:
        print_table(results, sc, methods_of[sc], vars_list)

    # ---- files ---------------------------------------------------------------------------------
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    fields = ["scenario", "method", "variable", "n", "rmse_norm", "mae_norm", "bias_norm", "skill",
              "rmse_phys", "mae_phys", "bias_phys"]
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, restval="")
        wr.writeheader()
        for sc, d in results.items():
            for m, r in d.items():
                for v, s in r["per_variable"].items():
                    wr.writerow({"scenario": sc, "method": m, "variable": v, **s})
                if r["wind_speed"] is not None:
                    wr.writerow({"scenario": sc, "method": m, "variable": "wind_speed (derived)",
                                 **r["wind_speed"]})

    plot_summary(results, vars_list, groups, has_regr, out_dir / "summary.png")

    if example_idx is None:
        example_idx = pick_example_idx(test_ds, mean_t, std_t, vars_list, batch_size, device)
    x1 = test_ds[example_idx][0].unsqueeze(0).to(device)
    plot_example_spatial(model, x1, mean_t, std_t, vars_list, interp_iters, seed,
                         out_dir / "examples_spatial.png")
    plot_example_variable(model, x1, regress, groups, vars_list, mean_t, std_t,
                          out_dir / "examples_variable.png")
    print(f"\nExample sample index: {example_idx}\nFiles written to {out_dir.resolve()}")
    return results


if __name__ == "__main__":
    # >>> adapt this import to the file where your dataset class and constants live <<<
    from ..models.vae_case1_case2 import (
        CHECKPOINT,
        TEST_DIR,
        TRAIN_DIR,
        VARS_LIST,
        CyclonePatchDataset,
    )

    ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    test_ds = CyclonePatchDataset(TEST_DIR, variables=VARS_LIST, stats_dict=ck["stats"])
    run_evaluation(CHECKPOINT, test_ds, out_dir="eval_out")

    # Optional extra baseline for the "variable missing everywhere" case (linear regression from the
    # other variables, fitted on the train set). Not an interpolation; uncomment only if you want it:
    # train_ds = CyclonePatchDataset(TRAIN_DIR, variables=VARS_LIST, stats_dict=ck["stats"])
    # run_evaluation(CHECKPOINT, test_ds, train_ds, out_dir="eval_out")