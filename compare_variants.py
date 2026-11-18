"""
compare_variants.py - tables AND plots of the RMSE (physical units) of the interpolation baseline and every ocean-model
variant, read from the ocean_eval_metrics_<split>.json that `ocean_model.py --stage eval` already wrote.

    python compare_variants.py                                  # tables, val, all variants found
    python compare_variants.py --plot                           # + figures for the middle day (mid) and a skill heatmap
    python compare_variants.py --plot --plot-scenarios mid,last,first
    python compare_variants.py --split test --scope core --plot

It only reads files: no model, no data. Variants whose folder or json is missing are skipped (and listed).
Folder of a variant = <root>/ocean_residual<tag>/  (root = your CYCLONE_OUT_DIR, i.e. cfg.CHECKPOINT_DIR).
Figures go to <root>/comparison_plots/ (or --outdir):
    <split>_<scope>_<scenario>_rmse.png      absolute RMSE per variable (own units), interpolation in grey
    <split>_<scope>_<scenario>_relative.png  RMSE / interpolation RMSE, all variables on one axis (1 = interpolation)
    <split>_<scope>_skill_heatmap.png        skill of every variant in every scenario (blue = better than the baseline)
"""
import argparse
import csv
import json
from pathlib import Path

import config as cfg

OCN = ["avg_tos", "avg_zos", "avg_sos", "avg_mlotst125"]
UNITS = {"avg_tos": "°C", "avg_zos": "m", "avg_sos": "psu", "avg_mlotst125": "m"}
# tag -> short description. Add your own tags here or pass --tags.
VARIANTS = {
    "": "A  residual, U-Net with skips, ocean+atm",
    "_fullfield": "B  full field, U-Net with skips, ocean+atm",
    "_fullfield_bn8": "C  full field, no skips + 8-ch bottleneck",
    "_residual_bn8": "D  residual, no skips + 8-ch bottleneck",
    "_residual_noatm": "E  residual, U-Net with skips, ocean only",
    "_fullfield_noatm": "F  full field, U-Net with skips, ocean only",
}
SCENARIOS = ["mid", "last", "first", "last2", "first2", "ends"]
SCEN_TEXT = {
    "mid": "middle day missing (baseline: linear interpolation)",
    "last": "last day missing (baseline: persistence of the nearest day)",
    "first": "first day missing (baseline: persistence of the nearest day)",
    "last2": "last two days missing (baseline: persistence)",
    "first2": "first two days missing (baseline: persistence)",
    "ends": "first and last day missing (baseline: persistence of the middle day)",
}

# ---- colours: validated categorical palette (light surface), fixed order = fixed colour per variant ----------------
SURFACE, INK, INK2, GRID, BASE_GRAY = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8d8c87"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DIV_LOW, DIV_MID, DIV_HIGH = "#e34948", "#f0efec", "#2a78d6"          # diverging: worse (red) <- baseline -> better (blue)


def letter(tag):
    return VARIANTS[tag].split()[0] if tag in VARIANTS else (tag.strip("_") or "A")


def description(tag):
    return VARIANTS[tag][3:] if tag in VARIANTS else f"tag {tag!r}"


def color_of(tag, extra=[]):
    """the colour follows the variant, never its position in the current selection"""
    if tag in VARIANTS:
        return SLOTS[list(VARIANTS).index(tag)]
    if tag not in extra:
        extra.append(tag)
    return SLOTS[min(len(VARIANTS) + extra.index(tag), len(SLOTS) - 1)]


def load(root: Path, tag: str, split: str):
    path = root / f"ocean_residual{tag}" / f"ocean_eval_metrics_{split}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ------------------------------------------------------------------------------------------------------------------
# plots
# ------------------------------------------------------------------------------------------------------------------
def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=12, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _legend(fig, found, with_baseline=True, swatches=True):
    """swatches=False -> a plain text key (letters only), for figures where colour does not identify the variant"""
    import matplotlib.patches as mp
    from matplotlib.lines import Line2D
    if swatches:
        handles = [mp.Patch(facecolor=BASE_GRAY, label="interpolation / persistence (baseline)")] if with_baseline else []
        handles += [mp.Patch(facecolor=color_of(t), label=f"{letter(t)}  {description(t)}") for t in found]
    else:
        handles = [Line2D([], [], linestyle="none", label=f"{letter(t)}  {description(t)}") for t in found]
    leg = fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=13,
                     bbox_to_anchor=(0.5, -0.005), handlelength=0 if not swatches else 2.0)
    for txt in leg.get_texts():
        txt.set_color(INK2)
    return len(handles)


def _frame_rows(n_items):
    return 0.045 * ((n_items + 1) // 2) + 0.03          # figure fraction taken by a 2-column legend


def plot_rmse(found, ref, scen, scope, split, outdir):
    """absolute RMSE in physical units: one small panel per variable (each in its own units)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tags = list(found)
    fig, axes = plt.subplots(1, len(OCN), figsize=(15, 5.6), facecolor=SURFACE)
    for ax, v in zip(axes, OCN):
        _style(ax)
        vals = [ref[scen][v][scope]["rmse_baseline"]] + [found[t][scen][v][scope]["rmse_model"] for t in tags]
        cols = [BASE_GRAY] + [color_of(t) for t in tags]
        names = ["base"] + [letter(t) for t in tags]
        x = range(len(vals))
        ax.bar(x, vals, width=0.62, color=cols, edgecolor=SURFACE, linewidth=1.2)
        best = 1 + min(range(len(tags)), key=lambda i: vals[1 + i])
        for i in (0, best):                                        # label only the baseline and the best variant
            ax.text(i, vals[i], f"{vals[i]:.3g}", ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_xticks(list(x))
        ax.set_xticklabels(names)
        ax.set_title(f"{v.replace('avg_', '')} ({UNITS[v]})", color=INK, fontsize=13, loc="left")
        ax.set_ylim(0, max(vals) * 1.15)
    fig.suptitle(f"RMSE on {split.upper()} ({scope}) - {SCEN_TEXT.get(scen, scen)}", color=INK, fontsize=14,
                 ha="center") #, x=0.01
    n = _legend(fig, tags)
    fig.tight_layout(rect=(0, _frame_rows(n), 1, 0.95))
    path = outdir / f"{split}_{scope}_{scen}_rmse.png"
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_relative(found, ref, scen, scope, split, outdir):
    """RMSE / baseline RMSE: all four variables on ONE axis, 1 = as good as the baseline, lower is better"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tags = list(found)
    fig, ax = plt.subplots(figsize=(11, 5.6), facecolor=SURFACE)
    _style(ax)
    width = min(0.8 / len(tags), 0.16)
    top = 1.0
    for gi, v in enumerate(OCN):
        base = ref[scen][v][scope]["rmse_baseline"]
        rel = [found[t][scen][v][scope]["rmse_model"] / base for t in tags]
        top = max(top, max(rel))
        best = min(range(len(tags)), key=lambda i: rel[i])
        for i, t in enumerate(tags):
            xi = gi + (i - (len(tags) - 1) / 2) * (width + 0.02)
            ax.bar(xi, rel[i], width=width, color=color_of(t), edgecolor=SURFACE, linewidth=1.2)
            if i == best:                                          # label only the best bar of each group
                ax.text(xi, rel[i], f"{rel[i]:.2f}", ha="center", va="bottom", fontsize=12, color=INK)
    ax.axhline(1.0, color=INK2, linewidth=1.2)
    ax.text(1.008, 1.0, "baseline = 1", transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=12,
            color=INK2)                                              # in the right margin, never on top of a bar
    ax.set_xticks(range(len(OCN)))
    ax.set_xticklabels([v.replace("avg_", "") for v in OCN], fontsize=13)
    ax.set_ylabel("RMSE / baseline RMSE   (lower is better)", color=INK2, fontsize=13)
    ax.set_ylim(0, top * 1.12)
    ax.set_title(f"Error relative to the baseline on {split.upper()} ({scope}) - {SCEN_TEXT.get(scen, scen)}",
                 color=INK, fontsize=14, loc="left")
    n = _legend(fig, tags, with_baseline=False)
    fig.tight_layout(rect=(0, _frame_rows(n), 0.93, 1))
    path = outdir / f"{split}_{scope}_{scen}_relative.png"
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_skill_heatmap(found, ref, scenarios, scope, split, outdir):
    """skill = 1 - MSE_model / MSE_baseline for every variant (rows) and scenario (columns), one panel per variable"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb
    cmap = LinearSegmentedColormap.from_list("div", [DIV_LOW, DIV_MID, DIV_HIGH])
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)          # |skill| > 1 saturates; the number is still printed
    tags = list(found)
    scenarios = [s for s in scenarios if s in ref]
    panels = OCN + ["mean"]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 0.62 * len(tags) + 3.0), facecolor=SURFACE,
                             sharey=True)
    for ax, v in zip(axes, panels):
        mat = []
        for t in tags:
            row = []
            for s in scenarios:
                if v == "mean":
                    row.append(sum(found[t][s][o][scope]["skill"] for o in OCN) / len(OCN))
                else:
                    row.append(found[t][s][v][scope]["skill"])
            mat.append(row)
        ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
        for i, row in enumerate(mat):
            for j, val in enumerate(row):
                r, g, b = to_rgb(cmap(norm(val)))
                lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=9, color="#ffffff" if lum < 0.42 else "#0b0b0b")
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(scenarios, rotation=45, ha="right", fontsize=12, color=INK2)
        ax.set_yticks(range(len(tags)))
        ax.set_yticklabels([letter(t) for t in tags], fontsize=13, color=INK)
        ax.set_title("mean of the 4" if v == "mean" else v.replace("avg_", ""), color=INK, fontsize=14, loc="left")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks([k - 0.5 for k in range(1, len(scenarios))], minor=True)
        ax.set_yticks([k - 0.5 for k in range(1, len(tags))], minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)
    fig.suptitle(f"Skill vs the baseline on {split.upper()} ({scope}): blue = better than interpolation/persistence, "
                 f"red = worse, grey = same.  Colour saturates at +-1", #; the printed numbers are exact
                 color=INK, fontsize=14, ha="center") #, x=0.01
    n = _legend(fig, tags, with_baseline=False, swatches=False)
    fig.tight_layout(rect=(0, _frame_rows(n), 1, 0.93))
    path = outdir / f"{split}_{scope}_skill_heatmap.png"
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


# ------------------------------------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(cfg.CHECKPOINT_DIR), help="folder that contains the ocean_residual<tag> folders")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--scope", default="all", choices=["all", "core"])
    p.add_argument("--scenarios", default=",".join(SCENARIOS))
    p.add_argument("--tags", nargs="*", help="tags to compare (default: every known tag whose json exists); use '' for A")
    p.add_argument("--out", help="optional csv path for the long table")
    p.add_argument("--plot", action="store_true", help="also save the figures")
    p.add_argument("--plot-scenarios", default="mid", help="scenarios that get their own rmse/relative figures")
    p.add_argument("--outdir", help="figure folder (default: <root>/comparison_plots)")
    args = p.parse_args()

    root = Path(args.root)
    tags = args.tags if args.tags is not None else list(VARIANTS)
    found, missing = {}, []
    for t in tags:
        r = load(root, t, args.split)
        (found.__setitem__(t, r) if r is not None else missing.append(t))
    if not found:
        raise SystemExit(f"no ocean_eval_metrics_{args.split}.json found under {root}")
    rows_csv = []
    print(f"\nsplit: {args.split.upper()} | scope: {args.scope} | RMSE in physical units (lower is better) | "
          f"skill = 1 - MSE_model/MSE_baseline (higher is better, 0 = same as the baseline)\n")
    for t in found:
        print(f"  {letter(t)}: {description(t)}")
    if missing:
        print("  (skipped, not found: " + ", ".join(repr(t) for t in missing) + ")")

    ref = next(iter(found.values()))
    scen_list = [s for s in args.scenarios.split(",") if s in ref]
    for s in scen_list:
        print(f"\n=== {s} ===")
        names = [letter(t) for t in found]
        print(f"{'RMSE':16s} {'baseline':>11s} " + " ".join(f"{n:>11s}" for n in names))
        for v in OCN:
            base = ref[s][v][args.scope]["rmse_baseline"]
            vals = [r[s][v][args.scope]["rmse_model"] for r in found.values()]
            best = min(vals)
            cells = " ".join(f"{x:11.4g}" + ("*" if x == best else " ") for x in vals)
            print(f"{v:16s} {base:11.4g} {cells}")
            for t, r in found.items():
                m = r[s][v][args.scope]
                rows_csv.append([args.split, args.scope, s, v, letter(t), m["rmse_baseline"], m["rmse_model"], m["skill"]])
        print(f"{'skill':16s} {'':>11s} " + " ".join(f"{n:>11s}" for n in names))
        for v in OCN:
            print(f"{v:16s} {'':>11s} " + " ".join(f"{r[s][v][args.scope]['skill']:+11.3f}" for r in found.values()))
        mean_skill = [sum(r[s][v][args.scope]["skill"] for v in OCN) / len(OCN) for r in found.values()]
        print(f"{'mean skill':16s} {'':>11s} " + " ".join(f"{m:+11.3f}" for m in mean_skill))
    print("\n* = lowest RMSE in the row")
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["split", "scope", "scenario", "variable", "variant", "rmse_baseline", "rmse_model", "skill"])
            w.writerows(rows_csv)
        print(f"[compare] wrote {args.out}")

    if args.plot:
        outdir = Path(args.outdir) if args.outdir else root / "comparison_plots"
        outdir.mkdir(parents=True, exist_ok=True)
        saved = []
        for s in [s for s in args.plot_scenarios.split(",") if s in ref]:
            saved.append(plot_rmse(found, ref, s, args.scope, args.split, outdir))
            saved.append(plot_relative(found, ref, s, args.scope, args.split, outdir))
        saved.append(plot_skill_heatmap(found, ref, scen_list, args.scope, args.split, outdir))
        print("\n[compare] figures:\n  " + "\n  ".join(str(x) for x in saved))


if __name__ == "__main__":
    main()

    
# """
# compare_variants.py - ONE table with the RMSE (physical units) of the interpolation baseline and every ocean-model
# variant, read from the ocean_eval_metrics_<split>.json that `ocean_model.py --stage eval` already wrote.

#     python compare_variants.py                                  # val, scenarios mid,last,first,..., all variants found
#     python compare_variants.py --split test --scenarios mid,last
#     python compare_variants.py --scope core                     # storm core instead of the whole patch

# It only reads files: no model, no data. Variants whose folder or json is missing are skipped (and listed).
# Folder of a variant = <root>/ocean_residual<tag>/  (root = your CYCLONE_OUT_DIR, i.e. cfg.CHECKPOINT_DIR).
# """
# import argparse
# import csv
# import json
# from pathlib import Path

# import config as cfg

# OCN = ["avg_tos", "avg_zos", "avg_sos", "avg_mlotst125"]
# # tag -> short description. Add your own tags here or pass --tags.
# VARIANTS = {
#     "": "A  residual, U-Net with skips, ocean+atm",
#     "_fullfield": "B  full field, U-Net with skips, ocean+atm",
#     "_fullfield_bn8": "C  full field, no skips + 8-ch bottleneck",
#     "_residual_bn8": "D  residual, no skips + 8-ch bottleneck",
#     "_residual_noatm": "E  residual, U-Net with skips, ocean only",
#     "_fullfield_noatm": "F  full field, U-Net with skips, ocean only",
# }
# SCENARIOS = ["mid", "last", "first", "last2", "first2", "ends"]


# def load(root: Path, tag: str, split: str):
#     path = root / f"ocean_residual{tag}" / f"ocean_eval_metrics_{split}.json"
#     if not path.exists():
#         return None
#     with open(path) as f:
#         return json.load(f)


# def main():
#     p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
#     p.add_argument("--root", default=str(cfg.CHECKPOINT_DIR), help="folder that contains the ocean_residual<tag> folders")
#     p.add_argument("--split", default="val", choices=["val", "test"])
#     p.add_argument("--scope", default="all", choices=["all", "core"])
#     p.add_argument("--scenarios", default=",".join(SCENARIOS))
#     p.add_argument("--tags", nargs="*", help="tags to compare (default: every known tag whose json exists); use '' for A")
#     p.add_argument("--out", help="optional csv path for the long table")
#     args = p.parse_args()

#     root = Path(args.root)
#     tags = args.tags if args.tags is not None else list(VARIANTS)
#     found, missing = {}, []
#     for t in tags:
#         r = load(root, t, args.split)
#         (found.__setitem__(t, r) if r is not None else missing.append(t))
#     if not found:
#         raise SystemExit(f"no ocean_eval_metrics_{args.split}.json found under {root}")
#     letter = lambda t: VARIANTS.get(t, f"tag '{t}'").split()[0] if t in VARIANTS else (t or "A")
#     rows_csv = []
#     print(f"\nsplit: {args.split.upper()} | scope: {args.scope} | RMSE in physical units (lower is better) | "
#           f"skill = 1 - MSE_model/MSE_baseline (higher is better, 0 = same as the baseline)\n")
#     for t in found:
#         print(f"  {letter(t)}: {VARIANTS.get(t, 'tag ' + repr(t))[3:] if t in VARIANTS else t}")
#     if missing:
#         print("  (skipped, not found: " + ", ".join(repr(t) for t in missing) + ")")

#     for s in args.scenarios.split(","):
#         ref = next(iter(found.values()))
#         if s not in ref:
#             continue
#         print(f"\n=== {s} ===")
#         names = [letter(t) for t in found]
#         print(f"{'RMSE':16s} {'baseline':>11s} " + " ".join(f"{n:>11s}" for n in names))
#         for v in OCN:
#             base = ref[s][v][args.scope]["rmse_baseline"]
#             vals = [r[s][v][args.scope]["rmse_model"] for r in found.values()]
#             best = min(vals)
#             cells = " ".join(f"{x:11.4g}" + ("*" if x == best else " ") for x in vals)
#             print(f"{v:16s} {base:11.4g} {cells}")
#             for t, r in found.items():
#                 m = r[s][v][args.scope]
#                 rows_csv.append([args.split, args.scope, s, v, letter(t), m["rmse_baseline"], m["rmse_model"], m["skill"]])
#         print(f"{'skill':16s} {'':>11s} " + " ".join(f"{n:>11s}" for n in names))
#         for v in OCN:
#             print(f"{v:16s} {'':>11s} " + " ".join(f"{r[s][v][args.scope]['skill']:+11.3f}" for r in found.values()))
#         mean_skill = [sum(r[s][v][args.scope]["skill"] for v in OCN) / len(OCN) for r in found.values()]
#         print(f"{'mean skill':16s} {'':>11s} " + " ".join(f"{m:+11.3f}" for m in mean_skill))
#     print("\n* = lowest RMSE in the row")
#     if args.out:
#         with open(args.out, "w", newline="") as f:
#             w = csv.writer(f)
#             w.writerow(["split", "scope", "scenario", "variable", "variant", "rmse_baseline", "rmse_model", "skill"])
#             w.writerows(rows_csv)
#         print(f"[compare] wrote {args.out}")


# if __name__ == "__main__":
#     main()
