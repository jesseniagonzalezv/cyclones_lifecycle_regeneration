"""
residual_test.py - is there anything for a model to learn beyond temporal interpolation, and does the OTHER group help?

    r = target(t1) - (target(t0) + target(t2)) / 2          # what interpolation gets wrong at the middle day
A small U-Net (pixel space, no VAE, no diffusion) predicts r. --target chooses the group to predict:
    --target ocean : tos, zos, sos, mlotst   (the other group = atmosphere: 10u, 10v, msl)
    --target atm   : 10u, 10v, msl           (the other group = ocean)
Two runs per target:
    +other : input = target t0, t2  +  the OTHER group at t0, t1, t2   (cross-group: the other group of the missing day IS known)
    alone  : input = target t0, t2 only
It starts exactly at the interpolation (zero-init output), so skill = 1 - MSE_model / MSE_interpolation:
    ~0 -> nothing learnable beyond interpolation | > 0 -> predictable | compare "+other" with "alone": does the other group help?
Reported per variable (physical units, msl in hPa), on the whole patch and on the storm core.
NOTE: "+other" assumes ONLY the target group is missing on that day. If the whole frame is missing (all 7 variables),
the realistic number is "alone" (or "+other" fed with the other group of the neighbouring days only).

    python residual_test.py --target ocean        # train on TRAIN, monitor on VAL, final numbers on TEST
    python residual_test.py --target atm
    python residual_test.py --target atm --epochs 30 --core 32
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from .. import config as cfg

GROUPS = {"atm": ["10u", "10v", "msl"], "ocean": ["avg_tos", "avg_zos", "avg_sos", "avg_mlotst125"]}
OTHER = {"atm": "ocean", "ocean": "atm"}


def conv(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, padding_mode="replicate"), nn.GroupNorm(8, o), nn.SiLU())


class SmallUNet(nn.Module):
    def __init__(self, c_in, c_out, base=32):
        super().__init__()
        self.e1 = nn.Sequential(conv(c_in, base), conv(base, base))
        self.e2 = nn.Sequential(conv(base, 2 * base), conv(2 * base, 2 * base))
        self.e3 = nn.Sequential(conv(2 * base, 4 * base), conv(4 * base, 4 * base))
        self.pool = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.d2 = nn.Sequential(conv(6 * base, 2 * base), conv(2 * base, 2 * base))
        self.d1 = nn.Sequential(conv(3 * base, base), conv(base, base))
        self.out = nn.Conv2d(base, c_out, 1)
        nn.init.zeros_(self.out.weight)                 # starts as "residual = 0" = plain interpolation
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        d2 = self.d2(torch.cat([self.up(e3), e2], 1))
        d1 = self.d1(torch.cat([self.up(d2), e1], 1))
        return self.out(d1)


def make_xy(win, vars_list, tgt, oth, use_other):
    """win: (B, 3, C, H, W), normalised -> (input (B, Cin, H, W), interpolation (B, Ct, H, W), truth (B, Ct, H, W))"""
    it = [vars_list.index(v) for v in tgt]
    io = [vars_list.index(v) for v in oth]
    a0, a1, a2 = win[:, 0, it], win[:, 1, it], win[:, 2, it]
    interp = 0.5 * (a0 + a2)
    parts = [a0, a2]
    if use_other:
        parts += [win[:, t, io] for t in range(3)]
    return torch.cat(parts, 1), interp, a1


@torch.no_grad()
def evaluate(net, data, vars_list, tgt, oth, use_other, device, scales, core, bs=64):
    net.eval()
    H = data.shape[-1]
    lo, hi = (H - core) // 2, (H + core) // 2
    se = {k: torch.zeros(len(tgt), dtype=torch.float64) for k in ("model", "interp", "model_core", "interp_core")}
    n_all = n_core = 0
    for i in range(0, len(data), bs):
        win = torch.from_numpy(data[i:i + bs]).to(device)
        x, interp, truth = make_xy(win, vars_list, tgt, oth, use_other)
        pred = interp + net(x)
        for name, p in (("model", pred), ("interp", interp)):
            e2 = (p - truth) ** 2
            se[name] += e2.sum(dim=(0, 2, 3)).double().cpu()
            se[name + "_core"] += e2[..., lo:hi, lo:hi].sum(dim=(0, 2, 3)).double().cpu()
        n_all += truth.shape[0] * H * H
        n_core += truth.shape[0] * core * core
    net.train()
    out = {}
    for j, v in enumerate(tgt):
        r = {}
        for tag, n in (("all", n_all), ("core", n_core)):
            sfx = "" if tag == "all" else "_core"
            m = (se["model" + sfx][j] / n).sqrt().item()
            b = (se["interp" + sfx][j] / n).sqrt().item()
            r[tag] = {"rmse_interp": b * scales[j], "rmse_model": m * scales[j], "skill": 1 - (m / b) ** 2}
        out[v] = r
    return out


def run(train, val, test, vars_list, stats, target, use_other, epochs, bs, lr, core, device, base=32):
    """monitors on VAL every few epochs; TEST is evaluated once, after the last epoch (nothing is selected on it)"""
    tgt, oth = GROUPS[target], GROUPS[OTHER[target]]
    # RMSE of a normalised difference x std = physical RMSE (the mean cancels); msl Pa -> hPa
    scales = [(stats[v]["std"] if stats[v]["std"] > 1e-6 else 1.0) / cfg.PHYS_SCALE.get(v, 1.0) for v in tgt]
    c_in = 2 * len(tgt) + (3 * len(oth) if use_other else 0)
    net = SmallUNet(c_in, len(tgt), base).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * ((len(train) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    name = f"{target} +{OTHER[target]}" if use_other else f"{target} alone"
    print(f"\n[{name}] input channels {c_in}, parameters {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M")
    t0 = time.time()
    for ep in range(epochs):
        perm = np.random.permutation(len(train))
        tot = 0.0
        for i in range(0, len(train), bs):
            win = torch.from_numpy(train[np.sort(perm[i:i + bs])]).to(device)
            x, interp, truth = make_xy(win, vars_list, tgt, oth, use_other)
            loss = ((interp + net(x) - truth) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * win.shape[0]
        msg = f"  epoch {ep + 1:02d}/{epochs} train MSE {tot / len(train):.5f}"
        if (ep + 1) % max(1, epochs // 5) == 0 or ep + 1 == epochs:
            r = evaluate(net, val, vars_list, tgt, oth, use_other, device, scales, core)
            msg += " | val skill (all) " + " ".join(f"{v.replace('avg_', '')[:6]}={r[v]['all']['skill']:+.3f}" for v in tgt)
        print(msg + f" | {time.time() - t0:.0f}s")
    return evaluate(net, test, vars_list, tgt, oth, use_other, device, scales, core)


def report(results, target):
    tgt = GROUPS[target]
    for scope in ("all", "core"):
        print(f"\n=== TEST, target={target}, {scope}: RMSE in physical units (interpolation -> model), "
              f"skill = 1 - MSE_model/MSE_interp ===")
        print(f"{'variable':16s} {'interp':>10s} " + " ".join(f"{k + ' rmse':>20s} {k + ' skill':>16s}" for k in results))
        for v in tgt:
            row = f"{v:16s} {next(iter(results.values()))[v][scope]['rmse_interp']:10.4g} "
            row += " ".join(f"{r[v][scope]['rmse_model']:20.4g} {r[v][scope]['skill']:+16.3f}" for r in results.values())
            print(row)
    print("\nskill ~ 0: nothing learnable beyond interpolation | > 0: predictable | compare columns: does the other group help?")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=["ocean", "atm"], default="ocean", help="group to predict at the middle day")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--core", type=int, default=32, help="side (pixels) of the central storm-core box")
    p.add_argument("--base", type=int, default=32)
    args = p.parse_args()

    from ..dataset import CyclonePatchDataset
    train_ds = CyclonePatchDataset(cfg.TRAIN_DIR, variables=cfg.VARS_LIST)
    val_ds = CyclonePatchDataset(cfg.VAL_DIR, variables=cfg.VARS_LIST, stats_dict=train_ds.stats)
    test_ds = CyclonePatchDataset(cfg.TEST_DIR, variables=cfg.VARS_LIST, stats_dict=train_ds.stats)
    results = {}
    for use_other in (True, False):
        key = f"+{OTHER[args.target]}" if use_other else "alone"
        results[key] = run(train_ds.data, val_ds.data, test_ds.data, list(cfg.VARS_LIST), train_ds.stats, args.target,
                           use_other, args.epochs, args.batch_size, args.lr, args.core, cfg.DEVICE, args.base)
    report(results, args.target)


if __name__ == "__main__":
    main()
