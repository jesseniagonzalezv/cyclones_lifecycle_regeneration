"""
run.py - run any subset of the pipeline. Each stage reads its inputs from disk, so you never repeat earlier work.

    python run.py --stage all                    # vae -> latents -> diffusion -> eval (skips what already exists)
    python run.py --stage diffusion              # only (re)train the diffusion: needs just the latent cache
    python run.py --stage eval                   # only evaluate: whole test set metrics + N_PLOT figures
    python run.py --stage latents,diffusion      # new VAE checkpoint -> recompute latents and retrain diffusion
    python run.py --stage eval --scenarios mid,last --n-ensemble 4 --ddim-steps 50
    python run.py --stage diffusion --force      # retrain even if a checkpoint exists

Stages: vae | latents | diffusion | eval | all
Stage inputs   : vae        <- NetCDF train/test
                 latents    <- VAE checkpoint + NetCDF           (re-done automatically if the VAE checkpoint changed)
                 diffusion  <- latent caches ONLY (no VAE, no NetCDF)
                 eval       <- VAE decoder + diffusion checkpoint + test latents + test NetCDF
"""
import argparse
from typing import Optional

import config as cfg
import torch
from diffusion import DiffusionSchedule, LatentUNetDenoiser, train_diffusion
from evaluation import evaluate_latent_diffusion
from latents import (
    CachedLatentDataset,
    build_latent_cache,
    check_split_leakage,
    latent_cache_exists,
    latent_cache_is_current,
    load_latent_cache,
)
from torch.utils.data import DataLoader
from vae import CycloneVAE_Diffusion, load_vae, train_vae

from data import CyclonePatchDataset, PatchMeanTransform, fit_anomaly_scale

STAGES = ["vae", "latents", "diffusion", "eval"]


class LazyData:
    """Loads the NetCDF datasets only when a stage really needs them, and only once."""
    def __init__(self):
        self._train: Optional[CyclonePatchDataset] = None
        self._test: Optional[CyclonePatchDataset] = None

    @property
    def train(self):
        if self._train is None:
            self._train = CyclonePatchDataset(cfg.TRAIN_DIR, variables=cfg.VARS_LIST)
        return self._train

    @property
    def test(self):
        if self._test is None:
            self._test = CyclonePatchDataset(cfg.TEST_DIR, variables=cfg.VARS_LIST, stats_dict=self.train.stats)
        return self._test


def stage_vae(args, data):
    if cfg.VAE_CHECKPOINT.exists() and not args.force:
        print(f"[vae] checkpoint exists -> skipped ({cfg.VAE_CHECKPOINT})")
        return
    vae = CycloneVAE_Diffusion(in_channels=len(cfg.VARS_LIST), latent_channels=cfg.LATENT_CHANNELS).to(cfg.DEVICE)
    if cfg.ANOMALY_MODE:
        vae.patch_transform = PatchMeanTransform(fit_anomaly_scale(data.train.data, cfg.VARS_LIST)).to(cfg.DEVICE)
    print(f"[vae] anomaly mode: {cfg.ANOMALY_MODE}")
    train_vae(vae, DataLoader(data.train, batch_size=cfg.VAE_BATCH_SIZE, shuffle=True),
              epochs=args.epochs_vae or cfg.VAE_EPOCHS, stats=data.train.stats, vars_list=cfg.VARS_LIST,
              latent_channels=cfg.LATENT_CHANNELS,
              val_loader=DataLoader(data.test, batch_size=cfg.VAE_BATCH_SIZE, shuffle=False))


def stage_latents(args, data):
    if not cfg.VAE_CHECKPOINT.exists():
        raise FileNotFoundError("No VAE checkpoint: run --stage vae first")
    vae, _, _ = load_vae(cfg.VAE_CHECKPOINT, cfg.DEVICE)
    for path, ds_fn, fit in ((cfg.LATENTS_TRAIN, lambda: data.train, True), (cfg.LATENTS_TEST, lambda: data.test, False)):
        if latent_cache_is_current(path) and not args.force:
            print(f"[latents] {path} is up to date -> skipped")
        else:
            build_latent_cache(vae, ds_fn(), path, cfg.DEVICE, fit_normalizer=fit)
    check_split_leakage(load_latent_cache(cfg.LATENTS_TRAIN)["meta"], load_latent_cache(cfg.LATENTS_TEST)["meta"])


def stage_diffusion(args, data):
    if cfg.DIFFUSION_CHECKPOINT.exists() and not args.force:
        print(f"[diffusion] checkpoint exists -> skipped ({cfg.DIFFUSION_CHECKPOINT})")
        return
    if not latent_cache_exists(cfg.LATENTS_TRAIN):
        raise FileNotFoundError("No latent cache: run --stage latents first")
    train_ds = CachedLatentDataset(cfg.LATENTS_TRAIN, sample_latents=cfg.SAMPLE_LATENTS_TRAIN)
    val_ds = CachedLatentDataset(cfg.LATENTS_TEST, normalizer=train_ds.normalizer) if latent_cache_exists(cfg.LATENTS_TEST) else None
    C = train_ds[0].shape[1]            # latent channels (+ patch-mean channels in anomaly mode)
    denoiser = LatentUNetDenoiser(C, cfg.N_FRAMES, cfg.DIFFUSION_BASE_CHANNELS).to(cfg.DEVICE)
    print(f"[diffusion] denoiser parameters: {sum(p.numel() for p in denoiser.parameters()) / 1e6:.1f}M")
    schedule = DiffusionSchedule(cfg.DIFFUSION_STEPS, cfg.DEVICE)
    train_diffusion(denoiser, schedule, train_ds, train_ds.normalizer, val_ds,
                    epochs=args.epochs_diffusion or cfg.DIFFUSION_EPOCHS,
                    extra_meta={"vae_stamp": train_ds.vae_stamp})


def stage_eval(args, data):
    evaluate_latent_diffusion(
        scenarios=args.scenarios.split(",") if args.scenarios else cfg.EVAL_SCENARIOS,
        ddim_steps=args.ddim_steps or cfg.DDIM_STEPS, eta=args.eta if args.eta is not None else cfg.DDIM_ETA,
        n_ensemble=args.n_ensemble or cfg.N_ENSEMBLE, n_plot=cfg.N_PLOT if args.n_plot is None else args.n_plot,
        plot_scenarios=args.plot_scenarios.split(",") if args.plot_scenarios else cfg.PLOT_SCENARIOS,
        batch_size=args.eval_batch_size or cfg.EVAL_BATCH_SIZE,
        test_dataset=data._test)          # reuse the test set if an earlier stage already loaded it


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="all", help="comma list of: vae,latents,diffusion,eval,all")
    p.add_argument("--force", action="store_true", help="redo the selected stages even if their output exists")
    p.add_argument("--epochs-vae", type=int)
    p.add_argument("--epochs-diffusion", type=int)
    p.add_argument("--scenarios", help="eval scenarios, e.g. mid,last,first (default: all)")
    p.add_argument("--plot-scenarios", help="scenarios to plot (default mid,last,first)")
    p.add_argument("--n-plot", type=int, help="figures per scenario (metrics always use the whole test set)")
    p.add_argument("--ddim-steps", type=int)
    p.add_argument("--eta", type=float)
    p.add_argument("--n-ensemble", type=int)
    p.add_argument("--eval-batch-size", type=int)
    args = p.parse_args(argv)

    stages = STAGES if args.stage == "all" else [s.strip() for s in args.stage.split(",")]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        p.error(f"unknown stage(s) {bad}; choose from {STAGES}")
    cfg.ensure_dirs()
    print(f"[run] device {cfg.DEVICE} | stages {stages}")

    data = LazyData()
    fns = {"vae": stage_vae, "latents": stage_latents, "diffusion": stage_diffusion, "eval": stage_eval}
    for s in STAGES:                       # always run in pipeline order
        if s in stages:
            print(f"\n################ stage: {s} ################")
            fns[s](args, data)
    print("\n[run] done")


if __name__ == "__main__":
    main()
