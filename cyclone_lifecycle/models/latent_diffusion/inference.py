"""
inference.py - predict missing day(s) of ONE latent window file, for plotting / comparing with other models.

    # once, outside the loop (as before):
    vae, stats, vars_list = load_vae(vae_ckpt_path, device)
    denoiser, schedule, normalizer, dckpt = load_trained_diffusion(diffusion_ckpt_path, device)
    imputer = LatentImputer(vae, stats, vars_list, denoiser, schedule, normalizer, dckpt)
    #   (or simply LatentImputer() to load from the paths in config.py)
    payload = torch.load(latent_file, weights_only=False)       # one latent_track*.pt file, as before
    x_hat = evaluate_sample(imputer, payload, target_frame_idx=1)          # (3, C, H, W), normalised pixel space
    x_hat = evaluate_sample(imputer, payload, missing=(2,))                # forecast t2 from t0, t1
    x_hat = evaluate_sample(imputer, payload, missing=(1, 2))              # t1 and t2 from t0 only
    x_phys = imputer.to_physical(x_hat)                                    # numpy, physical units (msl in hPa)

The returned tensor always has all days: missing days are the diffusion prediction, observed days are the VAE decoding
of their cached latents. Take x_hat[list(missing)] for the predicted days only.
"""
from typing import Optional, Sequence, Union

from ... import config as cfg
import numpy as np
import torch
from .diffusion import ddim_sample, load_trained_diffusion
from .latents import augment_with_patch_mean
from .vae import load_vae

from ...dataset import denormalize


class LatentImputer:
    def __init__(self, vae=None, stats=None, vars_list=None, denoiser=None, schedule=None, normalizer=None,
                 diffusion_ckpt=None, vae_checkpoint=cfg.VAE_CHECKPOINT,
                 diffusion_checkpoint=cfg.DIFFUSION_CHECKPOINT, device=cfg.DEVICE):
        """
        Pass the models you already loaded (nothing is read from disk):
            vae, stats, vars_list = load_vae(vae_ckpt_path, device)
            denoiser, schedule, normalizer, dckpt = load_trained_diffusion(diffusion_ckpt_path, device)
            imputer = LatentImputer(vae, stats, vars_list, denoiser, schedule, normalizer, dckpt)
        or pass nothing and it loads from the checkpoint paths (do that once, outside any loop).
        """
        self.device = device
        if vae is None:
            vae, stats, vars_list = load_vae(vae_checkpoint, device)
        if denoiser is None:
            denoiser, schedule, normalizer, diffusion_ckpt = load_trained_diffusion(diffusion_checkpoint, device)
        self.vae, self.stats, self.vars_list = vae.eval(), stats, list(vars_list)
        self.denoiser, self.schedule, self.normalizer = denoiser.eval(), schedule, normalizer
        self.ckpt = diffusion_ckpt or {}
        self.prediction = self.ckpt.get("prediction", "eps")

    # -- decoding -----------------------------------------------------------------------------------------
    def _decode(self, z_norm: torch.Tensor) -> torch.Tensor:
        """(F, C', h, w) normalised latents -> (F, Cv, H, W) in normalised pixel space (same space as the dataset)."""
        z = self.normalizer.denormalize(z_norm)
        pm = None
        if self.vae.patch_transform is not None:               # anomaly mode: last Cv channels are patch-mean maps
            Cv = len(self.vars_list)
            z, pm = z[:, :-Cv], z[:, -Cv:].mean(dim=(-2, -1))
        x = self.vae.decode(z)
        return x if pm is None else self.vae.patch_transform.merge(x, pm)

    # -- prediction ---------------------------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, payload: dict, missing: Union[int, Sequence[int]] = (1,), ddim_steps=cfg.DDIM_STEPS,
                eta=cfg.DDIM_ETA, n_samples=1, reduce="first", seed: Optional[int] = 42) -> torch.Tensor:
        """
        payload : one latent window file (dict with 'z' (F, Lc, h, w), and 'patch_mean' (F, C) in anomaly mode)
        missing : day index or indices to predict (any non-empty proper subset of 0..F-1)
        reduce  : 'first' = one sample (sharp), 'mean' = mean of n_samples decoded predictions (lower RMSE, smoother),
                  'all' = tensor (n_samples, F, C, H, W)
        """
        missing = [missing] if isinstance(missing, int) else sorted(set(int(m) for m in missing))
        z = payload["z"].float()
        Fr = z.shape[0]
        assert 0 < len(missing) < Fr and all(0 <= m < Fr for m in missing), f"missing={missing} invalid for {Fr} days"
        assert Fr == self.denoiser.n_frames, f"window has {Fr} days, the denoiser was trained on {self.denoiser.n_frames}"
        pm = payload.get("patch_mean")
        if (pm is None) != (self.vae.patch_transform is None):
            raise RuntimeError("payload and VAE disagree about anomaly mode: rebuild the latents with the current VAE")

        z_aug = augment_with_patch_mean(z, None if pm is None else pm.float())             # (F, C', h, w)
        z_n = self.normalizer.normalize(z_aug.to(self.device)).unsqueeze(0)                # (1, F, C', h, w)
        miss = torch.zeros(1, Fr, dtype=torch.bool, device=self.device)
        miss[0, missing] = True
        z_obs = z_n.masked_fill(miss.view(1, Fr, 1, 1, 1), 0.0).repeat(n_samples, 1, 1, 1, 1)

        gen = None
        if seed is not None:
            gen = torch.Generator(self.device)
            gen.manual_seed(seed)
        z_hat = ddim_sample(self.denoiser, self.schedule, z_obs, miss.repeat(n_samples, 1), ddim_steps, eta, gen,
                            prediction=self.prediction)                                    # (n, F, C', h, w)
        x = torch.stack([self._decode(s) for s in z_hat]).cpu()                            # (n, F, Cv, H, W)
        if reduce == "all":
            return x
        return x.mean(0) if reduce == "mean" else x[0]

    # -- units --------------------------------------------------------------------------------------------
    def to_physical(self, x: torch.Tensor, msl_hpa: bool = True) -> np.ndarray:
        """(..., C, H, W) normalised pixel space -> numpy in physical units (msl in hPa if msl_hpa)."""
        a = x.detach().cpu().numpy().astype(np.float64)
        out = np.empty_like(a)
        for i, v in enumerate(self.vars_list):
            out[..., i, :, :] = denormalize(a[..., i, :, :], v, self.stats)
            if v == "msl" and msl_hpa:
                out[..., i, :, :] /= 100.0
        return out


def evaluate_sample(imputer: LatentImputer, payload: dict, target_frame_idx: Optional[int] = 1,
                    missing: Optional[Union[int, Sequence[int]]] = None, ddim_steps=cfg.DDIM_STEPS, eta=cfg.DDIM_ETA,
                    n_samples=1, reduce="first", seed: Optional[int] = 42) -> torch.Tensor:
    """Drop-in replacement of the old evaluate_sample: -> (F, C, H, W) normalised pixel space, all days.
    Use `missing=` for several days; `target_frame_idx` is kept for the single-day case."""
    miss = missing if missing is not None else target_frame_idx
    return imputer.predict(payload, miss, ddim_steps, eta, n_samples, reduce, seed)
# """
# inference.py - predict missing day(s) of ONE latent window file, for plotting / comparing with other models.

#     imputer = LatentImputer()                                   # loads VAE + diffusion once
#     payload = torch.load(latent_file, weights_only=False)       # one latent_track*.pt file, as before
#     x_hat = evaluate_sample(imputer, payload, target_frame_idx=1)          # (3, C, H, W), normalised pixel space
#     x_hat = evaluate_sample(imputer, payload, missing=(2,))                # forecast t2 from t0, t1
#     x_hat = evaluate_sample(imputer, payload, missing=(1, 2))              # t1 and t2 from t0 only
#     x_phys = imputer.to_physical(x_hat)                                    # numpy, physical units (msl in hPa)

# The returned tensor always has all days: missing days are the diffusion prediction, observed days are the VAE decoding
# of their cached latents. Take x_hat[list(missing)] for the predicted days only.
# """
# from typing import Optional, Sequence, Union

# import numpy as np
# import torch

# import config as cfg
# from data import denormalize
# from diffusion import ddim_sample, load_trained_diffusion
# from latents import augment_with_patch_mean
# from vae import load_vae


# class LatentImputer:
#     def __init__(self, vae_checkpoint=cfg.VAE_CHECKPOINT, diffusion_checkpoint=cfg.DIFFUSION_CHECKPOINT,
#                  device=cfg.DEVICE):
#         self.device = device
#         self.vae, self.stats, self.vars_list = load_vae(vae_checkpoint, device)
#         self.denoiser, self.schedule, self.normalizer, self.ckpt = load_trained_diffusion(diffusion_checkpoint, device)
#         self.prediction = self.ckpt.get("prediction", "eps")

#     # -- decoding -----------------------------------------------------------------------------------------
#     def _decode(self, z_norm: torch.Tensor) -> torch.Tensor:
#         """(F, C', h, w) normalised latents -> (F, Cv, H, W) in normalised pixel space (same space as the dataset)."""
#         z = self.normalizer.denormalize(z_norm)
#         pm = None
#         if self.vae.patch_transform is not None:               # anomaly mode: last Cv channels are patch-mean maps
#             Cv = len(self.vars_list)
#             z, pm = z[:, :-Cv], z[:, -Cv:].mean(dim=(-2, -1))
#         x = self.vae.decode(z)
#         return x if pm is None else self.vae.patch_transform.merge(x, pm)

#     # -- prediction ---------------------------------------------------------------------------------------
#     @torch.no_grad()
#     def predict(self, payload: dict, missing: Union[int, Sequence[int]] = (1,), ddim_steps=cfg.DDIM_STEPS,
#                 eta=cfg.DDIM_ETA, n_samples=1, reduce="first", seed: Optional[int] = 42) -> torch.Tensor:
#         """
#         payload : one latent window file (dict with 'z' (F, Lc, h, w), and 'patch_mean' (F, C) in anomaly mode)
#         missing : day index or indices to predict (any non-empty proper subset of 0..F-1)
#         reduce  : 'first' = one sample (sharp), 'mean' = mean of n_samples decoded predictions (lower RMSE, smoother),
#                   'all' = tensor (n_samples, F, C, H, W)
#         """
#         missing = [missing] if isinstance(missing, int) else sorted(set(int(m) for m in missing))
#         z = payload["z"].float()
#         Fr = z.shape[0]
#         assert 0 < len(missing) < Fr and all(0 <= m < Fr for m in missing), f"missing={missing} invalid for {Fr} days"
#         assert Fr == self.denoiser.n_frames, f"window has {Fr} days, the denoiser was trained on {self.denoiser.n_frames}"
#         pm = payload.get("patch_mean")
#         if (pm is None) != (self.vae.patch_transform is None):
#             raise RuntimeError("payload and VAE disagree about anomaly mode: rebuild the latents with the current VAE")

#         z_aug = augment_with_patch_mean(z, None if pm is None else pm.float())             # (F, C', h, w)
#         z_n = self.normalizer.normalize(z_aug.to(self.device)).unsqueeze(0)                # (1, F, C', h, w)
#         miss = torch.zeros(1, Fr, dtype=torch.bool, device=self.device)
#         miss[0, missing] = True
#         z_obs = z_n.masked_fill(miss.view(1, Fr, 1, 1, 1), 0.0).repeat(n_samples, 1, 1, 1, 1)

#         gen = None
#         if seed is not None:
#             gen = torch.Generator(self.device)
#             gen.manual_seed(seed)
#         z_hat = ddim_sample(self.denoiser, self.schedule, z_obs, miss.repeat(n_samples, 1), ddim_steps, eta, gen,
#                             prediction=self.prediction)                                    # (n, F, C', h, w)
#         x = torch.stack([self._decode(s) for s in z_hat]).cpu()                            # (n, F, Cv, H, W)
#         if reduce == "all":
#             return x
#         return x.mean(0) if reduce == "mean" else x[0]

#     # -- units --------------------------------------------------------------------------------------------
#     def to_physical(self, x: torch.Tensor, msl_hpa: bool = True) -> np.ndarray:
#         """(..., C, H, W) normalised pixel space -> numpy in physical units (msl in hPa if msl_hpa)."""
#         a = x.detach().cpu().numpy().astype(np.float64)
#         out = np.empty_like(a)
#         for i, v in enumerate(self.vars_list):
#             out[..., i, :, :] = denormalize(a[..., i, :, :], v, self.stats)
#             if v == "msl" and msl_hpa:
#                 out[..., i, :, :] /= 100.0
#         return out


# def evaluate_sample(imputer: LatentImputer, payload: dict, target_frame_idx: Optional[int] = 1,
#                     missing: Optional[Union[int, Sequence[int]]] = None, ddim_steps=cfg.DDIM_STEPS, eta=cfg.DDIM_ETA,
#                     n_samples=1, reduce="first", seed: Optional[int] = 42) -> torch.Tensor:
#     """Drop-in replacement of the old evaluate_sample: -> (F, C, H, W) normalised pixel space, all days.
#     Use `missing=` for several days; `target_frame_idx` is kept for the single-day case."""
#     miss = missing if missing is not None else target_frame_idx
#     return imputer.predict(payload, miss, ddim_steps, eta, n_samples, reduce, seed)
