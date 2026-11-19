"""
vae.py - per-frame VAE (your architecture), its loss, training and loading.
The VAE sees every day independently; all temporal reasoning is left to the diffusion model.
"""
import os
from typing import Optional, Tuple

from ... import config as cfg
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ...dataset import PatchMeanTransform


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


class CycloneVAE_Diffusion(nn.Module):
    def __init__(self, in_channels: int = len(cfg.VARS_LIST), latent_channels: int = cfg.LATENT_CHANNELS):
        super().__init__()
        self.patch_transform = None      # PatchMeanTransform when the VAE works on anomalies (not a parameter)
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
    """MSE (mean over pixels) + beta * KL (summed over latent dims, averaged over samples). Checked: correct."""
    def __init__(self, beta: float = cfg.VAE_BETA):
        super().__init__()
        self.beta = beta
        self.mse = nn.MSELoss()

    def forward(self, x_hat, x, mu, logvar):
        recon = self.mse(x_hat, x)
        kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=(1, 2, 3))
        kl = torch.mean(kl_per_sample)
        return recon + self.beta * kl, recon.detach(), kl.detach()


def _frames(model, window, device):
    """(B,T,C,H,W) window -> (B*T,C,H,W) VAE input: anomalies if the model has a patch-mean transform, else the data."""
    B, T, C, H, W = window.shape
    x = window.to(device)
    if model.patch_transform is not None:
        x, _ = model.patch_transform.split(x)
    return x.reshape(B * T, C, H, W)


@torch.no_grad()
def _validate(model, loader, criterion, device):
    model.eval()
    tot_recon, n = 0.0, 0
    for window, _, _ in loader:
        B, T, C, H, W = window.shape
        x = _frames(model, window, device)
        x_hat, mu, logvar = model(x)
        _, recon, _ = criterion(x_hat, x, mu, logvar)
        tot_recon += recon.item() * B * T
        n += B * T
    model.train()
    return tot_recon / max(n, 1)


def train_vae(model: nn.Module, dataloader: DataLoader, epochs: int, stats: dict, vars_list,
              latent_channels: int, val_loader: Optional[DataLoader] = None,
              checkpoint_path=cfg.VAE_CHECKPOINT, device=cfg.DEVICE, lr=cfg.VAE_LR, beta=cfg.VAE_BETA) -> nn.Module:
    print("\n[VAE Training] Starting...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = VAELoss(beta=beta)

    model.train()
    for epoch in range(epochs):
        epoch_loss, epoch_recon = 0.0, 0.0
        for window, _, _ in dataloader:
            B, T, C, H, W = window.shape
            x = _frames(model, window, device)

            optimizer.zero_grad()
            x_hat, mu, logvar = model(x)
            loss, recon, kl = criterion(x_hat, x, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * (B * T)
            epoch_recon += recon.item() * (B * T)

        n = len(dataloader.dataset) * T
        msg = f"Epoch {epoch+1:02d}/{epochs:02d} | Loss: {epoch_loss / n:.5f} | Recon MSE: {epoch_recon / n:.5f}"
        val = None
        if val_loader is not None:
            val = _validate(model, val_loader, criterion, device)
            msg += f" | Test recon MSE: {val:.5f}"
        print(msg)

        # the test curve is only MONITORED; the checkpoint is always the latest epoch, so the test set is
        # never used to choose anything
        torch.save({"model_state": model.state_dict(), "stats": stats, "vars_list": list(vars_list),
                    "latent_channels": latent_channels, "epoch": epoch + 1, "val_recon_mse": val,
                    "anomaly_scale": None if model.patch_transform is None else model.patch_transform.scale.cpu()},
                   checkpoint_path)
    print(f"[VAE Training] Saved checkpoint to {checkpoint_path}")
    return model


def load_vae(checkpoint_path=cfg.VAE_CHECKPOINT, device=cfg.DEVICE):
    """-> (vae in eval mode, stats, vars_list)"""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vars_list = list(ckpt.get("vars_list", cfg.VARS_LIST))
    latent_channels = ckpt.get("latent_channels", ckpt["model_state"]["to_mu.weight"].shape[0])
    vae = CycloneVAE_Diffusion(in_channels=len(vars_list), latent_channels=latent_channels).to(device)
    vae.load_state_dict(ckpt["model_state"])
    if ckpt.get("anomaly_scale") is not None:
        vae.patch_transform = PatchMeanTransform(ckpt["anomaly_scale"]).to(device)
    vae.eval()
    return vae, ckpt["stats"], vars_list


def file_stamp(path) -> float:
    """modification time, used to detect that a cache/checkpoint is older than the VAE it came from"""
    return os.path.getmtime(path) if os.path.exists(path) else -1.0
