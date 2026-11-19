import numpy as np
import torch

# from difussion_3days import VARS_LIST, impute_missing_timestep

VARS_LIST = ["10u", "10v", "msl", "avg_tos", "avg_zos", "avg_sos", "avg_mlotst125"]
LATENT_CHANNELS = 128  # Matched with evaluation script expectation


def impute_missing_timestep(denoiser, schedule, context, device, scale_factor=1.0, ddim=True, ddim_steps=200):
    denoiser.eval()
    B, _, H, W = context.shape
    shape = (B, LATENT_CHANNELS, H, W)
    x = torch.randn(shape, device=device) * scale_factor

    timesteps = torch.linspace(schedule.T - 1, 0, ddim_steps, dtype=torch.long, device=device) if ddim else range(schedule.T - 1, -1, -1)

    for i in timesteps:
        t = torch.full((B,), i, device=device, dtype=torch.long)
        t_norm = t.float() / schedule.T
        pred_noise = denoiser(x, t_norm, context)

        alpha_bar = schedule.alpha_bar[i]
        alpha = schedule.alpha[i]

        if ddim:
            # DDIM step update
            pred_x0 = (x - torch.sqrt(1.0 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)
            if i > 0:
                alpha_bar_prev = schedule.alpha_bar[timesteps[torch.where(timesteps == i)[0] + 1]] if (timesteps == i).nonzero()[0] < len(timesteps) - 1 else torch.tensor(1.0, device=device)
                dir_xt = torch.sqrt(1.0 - alpha_bar_prev) * pred_noise
                x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt
            else:
                x = pred_x0
        else:
            # DDPM step update
            beta = schedule.beta[i]
            noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
            x = 1.0 / torch.sqrt(alpha) * (x - (beta / torch.sqrt(1.0 - alpha_bar)) * pred_noise) + torch.sqrt(beta) * noise
            
    return x

def check_times_match(payload, metadata, sample_idx):
    # ----Evaluate the sample and get predicted and target times
    payload_times = np.asarray(payload["time_strs"], dtype="datetime64[m]")
    metadata_times = np.asarray(metadata.time_strs, dtype="datetime64[m]")

    print(f"Payload latent times for sample {sample_idx}: {payload_times}")
    print(f"Metadata ds times for sample {sample_idx}: {metadata_times}")
    if payload_times is None:
        raise ValueError(f"No times found in payload for sample {sample_idx}")
    if metadata_times is None:
        raise ValueError(f"No times found in metadata for sample {sample_idx}")
    if len(payload_times) != len(metadata_times):
        raise ValueError(
            f"Mismatch in number of times for sample {sample_idx}: "
            f"payload={len(payload_times)}, metadata={len(metadata_times)}"
        )
    if not all(p == m for p, m in zip(payload_times, metadata_times)):
        raise ValueError(
            f"Mismatch in times for sample {sample_idx}: "
            f"payload={payload_times}, metadata={metadata_times}"
        )
    
    assert payload_times is not None, f"No times found in payload for sample {sample_idx}"
    assert metadata_times is not None, f"No times found in metadata for sample {sample_idx}"

    assert len(payload_times) == len(metadata_times), (
        f"Mismatch in number of times for sample {sample_idx}: "
        f"payload={len(payload_times)}, metadata={len(metadata_times)}"
    )

    assert all(
        p == m for p, m in zip(payload_times, metadata_times)
    ), (
        f"Mismatch in times for sample {sample_idx}: "
        f"payload={payload_times}, metadata={metadata_times}"
    )


# replace in the inference function with this evaluate_sample function
def evaluate_sample(stats,vae, denoiser, payload, device, schedule, vars_list=VARS_LIST, scale_factor=1.0, ddim_steps=200, target_frame_idx=1):

    z = payload["z"]  # (3, 128, 16, 16)

    z_prev, z_true, z_next = z[0], z[1], z[2] # !!!! JGV it need to adapt when i calculate other that the middle frame, for example the first frame or the last frame, then the z_prev and z_next will be different
    context = torch.cat([z_prev, z_next], dim=0).unsqueeze(0)
    

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    z_hat = impute_missing_timestep(
        denoiser, 
        schedule, 
        context, 
        device, 
        scale_factor=scale_factor,
        ddim=True,          # <--- Determinist
        ddim_steps=ddim_steps
    )

    print(f"z_true std: {z_true.std().item():.4f}")
    print(f"z_hat std:  {z_hat.std().item():.4f}")

    # x_hat = vae.decode(z_hat).detach().cpu().numpy()[0] # remove the batch dimension:
    x_hat = vae.decode(z_hat).detach()  


    # only for testing!!!!!!!!!!!!!!
    x_hat = x_hat.repeat(3, 1, 1, 1)
    return x_hat