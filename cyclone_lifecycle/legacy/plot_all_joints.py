"""Plotting helpers that wrap VideoMAE's comparison plot for joint use
across the AE_difussion scripts.

This module provides a thin wrapper around the existing
`VIDEOMAE.videomae_with_baseline_comparison.plot_prediction_comparison`
so callers in this package can easily compare Target vs Baseline vs
VideoMAE vs Diffusion/VAE predictions without copying code.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os

import torch


def gather_case3_predictions(
	target_ctwh_norm: torch.Tensor,
	cfg=None,
	device: Optional[torch.device] = None,
	videomae_model: Optional[object] = None,
	vae_model: Optional[torch.nn.Module] = None,
	diffusion_denoiser: Optional[object] = None,
	diffusion_schedule: Optional[object] = None,
	latents_dir: Optional[str] = None,
	norm_stats: Optional[Dict[str, Dict[str, float]]] = None,
	vars_list: Optional[List[str]] = None,
	run_baseline: bool = True,
	labels: Optional[Dict[str, str]] = None,
	target_frame_idx: int = 1,
) -> Tuple[Dict[str, torch.Tensor], List[Tuple[int, int, str]]]:
	"""Collect predictions for Case 3 (missing timestep).

	For diffusion, this will attempt to call `evaluate_diffusion_imputation`
	(from the AE_difussion training scripts) and use the first returned
	example as the prediction for the missing frame.
	"""
	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	preds = {}
	preds[labels.get("target", "Target") if labels else "Target"] = target_ctwh_norm

	# VideoMAE prediction for missing timestep
	if videomae_model is not None and vcomp is not None and cfg is not None:
		recon = vcomp.reconstruct_case3_model_prediction(videomae_model, cfg, target_ctwh_norm, target_frame_idx, device)
		preds[labels.get("videomae", "VideoMAE") if labels else "VideoMAE"] = recon

	# Baseline (temporal linear interpolation)
	if run_baseline and vcomp is not None:
		baseline = vcomp.embed_case3_baseline_prediction(target_ctwh_norm, target_frame_idx)
		preds[labels.get("baseline", "Baseline") if labels else "Baseline"] = baseline

	# VAE: reconstruct all frames via VAE (per-frame autoencoding)
	if vae_model is not None:
		vae_recon = _vae_reconstruct_full_tensor(vae_model, target_ctwh_norm, device)
		preds[labels.get("vae", "VAE") if labels else "VAE"] = vae_recon

	# Diffusion: attempt to call the repo's evaluate_diffusion_imputation helper
	if diffusion_denoiser is not None and diffusion_schedule is not None and latents_dir is not None and vars_list is not None and norm_stats is not None:
		try:
			# Import locally to avoid hard dependency at module import time
			from AE_difussion.train_test_AE_VAE_masked_difussion_3days import evaluate_diffusion_imputation
			# evaluate_diffusion_imputation returns (summary, results)
			summary, results = evaluate_diffusion_imputation(
				vae_model, diffusion_denoiser, diffusion_schedule, latents_dir,
				norm_stats, vars_list, device, n_eval_tracks=1, save_dir=None
			)
			if results:
				example = results[0]
				# example['x_hat'] is numpy array [C, H, W] for the predicted frame
				x_hat = torch.from_numpy(example["x_hat"]).unsqueeze(0)  # [1, C, H, W]
				# Insert into a full-tensor copy of the original: [1, C, T, H, W]
				recon_full = target_ctwh_norm.clone()
				recon_full[:, :, target_frame_idx, :, :] = x_hat.to(recon_full.dtype)
				preds[labels.get("diffusion", "Diffusion") if labels else "Diffusion"] = recon_full
		except Exception:
			# If diffusion evaluation isn't available or fails, skip gracefully
			pass

	# Build rows: one row per variable for the fixed target frame
	rows = vcomp.case3_comparison_rows(cfg, target_frame_idx) if (vcomp is not None and cfg is not None) else [(c, target_frame_idx, str(c)) for c in range(target_ctwh_norm.shape[1])]

	return preds, rows


def plot_case1_joint(
	target_ctwh_norm: torch.Tensor,
	save_path: str,
	cfg=None,
	device: Optional[torch.device] = None,
	videomae_model: Optional[object] = None,
	vae_model: Optional[torch.nn.Module] = None,
	run_baseline: bool = True,
	labels: Optional[Dict[str, str]] = None,
	channel_idx: int = 0,
	metadata: Optional[SampleMetadata] = None,
	norm_stats: Optional[Dict[str, Dict[str, float]]] = None,
) -> None:
	"""Gather Case 1 predictions and produce the comparison plot to `save_path`."""
	preds, rows, pixel_box = gather_case1_predictions(
		target_ctwh_norm, cfg=cfg, device=device, videomae_model=videomae_model,
		vae_model=vae_model, run_baseline=run_baseline, labels=labels, channel_idx=channel_idx,
	)

	# Forward to the plotting helper
	plot_prediction_comparison_joint(
		target_ctwh_norm=target_ctwh_norm,
		predictions=preds,
		rows=rows,
		variable_names=(cfg.variable_names if cfg is not None else []),
		norm_stats=(norm_stats or {}),
		save_path=save_path,
		metadata=metadata,
		highlight_pixel_box=pixel_box,
		title_prefix="Case 1 - ",
	)


def plot_case3_joint(
	target_ctwh_norm: torch.Tensor,
	save_path: str,
	cfg=None,
	device: Optional[torch.device] = None,
	videomae_model: Optional[object] = None,
	vae_model: Optional[torch.nn.Module] = None,
	diffusion_denoiser: Optional[object] = None,
	diffusion_schedule: Optional[object] = None,
	latents_dir: Optional[str] = None,
	norm_stats: Optional[Dict[str, Dict[str, float]]] = None,
	vars_list: Optional[List[str]] = None,
	run_baseline: bool = True,
	labels: Optional[Dict[str, str]] = None,
	target_frame_idx: int = 1,
	metadata: Optional[SampleMetadata] = None,
) -> None:
	"""Gather Case 3 predictions and produce the comparison plot to `save_path`."""
	preds, rows = gather_case3_predictions(
		target_ctwh_norm, cfg=cfg, device=device, videomae_model=videomae_model,
		vae_model=vae_model, diffusion_denoiser=diffusion_denoiser, diffusion_schedule=diffusion_schedule,
		latents_dir=latents_dir, norm_stats=norm_stats, vars_list=vars_list, run_baseline=run_baseline,
		labels=labels, target_frame_idx=target_frame_idx,
	)

	plot_prediction_comparison_joint(
		target_ctwh_norm=target_ctwh_norm,
		predictions=preds,
		rows=rows,
		variable_names=(cfg.variable_names if cfg is not None else (vars_list or [])),
		norm_stats=(norm_stats or {}),
		save_path=save_path,
		metadata=metadata,
		highlight_pixel_box=None,
		title_prefix="Case 3 - ",
	)


def gather_case2_predictions(
	target_ctwh_norm: torch.Tensor,
	cfg=None,
	device: Optional[torch.device] = None,
	videomae_model: Optional[object] = None,
	vae_model: Optional[torch.nn.Module] = None,
	channel_idx: int = 0,
	labels: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, torch.Tensor], List[Tuple[int, int, str]]]:
	"""Collect predictions for Case 2: missing variable (entire channel removed).

	Returns (preds, rows) where rows are one per timestep for `channel_idx`.
	"""
	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	preds = {}
	preds[labels.get("target", "Target") if labels else "Target"] = target_ctwh_norm

	# VideoMAE: corrupt channel and run model
	if videomae_model is not None and vcomp is not None and cfg is not None:
		corrupted = vcomp.zero_out_channel(target_ctwh_norm, channel_idx)
		bool_masked_pos = vcomp.no_mask_mode_b(corrupted.shape[0], cfg.num_patches, device)
		with torch.no_grad():
			logits = vcomp.forward_full_sequence(videomae_model, corrupted, bool_masked_pos)
		reconstructed_tchw = vcomp.unpatchify(logits, cfg.tubelet_size, cfg.patch_size, cfg.num_frames, cfg.image_size, cfg.num_channels)
		reconstructed_ctwh = vcomp.to_project_layout(reconstructed_tchw)
		preds[labels.get("videomae", "VideoMAE") if labels else "VideoMAE"] = reconstructed_ctwh

	# VAE: simple per-frame reconstruction
	if vae_model is not None:
		vae_recon = _vae_reconstruct_full_tensor(vae_model, target_ctwh_norm, device)
		preds[labels.get("vae", "VAE") if labels else "VAE"] = vae_recon

	rows = vcomp.case1_comparison_rows(cfg, channel_idx) if (vcomp is not None and cfg is not None) else [(channel_idx, t, f"t={t}") for t in range(target_ctwh_norm.shape[2])]
	return preds, rows


def plot_case2_joint(
	target_ctwh_norm: torch.Tensor,
	save_path: str,
	cfg=None,
	device: Optional[torch.device] = None,
	videomae_model: Optional[object] = None,
	vae_model: Optional[torch.nn.Module] = None,
	channel_idx: int = 0,
	metadata: Optional[SampleMetadata] = None,
	norm_stats: Optional[Dict[str, Dict[str, float]]] = None,
	labels: Optional[Dict[str, str]] = None,
) -> None:
	preds, rows = gather_case2_predictions(target_ctwh_norm, cfg=cfg, device=device, videomae_model=videomae_model, vae_model=vae_model, channel_idx=channel_idx, labels=labels)

	plot_prediction_comparison_joint(
		target_ctwh_norm=target_ctwh_norm,
		predictions=preds,
		rows=rows,
		variable_names=(cfg.variable_names if cfg is not None else []),
		norm_stats=(norm_stats or {}),
		save_path=save_path,
		metadata=metadata,
		highlight_pixel_box=None,
		title_prefix="Case 2 - ",
	)


# CLI entrypoint — placed at bottom of file for clearer organization


def compare_sample_across_methods(
	cfg,
	sample_idx: int = 0,
	case: int = 3,
	checkpoint_dir: Optional[str] = None,
	diffusion_checkpoint: Optional[str] = None,
	latents_dir: Optional[str] = None,
	images_dir: Optional[str] = None,
	save_filename: Optional[str] = None,
	run_baselines: bool = True,
	target_frame_idx: int = 1,
) -> None:
	"""Run VideoMAE inference and diffusion evaluation for a single sample,
	assemble predictions and produce a comparison plot.

	This convenience function calls `VIDEOMAE.videomae_with_baseline_comparison.run_inference_tests`
	(with `num_samples=sample_idx+1` so the final returned tensors correspond to
	`sample_idx`), then calls `AE_difussion.train_test_AE_VAE_masked_difussion_3days.evaluate_latent_diffusion`
	to obtain diffusion results (expects it to return `results`). It then plots
	Target vs VideoMAE vs Baseline vs Diffusion for the requested `case`.
	"""
	# Lazy imports to avoid heavy import-time dependencies
	try:
		from VIDEOMAE.videomae_with_baseline_comparison import run_inference_tests, ProjectConfig
	except Exception as e:
		raise RuntimeError("Could not import VideoMAE inference utilities: " + str(e))

	try:
		from AE_difussion.train_test_AE_VAE_masked_difussion_3days import evaluate_latent_diffusion
	except Exception:
		evaluate_latent_diffusion = None

	images_dir = images_dir or (os.path.join(getattr(cfg, "output_dir", "."), "comparison_images") if cfg is not None else ".")
	save_filename = save_filename or f"comparison_sample{sample_idx}_case{case}.png"
	save_path = os.path.join(images_dir, save_filename)

	# 1) Run VideoMAE inference for the requested sample index (num_samples = sample_idx+1)
	clean, recon, baseline = run_inference_tests(cfg, checkpoint_dir=checkpoint_dir, num_samples=sample_idx + 1, save_images=False, images_dir=images_dir, run_baselines=run_baselines)

	# clean/recon/baseline are [1,C,T,H,W] tensors (normalized)
	target_ctwh_norm = clean
	videomae_recon = recon
	baseline_recon = baseline

	# 2) Try to obtain diffusion results (if available)
	diffusion_recon_full = None
	if evaluate_latent_diffusion is not None and latents_dir is not None:
		# Call evaluate_latent_diffusion expecting it to return (denoiser, schedule, summary, results)
		try:
			out = evaluate_latent_diffusion(vars_list=getattr(cfg, "variable_names", None), checkpoint=checkpoint_dir, diffusion_checkpoint=diffusion_checkpoint, latents_dir=latents_dir, save_dir=None)
		except TypeError:
			# Some variants may accept positional args differently
			out = evaluate_latent_diffusion()

		# Accept either (denoiser, schedule, summary) or with results appended
		results = None
		if isinstance(out, tuple) or isinstance(out, list):
			# find any element that looks like results (list of dicts containing 'x_hat')
			for item in out:
				if isinstance(item, list) and item and isinstance(item[0], dict) and "x_hat" in item[0]:
					results = item
					break
		if results and len(results) > 0:
			example = results[0]
			x_hat = example.get("x_hat")
			if x_hat is not None:
				# x_hat: numpy [C, H, W]
				x_hat_t = torch.from_numpy(x_hat).unsqueeze(0)
				# build full tensor by inserting predicted frame into target
				recon_full = target_ctwh_norm.clone()
				if case == 3:
					recon_full[:, :, target_frame_idx, :, :] = x_hat_t.to(recon_full.dtype)
				diffusion_recon_full = recon_full

	# 3) Assemble predictions dict
	preds = {"VideoMAE": videomae_recon, "Baseline": baseline_recon}
	if diffusion_recon_full is not None:
		preds["Diffusion"] = diffusion_recon_full

	# 4) Build rows and call plotting
	norm_stats = vcomp.load_norm_stats(cfg.norm_stats_path) if (vcomp is not None and hasattr(cfg, "norm_stats_path")) else {}

	if case == 1:
		rows = vcomp.case1_comparison_rows(cfg, 0)
	else:
		rows = vcomp.case3_comparison_rows(cfg, target_frame_idx)

	plot_prediction_comparison_joint(
		target_ctwh_norm=target_ctwh_norm,
		predictions=preds,
		rows=rows,
		variable_names=cfg.variable_names if cfg is not None else [],
		norm_stats=norm_stats,
		save_path=save_path,
		metadata=None,
		highlight_pixel_box=None,
		title_prefix=f"Sample {sample_idx} comparison (case {case}) - ",
	)



def create_predictions_dict(
	videomae: Optional[torch.Tensor] = None,
	baseline: Optional[torch.Tensor] = None,
	diffusion: Optional[torch.Tensor] = None,
	vae: Optional[torch.Tensor] = None,
	labels: Optional[Dict[str, str]] = None,
) -> Dict[str, torch.Tensor]:
	"""Build an ordered predictions dict for plotting.

	Provide any subset of predictions; order in the returned dict will be
		if vcomp is None:
			raise RuntimeError("VIDEOMAE utilities not available")
		vcfg = ProjectConfig()
		vcfg.output_dir = str(images_dir)
		clean, recon, baseline = vcomp.run_inference_tests(vcfg, checkpoint_dir=checkpoint_dir, num_samples=sample_idx + 1, save_images=False, images_dir=images_dir, run_baselines=run_baselines)
	"""
	out: Dict[str, torch.Tensor] = {}
	labels = labels or {}
	if videomae is not None:
		out[labels.get("videomae", "VideoMAE")] = videomae
	if baseline is not None:
		out[labels.get("baseline", "Baseline")] = baseline
	if diffusion is not None:
		out[labels.get("diffusion", "Diffusion")] = diffusion
	if vae is not None:
		out[labels.get("vae", "VAE")] = vae
	return out


def plot_prediction_comparison_joint(
	target_ctwh_norm: torch.Tensor,
	predictions: Dict[str, torch.Tensor],
	rows: List[Tuple[int, int, str]],
	variable_names: List[str],
	norm_stats: Dict[str, Dict[str, float]],
	save_path: str,
	metadata: Optional[SampleMetadata] = None,
	highlight_pixel_box: Optional[Tuple[int, int, int, int]] = None,
	title_prefix: str = "",
) -> None:
	"""Thin wrapper that forwards to the VideoMAE comparison plot.

	This keeps plotting consistent across the repository while allowing
	AE/diffusion code to call a single function in this package.
	"""
	if vcomp is None:
		raise RuntimeError(
			"VIDEOMAE plotting utilities are not importable. "
			"Ensure the repository path allows importing VIDEOMAE.* modules."
		)

	# Forward to the well-tested plotting function.
	vcomp.plot_prediction_comparison(
		target_ctwh_norm=target_ctwh_norm,
		predictions=predictions,
		rows=rows,
		variable_names=variable_names,
		norm_stats=norm_stats,
		save_path=save_path,
		metadata=metadata,
		highlight_pixel_box=highlight_pixel_box,
		title_prefix=title_prefix,
	)




if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="Produce comparison plots for Case 1/2/3")
	parser.add_argument("--case", choices=["1", "2", "3", "all"], default="1", help="Which case to plot")
	parser.add_argument("--sample-idx", type=int, default=0, help="Index of sample to use (0-based)")
	parser.add_argument("--target-frame", type=int, default=1, help="Target frame index for Case 3")
	parser.add_argument("--output-dir", default=None, help="Directory to save plots (overrides cfg.output_dir)")
	parser.add_argument("--no-vae", action="store_true", help="Skip loading VAE model")
	parser.add_argument("--no-videomae", action="store_true", help="Skip loading VideoMAE model")
	parser.add_argument("--no-diffusion", action="store_true", help="Skip loading diffusion denoiser")
	args = parser.parse_args()

	# Local config (paths for checkpoints / latents used by diffusion/VAE)
	local_cfg = ProjectConfig()
	if args.output_dir:
		local_cfg.output_dir = Path(args.output_dir)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# 1) Use VideoMAE's ProjectConfig + run_inference_tests to fetch the requested sample and metadata
	sample = None
	metadata = None
	vcfg = None
	try:
		if vcomp is None:
			raise RuntimeError("VIDEOMAE utilities not importable")
		vcfg = vcomp.ProjectConfig()
		# prefer saving outputs to the requested output dir
		vcfg.output_dir = str(local_cfg.output_dir)
		clean, recon, baseline = vcomp.run_inference_tests(vcfg, checkpoint_dir=None, num_samples=args.sample_idx + 1, save_images=False, images_dir=str(local_cfg.output_dir))
		sample = clean
		try:
			norm_stats_v = vcomp.load_norm_stats(vcfg.norm_stats_path) if hasattr(vcfg, "norm_stats_path") else None
			eval_ds = vcomp.NetCDFSpatioTemporalDataset(vcfg.eval_dir, vcfg.variable_names, vcfg.file_num_frames, vcfg.image_size, norm_stats_v, vcfg.flatten_time)
			metadata = eval_ds.get_metadata(args.sample_idx)
		except Exception:
			metadata = None
	except Exception as e:
		raise SystemExit(f"Failed to obtain sample via VideoMAE inference helper: {e}")

	# 2) Load optional models: VAE, VideoMAE (for reconstruct helpers), Diffusion
	vae_model = None
	if not args.no_vae:
		try:
			from AE_difussion.train_test_AE_VAE_masked_difussion_3days import CycloneVAE
			ckpt = torch.load(local_cfg.CHECKPOINT, map_location=device)
			vae_model = CycloneVAE(in_channels=len(local_cfg.variable_names), latent_channels=128).to(device)
			vae_model.load_state_dict(ckpt.get("model_state", ckpt))
			vae_model.eval()
			print("Loaded VAE model")
		except Exception:
			print("Could not load VAE model; proceeding without VAE reconstructions")

	videomae_model = None
	if not args.no_videomae and vcfg is not None:
		try:
			from VIDEOMAE.videomae_with_baseline_comparison import load_trained_model, get_device
			videomae_device = get_device(vcfg)
			videomae_model = load_trained_model(vcfg, checkpoint_dir=None).to(videomae_device)
			videomae_model.eval()
			print("Loaded VideoMAE model")
		except Exception:
			print("Could not load VideoMAE model; some plots may be skipped")

	diffusion_denoiser = None
	diffusion_schedule = None
	if not args.no_diffusion:
		try:
			from AE_difussion.train_test_AE_VAE_masked_difussion_3days import load_trained_diffusion
			diffusion_denoiser, diffusion_schedule, _ = load_trained_diffusion(local_cfg.DIFFUSION_CHECKPOINT, device=device)
			print("Loaded diffusion denoiser")
		except Exception:
			print("Could not load diffusion denoiser; diffusion results will be skipped")

	os.makedirs(local_cfg.output_dir, exist_ok=True)

	# 3) Dispatch plotting
	to_run = [args.case] if args.case != "all" else ["1", "2", "3"]
	use_cfg = vcfg if vcfg is not None else local_cfg
	norm_stats_to_pass = (vcomp.load_norm_stats(use_cfg.norm_stats_path) if (vcomp is not None and hasattr(use_cfg, "norm_stats_path")) else {})

	if "1" in to_run:
		out1 = os.path.join(local_cfg.output_dir, "case1_comparison.png")
		plot_case1_joint(sample, out1, cfg=use_cfg, device=device, videomae_model=videomae_model, vae_model=vae_model, norm_stats=norm_stats_to_pass, metadata=metadata)
		print(f"Saved Case1 plot to {out1}")
	if "2" in to_run:
		out2 = os.path.join(local_cfg.output_dir, "case2_comparison.png")
		plot_case2_joint(sample, out2, cfg=use_cfg, device=device, videomae_model=videomae_model, vae_model=vae_model, channel_idx=0, norm_stats=norm_stats_to_pass, metadata=metadata)
		print(f"Saved Case2 plot to {out2}")
	if "3" in to_run:
		out3 = os.path.join(local_cfg.output_dir, "case3_comparison.png")
		plot_case3_joint(sample, out3, cfg=use_cfg, device=device, videomae_model=videomae_model, vae_model=vae_model, diffusion_denoiser=diffusion_denoiser, diffusion_schedule=diffusion_schedule, latents_dir=str(local_cfg.LATENTS_DIR_TEST), norm_stats=norm_stats_to_pass, vars_list=(use_cfg.variable_names if hasattr(use_cfg, "variable_names") else local_cfg.variable_names), target_frame_idx=args.target_frame, metadata=metadata)
		print(f"Saved Case3 plot to {out3}")
	if run_baseline and vcomp is not None and pixel_box is not None:
		baseline = vcomp.embed_case1_baseline_prediction(target_ctwh_norm, pixel_box)
		preds[labels.get("baseline", "Baseline") if labels else "Baseline"] = baseline

	# VAE per-frame reconstruction (if provided)
	if vae_model is not None:
		vae_recon = _vae_reconstruct_full_tensor(vae_model, target_ctwh_norm, device)
		preds[labels.get("vae", "VAE") if labels else "VAE"] = vae_recon

	# Build rows: one row per timestep for the selected channel
	rows = vcomp.case1_comparison_rows(cfg, channel_idx) if (vcomp is not None and cfg is not None) else [(channel_idx, t, f"t={t}") for t in range(target_ctwh_norm.shape[2])]

	# return preds, rows, pixel_box

