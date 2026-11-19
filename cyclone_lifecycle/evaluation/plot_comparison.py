import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from transformers import VideoMAEForPreTraining

from ..helper import check_times_match
from ..models.latent_diffusion.diffusion import load_trained_diffusion
from ..models.latent_diffusion.inference import LatentImputer, evaluate_sample
from ..models.latent_diffusion.vae import CycloneVAE_Diffusion, load_vae
from ..models.ocean_model import (  # ocean fields of the missing day (pixel-space residual U-Net)
    OceanImputer,
    fill_ocean_of_prediction,
)

# from VAE_masked_var_spatial import (
#     # CycloneVAE,
#     inferencia_by_sample_case1,
#     inferencia_by_sample_case2,
# )
from ..models.vae_case1_case2 import (
    CycloneVAE,
    inferencia_by_sample_case1,
    inferencia_by_sample_case2,
)
from ..models.videomae_bs_3days import (
    ProjectConfig,
    _frame_to_numpy,
    build_comparison_filename,
    case3_comparison_rows,
    case_comparison_rows,
    embed_case1_baseline_prediction,
    embed_case3_baseline_prediction,
    get_device,
    load_trained_model,
    plot_prediction_comparison,
    reconstruct_case1_model_prediction,
    reconstruct_case2_model_prediction,
    reconstruct_case3_model_prediction,
)
from ..dataset import (
    NetCDFSpatioTemporalDataset,
    SampleMetadata,
    denormalize_channels as denormalize,
    load_norm_stats,
    parse_patch_filename,
)
from ..metrics import (
    UNIT_SCALE_FOR_STD,
    compute_row_metrics as _compute_row_metrics,
    get_dataset_std as _get_dataset_std,
)
from ..plotting import convert_for_display

DATA_ROOT  = Path("/scale/project/ifremer-isi-jumeaunumerique/NEXTGEMS_PROCESSED_LAT_LON")
CHECKPOINT_DIR = DATA_ROOT / "models"


EVAL_DIR_TEST = DATA_ROOT / "data_128x128_3days/test_patches"

#------------- Diffusion + VAE
TYPE = "vae_diffusion_anomaly"
CHECKPOINT_DIR = Path(os.environ.get("CYCLONE_OUT_DIR", DATA_ROOT / "ml_result/vae_diffusion_anomaly"))
# CHECKPOINT_DIR = Path(os.environ.get("CYCLONE_OUT_DIR", DATA_ROOT / "ml_result/vae_diffusion_anomaly_vaebeta6")) # higher error

SAVE_DIR = CHECKPOINT_DIR / "plots" / TYPE
VAE_DIFFUSION_CHECKPOINT = CHECKPOINT_DIR / f"checkpoint_model_{TYPE}.pt"
DIFFUSION_TYPE = f"unet_{TYPE}"
DIFFUSION_CHECKPOINT = CHECKPOINT_DIR / f"checkpoint_{DIFFUSION_TYPE}.pt"

#------------- Ocean model (ocean_model.py). Use the folder of the variant you chose on VAL:
#   ocean_residual                -> model A (sees the true atmosphere of the missing day when trained)
#   ocean_residual_maskatm        -> model trained with --mask-atm-missing --tag _maskatm (never sees it)
OCEAN_CHECKPOINT = DATA_ROOT / "result_ocean_model/diffusion/ocean_residual/checkpoint_ocean_unet.pt"

LATENTS_TRAIN = CHECKPOINT_DIR / "latents" / TYPE       # train.pt, test.pt, (normalisation lives in train.pt)
LATENTS_DIR_TEST = CHECKPOINT_DIR / "latents" / f"test_{TYPE}"




#------------- VideoMAE checkpoint
# VIDEOMAE_CHECKPOINT = "/scale/project/ifremer-isi-jumeaunumerique/NEXTGEMS_PROCESSED_LAT_LON/videomae_missing_timestep/"
VIDEOMAE_CHECKPOINT = "/scale/project/ifremer-isi-jumeaunumerique/NEXTGEMS_PROCESSED_LAT_LON/ml_result/videomae/"

#------------- VAE checkpoint
# VAE_CHECKPOINT_DIR = DATA_ROOT / "models_VAE_masked_var_spatial"
# VAE_MASKED_TYPE = "VAE_spatial_or_variable_masking"
# CHECKPOINT_MASKED_VAE = VAE_CHECKPOINT_DIR / f"checkpoint_model_{VAE_MASKED_TYPE}.pt"

RESULT_CHECKPOINT_DIR = DATA_ROOT / "ml_result"


VAE_CHECKPOINT_DIR = RESULT_CHECKPOINT_DIR /"vae_case1_case2"
VAE_MASKED_TYPE = "vae_case1_case2"
CHECKPOINT_MASKED_VAE = VAE_CHECKPOINT_DIR / f"checkpoint_model_{VAE_MASKED_TYPE}.pt"


SAVE_ALL_DIR = DATA_ROOT / "result_ocean_model/result_metrics" # RESULT_CHECKPOINT_DIR /"join_plots"


VAR_LABELS = {
    "10u":             "Zonal wind 10u, m/s",
    "10v":             "Meridional wind 10v, m/s",
    "msl":             "Mean sea level pressure, hPa", #raw data is Pa but plot in hPa 
    "avg_tos":         "Sea surface temperature, °C",
    "avg_zos":         "Sea surface height, m",
    "avg_sos":         "Sea surface salinity, psu",
    "avg_mlotst125":   "Mixed layer depth, m",
}


def compute_prediction_metrics(
    target_ctwh_norm: torch.Tensor,
    predictions: dict[str, torch.Tensor],
    rows: list[tuple[int, int, str]],
    variable_names: list[str],
    norm_stats: dict[str, Any],
    highlight_pixel_box: Optional[tuple[int, int, int, int]] = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute per-row metrics without creating or saving a plot."""
    def get_stats(label: str) -> dict[str, dict[str, float]]:
        if label in norm_stats and isinstance(norm_stats[label], dict):
            return norm_stats[label]
        if "target" in norm_stats and isinstance(norm_stats["target"], dict):
            return norm_stats["target"]
        return norm_stats

    target_physical = denormalize(target_ctwh_norm, get_stats("target"), variable_names)
    predictions_physical = {
        label: denormalize(prediction, get_stats(label), variable_names)
        for label, prediction in predictions.items()
    }

    all_metrics = {}
    for channel_idx, frame_idx, row_label in rows:
        var_name = variable_names[channel_idx]
        target_raw = _frame_to_numpy(target_physical, channel_idx, frame_idx)
        target_frame, _ = convert_for_display(var_name, target_raw)
        row_metrics = {}

        for label, prediction in predictions_physical.items():
            pred_raw = _frame_to_numpy(prediction, channel_idx, frame_idx)
            pred_frame, _ = convert_for_display(var_name, pred_raw)
            dataset_std = _get_dataset_std(
                var_name=var_name,
                norm_stats=norm_stats,
                label=label,
                unit_scale=UNIT_SCALE_FOR_STD.get(var_name, 1.0),
            )
            row_metrics[label] = _compute_row_metrics(
                target_frame=target_frame,
                pred_frame=pred_frame,
                box=highlight_pixel_box,
                dataset_std=dataset_std,
            )

        all_metrics[row_label] = row_metrics

    return all_metrics


def _flatten_sample_metrics(
    metrics: dict[str, dict[str, dict[str, float]]],
    sample_index: int,
    type_case: str,
    subcase: str,
    metadata: "SampleMetadata",
) -> list[dict[str, Any]]:
    records = []
    for row_label, model_metrics in metrics.items():
        for model_label, values in model_metrics.items():
            records.append({
                "sample_index": sample_index,
                "case": type_case,
                "track_id": metadata.track_id,
                "season": metadata.season,
                "basin": metadata.basin,
                "rank": metadata.rank,
                "win_idx": metadata.win_idx,
                "subcase": subcase,
                # "time": metadata.time_str,
                "row": row_label,
                "model": model_label,
                "source_file": str(Path(metadata.source_file).name),
                **values,
            })
    return records


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _save_eval_metrics(
    metrics_dir: str,
    type_case: str,
    all_records: list[dict[str, Any]],
    column_final: str,
) -> None:
    os.makedirs(metrics_dir, exist_ok=True)
    sample_path = os.path.join(metrics_dir, f"{type_case}_sample_metrics.json")
    with open(sample_path, "w", encoding="utf-8") as metrics_file:
        json.dump(all_records, metrics_file, indent=2, default=_json_default)

    sample_dir = os.path.join(metrics_dir, f"{type_case}_samples")
    os.makedirs(sample_dir, exist_ok=True)
    records_by_sample = defaultdict(list)
    for record in all_records:
        records_by_sample[record["sample_index"]].append(record)
    for sample_index, records in records_by_sample.items():
        sample_path = os.path.join(sample_dir, f"sample_{sample_index:05d}_metrics.json")
        with open(sample_path, "w", encoding="utf-8") as metrics_file:
            json.dump(records, metrics_file, indent=2, default=_json_default)

    record_fields = [
    #     "sample_index", "case", "subcase", "source_file", "track_id", "season",
    #     "basin", "rank", "win_idx", "time", "row", "model", "mae", "rmse",
    #     "bias", "nrmse",
    # ]
            "sample_index", "case", "track_id", "season",
        "basin", "rank", "win_idx", "subcase", "row", "model", "mae", "rmse",
        "bias", "nrmse", "source_file"
    ]
    records_path = os.path.join(metrics_dir, f"{type_case}_all_samples_metrics.csv")
    with open(records_path, "w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=record_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_records)

    grouped_values = defaultdict(list)
    for record in all_records:
        group = (record["case"], record[column_final], record["model"]) #, record["subcase"]
        for metric_name in ("mae", "rmse", "bias", "nrmse"):
            grouped_values[(*group, metric_name)].append(record[metric_name])

    summary_path = os.path.join(metrics_dir, f"{type_case}_dataset_summary.csv")
    summary_fields = ["case", column_final, "model", "metric", "sample_count", "mean", "std"] #"subcase",
    with open(summary_path, "w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=summary_fields)
        writer.writeheader()
        for (case, column, model, metric_name), values in sorted(grouped_values.items()): # subcase,
            writer.writerow({
                "case": case,
                # "subcase": subcase,
                column_final: column,
                "model": model,
                "metric": metric_name,
                "sample_count": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            })


def visualize_case1_multi_model_vs_baseline(
    videomae_model: VideoMAEForPreTraining,
    vae_model: CycloneVAE,
    cfg: ProjectConfig,
    clean_sample_cthw: torch.Tensor,
    target_frame_idx: int,
    device: torch.device,
    videomae_norm_stats: dict[str, dict[str, float]],
    other_model_norm_stats: dict[str, dict[str, float]],
    save_path: str,
    metadata: Optional["SampleMetadata"] = None,
    videomae_label: str = "VideoMAE",
    other_model_label: str = "masked VAE",
    baseline_label: str = "Linear Interpolation",
    make_plot: bool = True,
    # only_box: bool= True,
) -> tuple[dict[str, dict[str, dict[str, float]]], torch.Tensor, torch.Tensor, torch.Tensor]:

    # clean_sample_ctwh=sample, #[1, C, T, H, W] # aqui se envio un 1 sample
    # for the masked VAE i need B, C, H, W where B= Txn_samples=3x1 flatted
    # clean_samples_cthw = eval_dataset[i].unsqueeze(0) #[1, C, T, H, W]

    # 1. Predicción VideoMAE (salida en espacio normalizado de VideoMAE)
    reconstructed_videomae, pixel_box = reconstruct_case1_model_prediction(
        videomae_model, cfg, clean_sample_cthw, device
    )
    print(f"pixel_box: {pixel_box} pixel_box[0]: {pixel_box[0]}")

    # 3. Baseline (calculado sobre el tensor de entrada, usa norm_stats de VideoMAE)
    baseline_ctwh = embed_case1_baseline_prediction(clean_sample_cthw, pixel_box)


    # 2. Predicción VAE (salida en espacio normalizado de VAE)
    pred, pred_box = inferencia_by_sample_case1(
        model=vae_model,
        clean_samples_cthw=clean_sample_cthw,
        pixel_box=pixel_box,
        device=device,
    )

    name_plot = f"case1_t{target_frame_idx}_multi_model_vs_baseline"
    reconstructed_vae = pred_box.permute(1, 0, 2, 3).unsqueeze(0)

    # if only_box:
    #     reconstructed_vae = pred_box.permute(1, 0, 2, 3).unsqueeze(0)
    # else:
    #     reconstructed_vae = pred.permute(1, 0, 2, 3).unsqueeze(0)
    #     pixel_box = None
    #     name_plot = f"case1_allarea_t{target_frame_idx}_multi_model_vs_baseline"


    print(f"clean_sample_ctwh shape: {clean_sample_cthw.shape}") #all has[1, 7, 3, 128, 128]
    print(f"reconstructed_videomae shape: {reconstructed_videomae.shape}")
    # print(f"baseline_ctwh shape: {baseline_ctwh.shape}")

    # 4. Agrupar Predicciones
    predictions = {
        videomae_label: reconstructed_videomae,
        other_model_label: reconstructed_vae,
        baseline_label: baseline_ctwh,
    }

    print(f"VIDEOMAE statistic: {videomae_norm_stats}")

    # 5. Mapear cada etiqueta a sus estadísticas correspondientes
    per_model_norm_stats = {
        "target": videomae_norm_stats,        # Target viene del dataset de VideoMAE
        videomae_label: videomae_norm_stats,
        other_model_label: videomae_norm_stats, #vae_norm_stats,
        baseline_label: videomae_norm_stats,  # El baseline comparte norm con el dataset de entrada
    }

    save_path = build_comparison_filename(
        metadata, 
        name_plot, # f"case1_t{target_frame_idx}_multi_model_vs_baseline", 
        save_path, 
        type_model="unified", 
        frame_idx=target_frame_idx
    )

    rows = case3_comparison_rows(cfg, target_frame_idx)
    all_metrics = compute_prediction_metrics(
        target_ctwh_norm=clean_sample_cthw,
        predictions=predictions,
        rows=rows,
        variable_names=cfg.variable_names,
        norm_stats=per_model_norm_stats,
        highlight_pixel_box=pixel_box,
    )
    if make_plot:
        plot_prediction_comparison(
            target_ctwh_norm=clean_sample_cthw,
            predictions=predictions,
            rows=rows,
            variable_names=cfg.variable_names,
            norm_stats=per_model_norm_stats,
            save_path=save_path,
            metadata=metadata,
            highlight_pixel_box=pixel_box,
            title_prefix=f"Case 1 (spatial recovery, t={target_frame_idx}) - Time: {metadata.time_strs[target_frame_idx]} - ",
        )

    return all_metrics, clean_sample_cthw, reconstructed_videomae, baseline_ctwh #, reconstructed_diffusion



def visualize_case2_multi_model_vs_baseline(
    videomae_model: VideoMAEForPreTraining,
    vae_model: CycloneVAE,
    cfg: ProjectConfig,
    clean_sample_cthw: torch.Tensor,
    channel_idx: int,
    device: torch.device,
    videomae_norm_stats: dict[str, dict[str, float]],
    other_model_norm_stats: dict[str, dict[str, float]],
    save_path: str,
    metadata: Optional["SampleMetadata"] = None,
    videomae_label: str = "VideoMAE",
    other_model_label: str = "masked VAE",
    make_plot: bool = True,
    plot_frame_indices: Optional[list[int]] = None,
    # baseline_label: str = "Linear Interpolation",
) -> tuple[dict[str, dict[str, dict[str, float]]], torch.Tensor, torch.Tensor]:

    # clean_sample_ctwh=sample, #[1, C, T, H, W] # aqui se envio un 1 sample
    # for the masked VAE i need B, C, H, W where B= Txn_samples=3x1 flatted
    # clean_samples_cthw = eval_dataset[i].unsqueeze(0) #[1, C, T, H, W]

    # 1. Predicción VideoMAE (salida en espacio normalizado de VideoMAE) provide the prediction of 3 timesteps 
    reconstructed_videomae = reconstruct_case2_model_prediction(
        videomae_model, cfg, clean_sample_cthw, channel_idx, device
    )

    # 2. Predicción VAE (salida en espacio normalizado de VAE) provide the 3 timesteps 
    pred = inferencia_by_sample_case2(
        model=vae_model,
        clean_samples_cthw=clean_sample_cthw,
        channel_idx=channel_idx,
        device=device,
    )
    reconstructed_vae = pred.permute(1, 0, 2, 3).unsqueeze(0)




    print(f"clean_sample_ctwh shape: {clean_sample_cthw.shape}") #all has[1, 7, 3, 128, 128]
    print(f"reconstructed_videomae shape: {reconstructed_videomae.shape}")
    # print(f"baseline_ctwh shape: {baseline_ctwh.shape}")

    # 4. Agrupar Predicciones
    predictions = {
        videomae_label: reconstructed_videomae, 
        other_model_label: reconstructed_vae,
        # baseline_label: baseline_ctwh,
    }

    print(f"VIDEOMAE statistic: {videomae_norm_stats}")

    # 5. Mapear cada etiqueta a sus estadísticas correspondientes
    per_model_norm_stats = {
        "target": videomae_norm_stats,        # Target viene del dataset de VideoMAE
        videomae_label: videomae_norm_stats,
        other_model_label: videomae_norm_stats, #vae_norm_stats,
        # baseline_label: videomae_norm_stats,  # El baseline comparte norm con el dataset de entrada
    }

    save_path = build_comparison_filename(
        metadata, 
        f"case2_{cfg.variable_names[channel_idx]}_multi_model", 
        save_path, 
        type_model="unified", 
    )

    rows = case_comparison_rows(cfg, channel_idx, metadata)
    all_metrics = compute_prediction_metrics(
        target_ctwh_norm=clean_sample_cthw,
        predictions=predictions,
        rows=rows,
        variable_names=cfg.variable_names,
        norm_stats=per_model_norm_stats,
    )
    plot_rows = rows if plot_frame_indices is None else [
        row for row in rows if row[1] in plot_frame_indices
    ]
    if make_plot and plot_rows:
        plot_prediction_comparison(
            target_ctwh_norm=clean_sample_cthw,
            predictions=predictions,
            rows=plot_rows,
            variable_names=cfg.variable_names,
            norm_stats=per_model_norm_stats,
            save_path=save_path,
            metadata=metadata,
            highlight_pixel_box=None,
            title_prefix=f"Case 2 (missing variable) - {cfg.variable_names[channel_idx]} - ",
        )

    return all_metrics, clean_sample_cthw, reconstructed_videomae #, reconstructed_diffusion




def visualize_case3_multi_model_vs_baseline(
    videomae_model: VideoMAEForPreTraining,
    # vae_model: CycloneVAE,
    # denoiser: torch.nn.Module,
    # schedule: Any, #schedule: DiffusionSchedule from difussion_3days import DiffusionSchedule
    # scale_factor: float,
    imputer: LatentImputer,
    payload: dict,
    cfg: ProjectConfig,
    clean_sample_cthw: torch.Tensor,
    target_frame_idx: int,
    device: torch.device,
    videomae_norm_stats: dict[str, dict[str, float]],
    diffusion_norm_stats: dict[str, dict[str, float]],
    save_path: str,
    metadata: Optional["SampleMetadata"] = None,
    videomae_label: str = "VideoMAE",
    diffusion_label: str = "Diffusion",
    baseline_label: str = "Linear Interpolation",
    make_plot: bool = True,
    ocean_imputer: Optional[OceanImputer] = None,
    hybrid_label: str = "Diffusion + ocean residual",
) -> tuple[dict[str, dict[str, dict[str, float]]], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns metrics as {row_label: {pred_label: {"mae":..., "rmse":..., "bias":..., "nrmse":...}}}
    """
    # 1. Predicción VideoMAE (salida en espacio normalizado de VideoMAE)
    reconstructed_videomae = reconstruct_case3_model_prediction(
        videomae_model, cfg, clean_sample_cthw, target_frame_idx, device
    )

    # 2. Predicción Diffusion (salida en espacio normalizado de Diffusion/VAE)
    # pred = evaluate_sample(
    #     stats=diffusion_norm_stats,
    #     vae=vae_model,
    #     denoiser=denoiser,
    #     schedule=schedule,
    #     payload=payload,
    #     device=device,
    #     # test_dir=cfg.eval_dir,
    #     vars_list=cfg.variable_names,
    #     scale_factor=scale_factor,
    #     target_frame_idx=target_frame_idx
    # )
    pred = evaluate_sample(imputer, payload, target_frame_idx=target_frame_idx)   # t1 from t0, t2 # (3, C, H, W)
    reconstructed_diffusion = pred.permute(1, 0, 2, 3).unsqueeze(0)


    # 3. Baseline (calculado sobre el tensor de entrada, usa norm_stats de VideoMAE)
    baseline_ctwh = embed_case3_baseline_prediction(clean_sample_cthw, target_frame_idx)

    # 4. Agrupar Predicciones
    predictions = {
        videomae_label: reconstructed_videomae,
        diffusion_label: reconstructed_diffusion,
        baseline_label: baseline_ctwh,
    }

    # 5. Mapear cada etiqueta a sus estadísticas correspondientes
    per_model_norm_stats = {
        "target": videomae_norm_stats,        # Target viene del dataset de VideoMAE
        videomae_label: videomae_norm_stats,
        diffusion_label: diffusion_norm_stats,
        baseline_label: videomae_norm_stats,  # El baseline comparte norm con el dataset de entrada
    }

    # 4b. Hybrid: the diffusion prediction, but with the OCEAN channels of the missing day from the ocean model.
    # The whole day is missing in case 3, so the ocean model gets the diffusion's atmosphere of that day (never the
    # true one) and the observed days of the true sample. The atmosphere rows of the hybrid equal the diffusion's.
    if ocean_imputer is not None:
        hybrid_tchw = fill_ocean_of_prediction(
            ocean_imputer,
            true_tchw=clean_sample_cthw[0].permute(1, 0, 2, 3),     # [1, C, T, H, W] -> [T, C, H, W]
            pred_tchw=pred,                                          # diffusion output [T, C, H, W]
            missing=target_frame_idx,
        )
        predictions[hybrid_label] = hybrid_tchw.permute(1, 0, 2, 3).unsqueeze(0).to(clean_sample_cthw.device)
        per_model_norm_stats[hybrid_label] = diffusion_norm_stats    # same train stats (checked at load time)

    save_path = build_comparison_filename(
        metadata, 
        f"case3_time{target_frame_idx}_multi_model_vs_baseline", 
        save_path, 
        type_model="unified", 
        frame_idx=target_frame_idx
    )

                                                            #case2  chan_idx, [t0, t1,t2], [str_t0, str_t1, str_t2] 
    rows = case3_comparison_rows(cfg, target_frame_idx)
    all_metrics = compute_prediction_metrics(
        target_ctwh_norm=clean_sample_cthw,
        predictions=predictions,
        rows=rows,
        variable_names=cfg.variable_names,
        norm_stats=per_model_norm_stats,
    )
    if make_plot:
        plot_prediction_comparison(
            target_ctwh_norm=clean_sample_cthw,
            predictions=predictions,
            rows=rows,
            variable_names=cfg.variable_names,
            norm_stats=per_model_norm_stats,
            save_path=save_path,
            metadata=metadata,
            highlight_pixel_box=None,
            title_prefix=f"Case 3 (full timestep recovery, t={target_frame_idx}) - Time: {metadata.time_strs[target_frame_idx]} - ",
        )

    return all_metrics, clean_sample_cthw, reconstructed_videomae, reconstructed_diffusion, baseline_ctwh


def check_same_norm_stats(stats_a, stats_b, variable_names, name_a="videomae", name_b="ocean model", rtol=1e-3):
    """The ocean model normalises with the TRAIN stats saved in its checkpoint; the samples here are normalised with
    the VideoMAE stats. They must be the same numbers (up to rounding), otherwise the ocean prediction is off."""
    try:
        for v in variable_names:
            for k in ("mean", "std"):
                if not np.isclose(float(stats_a[v][k]), float(stats_b[v][k]), rtol=rtol, atol=1e-8):
                    raise ValueError(f"normalisation stats differ for {v}/{k}: {name_a}={stats_a[v][k]} {name_b}={stats_b[v][k]}")
    except (KeyError, TypeError) as e:
        print(f"[WARN] could not compare the {name_a} and {name_b} stats automatically ({e!r}); check them by hand")


def run_unified_inference_tests(
    cfg: ProjectConfig,
    eval_dir,
    latent_dir,
    vae_diffusion_checkpoint,
    vae_checkpoint,
    diffusion_checkpoint = DIFFUSION_CHECKPOINT,
    videomae_checkpoint = None,
    ocean_checkpoint = None,
    images_dir = None,
    type_case = "case3",  # or "case1"
    plot_sample_indices: Optional[list[int]] = None,
    plot_case2_frame_indices: Optional[list[int]] = None,
    # only_box= True,

):
    device = get_device(cfg)
    images_dir = images_dir or os.path.join(cfg.output_dir, "unified_inference_images")
    os.makedirs(images_dir, exist_ok=True)

    # Load VideoMAE
    videomae_model = load_trained_model(cfg, videomae_checkpoint).to(device)
    videomae_norm_stats = load_norm_stats(cfg.norm_stats_path)

    # Load Diffusion & VAE
    # vae_ckpt = torch.load(vae_diffusion_checkpoint, map_location=device, weights_only=False)
    # diffusion_norm_stats = vae_ckpt["stats"]

    # vae_model_diffusion = CycloneVAE_Diffusion(in_channels=len(cfg.variable_names), latent_channels=128).to(device)
    # vae_model_diffusion.load_state_dict(vae_ckpt["model_state"])
    # vae_model_diffusion.eval()

    # denoiser, schedule, scale_factor, _ = load_trained_diffusion(diffusion_checkpoint, device=device)
    # denoiser.eval()

    vae, stats, vars_list = load_vae(vae_diffusion_checkpoint, device)
    denoiser, schedule, normalizer, dckpt = load_trained_diffusion(diffusion_checkpoint, device)
    imputer = LatentImputer(vae, stats, vars_list, denoiser, schedule, normalizer, dckpt)
    diffusion_norm_stats = stats

    # Ocean model, loaded ONCE here (not per sample)
    ocean_imputer = None
    if ocean_checkpoint is not None:
        ocean_imputer = OceanImputer(checkpoint=ocean_checkpoint, device=device)
        assert list(ocean_imputer.vars_list) == list(cfg.variable_names), "channel order differs from the ocean model"
        check_same_norm_stats(videomae_norm_stats, ocean_imputer.stats, cfg.variable_names)
        print(f"[Ocean] loaded {ocean_checkpoint} (epoch {ocean_imputer.ck['epoch']}, "
              f"mask_atm={ocean_imputer.ck.get('mask_atm', False)})")

    # Datasets
    eval_dataset = NetCDFSpatioTemporalDataset(
        eval_dir, cfg.variable_names, cfg.file_num_frames, cfg.image_size, videomae_norm_stats, cfg.flatten_time
    )  #cfg.eval_dir
    latent_files = sorted(Path(latent_dir).glob("*.pt"))

    # VAE  case 1 for now it is different that the diffusion model
    # model_checkpoint = torch.load(vae_checkpoint, map_location=device)
    # # saved_stats = checkpoint["stats"]
    # # print(f"masked VAE statistic: {saved_stats}")
    # masked_vae_model = CycloneVAE(in_channels=len(cfg.variable_names), latent_channels=128).to(device)
    # masked_vae_model.load_state_dict(model_checkpoint["model_state"])

    ckpt = torch.load(vae_checkpoint, map_location=device)
    C = len(cfg.variable_names)

    # channel order must match training, otherwise the variables get swapped silently
    assert list(ckpt["vars_list"]) == list(cfg.variable_names)

    masked_vae_model = CycloneVAE(
        in_channels=ckpt.get("in_channels", 2 * C),
        out_channels=ckpt.get("out_channels", C),
        latent_channels=128,
    ).to(device)
    masked_vae_model.load_state_dict(ckpt["model_state"])
    masked_vae_model.eval()

    vae_saved_stats = ckpt["stats"]   # mean/std used for normalisation


    all_records = []

    for i in range(len(eval_dataset)):
    # for i in range(2):
        sample = eval_dataset[i].unsqueeze(0)
        make_plot = plot_sample_indices is not None and i in plot_sample_indices

        if type_case == "case3":
            target_frame_idx = 1 #i % cfg.num_frames here i use the middle timestep because of the difusion model
            metadata = eval_dataset.get_metadata(i, frame_idx=target_frame_idx)
            payload = torch.load(latent_files[i], map_location=device, weights_only=False)
            print(f"source_file eval_dataset: {Path(metadata.source_file).name}")
            print(f"source_file payload: {Path(payload['source_file']).name}")
            check_times_match(payload, metadata, target_frame_idx)

            case3_metrics, _, _, _, _ = visualize_case3_multi_model_vs_baseline(
                videomae_model=videomae_model,
                # vae_model=vae_model_diffusion,
                # denoiser=denoiser,
                # schedule=schedule,
                # scale_factor=scale_factor,
                imputer=imputer,
                payload=payload,
                cfg=cfg,
                clean_sample_cthw=sample, #[1, C, T, H, W]
                target_frame_idx=target_frame_idx,
                device=device,
                videomae_norm_stats=videomae_norm_stats,
                diffusion_norm_stats=diffusion_norm_stats,
                save_path=images_dir,
                metadata=metadata,
                make_plot=make_plot,
                ocean_imputer=ocean_imputer,
            )
            all_records.extend(_flatten_sample_metrics(
                case3_metrics, i, type_case, metadata.time_strs[target_frame_idx], metadata
            ))
            _save_eval_metrics(images_dir, type_case, all_records, column_final="row")


        elif type_case == "case1":
            print(f"sample.shape[2]: {sample.shape[2]}")
            # update because now is predicting 3 timesteps and only obtain one each time instead of directly
            # after the prediction obtaine the metrics
            for target_frame_idx in range(3): # sample.shape[2] n_timesteps by sample = 3
                metadata = eval_dataset.get_metadata(i, frame_idx=target_frame_idx)
                case1_metrics, _, _, _= visualize_case1_multi_model_vs_baseline(
                    videomae_model=videomae_model,
                    vae_model=masked_vae_model,
                    cfg=cfg,
                    clean_sample_cthw=sample,
                    target_frame_idx=target_frame_idx,
                    device=device,
                    videomae_norm_stats=videomae_norm_stats,
                    other_model_norm_stats=vae_saved_stats, #videomae_norm_stats, # they are the same 99.99999......
                    save_path=images_dir,
                    metadata=metadata,
                    # only_box=only_box
                    make_plot=make_plot,
                )
                all_records.extend(_flatten_sample_metrics(
                    case1_metrics, i, type_case, metadata.time_strs[target_frame_idx], metadata
                ))

            # columns_row: channel, frame, label   subcase fix here frame  row variable channel
            _save_eval_metrics(images_dir, type_case, all_records, column_final="row")

        elif type_case == "case2":
            metadata = eval_dataset.get_metadata(i)

            for channel_idx in range(7): # along all variables
                # channel_idx = 0  #variable 10u
                case2_metrics, _, _= visualize_case2_multi_model_vs_baseline(
                    videomae_model=videomae_model,
                    vae_model=masked_vae_model,
                    cfg=cfg,
                    clean_sample_cthw=sample,
                    channel_idx=channel_idx,
                    device=device,
                    videomae_norm_stats=videomae_norm_stats,
                    other_model_norm_stats=videomae_norm_stats, # they are the same 99.99999......
                    save_path=images_dir,
                    metadata=metadata,
                    make_plot=make_plot,
                    plot_frame_indices=plot_case2_frame_indices,
                )
                all_records.extend(_flatten_sample_metrics(
                    case2_metrics,
                    i,
                    type_case,
                    cfg.variable_names[channel_idx],
                    metadata,
                ))
            # columns_row: channel, frame, label   subcase fix here channel  row variable timesteps 
            _save_eval_metrics(images_dir, type_case, all_records, column_final="subcase")


if __name__ == "__main__":
    # type_case = "case1"  
    # type_case = "case2"  
    type_case = "case3"  

    # for type_case in ["case2", "case3", "case1",]:
    # for type_case in ["case1"]: # run metrics 50 min
    # for type_case in ["case1", "case2"]: # run metrics 12 min
    for type_case in ["case3"]: # run metrics 12 min
        cfg = ProjectConfig()

        run_unified_inference_tests(cfg,
        eval_dir=EVAL_DIR_TEST,
        latent_dir=LATENTS_DIR_TEST,
        vae_diffusion_checkpoint=VAE_DIFFUSION_CHECKPOINT,
        vae_checkpoint=CHECKPOINT_MASKED_VAE,
        diffusion_checkpoint=DIFFUSION_CHECKPOINT,
        videomae_checkpoint=VIDEOMAE_CHECKPOINT,
        ocean_checkpoint=OCEAN_CHECKPOINT,
        images_dir=SAVE_ALL_DIR,
        type_case=type_case,
        plot_sample_indices = [0,1], #None,
        # plot_sample_indices = None,
        # only_box= True,
        # only_box= False,

    )
        
# import csv
# import json
# import os
# import sys
# from collections import defaultdict
# from pathlib import Path
# from typing import Any, Optional

# import numpy as np
# import torch
# from inference import LatentImputer, evaluate_sample
# from transformers import VideoMAEForPreTraining

# from diffusion import load_trained_diffusion
# from helper import check_times_match
# from vae import CycloneVAE_Diffusion, load_vae

# # from VAE_masked_var_spatial import (
# #     # CycloneVAE,
# #     inferencia_by_sample_case1,
# #     inferencia_by_sample_case2,
# # )
# from vae_case1_case2 import (
#     CycloneVAE,
#     inferencia_by_sample_case1,
#     inferencia_by_sample_case2,
# )
# from videomae_bs_3days import (
#     UNIT_SCALE_FOR_STD,
#     NetCDFSpatioTemporalDataset,
#     ProjectConfig,
#     SampleMetadata,
#     _compute_row_metrics,
#     _frame_to_numpy,
#     _get_dataset_std,
#     build_comparison_filename,
#     case3_comparison_rows,
#     case_comparison_rows,
#     convert_for_display,
#     denormalize,
#     embed_case1_baseline_prediction,
#     embed_case3_baseline_prediction,
#     get_device,
#     load_norm_stats,
#     load_trained_model,
#     
#     plot_prediction_comparison,
#     reconstruct_case1_model_prediction,
#     reconstruct_case2_model_prediction,
#     reconstruct_case3_model_prediction,
# )

# DATA_ROOT  = Path("/scale/project/ifremer-isi-jumeaunumerique/NEXTGEMS_PROCESSED_LAT_LON")
# CHECKPOINT_DIR = DATA_ROOT / "models"


# #------------- Diffusion checkpoint  old VAE
# # TYPE = "VAE_no_masking"  #!!!!!!!!! later change the VAE in the diffusion model
# # TYPE = "VAE"  #!!!!!!!!! later change the VAE in the diffusion model
# # VAE_DIFFUSION_CHECKPOINT = CHECKPOINT_DIR / f"checkpoint_model_{TYPE}.pt"

# #------------- Diffusion checkpoint  new VAE
# # VAE_CHECKPOINT_DIR = DATA_ROOT / "models_VAE_masked_var_spatial"
# # TYPE = "VAE_spatial_or_variable_masking"
# # VAE_DIFFUSION_CHECKPOINT = VAE_CHECKPOINT_DIR / f"checkpoint_model_{TYPE}.pt"
# #------------- latent files for inference
# # LATENTS_DIR = CHECKPOINT_DIR / "latents" / TYPE
# # -------------------  paths for plotting ------------------
# # EVAL_DIR_TEST = DATA_ROOT / "data_128x128_3days/test_patches/test_plot"
# # LATENTS_DIR_TEST = CHECKPOINT_DIR / "latents" / f"test_{TYPE}" / "eval"/"plot"
# # # # -------------------  path all test ------------------
# EVAL_DIR_TEST = DATA_ROOT / "data_128x128_3days/test_patches"
# # LATENTS_DIR_TEST = CHECKPOINT_DIR / "latents" / f"test_{TYPE}" / "eval"
# # # --------------------------------------------------------------
# # DIFFUSION_TYPE = f"diffusion_with_unet_{TYPE}"
# # DIFFUSION_CHECKPOINT = CHECKPOINT_DIR / f"checkpoint_{DIFFUSION_TYPE}.pt"

# #------------- Diffusion + VAE
# TYPE = "vae_diffusion_anomaly"
# # CHECKPOINT_DIR = Path(os.environ.get("CYCLONE_OUT_DIR", DATA_ROOT / "ml_result/vae_diffusion_anomaly"))
# CHECKPOINT_DIR = Path(os.environ.get("CYCLONE_OUT_DIR", DATA_ROOT / "result_vf/vae_diffusion_anomaly_vaebeta6"))

# SAVE_DIR = CHECKPOINT_DIR / "plots" / TYPE
# VAE_DIFFUSION_CHECKPOINT = CHECKPOINT_DIR / f"checkpoint_model_{TYPE}.pt"
# DIFFUSION_TYPE = f"unet_{TYPE}"
# DIFFUSION_CHECKPOINT = CHECKPOINT_DIR / f"checkpoint_{DIFFUSION_TYPE}.pt"

# LATENTS_TRAIN = CHECKPOINT_DIR / "latents" / TYPE       # train.pt, test.pt, (normalisation lives in train.pt)
# LATENTS_DIR_TEST = CHECKPOINT_DIR / "latents" / f"test_{TYPE}"




# #------------- VideoMAE checkpoint
# # VIDEOMAE_CHECKPOINT = "/scale/project/ifremer-isi-jumeaunumerique/NEXTGEMS_PROCESSED_LAT_LON/videomae_missing_timestep/"
# VIDEOMAE_CHECKPOINT = "/scale/project/ifremer-isi-jumeaunumerique/NEXTGEMS_PROCESSED_LAT_LON/result_vf/videomae/"


# #------------- VAE checkpoint
# # VAE_CHECKPOINT_DIR = DATA_ROOT / "models_VAE_masked_var_spatial"
# # VAE_MASKED_TYPE = "VAE_spatial_or_variable_masking"
# # CHECKPOINT_MASKED_VAE = VAE_CHECKPOINT_DIR / f"checkpoint_model_{VAE_MASKED_TYPE}.pt"

# RESULT_CHECKPOINT_DIR = DATA_ROOT / "result_vf"


# VAE_CHECKPOINT_DIR = RESULT_CHECKPOINT_DIR /"vae_case1_case2"
# VAE_MASKED_TYPE = "vae_case1_case2"
# CHECKPOINT_MASKED_VAE = VAE_CHECKPOINT_DIR / f"checkpoint_model_{VAE_MASKED_TYPE}.pt"


# SAVE_ALL_DIR = "/home/jgonzale/cyclone_atm/all_models/adaptations_models/result_metrics" # RESULT_CHECKPOINT_DIR /"join_plots"


# VAR_LABELS = {
#     "10u":             "Zonal wind 10u, m/s",
#     "10v":             "Meridional wind 10v, m/s",
#     "msl":             "Mean sea level pressure, hPa", #raw data is Pa but plot in hPa 
#     "avg_tos":         "Sea surface temperature, °C",
#     "avg_zos":         "Sea surface height, m",
#     "avg_sos":         "Sea surface salinity, psu",
#     "avg_mlotst125":   "Mixed layer depth, m",
# }


# def compute_prediction_metrics(
#     target_ctwh_norm: torch.Tensor,
#     predictions: dict[str, torch.Tensor],
#     rows: list[tuple[int, int, str]],
#     variable_names: list[str],
#     norm_stats: dict[str, Any],
#     highlight_pixel_box: Optional[tuple[int, int, int, int]] = None,
# ) -> dict[str, dict[str, dict[str, float]]]:
#     """Compute per-row metrics without creating or saving a plot."""
#     def get_stats(label: str) -> dict[str, dict[str, float]]:
#         if label in norm_stats and isinstance(norm_stats[label], dict):
#             return norm_stats[label]
#         if "target" in norm_stats and isinstance(norm_stats["target"], dict):
#             return norm_stats["target"]
#         return norm_stats

#     target_physical = denormalize(target_ctwh_norm, get_stats("target"), variable_names)
#     predictions_physical = {
#         label: denormalize(prediction, get_stats(label), variable_names)
#         for label, prediction in predictions.items()
#     }

#     all_metrics = {}
#     for channel_idx, frame_idx, row_label in rows:
#         var_name = variable_names[channel_idx]
#         target_raw = _frame_to_numpy(target_physical, channel_idx, frame_idx)
#         target_frame, _ = convert_for_display(var_name, target_raw)
#         row_metrics = {}

#         for label, prediction in predictions_physical.items():
#             pred_raw = _frame_to_numpy(prediction, channel_idx, frame_idx)
#             pred_frame, _ = convert_for_display(var_name, pred_raw)
#             dataset_std = _get_dataset_std(
#                 var_name=var_name,
#                 norm_stats=norm_stats,
#                 label=label,
#                 unit_scale=UNIT_SCALE_FOR_STD.get(var_name, 1.0),
#             )
#             row_metrics[label] = _compute_row_metrics(
#                 target_frame=target_frame,
#                 pred_frame=pred_frame,
#                 box=highlight_pixel_box,
#                 dataset_std=dataset_std,
#             )

#         all_metrics[row_label] = row_metrics

#     return all_metrics


# def _flatten_sample_metrics(
#     metrics: dict[str, dict[str, dict[str, float]]],
#     sample_index: int,
#     type_case: str,
#     subcase: str,
#     metadata: "SampleMetadata",
# ) -> list[dict[str, Any]]:
#     records = []
#     for row_label, model_metrics in metrics.items():
#         for model_label, values in model_metrics.items():
#             records.append({
#                 "sample_index": sample_index,
#                 "case": type_case,
#                 "track_id": metadata.track_id,
#                 "season": metadata.season,
#                 "basin": metadata.basin,
#                 "rank": metadata.rank,
#                 "win_idx": metadata.win_idx,
#                 "subcase": subcase,
#                 # "time": metadata.time_str,
#                 "row": row_label,
#                 "model": model_label,
#                 "source_file": str(Path(metadata.source_file).name),
#                 **values,
#             })
#     return records


# def _json_default(value: Any) -> Any:
#     if isinstance(value, np.generic):
#         return value.item()
#     raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


# def _save_eval_metrics(
#     metrics_dir: str,
#     type_case: str,
#     all_records: list[dict[str, Any]],
#     column_final: str,
# ) -> None:
#     os.makedirs(metrics_dir, exist_ok=True)
#     sample_path = os.path.join(metrics_dir, f"{type_case}_sample_metrics.json")
#     with open(sample_path, "w", encoding="utf-8") as metrics_file:
#         json.dump(all_records, metrics_file, indent=2, default=_json_default)

#     sample_dir = os.path.join(metrics_dir, f"{type_case}_samples")
#     os.makedirs(sample_dir, exist_ok=True)
#     records_by_sample = defaultdict(list)
#     for record in all_records:
#         records_by_sample[record["sample_index"]].append(record)
#     for sample_index, records in records_by_sample.items():
#         sample_path = os.path.join(sample_dir, f"sample_{sample_index:05d}_metrics.json")
#         with open(sample_path, "w", encoding="utf-8") as metrics_file:
#             json.dump(records, metrics_file, indent=2, default=_json_default)

#     record_fields = [
#     #     "sample_index", "case", "subcase", "source_file", "track_id", "season",
#     #     "basin", "rank", "win_idx", "time", "row", "model", "mae", "rmse",
#     #     "bias", "nrmse",
#     # ]
#             "sample_index", "case", "track_id", "season",
#         "basin", "rank", "win_idx", "subcase", "row", "model", "mae", "rmse",
#         "bias", "nrmse", "source_file"
#     ]
#     records_path = os.path.join(metrics_dir, f"{type_case}_all_samples_metrics.csv")
#     with open(records_path, "w", newline="", encoding="utf-8") as metrics_file:
#         writer = csv.DictWriter(metrics_file, fieldnames=record_fields, extrasaction="ignore")
#         writer.writeheader()
#         writer.writerows(all_records)

#     grouped_values = defaultdict(list)
#     for record in all_records:
#         group = (record["case"], record[column_final], record["model"]) #, record["subcase"]
#         for metric_name in ("mae", "rmse", "bias", "nrmse"):
#             grouped_values[(*group, metric_name)].append(record[metric_name])

#     summary_path = os.path.join(metrics_dir, f"{type_case}_dataset_summary.csv")
#     summary_fields = ["case", column_final, "model", "metric", "sample_count", "mean", "std"] #"subcase",
#     with open(summary_path, "w", newline="", encoding="utf-8") as metrics_file:
#         writer = csv.DictWriter(metrics_file, fieldnames=summary_fields)
#         writer.writeheader()
#         for (case, column, model, metric_name), values in sorted(grouped_values.items()): # subcase,
#             writer.writerow({
#                 "case": case,
#                 # "subcase": subcase,
#                 column_final: column,
#                 "model": model,
#                 "metric": metric_name,
#                 "sample_count": len(values),
#                 "mean": float(np.mean(values)),
#                 "std": float(np.std(values)),
#             })


# def visualize_case1_multi_model_vs_baseline(
#     videomae_model: VideoMAEForPreTraining,
#     vae_model: CycloneVAE,
#     cfg: ProjectConfig,
#     clean_sample_cthw: torch.Tensor,
#     target_frame_idx: int,
#     device: torch.device,
#     videomae_norm_stats: dict[str, dict[str, float]],
#     other_model_norm_stats: dict[str, dict[str, float]],
#     save_path: str,
#     metadata: Optional["SampleMetadata"] = None,
#     videomae_label: str = "VideoMAE",
#     other_model_label: str = "masked VAE",
#     baseline_label: str = "Linear Interpolation",
#     make_plot: bool = True,
#     # only_box: bool= True,
# ) -> tuple[dict[str, dict[str, dict[str, float]]], torch.Tensor, torch.Tensor, torch.Tensor]:

#     # clean_sample_ctwh=sample, #[1, C, T, H, W] # aqui se envio un 1 sample
#     # for the masked VAE i need B, C, H, W where B= Txn_samples=3x1 flatted
#     # clean_samples_cthw = eval_dataset[i].unsqueeze(0) #[1, C, T, H, W]

#     # 1. Predicción VideoMAE (salida en espacio normalizado de VideoMAE)
#     reconstructed_videomae, pixel_box = reconstruct_case1_model_prediction(
#         videomae_model, cfg, clean_sample_cthw, device
#     )
#     print(f"pixel_box: {pixel_box} pixel_box[0]: {pixel_box[0]}")

#     # 3. Baseline (calculado sobre el tensor de entrada, usa norm_stats de VideoMAE)
#     baseline_ctwh = embed_case1_baseline_prediction(clean_sample_cthw, pixel_box)


#     # 2. Predicción VAE (salida en espacio normalizado de VAE)
#     pred, pred_box = inferencia_by_sample_case1(
#         model=vae_model,
#         clean_samples_cthw=clean_sample_cthw,
#         pixel_box=pixel_box,
#         device=device,
#     )

#     name_plot = f"case1_t{target_frame_idx}_multi_model_vs_baseline"
#     reconstructed_vae = pred_box.permute(1, 0, 2, 3).unsqueeze(0)

#     # if only_box:
#     #     reconstructed_vae = pred_box.permute(1, 0, 2, 3).unsqueeze(0)
#     # else:
#     #     reconstructed_vae = pred.permute(1, 0, 2, 3).unsqueeze(0)
#     #     pixel_box = None
#     #     name_plot = f"case1_allarea_t{target_frame_idx}_multi_model_vs_baseline"


#     print(f"clean_sample_ctwh shape: {clean_sample_cthw.shape}") #all has[1, 7, 3, 128, 128]
#     print(f"reconstructed_videomae shape: {reconstructed_videomae.shape}")
#     # print(f"baseline_ctwh shape: {baseline_ctwh.shape}")

#     # 4. Agrupar Predicciones
#     predictions = {
#         videomae_label: reconstructed_videomae,
#         other_model_label: reconstructed_vae,
#         baseline_label: baseline_ctwh,
#     }

#     print(f"VIDEOMAE statistic: {videomae_norm_stats}")

#     # 5. Mapear cada etiqueta a sus estadísticas correspondientes
#     per_model_norm_stats = {
#         "target": videomae_norm_stats,        # Target viene del dataset de VideoMAE
#         videomae_label: videomae_norm_stats,
#         other_model_label: videomae_norm_stats, #vae_norm_stats,
#         baseline_label: videomae_norm_stats,  # El baseline comparte norm con el dataset de entrada
#     }

#     save_path = build_comparison_filename(
#         metadata, 
#         name_plot, # f"case1_t{target_frame_idx}_multi_model_vs_baseline", 
#         save_path, 
#         type_model="unified", 
#         frame_idx=target_frame_idx
#     )

#     rows = case3_comparison_rows(cfg, target_frame_idx)
#     all_metrics = compute_prediction_metrics(
#         target_ctwh_norm=clean_sample_cthw,
#         predictions=predictions,
#         rows=rows,
#         variable_names=cfg.variable_names,
#         norm_stats=per_model_norm_stats,
#         highlight_pixel_box=pixel_box,
#     )
#     if make_plot:
#         plot_prediction_comparison(
#             target_ctwh_norm=clean_sample_cthw,
#             predictions=predictions,
#             rows=rows,
#             variable_names=cfg.variable_names,
#             norm_stats=per_model_norm_stats,
#             save_path=save_path,
#             metadata=metadata,
#             highlight_pixel_box=pixel_box,
#             title_prefix=f"Case 1 (spatial recovery, t={target_frame_idx}) - Time: {metadata.time_strs[target_frame_idx]} - ",
#         )

#     return all_metrics, clean_sample_cthw, reconstructed_videomae, baseline_ctwh #, reconstructed_diffusion



# def visualize_case2_multi_model_vs_baseline(
#     videomae_model: VideoMAEForPreTraining,
#     vae_model: CycloneVAE,
#     cfg: ProjectConfig,
#     clean_sample_cthw: torch.Tensor,
#     channel_idx: int,
#     device: torch.device,
#     videomae_norm_stats: dict[str, dict[str, float]],
#     other_model_norm_stats: dict[str, dict[str, float]],
#     save_path: str,
#     metadata: Optional["SampleMetadata"] = None,
#     videomae_label: str = "VideoMAE",
#     other_model_label: str = "masked VAE",
#     make_plot: bool = True,
#     plot_frame_indices: Optional[list[int]] = None,
#     # baseline_label: str = "Linear Interpolation",
# ) -> tuple[dict[str, dict[str, dict[str, float]]], torch.Tensor, torch.Tensor]:

#     # clean_sample_ctwh=sample, #[1, C, T, H, W] # aqui se envio un 1 sample
#     # for the masked VAE i need B, C, H, W where B= Txn_samples=3x1 flatted
#     # clean_samples_cthw = eval_dataset[i].unsqueeze(0) #[1, C, T, H, W]

#     # 1. Predicción VideoMAE (salida en espacio normalizado de VideoMAE) provide the prediction of 3 timesteps 
#     reconstructed_videomae = reconstruct_case2_model_prediction(
#         videomae_model, cfg, clean_sample_cthw, channel_idx, device
#     )

#     # 2. Predicción VAE (salida en espacio normalizado de VAE) provide the 3 timesteps 
#     pred = inferencia_by_sample_case2(
#         model=vae_model,
#         clean_samples_cthw=clean_sample_cthw,
#         channel_idx=channel_idx,
#         device=device,
#     )
#     reconstructed_vae = pred.permute(1, 0, 2, 3).unsqueeze(0)




#     print(f"clean_sample_ctwh shape: {clean_sample_cthw.shape}") #all has[1, 7, 3, 128, 128]
#     print(f"reconstructed_videomae shape: {reconstructed_videomae.shape}")
#     # print(f"baseline_ctwh shape: {baseline_ctwh.shape}")

#     # 4. Agrupar Predicciones
#     predictions = {
#         videomae_label: reconstructed_videomae, 
#         other_model_label: reconstructed_vae,
#         # baseline_label: baseline_ctwh,
#     }

#     print(f"VIDEOMAE statistic: {videomae_norm_stats}")

#     # 5. Mapear cada etiqueta a sus estadísticas correspondientes
#     per_model_norm_stats = {
#         "target": videomae_norm_stats,        # Target viene del dataset de VideoMAE
#         videomae_label: videomae_norm_stats,
#         other_model_label: videomae_norm_stats, #vae_norm_stats,
#         # baseline_label: videomae_norm_stats,  # El baseline comparte norm con el dataset de entrada
#     }

#     save_path = build_comparison_filename(
#         metadata, 
#         f"case2_{cfg.variable_names[channel_idx]}_multi_model", 
#         save_path, 
#         type_model="unified", 
#     )

#     rows = case_comparison_rows(cfg, channel_idx, metadata)
#     all_metrics = compute_prediction_metrics(
#         target_ctwh_norm=clean_sample_cthw,
#         predictions=predictions,
#         rows=rows,
#         variable_names=cfg.variable_names,
#         norm_stats=per_model_norm_stats,
#     )
#     plot_rows = rows if plot_frame_indices is None else [
#         row for row in rows if row[1] in plot_frame_indices
#     ]
#     if make_plot and plot_rows:
#         plot_prediction_comparison(
#             target_ctwh_norm=clean_sample_cthw,
#             predictions=predictions,
#             rows=plot_rows,
#             variable_names=cfg.variable_names,
#             norm_stats=per_model_norm_stats,
#             save_path=save_path,
#             metadata=metadata,
#             highlight_pixel_box=None,
#             title_prefix=f"Case 2 (missing variable) - {cfg.variable_names[channel_idx]} - ",
#         )

#     return all_metrics, clean_sample_cthw, reconstructed_videomae #, reconstructed_diffusion




# def visualize_case3_multi_model_vs_baseline(
#     videomae_model: VideoMAEForPreTraining,
#     # vae_model: CycloneVAE,
#     # denoiser: torch.nn.Module,
#     # schedule: Any, #schedule: DiffusionSchedule from difussion_3days import DiffusionSchedule
#     # scale_factor: float,
#     imputer: LatentImputer,
#     payload: dict,
#     cfg: ProjectConfig,
#     clean_sample_cthw: torch.Tensor,
#     target_frame_idx: int,
#     device: torch.device,
#     videomae_norm_stats: dict[str, dict[str, float]],
#     diffusion_norm_stats: dict[str, dict[str, float]],
#     save_path: str,
#     metadata: Optional["SampleMetadata"] = None,
#     videomae_label: str = "VideoMAE",
#     diffusion_label: str = "Diffusion",
#     baseline_label: str = "Linear Interpolation",
#     make_plot: bool = True,
# ) -> tuple[dict[str, dict[str, dict[str, float]]], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#     """Returns metrics as {row_label: {pred_label: {"mae":..., "rmse":..., "bias":..., "nrmse":...}}}
#     """
#     # 1. Predicción VideoMAE (salida en espacio normalizado de VideoMAE)
#     reconstructed_videomae = reconstruct_case3_model_prediction(
#         videomae_model, cfg, clean_sample_cthw, target_frame_idx, device
#     )

#     # 2. Predicción Diffusion (salida en espacio normalizado de Diffusion/VAE)
#     # pred = evaluate_sample(
#     #     stats=diffusion_norm_stats,
#     #     vae=vae_model,
#     #     denoiser=denoiser,
#     #     schedule=schedule,
#     #     payload=payload,
#     #     device=device,
#     #     # test_dir=cfg.eval_dir,
#     #     vars_list=cfg.variable_names,
#     #     scale_factor=scale_factor,
#     #     target_frame_idx=target_frame_idx
#     # )
#     pred = evaluate_sample(imputer, payload, target_frame_idx=target_frame_idx)   # t1 from t0, t2 # (3, C, H, W)
#     reconstructed_diffusion = pred.permute(1, 0, 2, 3).unsqueeze(0)


#     # 3. Baseline (calculado sobre el tensor de entrada, usa norm_stats de VideoMAE)
#     baseline_ctwh = embed_case3_baseline_prediction(clean_sample_cthw, target_frame_idx)

#     # 4. Agrupar Predicciones
#     predictions = {
#         videomae_label: reconstructed_videomae,
#         diffusion_label: reconstructed_diffusion,
#         baseline_label: baseline_ctwh,
#     }


#     # 5. Mapear cada etiqueta a sus estadísticas correspondientes
#     per_model_norm_stats = {
#         "target": videomae_norm_stats,        # Target viene del dataset de VideoMAE
#         videomae_label: videomae_norm_stats,
#         diffusion_label: diffusion_norm_stats,
#         baseline_label: videomae_norm_stats,  # El baseline comparte norm con el dataset de entrada
#     }

#     save_path = build_comparison_filename(
#         metadata, 
#         f"case3_time{target_frame_idx}_multi_model_vs_baseline", 
#         save_path, 
#         type_model="unified", 
#         frame_idx=target_frame_idx
#     )

#                                                             #case2  chan_idx, [t0, t1,t2], [str_t0, str_t1, str_t2] 
#     rows = case3_comparison_rows(cfg, target_frame_idx)
#     all_metrics = compute_prediction_metrics(
#         target_ctwh_norm=clean_sample_cthw,
#         predictions=predictions,
#         rows=rows,
#         variable_names=cfg.variable_names,
#         norm_stats=per_model_norm_stats,
#     )
#     if make_plot:
#         plot_prediction_comparison(
#             target_ctwh_norm=clean_sample_cthw,
#             predictions=predictions,
#             rows=rows,
#             variable_names=cfg.variable_names,
#             norm_stats=per_model_norm_stats,
#             save_path=save_path,
#             metadata=metadata,
#             highlight_pixel_box=None,
#             title_prefix=f"Case 3 (full timestep recovery, t={target_frame_idx}) - Time: {metadata.time_strs[target_frame_idx]} - ",
#         )

#     return all_metrics, clean_sample_cthw, reconstructed_videomae, reconstructed_diffusion, baseline_ctwh


# def run_unified_inference_tests(
#     cfg: ProjectConfig,
#     eval_dir,
#     latent_dir,
#     vae_diffusion_checkpoint,
#     vae_checkpoint,
#     diffusion_checkpoint = DIFFUSION_CHECKPOINT,
#     videomae_checkpoint = None,
#     images_dir = None,
#     type_case = "case3",  # or "case1"
#     plot_sample_indices: Optional[list[int]] = None,
#     plot_case2_frame_indices: Optional[list[int]] = None,
#     # only_box= True,

# ):
#     device = get_device(cfg)
#     images_dir = images_dir or os.path.join(cfg.output_dir, "unified_inference_images")
#     os.makedirs(images_dir, exist_ok=True)

#     # Load VideoMAE
#     videomae_model = load_trained_model(cfg, videomae_checkpoint).to(device)
#     videomae_norm_stats = load_norm_stats(cfg.norm_stats_path)

#     # Load Diffusion & VAE
#     # vae_ckpt = torch.load(vae_diffusion_checkpoint, map_location=device, weights_only=False)
#     # diffusion_norm_stats = vae_ckpt["stats"]

#     # vae_model_diffusion = CycloneVAE_Diffusion(in_channels=len(cfg.variable_names), latent_channels=128).to(device)
#     # vae_model_diffusion.load_state_dict(vae_ckpt["model_state"])
#     # vae_model_diffusion.eval()

#     # denoiser, schedule, scale_factor, _ = load_trained_diffusion(diffusion_checkpoint, device=device)
#     # denoiser.eval()

#     vae, stats, vars_list = load_vae(vae_diffusion_checkpoint, device)
#     denoiser, schedule, normalizer, dckpt = load_trained_diffusion(diffusion_checkpoint, device)
#     imputer = LatentImputer(vae, stats, vars_list, denoiser, schedule, normalizer, dckpt)
#     diffusion_norm_stats = stats

#     # Datasets
#     eval_dataset = NetCDFSpatioTemporalDataset(
#         eval_dir, cfg.variable_names, cfg.file_num_frames, cfg.image_size, videomae_norm_stats, cfg.flatten_time
#     )  #cfg.eval_dir
#     latent_files = sorted(Path(latent_dir).glob("*.pt"))

#     # VAE  case 1 for now it is different that the diffusion model
#     # model_checkpoint = torch.load(vae_checkpoint, map_location=device)
#     # # saved_stats = checkpoint["stats"]
#     # # print(f"masked VAE statistic: {saved_stats}")
#     # masked_vae_model = CycloneVAE(in_channels=len(cfg.variable_names), latent_channels=128).to(device)
#     # masked_vae_model.load_state_dict(model_checkpoint["model_state"])

#     ckpt = torch.load(vae_checkpoint, map_location=device)
#     C = len(cfg.variable_names)

#     # channel order must match training, otherwise the variables get swapped silently
#     assert list(ckpt["vars_list"]) == list(cfg.variable_names)

#     masked_vae_model = CycloneVAE(
#         in_channels=ckpt.get("in_channels", 2 * C),
#         out_channels=ckpt.get("out_channels", C),
#         latent_channels=128,
#     ).to(device)
#     masked_vae_model.load_state_dict(ckpt["model_state"])
#     masked_vae_model.eval()

#     vae_saved_stats = ckpt["stats"]   # mean/std used for normalisation


#     all_records = []

#     for i in range(len(eval_dataset)):
#     # for i in range(2):
#         sample = eval_dataset[i].unsqueeze(0)
#         make_plot = plot_sample_indices is not None and i in plot_sample_indices

#         if type_case == "case3":
#             target_frame_idx = 1 #i % cfg.num_frames here i use the middle timestep because of the difusion model
#             metadata = eval_dataset.get_metadata(i, frame_idx=target_frame_idx)
#             payload = torch.load(latent_files[i], map_location=device, weights_only=False)
#             print(f"source_file eval_dataset: {Path(metadata.source_file).name}")
#             print(f"source_file payload: {Path(payload['source_file']).name}")
#             check_times_match(payload, metadata, target_frame_idx)

#             case3_metrics, _, _, _, _ = visualize_case3_multi_model_vs_baseline(
#                 videomae_model=videomae_model,
#                 # vae_model=vae_model_diffusion,
#                 # denoiser=denoiser,
#                 # schedule=schedule,
#                 # scale_factor=scale_factor,
#                 imputer=imputer,
#                 payload=payload,
#                 cfg=cfg,
#                 clean_sample_cthw=sample, #[1, C, T, H, W]
#                 target_frame_idx=target_frame_idx,
#                 device=device,
#                 videomae_norm_stats=videomae_norm_stats,
#                 diffusion_norm_stats=diffusion_norm_stats,
#                 save_path=images_dir,
#                 metadata=metadata,
#                 make_plot=make_plot,
#             )
#             all_records.extend(_flatten_sample_metrics(
#                 case3_metrics, i, type_case, metadata.time_strs[target_frame_idx], metadata
#             ))
#             _save_eval_metrics(images_dir, type_case, all_records, column_final="row")


#         elif type_case == "case1":
#             print(f"sample.shape[2]: {sample.shape[2]}")
#             # update because now is predicting 3 timesteps and only obtain one each time instead of directly
#             # after the prediction obtaine the metrics
#             for target_frame_idx in range(3): # sample.shape[2] n_timesteps by sample = 3
#                 metadata = eval_dataset.get_metadata(i, frame_idx=target_frame_idx)
#                 case1_metrics, _, _, _= visualize_case1_multi_model_vs_baseline(
#                     videomae_model=videomae_model,
#                     vae_model=masked_vae_model,
#                     cfg=cfg,
#                     clean_sample_cthw=sample,
#                     target_frame_idx=target_frame_idx,
#                     device=device,
#                     videomae_norm_stats=videomae_norm_stats,
#                     other_model_norm_stats=vae_saved_stats, #videomae_norm_stats, # they are the same 99.99999......
#                     save_path=images_dir,
#                     metadata=metadata,
#                     # only_box=only_box
#                     make_plot=make_plot,
#                 )
#                 all_records.extend(_flatten_sample_metrics(
#                     case1_metrics, i, type_case, metadata.time_strs[target_frame_idx], metadata
#                 ))

#             # columns_row: channel, frame, label   subcase fix here frame  row variable channel
#             _save_eval_metrics(images_dir, type_case, all_records, column_final="row")

#         elif type_case == "case2":
#             metadata = eval_dataset.get_metadata(i)

#             for channel_idx in range(7): # along all variables
#                 # channel_idx = 0  #variable 10u
#                 case2_metrics, _, _= visualize_case2_multi_model_vs_baseline(
#                     videomae_model=videomae_model,
#                     vae_model=masked_vae_model,
#                     cfg=cfg,
#                     clean_sample_cthw=sample,
#                     channel_idx=channel_idx,
#                     device=device,
#                     videomae_norm_stats=videomae_norm_stats,
#                     other_model_norm_stats=videomae_norm_stats, # they are the same 99.99999......
#                     save_path=images_dir,
#                     metadata=metadata,
#                     make_plot=make_plot,
#                     plot_frame_indices=plot_case2_frame_indices,
#                 )
#                 all_records.extend(_flatten_sample_metrics(
#                     case2_metrics,
#                     i,
#                     type_case,
#                     cfg.variable_names[channel_idx],
#                     metadata,
#                 ))
#             # columns_row: channel, frame, label   subcase fix here channel  row variable timesteps 
#             _save_eval_metrics(images_dir, type_case, all_records, column_final="subcase")


# if __name__ == "__main__":
#     # type_case = "case1"  
#     # type_case = "case2"  
#     type_case = "case3"  

#     for type_case in ["case2", "case3", "case1",]:
#     # for type_case in ["case1"]: # run metrics 50 min
#     # for type_case in ["case1", "case2"]: # run metrics 12 min
#     # for type_case in ["case3"]: # run metrics 12 min
#         cfg = ProjectConfig()

#         run_unified_inference_tests(cfg,
#         eval_dir=EVAL_DIR_TEST,
#         latent_dir=LATENTS_DIR_TEST,
#         vae_diffusion_checkpoint=VAE_DIFFUSION_CHECKPOINT,
#         vae_checkpoint=CHECKPOINT_MASKED_VAE,
#         diffusion_checkpoint=DIFFUSION_CHECKPOINT,
#         videomae_checkpoint=VIDEOMAE_CHECKPOINT,
#         images_dir=SAVE_ALL_DIR,
#         type_case=type_case,
#         plot_sample_indices = [0,1], #None,
#         # plot_sample_indices = None,
#         # only_box= True,
#         # only_box= False,

#     )