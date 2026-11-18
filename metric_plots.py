
"""Create compact per-case plots from the saved evaluation metric CSV files."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update(
	{
		"font.family": "DejaVu Sans",
		"font.size": 12,
		"axes.titlesize": 14,
		"axes.labelsize": 13,
		"figure.titlesize": 16,
		"legend.fontsize": 15,
	}
)

METRICS_DIR = Path(__file__).parent / "metrics"
VARIABLE_ORDER = ["10u", "10v", "msl", "avg_tos", "avg_zos", "avg_sos", "avg_mlotst125"]
# MODEL_ORDER = ["VideoMAE", "masked VAE", "Diffusion", "Linear Interpolation"]
MODEL_ORDER = ["VideoMAE", "masked VAE", "Diffusion", "Diffusion + ocean residual", "Linear Interpolation"]
PLOT_STYLE = {
	"figure_size": (17, 5.2),
	"dpi": 180,
	"title_template": "{case}: average {metric} by variable and model in the test set",
	"title_y": 0.985,
	"model_legend_y": 0.105,
	"subplot_rect": (0, 0.16, 1, 0.95),
}
MODEL_COLORS = {
	"VideoMAE":"#2878B5", # blue
	"masked VAE": "#E07A32", # orange
	"Diffusion": "#3A9D70", # green
	"Diffusion + ocean residual": "#C768A3", # purple
	"Linear Interpolation": "#777777",
}

VARIABLE_LABELS = {
	"10u": "10u (m/s)",
	"10v": "10v (m/s)",
	"msl": "MSL (hPa)",
	"avg_tos": "SST (°C)",
	"avg_zos": "SSH (m)",
	"avg_sos": "SSS (psu)",
	"avg_mlotst125": "MLD (m)",
}




# METRIC_LABELS = {
# 	"nrmse": "NRMSE (normalized, lower is better)",
# 	"mae": "MAE (lower is better)",
# 	"rmse": "RMSE (lower is better)",
# 	"bias": "Bias (signed; closer to zero is better)",
# }

METRIC_LABELS = {
	"nrmse": "NRMSE (normalized)",
	"mae": "MAE",
	"rmse": "RMSE",
	"bias": "Bias",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
	with path.open(newline="", encoding="utf-8") as csv_file:
		return list(csv.DictReader(csv_file))


def _case1_summary(path: Path, metric: str) -> dict[tuple[str, str], tuple[float, float]]:
	values = defaultdict(list)
	for record in _read_csv(path):
		value = record.get(metric, "")
		if value:
			values[(record["row"], record["model"])].append(float(value))
	return {
		key: (float(np.mean(samples)), float(np.std(samples)))
		for key, samples in values.items()
	}


def _summary_case(path: Path, metric: str, variable_column: str) -> dict[tuple[str, str], tuple[float, float]]:
	summary = {}
	for record in _read_csv(path):
		if record["metric"] == metric:
			summary[(record[variable_column], record["model"])] = (
				float(record["mean"]),
				float(record["std"]),
			)
	return summary


def _plot_case(
	case_name: str,
	summary: dict[tuple[str, str], tuple[float, float]],
	metric: str,
	output_dir: Path,
) -> Path:
	figure, axes = plt.subplots(
		1,
		len(VARIABLE_ORDER),
		figsize=PLOT_STYLE["figure_size"],
		squeeze=False,
	)
	plotted_models = [
		model for model in MODEL_ORDER
		if any((variable, model) in summary for variable in VARIABLE_ORDER)
	]
	bar_width = 0.78 / max(len(plotted_models), 1)

	for variable_index, variable in enumerate(VARIABLE_ORDER):
		axis = axes[0, variable_index]
		for model_index, model in enumerate(plotted_models):
			result = summary.get((variable, model))
			if result is None:
				continue
			mean, std = result
			x_position = model_index - (len(plotted_models) - 1) / 2
			axis.bar(
				x_position * bar_width,
				mean,
				width=bar_width * 0.9,
				yerr=std,
				capsize=2,
				color=MODEL_COLORS[model],
				label=model if variable_index == 0 else None,
				error_kw={"elinewidth": 0.8},
			)
		axis.set_title(
			VARIABLE_LABELS.get(variable, variable),
		)
		axis.set_xticks([])
		axis.grid(axis="y", color="#D9DEE5", linewidth=0.7)
		axis.set_axisbelow(True)
		axis.spines[["top", "right"]].set_visible(False)
		if variable_index == 0:
			axis.set_ylabel(METRIC_LABELS[metric])

	model_handles = [
		Patch(facecolor=MODEL_COLORS[model], label=model)
		for model in plotted_models
	]
	figure.suptitle(
		PLOT_STYLE["title_template"].format(case=case_name.upper(), metric=metric.upper()),
		y=PLOT_STYLE["title_y"],
	)
	figure.legend(
		handles=model_handles,
		labels=plotted_models,
		loc="lower center",
		bbox_to_anchor=(0.5, PLOT_STYLE["model_legend_y"]),
		ncol=len(plotted_models),
		frameon=False,
	)
	figure.tight_layout(rect=PLOT_STYLE["subplot_rect"])
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / f"{case_name}_{metric}.png"
	figure.savefig(output_path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight")
	plt.close(figure)
	return output_path


def create_metric_plots(metrics_dir: Path, output_dir: Path, metric: str) -> list[Path]:
	case1 = _summary_case(metrics_dir / "case1_dataset_summary.csv", metric, "row")
	case2 = _summary_case(metrics_dir / "case2_dataset_summary.csv", metric, "subcase")
	case3 = _summary_case(metrics_dir / "case3_dataset_summary.csv", metric, "row")

	return [
		_plot_case("case1", case1, metric, output_dir),
		_plot_case("case2", case2, metric, output_dir),
		_plot_case("case3", case3, metric, output_dir),
	]


def main() -> None:
	parser = argparse.ArgumentParser(description="Plot compact model metric comparisons by case.")
	parser.add_argument("--metric", choices=METRIC_LABELS, default="rmse") # "nrmse"
	parser.add_argument("--metrics-dir", type=Path, default=METRICS_DIR)
	parser.add_argument("--output-dir", type=Path, default=METRICS_DIR / "plots")
	args = parser.parse_args()

	for output_path in create_metric_plots(args.metrics_dir, args.output_dir, args.metric):
		print(f"Saved {output_path}")


if __name__ == "__main__":
	main()