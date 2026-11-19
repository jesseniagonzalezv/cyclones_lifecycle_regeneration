#!/bin/bash

# source ~/.bashrc
#SBATCH --job-name=plot_comparison  
#SBATCH --partition=gpu
#SBATCH --exclude=compute-117-39
#SBATCH --mem=32GB 
#SBATCH --gpus=1                 
#SBATCH --cpus-per-gpu=8 #4
#SBATCH --time=16:00:00            #total run time
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jessenia.gv@gmail.com 
#SBATCH --output=/scale/project/ifremer-isi-jumeaunumerique/NEXTGEMS_PROCESSED_LAT_LON/ml_result/log_plot_comparison%j.txt   
#SBATCH --error=/scale/project/ifremer-isi-jumeaunumerique/NEXTGEMS_PROCESSED_LAT_LON/ml_result/log_plot_comparison%j_err.txt    # separate error file




echo "started"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1


# ---- plots comparison
# micromamba run -n cyclone_mamba python -m cyclone_lifecycle.evaluation.plot_comparison # case 1 took like 5 hours verify what is happening 
micromamba run -n cyclone_mamba python -m cyclone_lifecycle.evaluation.metric_plots --metrics-dir "/scale/project/ifremer-isi-jumeaunumerique/NEXTGEMS_PROCESSED_LAT_LON/result_ocean_model/result_metrics" --output-dir "/scale/project/ifremer-isi-jumeaunumerique/NEXTGEMS_PROCESSED_LAT_LON/result_ocean_model/result_metrics/metrics"


# micromamba run -n cyclone_mamba python metric_plots.py --metrics-dir "/home/jgonzale/cyclone_atm/all_models/adaptations_models/result_metrics" --output-dir "/home/jgonzale/cyclone_atm/all_models/adaptations_models/result_metrics/metrics"
# micromamba run -n cyclone_mamba python metric_plots.py --metrics-dir "/scale/project/ifremer-isi-jumeaunumerique/NEXTGEMS_PROCESSED_LAT_LON/ml_result/join_plots/" --output-dir "/scale/project/ifremer-isi-jumeaunumerique/NEXTGEMS_PROCESSED_LAT_LON/ml_result/join_plots/metrics" 
echo "end"

