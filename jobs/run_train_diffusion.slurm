#!/bin/bash

# source ~/.bashrc
#SBATCH --job-name=diffusion 
##SBATCH --partition=cpu
#SBATCH --partition=gpu
#SBATCH --exclude=compute-117-39
#SBATCH --mem=128GB #128GB train ml #20G #difussion #128G ML #64G
#SBATCH --gpus=1 #1                   
#SBATCH --cpus-per-gpu=8 #4
#SBATCH --time=12:00:00            #total run time
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jessenia.gv@gmail.com 
#SBATCH --output=/scale/project/ifremer-isi-jumeaunumerique/NEXTGEMS_PROCESSED_LAT_LON/ml_result/log_train_test_diffusion%j.txt   
#SBATCH --error=/scale/project/ifremer-isi-jumeaunumerique/NEXTGEMS_PROCESSED_LAT_LON/ml_result/log_train_test_diffusion%j_err.txt    # separate error file




echo "started"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1

micromamba run -n cyclone_mamba python -m cyclone_lifecycle.models.latent_diffusion.run --stage diffusion


echo "end"

