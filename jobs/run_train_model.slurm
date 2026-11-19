#!/bin/bash

# source ~/.bashrc
#SBATCH --job-name=videomae #vae 
##SBATCH --partition=cpu
#SBATCH --partition=gpu
#SBATCH --exclude=compute-117-39
#SBATCH --mem=128GB #128GB train ml #20G #difussion #128G ML #64G
#SBATCH --gpus=1 #1                   
#SBATCH --cpus-per-gpu=8 #4
#SBATCH --time=12:00:00            #total run time
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jessenia.gv@gmail.com 
##SBATCH --output=/scale/project/ifremer-isi-jumeaunumerique/NEXTGEMS_PROCESSED_LAT_LON/ml_result/log_train_test_vae%j.txt   
##SBATCH --error=/scale/project/ifremer-isi-jumeaunumerique/NEXTGEMS_PROCESSED_LAT_LON/ml_result/log_train_test_vae%j_err.txt    # separate error file
#SBATCH --output=/scale/project/ifremer-isi-jumeaunumerique/NEXTGEMS_PROCESSED_LAT_LON/ml_result/log_train_test_videomae%j.txt   
#SBATCH --error=/scale/project/ifremer-isi-jumeaunumerique/NEXTGEMS_PROCESSED_LAT_LON/ml_result/log_train_test_videomae%j_err.txt    # separate error file




echo "started"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1


# micromamba run -n cyclone_mamba python -m cyclone_lifecycle.models.vae_case1_case2

#1
micromamba run -n cyclone_mamba python -m cyclone_lifecycle.models.videomae_bs_3days
# micromamba run -n cyclone_mamba python -m cyclone_lifecycle.models.videomae_bs_3days --mode "infer"

# micromamba run -n cyclone_mamba python test_videomae_cnn.py 
# micromamba run -n cyclone_mamba python test_videomae_cnn.py --mode "infer" --ckpt "/scale/project/ifremer-isi-jumeaunumerique/NEXTGEMS_PROCESSED_LAT_LON/test_videomae_cnn/model_epoch_30.pt"
#  this
# micromamba run -n cyclone_mamba python test_videomae_cnn.py --mode "infer" 

#2
# bash
# micromamba run -n cyclone_mamba python -m cyclone_lifecycle.models.latent_diffusion.run --stage all

# ---- plots comparison
# micromamba run -n cyclone_mamba python -m cyclone_lifecycle.evaluation.plot_comparison


echo "end"


#bash

# micromamba activate cyclone_mamba
# squeue --me
# 

## el cnn parece que se ve mejor en tal caso train con eso pero agregar el missing timestep 3
# el difussion necesito 