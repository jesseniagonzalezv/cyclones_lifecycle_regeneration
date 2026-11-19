#!/bin/bash

# source ~/.bashrc
#SBATCH --job-name=3ocean_model
##SBATCH --partition=cpu
#SBATCH --partition=gpu
#SBATCH --exclude=compute-117-39
#SBATCH --mem=128GB #128GB train ml #20G #difussion #128G ML #64G
#SBATCH --gpus=1 #1                   
#SBATCH --cpus-per-gpu=8 #4
#SBATCH --time=12:00:00            #total run time
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jessenia.gv@gmail.com 
#SBATCH --output=/scale/project/ifremer-isi-jumeaunumerique/NEXTGEMS_PROCESSED_LAT_LON/result_ocean_model/log_train_test_ocean_model%j.txt   
#SBATCH --error=/scale/project/ifremer-isi-jumeaunumerique/NEXTGEMS_PROCESSED_LAT_LON/result_ocean_model/log_train_test_ocean_model%j_err.txt 


echo "started"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1



#1
# micromamba run -n cyclone_mamba python test_ocean_var.py --epochs 30 --core 32
epochs=60
# python -m cyclone_lifecycle.models.ocean_model --stage train,eval --epochs $epochs                                         # A: correction to baseline (default)
# python -m cyclone_lifecycle.models.ocean_model --stage train,eval --no-residual --tag _fullfield --epochs $epochs     # B: predicts the full field
# python -m cyclone_lifecycle.models.ocean_model --stage train,eval --no-residual --bottleneck 8 --tag _fullfield_bn8 --epochs $epochs  # C: full field + VAE-like bottleneck
# python -m cyclone_lifecycle.models.ocean_model --stage train,eval --bottleneck 8 --tag _residual_bn8 --epochs $epochs     # D: correction + bottleneck
# python -m cyclone_lifecycle.models.ocean_model --stage train,eval --no-atm --tag _residual_noatm --epochs $epochs                 # E
# python -m cyclone_lifecycle.models.ocean_model --stage train,eval --no-atm --no-residual --tag _fullfield_noatm --epochs $epochs  # F
# python -m cyclone_lifecycle.evaluation.compare_variants                          # all variants found, val
# python -m cyclone_lifecycle.evaluation.compare_variants --scope core --out val_core.csv

python -m cyclone_lifecycle.evaluation.compare_variants --plot                              # middle day + skill heatmap, on val
python -m cyclone_lifecycle.evaluation.compare_variants --plot --plot-scenarios mid,last    # more scenarios
python -m cyclone_lifecycle.evaluation.compare_variants --plot --scope core                 # storm core instead of the whole patch

# example: the winner turned out to be B
# python -m cyclone_lifecycle.models.ocean_model --stage eval --split test --tag _fullfield
echo "end"

