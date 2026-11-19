#!/bin/bash

#source ~/.bashrc
#SBATCH --job-name=regrid_lat_lot
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --exclude=compute-117-39
#SBATCH --mem=2G  #4G
#SBATCH --time=2:30:00 #08:00:00            #
##SBATCH --array=1       # test sample
#SBATCH --array=1-8%8                  # test Runs batches 1 to 8 (max 8 running at once)
##SBATCH --array=1-16%8                  # train Runs batches 1 to 8 (max 8 running at once)
##SBATCH --array=6-8%8                  # Runs batches 1 to 5 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jessenia.gv@gmail.com # Email address for notifications
#SBATCH --output=/scale/project/ifremer-isi-jumeaunumerique/NEXTGEMS_PROCESSED_LAT_LON/data_128x128_3days/log_regrid_batch_%a_%j.txt   # %a is the batch number, %j is the job ID
#SBATCH --error=/scale/project/ifremer-isi-jumeaunumerique/NEXTGEMS_PROCESSED_LAT_LON/data_128x128_3days/log_regrid_batch_%a_%j_err.txt    # separate error file


#eval "$(micromamba shell hook --shell bash)"
#micromamba activate cyclone_mamba

#micromamba activate /ontap/user/jgonzale/conda-env/cyclone_mamba

echo "started"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
# bash
# micromamba run -n cyclone_mamba python regrid_lat_lon.py
# Pass the batch number ($SLURM_ARRAY_TASK_ID) and batch size directly to Python
#micromamba run -n cyclone_mamba python regrid_lat_lon.py --batch $SLURM_ARRAY_TASK_ID --n_tracks 200

# this has a code that consider the selectiong given the center of the cyclone but we will have irregular times
# /home/jgonzale/conda-env/cyclone_mamba/bin/python regrid_lat_lon_irregular_time.py --batch $SLURM_ARRAY_TASK_ID --n_tracks 200

# here given one center of the cyclone then i will select the day previus and the next day
# micromamba run -n cyclone_mamba python regrid_lat_lon_3_days.py --batch $SLURM_ARRAY_TASK_ID --n_tracks 200


# -------------------- new code to create the patches for train, test and val
# micromamba run -n cyclone_mamba python regridding_3timesteps.py  --batch $SLURM_ARRAY_TASK_ID --n_tracks 400 --set "train" #

# micromamba run -n cyclone_mamba python regridding_3timesteps.py  --batch $SLURM_ARRAY_TASK_ID --n_tracks 200 --set "test" # All 8 Elapsed: 	0:15:26
micromamba run -n cyclone_mamba python -m cyclone_lifecycle.preprocessing.regridding_3timesteps --batch "$SLURM_ARRAY_TASK_ID" --n_tracks 200 --set "val"

echo "end"

