#!/bin/bash

#SBATCH --job-name=7fold                    # Job name
#SBATCH --nodes=7 --ntasks=7  --ntasks-per-node=1 --gres=gpu:RTX_6000:4 --mem-per-cpu=24GB                 # Number of no$
#SBATCH -t300
#SBATCH --output=logs/7fold.out
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=dmartin305@gatech.edu        # E-mail address for notifications

cd $SLURM_SUBMIT_DIR                            # Change to working directory
module load anaconda3                           # Load module dependencies
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1
pip install pandas numpy torch keras tensorflow pyarrow fastparquet
srun --nodes=1 --ntasks=1 python3 train_25_sign.py 0 &
srun --nodes=1 --ntasks=1 python3 train_25_sign.py 1 & 
srun --nodes=1 --ntasks=1 python3 train_25_sign.py 2 &
srun --nodes=1 --ntasks=1 python3 train_25_sign.py 3 &
srun --nodes=1 --ntasks=1 python3 train_25_sign.py 4 &
srun --nodes=1 --ntasks=1 python3 train_25_sign.py 5 &
srun --nodes=1 --ntasks=1 python3 train_25_sign.py 6 &
wait