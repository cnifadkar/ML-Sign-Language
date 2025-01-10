# team-44-cs-4641-project
Team 44 CS 4641 Project

**Note: we do not outline files that were no longer relevant in the scope of this project.**

/train_mediapipe_25_sign/: This directory is used on PACE
1. 7fold.sbatch: This file is used to deploy a SLURM job on pace
2. architechtures.py: This file contains the 3 models implemented
3. popsign_dataset.py: This file contains the DataLoader-esse class used to modulate loading and preprocessing data
4. train_25_sign.py: This file acts as our main script, it loads data, trains models, and saves results.

/data/: this directory contains subsets of the full dataset used for training
1. /filtered_data/: contains 25 signs from 21 users 
2. /filtered_data_5/: contains 5 signs from 21 users, used for troubleshooting
Nested in each of these data subsets: 
1. /data/filtered_data/train.csv: This file is a list of the metadata of all landmarks in the parquet files and its respective labels to train on.
2. /data/filtered_data/sign_to_prediction_index_map.json: This file matches the labels to a number to use.
3. /data/filtered_data/train_landmark_files: This directory contains all the data that is being used for this project.

/results/: This directory contains the results of: 
1. /eval_history/: evaluation across the folds and architectures of test
2. /pred_history/: the predictions across the folds and architectures of test samples
3. /train_history/: the training history across the folds and architectures 

filter_data.py: script that filtered out a subset of the data results 
/dir/results.html: This file contains the results of our models.
/dir/results.ipynb: This file contains the results of our models in a jupyter notebook.


