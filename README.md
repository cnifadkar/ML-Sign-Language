Here’s a more polished and concise **README** for your project:

---

# **ML Sign Language Project**

This repository contains our final project for **CS 4641: Machine Learning**. Below is an overview of the key files and directories relevant to our work.

---

## **Project Structure**

### **1. Main Directories**
- **`/train_mediapipe_25_sign/`**: Contains scripts and files used on PACE for running experiments.
- **`/data/`**: Subsets of the full dataset used for training.
  - **`/filtered_data/`**: 25 signs from 21 users.
  - **`/filtered_data_5/`**: 5 signs from 21 users, used for troubleshooting.

- **`/results/`**: Stores the results of model evaluations and training:
  - **`/eval_history/`**: Evaluation results across folds and architectures.
  - **`/pred_history/`**: Predictions for test samples across folds and architectures.
  - **`/train_history/`**: Training history across folds and architectures.

---

### **2. Key Files**
- **`7fold.sbatch`**: SLURM job script for deploying tasks on PACE.
- **`architectures.py`**: Contains the three models implemented for this project.
- **`popsign_dataset.py`**: Data loader class for loading and preprocessing data.
- **`train_25_sign.py`**: Main script for loading data, training models, and saving results.
- **`filter_data.py`**: Script for filtering and creating subsets of the dataset.

---

### **3. Data File Descriptions**
- **`/data/filtered_data/train.csv`**: Metadata of landmarks and their respective labels for training.
- **`/data/filtered_data/sign_to_prediction_index_map.json`**: Maps labels to numerical indices.
- **`/data/filtered_data/train_landmark_files/`**: Contains the actual data used in the project.

---

### **4. Results Files**
- **`results.html`**: A summary of model results.
- **`results.ipynb`**: Jupyter Notebook containing detailed results and visualizations.

---

## **Notes**
- Irrelevant files from earlier stages of the project are not included in this outline for clarity.

---
