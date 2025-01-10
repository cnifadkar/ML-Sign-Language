from popsign_dataset import PopsignDataModule as Popsign
from architectures import Architectures
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser(description="The Nth fold to train on.")

parser.add_argument('fold', type=int, help="Nth fold")

args = parser.parse_args()

popsignData = Popsign(
        data_dir= "data/filtered_data", 
        batch_size = 32, 
        classes= 25,
        num_of_samples= 10,
        feature_groups= ['hand'],
        fold_idx= args.fold,
        max_frames = 350)

X_train, Y_train, X_test, Y_test = popsignData.test_train_split()

architectures = Architectures()
cnn1d = architectures.create1dCNN(X_train, popsignData.classes)
cnn2d = architectures.create2dCNN(X_train, popsignData.classes)
lstm = architectures.createLSTM(X_train, popsignData.classes)

# Create Directories to store results from experiments 
HISTORY_ROOT = f'results/mediapipe_hands_25_sign/train_history'
EVAL_ROOT = f'results/mediapipe_hands_25_sign/eval_history'
PRED_ROOT = f'results/mediapipe_hands_25_sign/pred_history'
os.makedirs(HISTORY_ROOT, exist_ok=True)
os.makedirs(EVAL_ROOT, exist_ok=True)
os.makedirs(PRED_ROOT, exist_ok=True)

# Save the ground truth values of Y_test for later computation w DCG, MRR, etc.
ground_truth = pd.DataFrame(Y_test)
ground_truth.to_csv(PRED_ROOT + "/ground_truth.csv")

# ==========================================================================
# CNN 1D 
# ==========================================================================
cnn1d_history = cnn1d.fit(X_train, Y_train, epochs = 50)
cnn1d_history_df = pd.DataFrame(cnn1d_history.history)
cnn1d_history_df.to_csv(HISTORY_ROOT + f'/fold_{args.fold}_cnn1d_history.csv')

cnn1d_eval = cnn1d.evaluate(X_test, Y_test)
cnn1d_eval_df = pd.DataFrame(cnn1d_eval)
cnn1d_eval_df.to_csv(EVAL_ROOT + f'/fold_{args.fold}_cnn1d_history.csv')

cnn1d_pred = cnn1d.predict(X_test)
cnn1d_pred_df = pd.DataFrame(cnn1d_pred)
cnn1d_pred_df.to_csv(PRED_ROOT + f'/fold_{args.fold}_cnn1d_history.csv')


# ==========================================================================
# CNN 2D 
# ==========================================================================
cnn2d_history = cnn2d.fit(X_train, Y_train, epochs = 50)
cnn2d_history_df = pd.DataFrame(cnn2d_history.history)
cnn2d_history_df.to_csv(HISTORY_ROOT + '/cnn2d_history.csv')

cnn2d_eval = cnn2d.evaluate(X_test, Y_test)
cnn2d_eval_df = pd.DataFrame(cnn2d_eval)
cnn2d_eval_df.to_csv(EVAL_ROOT + f'/fold_{args.fold}_cnn2d_history.csv')

cnn2d_pred = cnn1d.predict(X_test)
cnn2d_pred_df = pd.DataFrame(cnn2d_pred)
cnn2d_pred_df.to_csv(PRED_ROOT + f'/fold_{args.fold}_cnn2d_history.csv')


# ==========================================================================
# LSTM
# ==========================================================================
lstm_history = lstm.fit(X_train, Y_train, epochs = 50)
lstm_history_df = pd.DataFrame(lstm_history.history)
lstm_history_df.to_csv(HISTORY_ROOT + '/lstm_history.csv')

lstm_eval = lstm.evaluate(X_test, Y_test)
lstm_eval_df = pd.DataFrame(lstm_eval)
lstm_eval_df.to_csv(EVAL_ROOT + f'/fold_{args.fold}_lstm_history.csv')

lstm_pred = lstm.predict(X_test)
lstm_pred_df = pd.DataFrame(lstm_pred)
lstm_pred_df.to_csv(PRED_ROOT + f'/fold_{args.fold}_lstm_history.csv')





