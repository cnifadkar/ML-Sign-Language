import pandas as pd 
import numpy as np 
import json
import shutil
import os

SAMPLES_PER_PARTICIPANT = 10
NUM_OF_SIGNS = 25

df = pd.read_csv('data/kaggle_data/train.csv')

####################################################
# 
# DO NOT RUN ANY OF THESE IT WILL OVERWRITE DATA 
#
####################################################

def obtain_filtered_data():
    
    FILTERED_ROOT = 'data/filtered_data/'
    
    # Eleminate any signs that dont meet a minimum number of samples across all participants
    pivot_table = df.pivot_table(index='sign', columns= 'participant_id', aggfunc= 'size', fill_value = 0)
    eligible_signs_mask = pivot_table.min(axis =1) >= SAMPLES_PER_PARTICIPANT
    eligible_signs = pivot_table[eligible_signs_mask].index.tolist()

    # Select signs that meet the mimimum critera 
    selected_signs = np.random.choice(eligible_signs, size= NUM_OF_SIGNS, replace = False)

    # Create JSON for sign_to_prediction_index_map.json
    json_map = {}
    
    # Get said data for selected signs
    filtered_samples = pd.DataFrame()
    i = 0
    for sign in selected_signs:
        
        # write to dictionary
        json_map[sign] = i
        i += 1
        
        sign_df = df[df['sign'] == sign]
        for participant in sign_df['participant_id'].unique():
            participant_sign_df = sign_df[sign_df['participant_id'] == participant]
            samples = participant_sign_df.sample(n = SAMPLES_PER_PARTICIPANT)
            filtered_samples = pd.concat([filtered_samples, samples])
    
    # Write a new train.csv file 
    filtered_samples.reset_index(drop= True, inplace= True)
    filtered_samples.to_csv(FILTERED_ROOT + 'train.csv')
    
    # Write a new sign_to_prediciton_index_map.json
    with open(FILTERED_ROOT + "sign_to_prediction_index_map.json", "w") as outfile:
        json.dump(json_map, outfile)
    
    return filtered_samples

def filter_out_samples(sample_df): 
    print(sample_df)
    KAGGLE_ROOT = 'data/kaggle_data/'
    FILTERED_ROOT = 'data/filtered_data/'
    
    # Copy landmark file to filtered data 
    paths = sample_df['path']
    for index, path in paths.items():   
        src = KAGGLE_ROOT + path
        dst = FILTERED_ROOT + path
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
    
def count_min_samples(df):
    # Get the number of samples provided by particpants for each sign
    pivot_table = df.pivot_table(index='sign', columns= 'participant_id', aggfunc= 'size', fill_value = 0)
    # Create a new column to store the min value
    pivot_table['min'] = pivot_table.min(axis=1)
    # Drop the count of samples for all participants
    pivot_table = pivot_table[['min']].reset_index()
    pivot_table.to_csv("min_sample_by_sign.csv", index = False)
    
