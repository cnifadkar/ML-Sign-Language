import pandas as pd


def get_features(parquet_path, feature_types = ['face','pose','hand']):
    
    data = pd.read_parquet(parquet_path)
    USE_FACE = 'face' in feature_types 
    USE_POSE = 'pose' in feature_types
    USE_HAND = 'hand' in feature_types

    right_hand = data[data['type'].str.contains('right_hand')]
    left_hand = data[data['type'].str.contains('left_hand')]
    pose = data[data['type'].str.contains('pose')]
    face = data[data['type'].str.contains('face')]

    # Create lists where the element contain the x,y,z 
    # values for specific landmarks at each frame
    right_hand_landmarks = []
    left_hand_landmarks = []
    pose_landmarks = []
    face_landmarks = []
    # merged contains all the landmarks that are requested 
    merged = []

    # Get a list of all of the right hand landmarks by the landmark index
    if USE_HAND:
        for i in right_hand['landmark_index'].unique():
            curr_right_hand_landmarks = pose[pose['landmark_index'] == i].copy()
            curr_right_hand_landmarks.rename(
                columns={
                    "x": "x_right_hand_" + str(i),
                    "y": "y_right_hand_" + str(i),
                    "z": "z_right_hand_" + str(i),
                },
                inplace=True,
            ) 
            curr_right_hand_landmarks.drop(
                    ["row_id", "type", "landmark_index"], axis=1, inplace=True
                )
            curr_right_hand_landmarks.reset_index(drop=True, inplace=True)
            curr_right_hand_landmarks.set_index("frame", inplace=True)
            right_hand_landmarks.append(curr_right_hand_landmarks)
        # Get a list of all of the left hand landmarks by the landmark index
        for i in left_hand['landmark_index'].unique():
            curr_left_hand_landmarks = pose[pose['landmark_index'] == i].copy() 
            curr_left_hand_landmarks.rename(
                columns={
                    "x": "x_left_hand_" + str(i),
                    "y": "y_left_hand_" + str(i),
                    "z": "z_left_hand_" + str(i),
                },
                inplace=True,
            ) 
            curr_left_hand_landmarks.drop(
                    ["row_id", "type", "landmark_index"], axis=1, inplace=True
                )
            curr_left_hand_landmarks.reset_index(drop=True, inplace=True)
            curr_left_hand_landmarks.set_index("frame", inplace=True)
            left_hand_landmarks.append(curr_left_hand_landmarks)
            
        merged_right_hand = pd.concat(right_hand_landmarks, axis = 1)
        merged_left_hand = pd.concat(left_hand_landmarks, axis = 1)
        
        # Handle hand dominance
        right_hand_nans = merged_right_hand.isna().sum().sum()
        left_hand_nans = merged_left_hand.isna().sum().sum()

        right_handed = left_hand_nans >= right_hand_nans
        
        if right_handed:
            merged_right_hand.columns = merged_right_hand.columns.str.replace("_right", "")
            merged.append(merged_right_hand)
        else: 
            center = 0.5
            x_col_left = [col for col in merged_left_hand.columns if "x" in col]
            for col in x_col_left:
                merged_left_hand[col] = center - (merged_left_hand[col] - center)
            merged_left_hand.columns = merged_left_hand.columns.str.replace("_left", "")
            merged.append(merged_left_hand)

    # Get a list of all of the pose landmarks by the landmark index    
    if USE_POSE:
        for i in pose['landmark_index'].unique():
            curr_pose_landmark = pose[pose['landmark_index'] == i].copy() 
            curr_pose_landmark.rename(
                columns={
                    "x": "x_pose_" + str(i),
                    "y": "y_pose_" + str(i),
                    "z": "z_pose_" + str(i),
                },
                inplace=True,
            ) 
            curr_pose_landmark.drop(
                    ["row_id", "type", "landmark_index"], axis=1, inplace=True
                )
            curr_pose_landmark.reset_index(drop=True, inplace=True)
            curr_pose_landmark.set_index("frame", inplace=True)
            pose_landmarks.append(curr_pose_landmark)
        merged_pose = pd.concat(pose_landmarks, axis = 1)
        merged.append(merged_pose)
        
    # Get a list of all of the face landmarks by the landmark index
    if USE_FACE:
        for i in face['landmark_index'].unique():
            curr_face_landmark = face[face['landmark_index'] == i].copy() 
            curr_face_landmark.rename(
                columns={
                    "x": "x_face_" + str(i),
                    "y": "y_face_" + str(i),
                    "z": "z_face_" + str(i),
                },
                inplace=True,
            ) 
            curr_face_landmark.drop(
                    ["row_id", "type", "landmark_index"], axis=1, inplace=True
                )
            curr_face_landmark.reset_index(drop=True, inplace=True)
            curr_face_landmark.set_index("frame", inplace=True)
            face_landmarks.append(curr_face_landmark)
        merged_face = pd.concat(face_landmarks, axis = 1)        
        merged.append(merged_face)
    
    return pd.concat(merged, axis = 1)

def preprocess_data(path, DROP_Z = True):
    """Pass in the path name of parquet file to 
    preprocess. The output is a pandas DataFrame of the preprocessed data
    to be used in training.

    Args:
        path (str): Path to Parquet File        
        DROP_Z (bool, optional): Drop Z axis from landmarks. Defaults to True.

    Returns:
        DataFrame: Preprocessed dataframe
    """    
    data = pd.read_parquet(path)
    data.fillna(0.0, inplace=True)
    if DROP_Z:
        return data.loc[:, ~data.columns.str.startswith("z")]
    return data