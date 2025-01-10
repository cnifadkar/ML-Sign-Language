import pandas as pd
import numpy as np 
import json
from keras.utils import to_categorical
from tqdm import tqdm

class PopsignDataModule():
      def __init__(
            self, 
            data_dir: str = "data", 
            batch_size: int = 32, 
            classes: int = 250,
            num_of_samples: int = 10,
            feature_groups: list = ['face','pose','hand'],
            fold_idx: int = 0,
            max_frames: int = 350,
      ):
            self.data_dir = data_dir
            self.batch_size = batch_size
            self.classes = classes
            self.num_samples = num_of_samples
            self.feature_groups = feature_groups
            self.fold_idx = fold_idx
            self.max_frames = max_frames
            self.test_folds = [
                  ['16069', '26734', '37055'],
                  ['49445', '34503', '62590'],
                  ['29302', '25571', '61333'],
                  ['2044', '32319', '28656'],
                  ['18796', '30680', '36257'],
                  ['22343', '55372', '37779'],
                  ['27610', '53618', '4718']
            ]
            self.train_folds = [ 
                  ['55372', '28656', '53618', '62590', '27610', '37779', '4718',
                  '25571', '2044', '36257', '29302', '32319', '61333', '30680',
                  '49445', '18796', '22343', '34503'],
                  ['55372', '28656', '53618', '27610', '37779', '4718', '25571',
                  '2044', '36257', '29302', '32319', '61333', '16069', '30680',
                  '18796', '37055', '26734', '22343'],
                  ['55372', '28656', '53618', '62590', '27610', '37779', '4718',
                  '2044', '36257', '32319', '16069', '30680', '49445', '18796',
                  '37055', '26734', '22343', '34503'],
                  ['55372', '53618', '62590', '27610', '37779', '4718', '25571',
                  '36257', '29302', '61333', '16069', '30680', '49445', '18796',
                  '37055', '26734', '22343', '34503'],
                  ['55372', '28656', '53618', '62590', '27610', '37779', '4718',
                  '25571', '2044', '29302', '32319', '61333', '16069', '49445',
                  '37055', '26734', '22343', '34503'],
                  ['28656', '53618', '62590', '27610', '4718', '25571', '2044',
                  '36257', '29302', '32319', '61333', '16069', '30680', '49445',
                  '18796', '37055', '26734', '34503'],
                  ['55372', '28656', '62590', '37779', '25571', '2044', '36257',
                  '29302', '32319', '61333', '16069', '30680', '49445', '18796',
                  '37055', '26734', '22343', '34503']
            ]
            json_map = self.data_dir + '/' + 'sign_to_prediction_index_map.json'
            with open(json_map, 'r') as file:
                  json_map = json.load(file)
            self.sign_to_prediction_map = json_map
            self.map_list = pd.read_csv(self.data_dir + '/' + 'train.csv')
      
      def test_train_split(self):
            # Get the users in each fold 
            test_users = self.test_folds[self.fold_idx]
            train_users = self.train_folds[self.fold_idx]
            map_list = self.map_list
            
            # create lists to store the values in the split
            X_test = []
            X_train = []
            Y_test = []
            Y_train = []
            
            for test_user in test_users: 
                  mask = self.map_list['participant_id'] == int(test_user)
                  for _, row in tqdm(map_list[mask].iterrows(), desc= f"Loading Participant {test_user} "):
                        path = row[1]
                        sign = row[4]
                        sign_class = self.sign_to_class(sign)
                        data = self.get_features(self.data_dir +'/' + path)
                        data = self.pad_samples(data)
                        data = self.preprocess_data(data)
                        X_test.append(data)
                        Y_test.append(sign_class)
                        
            for train_user in train_users:
                  mask = self.map_list['participant_id'] == int(train_user)
                  for _, row in tqdm(map_list[mask].iterrows(), desc= f"Loading Participant {train_user}"):
                        path = row[1]
                        sign = row[4]
                        sign_class = self.sign_to_class(sign)
                        data = self.get_features(self.data_dir +'/' + path)
                        data = self.pad_samples(data)
                        data = self.preprocess_data(data)        
                        X_train.append(data) 
                        Y_train.append(sign_class)

            # iterate over all the dataframes added to the X Train list and convert 
            # them into lists of numpy arrays, then stack the list into an 3d numpy array
            X_train = [df.to_numpy() for df in X_train]
            X_train = np.stack(X_train, axis = 0)

            # iterate over all the dataframes added to the X Test list and convert 
            # them into lists of numpy arrays, then stack the list into an 3d numpy array
            X_test = [df.to_numpy() for df in X_test]
            X_test = np.stack(X_test, axis = 0)

            # One Hot encode Y Train and Y test 
            Y_train = to_categorical(Y_train, num_classes=self.classes)
            Y_test = to_categorical(Y_test, num_classes= self.classes)
            
            return X_train, Y_train, X_test, Y_test
            
      def preprocess_data(self, data, drop_z = True):
            data.fillna(0.0, inplace=True)
            if drop_z:
                  return data.loc[:, ~data.columns.str.startswith("z")]
            return data
      
      def sign_to_class(self ,sign):
            return self.sign_to_prediction_map[sign] 

      def pad_samples(self, original_df):
            num_col = len(original_df.columns)
            original_frames = len(original_df)
            padding_before = (self.max_frames - original_frames) // 2
            padding_after = self.max_frames - original_frames - padding_before
            
            if self.max_frames <= original_frames:
                  start = original_frames / 2 - self.max_frames / 2 
                  end = start + self.max_frames
                  return original_df.iloc[start:end]
            else: 
                  pad_before_df = pd.DataFrame(
                        np.zeros((padding_before, num_col)), columns = original_df.columns
                  )
                  pad_after_df = pd.DataFrame(
                        np.zeros((padding_after, num_col)), columns = original_df.columns
                  )
            return pd.concat([pad_before_df, original_df, pad_after_df], ignore_index=True)

      def get_features(self, parquet_path):
            data = pd.read_parquet(parquet_path)
            USE_FACE = 'face' in self.feature_groups 
            USE_POSE = 'pose' in self.feature_groups 
            USE_HAND = 'hand' in self.feature_groups 

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