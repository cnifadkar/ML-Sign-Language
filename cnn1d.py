from typing import Any
import lightning as L
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, GlobalAveragePooling1D, Input
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler

class CNN1D(L.LightningModule):
    def __init__(
            self,
            input_shape,
            num_classifiers,
            k: int = 1,  
            data_dir: str = 'data/',
            split_seed: int = 24324,
            num_splits: int = 10,
            batch_size: int = 32
        ):
        
        model = Sequential()
        model.add(Input(input_shape))
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=3, padding='same'))
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=3, padding='same'))
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(units=64, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(units=num_classifiers, activation='softmax'))
        self.model = model
        super().__init__()
        
        
    def forward(self):
        return super().forward()
    
    def training_step(self):
        return super().training_step()
    
    def configure_optimizers(self):
        return super().configure_optimizers()

