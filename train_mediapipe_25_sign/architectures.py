import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, GlobalAveragePooling1D, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam

class Architectures():
    def create1dCNN(self, X_train, num_classes):
        INPUT_SHAPE = X_train.shape[1:]
        model = Sequential()
        model.add(Input(INPUT_SHAPE))
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=3, padding='same'))
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=3, padding='same'))
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(units=64, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(units=num_classes, activation='softmax'))

        model.compile(
            optimizer=Adam(), 
            loss='categorical_crossentropy',
            metrics=['accuracy'],
            )
        return model
    
    def create2dCNN(self, X_train, num_classes):
        INPUT_SHAPE = X_train.shape[1:]
        model = Sequential()
        model.add(Input((INPUT_SHAPE[0], INPUT_SHAPE[1], 1)))
        model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
        model.add(MaxPooling2D())
        model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
        model.add(MaxPooling2D())
        model.add(Conv2D(filters= 64, kernel_size=3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(units=64, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(units=num_classes, activation='softmax'))
        
        model.compile(
            optimizer=Adam(), 
            loss='categorical_crossentropy',
            metrics=['accuracy'],
            )
        return model
        
    def createLSTM(self, X_train, num_classes):
        INPUT_SHAPE = X_train.shape
        model = Sequential()
        model.add(Input((INPUT_SHAPE[1], INPUT_SHAPE[2])))
        
        model.add(keras.layers.Bidirectional(
            keras.layers.LSTM(128, return_sequences=True),
        ))
        model.add(keras.layers.Bidirectional(
            keras.layers.LSTM(128)
        ))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation="softmax"))
        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            optimizer="adam",
            metrics=["accuracy"],
        )
        return model