import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.model import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score


# Create
# 1. evaluation function
# 2. accuracy function
# 3. Save model function
# 4. Load model function
# 5. Load dataset function
# 6. Predict Function
# 7. Train Function
# 8. Train spli helper function


class Model:
    def __init__(self, actions):
        # Path for exported data, numpy arrays
        self.DATA_PATH = os.path.join("MP_DATA")

        # Actions that the code will try to detect
        self.actions = np.array(["hello", "thanks", "iloveyou"])

        self.log_dir = os.path.join("Logs")
        self.tb_callback = TensorBoard(log_dir=self.log_dir)

        # Build the neural network
        self.model = Sequential()
        # Add layers
        self.model.add(
            LSTM(64, return_sequences=True, activation="relu", input_shape=(30, 1662))
        )
        self.model.add(LSTM(128, return_sequences=True, activation="relu"))
        # Does not return sequences to the next layer
        self.model.add(LSTM(64, return_sequences=False, activation="relu"))
        # Fully connected layers
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dense(32, activation="relu"))
        # Returns values with 0-1 with the sum of all values return summing to 1
        self.model.add(Dense(self.actions.shape[0], activation="softmax"))

        # Multiclass classfication model uses categorical_crossentropy
        # Binary - Binary_crossentropy
        # Regression -Mean Square Error (MSE)
        self.model.compile(
            optimizer="Adam",
            loss="categorical_crossentropy",
            metrics=["categorical_accuracy"],
        )

    def load_trained_model(self, model_filename="action.h5"):
        # Load the trained model
        self.model.load_weights(model_filename)

    def save_trained_model(self, model_filename="action.h5"):
        # Save the trained model
        self.model.save(model_filename)

    def split_training_data(self, X, y, test_size=0.05):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size)
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train, epochs=2000):
        # Extract saved data for training the model

        label_map = {label: num for num, label in enumerate(self.actions)}
        sequences, labels = [], []
        # Thirty Videos worth of data
        no_sequences = 30
        # 30fps for each sequence
        sequence_length = 30

        for action in self.actions:
            for sequence in range(no_sequences):
                window = []
                for frame_num in range(sequence_length):
                    res = np.load(
                        os.path.join(
                            self.DATA_PATH, action, str(sequence), f"{frame_num}.npy"
                        )
                    )
                    window.append(res)
                sequences.append(window)
                labels.append(label_map[action])

        X = np.array(sequences)
        y = to_categorical(labels).astype(int)

        # The test partition will be 5% of our saved data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

        # Train the model
        self.model.fit(X_train, y_train, epochs, callbacks=[self.tb_callback])
        # self.evaluate(X_test, y_test)

    def evaluate(self, X_test, y_test):
        # Model evaluation
        yhat = self.model.predict(X_test)
        ytrue = np.argmax(y_test, axis=1).tolist()
        yhat = np.argmax(yhat, axis=1).tolist()

        # The numbers should be within the top left and bottom right range
        multilabel_confusion_matrix(ytrue, yhat)

        # The higher the number the better
        score = accuracy_score(ytrue, yhat)
        print(score)
