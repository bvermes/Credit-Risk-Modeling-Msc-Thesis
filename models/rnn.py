# import tensorflow as tf
# import tensorflow as tf

# print(tf.__version__)

# import keras

# print(keras.__version__)
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from models.model import Model
import numpy as np


class RNNModel(Model):
    def __init__(self, param_grid, output_path, random_state, visualization_weight):
        super().__init__(
            param_grid=param_grid,
            output_path=output_path + "random_forest_model/",
            random_state=random_state,
            visualization_weight=visualization_weight,
            modelname="random_forest",
        )

    def _build_model_rnn(self, X_train, X_test, y_train, y_test):
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        y = np.array(y)

        model = Sequential(
            [
                LSTM(
                    units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)
                ),
                Dropout(0.2),
                LSTM(units=50, return_sequences=True),
                Dropout(0.2),
                LSTM(units=50),
                Dropout(0.2),
                Dense(units=1),
            ]
        )

    def run(self, X_train, X_test, y_train, y_test):
        parameters, performance_metrics = self._build_model_rnn(
            X_train, X_test, y_train, y_test
        )
