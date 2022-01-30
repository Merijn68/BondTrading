from typing import Tuple, Dict
from pathlib import Path

import tensorflow.keras.layers as tfl
import tensorflow as tf
import numpy as np


def naivepredict(
    series: np.ndarray,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    y = series[1:]
    yhat = series[:-1]
    return y, yhat


def calc_mae_for_horizon(
    train_set: np.ndarray,
    horizon: int = 1,
) -> float:
    """calculate mean difference between naive prediction and y at horizon"""
    maelist = []
    for x, y in train_set:
        # get the last value of every batch
        x1 = x[:, -1]
        if x1.ndim > 1:
            # get only the signal
            x1 = x1[:, 0]
        # this will be the batchsize, so mostly 32
        size = tf.size(x1)
        # broadcast
        yhat = tf.broadcast_to(tf.reshape(x1, [size, 1]), [size, horizon])
        # calculate mae
        mae = np.mean(np.abs(yhat - y))
        maelist.append(mae)
    norm = np.mean(maelist)
    return norm


# def simple_model(
#     window_size: int = 32,
#     units: int = 1,
#     type="RNN",
#     horizon: int = 1,
# ):
#     """A simple model for time series analyses"""

#     model = Sequential()
#     model.add(tfl.Reshape((window_size, 1)))
#     if type == "RNN":
#         model.add(tfl.SimpleRNN(units)),
#     model.add(tfl.Dense(horizon))

#     return model


class BaseModel(tf.keras.Model):
    def __init__(self: tf.keras.Model, name: str, config: Dict) -> None:
        super().__init__(name=name)

        self.__path = Path("../data/models/")

        self.__config = config
        self.out = tfl.Dense(config["horizon"])

        # reshape 2D input into 3D data only if input is 2d.
        config.setdefault("features", 1)
        self.reshape = tfl.Reshape((config["window"], config["features"]))

    def call(self: tf.keras.Model, x: tf.Tensor) -> tf.Tensor:
        x = self.reshape(x)
        x = tfl.Flatten()(x)
        x = self.out(x)
        return x

    @property
    def config(self):
        return self.__config

    @config.setter
    def config(self: tf.keras.Model, config: Dict):
        self.__config = config

    def save_weights(self):
        # Save the weights
        path = Path(self.__path, self.name)
        super().save_weights(path)

    def load_weights(self):
        # load the weights
        path = Path(self.__path, self.name)
        super().load_weights(path)


class RnnModel(BaseModel):
    def __init__(self: BaseModel, name: str, config: Dict) -> None:
        super().__init__(name, config)

        config.setdefault("type", "RNN")
        layertype: str = config["type"]
        if layertype == "LSTM":
            self.cell = tfl.LSTM
        elif layertype == "GRU":
            self.cell = tfl.GRU
        else:
            self.cell = tfl.SimpleRNN

        config.setdefault("filters", 0)
        config.setdefault("kernel", 1)
        config.setdefault("activation", None)

        self.conv = tfl.Conv1D(
            filters=config["filters"],
            kernel_size=config["kernel"],
            activation=config["activation"],
            strides=1,
            padding="same",
        )

        config.setdefault("units", 1)
        config.setdefault("hidden", 0)
        config.setdefault("dropout", 0)
        self.hidden = []
        for _ in range(config["hidden"] - 1):
            self.hidden += [self.cell(units=config["units"], return_sequences=True)]

        # Try to see if adding a timedistributed dense layer helps
        config.setdefault("timeDistributed", False)
        if config["timeDistributed"]:
            self.hidden += [tfl.TimeDistributed(tfl.Dense(config["horizon"]))]

        # Last output cells should not return sequence
        self.hidden += [
            self.cell(
                config["units"], return_sequences=False, dropout=config["dropout"]
            )
        ]

    def call(self: BaseModel, x: tf.Tensor) -> tf.Tensor:

        x = self.reshape(x)
        if self.config["filters"] >= 1:
            x = self.conv(x)
        for layer in self.hidden:
            x = layer(x)
        x = self.out(x)

        return x


# def time_distributed(
#     window_size: int = 32,
#     units: int = 1,
#     type="RNN",
#     horizon: int = 1,
# ):
#     """A simple model for time series analyses with time distributed dense layer"""

#     model = Sequential()
#     model.add(tfl.Reshape((window_size, 1)))
#     if type == "RNN":
#         model.add(tfl.SimpleRNN(units, return_sequences=True, input_shape=[None, 1])),
#     model.add(tfl.TimeDistributed(tfl.Dense(horizon))),
#     model.add(tfl.SimpleRNN(units, return_sequences=False, input_shape=[None, 1])),
#     model.add(tfl.Dense(horizon))

#     return model
