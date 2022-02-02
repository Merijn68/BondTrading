from typing import Dict
from pathlib import Path

import tensorflow.keras.layers as tfl
import tensorflow as tf


class BaseModel(tf.keras.Model):
    def __init__(self: tf.keras.Model, name: str, config: Dict) -> None:
        super().__init__(name=name)

        self.__path = Path("../data/models/")

        config.setdefault("features", 1)
        config.setdefault("window", 1)

        self.__config = config
        self.out = tfl.Dense(config["horizon"])

        # reshape input
        self.reshape = tfl.Reshape((config["window"], config["features"]))

    def call(self: tf.keras.Model, x: tf.Tensor) -> tf.Tensor:

        x = self.reshape(x)
        x = tfl.Flatten()(x)
        x = self.out(x)
        return x

    def model(self: tf.keras.Model):

        config = self.config
        x = tfl.Input(shape=(config["window"], config["features"]))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    @property
    def config(self: tf.keras.Model):
        return self.__config

    @config.setter
    def config(self: tf.keras.Model, config: Dict):
        self.__config = config

    def save(self: tf.keras.Model):
        # Save the entire model
        path = Path(self.__path, self.name + ".h5")
        super().save(path)

    def load(self: tf.keras.Model):
        # load the model
        path = Path(self.__path, self.name + ".h5")
        super().load(path)


class RnnModel(BaseModel):
    def __init__(self: BaseModel, name: str, config: Dict) -> None:
        super().__init__(name, config)

        config.setdefault("features", 1)
        self.input_layer = tfl.Input([None, config["features"]])

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
        self.hidden = []
        for _ in range(config["hidden"] - 1):
            self.hidden += [self.cell(units=config["units"], return_sequences=True)]

        # Last output cells return_sequence is set to False
        self.hidden += [self.cell(config["units"], return_sequences=False)]

        config.setdefault("dropout", 0)
        self.dropout = tfl.Dropout(config["dropout"])

    def call(self: BaseModel, x: tf.Tensor, training=False) -> tf.Tensor:

        x = self.reshape(x)
        if self.config["filters"] >= 1:
            x = self.conv(x)
        for layer in self.hidden:
            x = layer(x)
        if training:
            x = self.dropout(x, training=training)
        x = self.out(x)

        return x


class RnnUpDownModel(RnnModel):
    def __init__(self: RnnModel, name: str, config: Dict) -> None:
        super().__init__(name, config)
        self.out = tfl.Dense(1, activation="sigmoid")
