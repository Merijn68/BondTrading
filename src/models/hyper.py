from typing import Dict

import tensorflow as tf

from ray import tune
from ray.tune import JupyterNotebookReporter
from ray.tune.integration.keras import TuneReportCallback
from ray.tune.schedulers import AsyncHyperBandScheduler
import numpy as np


from src.data import window
from src.models import base_model


# class HyperRnn(tf.keras.Model):
#     def __init__(self: tf.keras.Model, config: Dict) -> None:
#         super().__init__()

#         # reshape 2D input into 3D data only if input is 2d.
#         self.reshape = tfl.Reshape((config["window"], config["features"]))

#         # one convolution
#         if config["filters"] == 0:
#             self._convolutional_layers = False
#         else:
#             self._convolutional_layers = True

#         self.conv = tfl.Conv1D(
#             filters=config["filters"],
#             kernel_size=config["kernel"],
#             activation=config["activation"],
#             strides=1,
#             padding="same",
#         )

#         layertype: str = config["type"]
#         units: int = config["units"]
#         if layertype == "LSTM":
#             # self.cell = tfl.LSTM
#             self.cell = tfl.LSTM
#         elif layertype == "GRU":
#             # self.cell = tfl.GRU
#             self.cell = tfl.GRU
#         else:
#             self.cell = tfl.SimpleRNN

#         self.hidden = []

#         if config["hidden"] > 1:
#             for _ in range(config["hidden"] - 1):
#                 self.hidden += [self.cell(units=units, return_sequences=True)]

#             # Try to see if adding a timedistributed dense layer
#             if config["timeDistributed"]:
#                 self.hidden += [tfl.TimeDistributed(tfl.Dense(config["horizon"]))]

#         # dropout on last layer
#         if "dropout" in config:
#             dropout = config["dropout"]
#         else:
#             dropout = 0
#         # Last output cells should not return sequence
#         self.hidden = [self.cell(units, return_sequences=False, dropout=dropout)]

#         self.out = tfl.Dense(config["horizon"])
#         # self.out = tfl.TimeDistributed(tfl.Dense(config["horizon"]))

#     def call(self: tf.keras.Model, x: tf.Tensor) -> tf.Tensor:
#         x = self.reshape(x)
#         if self._convolutional_layers:
#             x = self.conv(x)
#         for layer in self.hidden:
#             x = layer(x)
#         x = self.out(x)
#         return x


def train_hypermodel(
    train: np.ndarray, test: np.ndarray, config: Dict
) -> base_model.RnnModel:

    window_size = config["window"]
    batch_size = 32
    shuffle_buffer = 2
    horizon = 1

    train_set = window.windowed_dataset(
        train, window_size, batch_size, shuffle_buffer, horizon=horizon
    )
    valid_set = window.windowed_dataset(
        test, window_size, batch_size, shuffle_buffer, horizon=horizon
    )

    def scheduler(epoch: int, lr: float) -> float:
        # first n epochs, the learning rate stays high
        stabletill = 5  # the model seems to start hit a plateau around 5,
        # so let's slowly start to decrease
        # then it drops exponentially untill
        droptill = 40

        if epoch < stabletill or epoch >= droptill:
            learning_rate = lr

        if stabletill <= epoch < droptill:
            learning_rate = lr * tf.math.exp(-0.1)

        # with this, we can check the lr in the history of the fit (or tensorboard)
        tf.summary.scalar("learning rate", data=learning_rate, step=epoch)
        return learning_rate

    lrs = tf.keras.callbacks.LearningRateScheduler(scheduler)
    model = base_model.RnnModel("hyper", config)
    model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(), metrics=["mae"])

    callbacks = [
        TuneReportCallback(
            {
                "val_loss": "val_loss",
                "val_mae": "val_mae",
            }
        ),
        lrs,
    ]
    model.fit(
        train_set,
        epochs=config["epochs"],
        validation_data=valid_set,
        callbacks=callbacks,
    )

    return model


def hypertune(
    train: np.array, test: np.array, config: Dict
) -> tune.analysis.experiment_analysis.ExperimentAnalysis:

    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration", max_t=200, grace_period=config["grace_period"]
    )

    reporter = JupyterNotebookReporter(overwrite=True)

    def tune_wrapper(config: Dict) -> None:
        from src.models import hyper

        _ = hyper.train_hypermodel(train, test, config)

    tune.register_trainable("wrapper", tune_wrapper)

    analysis = tune.run(
        "wrapper",
        name="hypertune",
        scheduler=sched,
        metric="val_loss",
        mode="min",
        progress_reporter=reporter,
        local_dir=config["local_dir"],
        stop={"training_iteration": config["epochs"]},
        num_samples=config["samples"],
        # resources_per_trial={"cpu": 2, "gpu": 1}, # Check if this is needed on COLAB
        config=config,
    )
    return analysis
