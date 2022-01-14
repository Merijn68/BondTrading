from typing import Dict

import tensorflow as tf
import tensorflow.keras.layers as tfl
from ray import tune
from ray.tune import JupyterNotebookReporter
from ray.tune.integration.keras import TuneReportCallback
from ray.tune.schedulers import AsyncHyperBandScheduler

from src.data import make_dataset


class HyperRnn(tf.keras.Model):
    def __init__(self: tf.keras.Model, config: Dict) -> None:
        super().__init__()

        # reshape 2D input into 3D data
        self.reshape = tfl.Reshape((config["window"], 1))

        # one convolution
        self.conv = tfl.Conv1D(
            filters=config["filters"],
            kernel_size=config["kernel"],
            strides=1,
            padding="same",
        )

        layertype: str = config["type"]
        units: int = config["units"]
        if layertype == "LSTM":
            self.cell = tfl.LSTM
        elif layertype == "GRU":
            self.cell = tfl.GRU
        else:
            self.cell = tfl.SimpleRNN

        self.hidden = []

        if config["hidden"] > 1:
            for _ in range(config["hidden"] - 1):
                self.hidden += [self.cell(units=units, return_sequences=True)]

        self.hidden = [self.cell(units, return_sequences=False)]
        self.out = tfl.Dense(config["horizon"])

    def call(self: tf.keras.Model, x: tf.Tensor) -> tf.Tensor:
        x = self.reshape(x)
        x = self.conv(x)
        for layer in self.hidden:
            x = layer(x)
        x = self.out(x)
        return x


def train_hypermodel(config: Dict) -> HyperRnn:
    data = make_dataset.get_sunspots(datadir=config["datadir"])
    train, test = make_dataset.preprocess(data["MonthlyMean"], split=2500)
    window_size = config["window"]
    train_set = make_dataset.windowed_dataset(
        train, window_size, batch_size=32, shuffle_buffer=25, horizon=1
    )
    valid_set = make_dataset.windowed_dataset(
        train, window_size, batch_size=32, shuffle_buffer=25, horizon=1
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
    model = HyperRnn(config)
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


def hypertune(config: Dict) -> tune.analysis.experiment_analysis.ExperimentAnalysis:

    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration", max_t=200, grace_period=config["grace_period"]
    )

    reporter = JupyterNotebookReporter(overwrite=True)

    def tune_wrapper(config: Dict) -> None:
        from src.models import hyper

        _ = hyper.train_hypermodel(config)

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
        resources_per_trial={"cpu": 4, "gpu": 1},
        config=config,
    )
    return analysis

