from typing import Dict

import tensorflow as tf

from ray import tune
from ray.tune import JupyterNotebookReporter
from ray.tune.integration.keras import TuneReportCallback
from ray.tune.schedulers import AsyncHyperBandScheduler
import numpy as np

from src.data import window
from src.models import evaluate
from src.models.base_model import RnnModel


def train_hypermodel(train: np.ndarray, test: np.ndarray, config: Dict) -> RnnModel:

    window_size = config["window"]
    batch_size = config["batch_size"]
    shuffle_buffer = config["shuffle_buffer"]
    horizon = config["horizon"]

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
    model = RnnModel("hyper", config)

    config.setdefault("loss", "mse")
    config.setdefault("loss_alpha", 0)

    if config["loss"] == "updown":
        loss = evaluate.directional_loss_with_alpha(config["loss_alpha"])
    else:
        loss = "mse"

    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(), metrics=["mae"])

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
        resources_per_trial={"cpu": 2, "gpu": 1},  # Check if this is needed on COLAB
        config=config,
    )
    return analysis
