import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import tensorflow as tf
from tqdm import tqdm
from src.models import base_model


def mse(y: np.ndarray, yhat: np.ndarray) -> float:
    return np.mean((y - yhat) ** 2)


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return np.mean(np.abs(y - yhat))


def mase(y: np.ndarray, yhat: np.ndarray) -> float:
    norm = mae(*base_model.naivepredict(y))
    return mae(y, yhat) / norm


class ScaledMAE(tf.keras.metrics.Metric):
    def __init__(self, scale: float = 1.0, name: str = "smae", **kwargs) -> None:
        super(ScaledMAE, self).__init__(name=name, **kwargs)
        self.scale = scale
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")

    def update_state(
        self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: tf.Tensor = None
    ) -> None:
        metric = self.ae(y_true, y_pred) / self.scale
        self.total.assign_add(metric)
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self) -> float:
        return self.total / self.count

    def ae(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return tf.reduce_sum(tf.abs(y_true - y_pred))


def naivepredict(
    series: np.ndarray,
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


def generate_prediction(
    model: tf.keras.Model,
    series: np.ndarray,
    window: int,
    horizon: int,
    figsize: Tuple[int, int] = (10, 10),
) -> np.ndarray:
    """
    After a model is trained, we can check the predictions.
    This function generates predictions, given a model and a timeseries,
     for a given window in the past
    and a horizon in the future.
    It returns both the prediction and a plot.
    """

    # make sure we have an np.array
    series = np.array(series)
    # calculate the amount of horizons we can predict in a given series
    batches = int(np.floor((len(series) - window) / horizon))

    # we might end up with some rest, where we don't have enough data for the
    # last prediction, so we stop just before that
    # end = batches * horizon - window
    yhat_ = []

    # for every batch
    for i in tqdm(range(batches)):
        # skip the horizons we already predicted
        shift = i * horizon

        # take the window from the past we need for predicting,
        # skipping what we already predicted
        X = series[0 + shift : window + shift]  # noqa: N806, E203

        # add a dimension, needed for the timeseries
        X = X[np.newaxis, :]  # noqa: N806
        if X.shape[1] == window:
            # predict the future horizon, given the past window
            y = model.predict(X).flatten()[:horizon]
            # collect as a list of predictions
            yhat_.append(y)

    # transform the appended results into a single numpy array for plotting
    yhat = np.concatenate(yhat_, axis=None)

    plt.figure(figsize=figsize)
    # plt.xlim(300, 600)
    plt.plot(yhat, label="prediction")
    if series.ndim > 1:
        plt.plot(series[window:][:, 0], label="actual")
    else:
        plt.plot(series[window:], label="actual")
    plt.legend()

    return yhat
