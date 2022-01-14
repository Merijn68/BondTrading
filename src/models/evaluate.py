
import re
import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt

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



def plot_results(
    result: Dict,
    ymin: float = 0.0,
    ymax: Optional[float] = None,
    yscale: str = "linear",
    moving: int = -1,
    alpha: float = 0.5,
    patience: int = 1,
    subset: str = ".",
    grid: bool = False,
    measure: str = 'loss',
    figsize: Tuple[int, int] = (15, 10),
) -> None:

    val_measure = 'val_' + measure
    if not grid:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        if moving < 0:
            move = True

        for key in result.keys():
            if bool(re.search(subset, key)):
                ms: List[float] = result[key].history[measure]
                if move:
                    z = movingaverage(ms, moving)
                    z = np.concatenate([[np.nan] * moving, z[moving:-moving]])
                    color = next(ax1._get_lines.prop_cycler)["color"]
                    ax1.plot(z, label=key, color=color)
                    ax1.plot(ms, label=key, alpha=alpha, color=color)
                else:
                    ax1.plot(ms, label=key)

                ax1.set_yscale(yscale)
                ax1.set_ylim(ymin, ymax)
                ax1.set_title("train")                
                valms = result[key].history[val_measure]

                if move:
                    z = movingaverage(valms, moving)
                    z = np.concatenate([[np.nan] * moving, z[moving:-moving]])[
                        :-patience
                    ]
                    color = next(ax2._get_lines.prop_cycler)["color"]
                    ax2.plot(z, label=key, color=color)
                    ax2.plot(valms, label=key, alpha=alpha, color=color)
                else:
                    ax2.plot(valms[:-patience], label=key)

                ax2.set_yscale(yscale)
                ax2.set_ylim(ymin, ymax)
                ax2.set_title("valid")

        plt.legend()
    if grid:

        keyset = list(filter(lambda x: re.search(subset, x), [*result.keys()]))
        gridsize = int(np.ceil(np.sqrt(len(keyset))))

        plt.figure(figsize=(15, 15))
        for i, key in enumerate(keyset):
            plt.subplot(gridsize, gridsize, i + 1)
            ms = result[key].history[measure]
            valms = result[key].history[val_measure]
            plt.plot(ms, label="train")
            plt.ylim(0, ymax)
            plt.plot(valms, label="valid")
            plt.title(key)
            plt.legend()



def movingaverage(
    interval: List[float],
    window_size: int = 32
)-> List[float]:
    ret = np.cumsum(interval, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    return ret[window_size - 1:] / window_size

# Alternative plot 

# — — — — — — — — — — — — — — — — — — — — — — — — — — — — — -
# Retrieve a list of list results on training and test data
# sets for each training epoch
# — — — — — — — — — — — — — — — — — — — — — — — — — — — — — -
#loss=history.history[‘loss’]
#epochs=range(len(loss)) # Get number of epochs
# — — — — — — — — — — — — — — — — — — — — — — — — 
# Plot training and validation loss per epoch
# — — — — — — — — — — — — — — — — — — — — — — — — 
#plt.plot(epochs, loss, ‘r’)
#plt.title(‘Training loss’)
#plt.xlabel(“Epochs”)
#plt.ylabel(“Loss”)
#plt.legend([“Loss”])
#plt.figure()
#zoomed_loss = loss[200:]
#zoomed_epochs = range(200,500)
# — — — — — — — — — — — — — — — — — — — — — — — — 
# Plot training and validation loss per epoch
# — — — — — — — — — — — — — — — — — — — — — — — — 
#plt.plot(zoomed_epochs, zoomed_loss, ‘r’)
#plt.title(‘Training loss’)
#plt.xlabel(“Epochs”)
#plt.ylabel(“Loss”)
#plt.legend([“Loss”])
#plt.figure()


def naive(
    result: Dict, 
    ylim: float = 2,
    subset: str = "."
) -> None:
    for key in result.keys():
        if bool(re.search(subset, key)):
            plt.plot(result[key].history["val_smae"], label=key)

    plt.axhline(1.0, color="r", label="naive norm")
    plt.ylim(0, ylim)
    plt.legend()
