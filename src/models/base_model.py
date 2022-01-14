
from typing import Dict, Tuple
from loguru import logger


import tensorflow as tf
from tensorflow.keras.layers import (    
    Flatten
)
from tensorflow.keras.layers.experimental.preprocessing import Rescaling


import numpy as np

def naivepredict(
  series: np.array,
  horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:    
    y = series[1:]
    yhat = series[:-1]
    return y, yhat


# Base line class predicts waarde t+1 = waarde t
class Baseline(tf.keras.Model):
  def __init__(self, label_index=None):
    super().__init__()
    self.label_index = label_index

  def call(self, inputs):
    if self.label_index is None:
      return inputs    
    result = inputs[:, :, self.label_index]
    return result[:, :, tf.newaxis]

class Basemodel(tf.keras.Model):
    def __init__(self: tf.keras.Model, config: Dict) -> None:
        super().__init__(config)
        logger.info("init base")

        self.hidden = []
        if "initializer" not in config:
            config["initializer"] = "glorot_uniform"

        self.out = tf.keras.layers.Dense(1, activation="relu")

    def call(self: tf.keras.Model, x: tf.Tensor) -> tf.Tensor:
        
        for layer in self.hidden:
            x = layer(x)

        x = self.out(x)
        return x