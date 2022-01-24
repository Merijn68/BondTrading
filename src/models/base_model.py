
from typing import Dict, Tuple, List
from loguru import logger
import tensorflow as tf
import numpy as np

def naivepredict(
  series: np.array,
  horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:    
    y = series[1:]
    yhat = series[:-1]
    return y, yhat

def calc_mae_for_horizon(
  train_set: np.ndarray,
  horizon: int = 1,
) -> float:
  ''' calculate mean difference between naive prediction and y at horizon '''
  maelist = []
  for x, y in train_set:        
      x1 = x[:, -1] # get the last value of every batch
      if x1.ndim > 1:
        x1 = x1[:, 0] # get only the signal
      size = tf.size(x1) # this will be the batchsize, so mostly 32
      yhat = tf.broadcast_to(tf.reshape(x1, [size,1]), [size, horizon]) # broadcast
      mae = np.mean(np.abs(yhat - y)) # calculate mae
      maelist.append(mae)
  norm = np.mean(maelist)
  return norm



# Base line class predicts waarde t+1 = waarde t
class Baseline(tf.keras.Model):
  def __init__(self: tf.keras.Model, horizon: int, size: int):
    super().__init__()    
    _horizon = horizon
    _size = size

  def call(self, inputs):    
    
    horizon = self._horizon
    size = self._size
    x1 = x[:, -1]

    yhat = tf.broadcast_to(tf.reshape(x1, [size,1]), [size, horizon]) 
    return outputs
    
def naivenorm(series: np.ndarray, horizon: int) -> float:
    X = series[:horizon]  # noqa: N806
    Y = series[1:]  # noqa: N806
    maelist: List[ndarray] = []
    for i, x in enumerate(X):
        y = Y[i : i + horizon] # noqa E203
        yhat = [x] * horizon
        mae = np.mean(np.abs(y - yhat))
        maelist.append(mae)
    return np.mean(maelist)
