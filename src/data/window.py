import tensorflow as tf
from loguru import logger
import numpy as np


def windowed_dataset(
    data: np.ndarray,
    window_size: int,
    batch_size: int,
    shuffle_buffer: int,
    horizon: int = 1,
) -> tf.data.Dataset:
    logger.info(
        f"Split windowed dataset window_size = {window_size} "
        + f"batch_size =  {batch_size}, "
        + f"shuffle_buffer =  {shuffle_buffer} ,"
        + f"horizon =  {horizon}"
    )
    # features = data.shape[1] - 1
    ds = tf.data.Dataset.from_tensor_slices(data)
    # shifted windows. +1 for target value
    ds = ds.window(window_size + horizon, shift=1, drop_remainder=True)
    # map into lists of size batch+target
    ds = ds.flat_map(lambda w: w.batch(window_size + horizon))
    ds = ds.shuffle(shuffle_buffer)

    # number of features > 1
    if data.ndim > 1:
        ds = ds.map(lambda w: (w[:-horizon], w[-horizon:, 0]))
    else:
        # split into data and target, x and y
        ds = ds.map(lambda w: (w[:-horizon], w[-horizon:]))
    return ds.batch(batch_size).prefetch(1)
