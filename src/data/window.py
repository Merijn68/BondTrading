import tensorflow as tf
from loguru import logger
import numpy as np


def windowed_dataset(
    data: np.ndarray,
    window_size: int,
    batch_size: int,
    shuffle_buffer: int,
    skip: int = 0,
    horizon: int = 1,
) -> tf.data.Dataset:
    """Generate a windowed dataset from timeseries"""

    logger.info(
        f"Split windowed dataset window_size = {window_size} "
        + f"batch_size =  {batch_size}, "
        + f"shuffle_buffer =  {shuffle_buffer} ,"
        + f"horizon =  {horizon}"
    )

    stretch = horizon + skip

    ds = tf.data.Dataset.from_tensor_slices(data)  # features = data.shape[1] - 1
    ds = ds.window(
        window_size + stretch, shift=1, drop_remainder=True
    )  # shifted windows. +1 for target value
    ds = ds.flat_map(
        lambda w: w.batch(window_size + stretch)
    )  # map into lists of size batch+target
    ds = ds.shuffle(shuffle_buffer)
    if data.ndim > 1:  # number of features > 1
        ds = ds.map(lambda w: (w[:-stretch], w[-horizon:, 0]))
    else:
        ds = ds.map(
            lambda w: (w[:-stretch], w[-horizon:])
        )  # split into data and target, x and y
    return ds.batch(batch_size).prefetch(1)
