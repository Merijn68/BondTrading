import tensorflow as tf
import pandas as pd
from loguru import logger
    
def windowed_dataset_from_df(
    df              : pd.DataFrame, 
    window_size     : int, 
    batch_size      : int, 
    shuffle_buffer  : int,
    horizon         : int =1
):
    logger.info(f'Split windowed dataset window_size = {window_size}, batch_size =  {batch_size}, shuffle_buffer =  {shuffle_buffer}, horizon =  {horizon}')
    
    features = len(df.columns)  - 1        
    print(f'We have {features} features!')
    
    ds = tf.data.Dataset.from_tensor_slices(df) 
    ds = ds.window(window_size + horizon, shift=1, drop_remainder=True) # shifted windows. +1 for target value
    ds = ds.flat_map(lambda w: w.batch(window_size + horizon)) # map into lists of size batch+target
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-horizon], w[-horizon:,0])) 
    return ds.batch(batch_size).prefetch(1)


def windowed_dataset_from_series(series, window_size, batch_size, shuffle_buffer, horizon=1):
    ds = tf.data.Dataset.from_tensor_slices(series) 
    ds = ds.window(window_size + horizon, shift=1, drop_remainder=True) # shifted windows. +1 for target value
    ds = ds.flat_map(lambda w: w.batch(window_size + horizon)) # map into lists of size batch+target
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-horizon], w[-horizon:])) # split into data and target, x and y
    return ds.batch(batch_size).prefetch(1)