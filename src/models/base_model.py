from typing import Tuple

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input


def basemodel(shape: Tuple) -> Model:
    """input shape, output a Dense model with one hidden layer

    Args:
        shape (Tuple): datashape, excluding batchsize

    Returns:
        Model: A keras model
    """
    input = Input(shape=shape)
    x = Dense(30, activation="relu")(input)
    output = Dense(1)(x)

    model = Model(inputs=[input], outputs=[output])
    return model

def naivemodel(shape: Tuple) -> Model:
    """Naive Forecasting to get a base line"""
    horsepower = np.array(train_features['Horsepower'])

    input = Input(shape=shape)
    output = Dense(1)
    model = tf.keras.Sequential([
        layers.Dense(units=1)
        ])
    return model

tf.keras.Sequential([
    horsepower_normalizer,
    layers.Dense(units=1)
])