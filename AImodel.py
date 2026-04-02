import tensorflow as tf
import numpy as np
import pandas as pd


def create_model():
    """Trains and evaluates model based off of landmarker coordinate values"""
    # Pray this works
    x_train = pd.read_csv('train.csv').values.astype('float32') # Extracts 2D array [x_coordinates][y_coordinates]
    y_train = pd.read_csv('trainKey.csv').values.astype('int32') # Extracts 1D array [Answer_Key]

    x_test = pd.read_csv('test.csv').values.astype('float32')
    y_test = pd.read_csv('testKey.csv').values.astype('int32')

    # x_train : x, y coordinates 
    # y_train : A -> 0, B -> 1, C -> 2 
    # x_train[0][0] = (x_coordinate, y_coordinate) -> y_train == 'answer'

    # Model structure uses 3 dense layers: 64 neurons -> 128 -> 26 (26 letters)
    staticModel = tf.keras.models.Sequential ([
        tf.keras.layers.Dense(64, input_shape = (42,)),
        tf.keras.layers.Dense(128, activation = 'relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(26)
    ])

    predictions = staticModel(x_train[:1]).numpy()

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy( from_logits = True )
    loss_fn(y_train[:1], predictions).numpy()

    staticModel.compile(optimizer = 'adam',
                        loss = loss_fn,
                        metrics = ['accuracy'])

    staticModel.fit(x_train, y_train, epochs = 45)

    staticModel.evaluate(x_test, y_test, verbose = 2)

    probability_model = tf.keras.Sequential([
        staticModel,
        tf.keras.layers.Softmax()
        ])

    probability_model(x_test[:5])


