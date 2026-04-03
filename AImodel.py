import tensorflow as tf
import numpy as np
import pandas as pd
import os

class ASL_model():
    def __init__ (self, model_file = 'asl_model.h5'):
        self.staticModel = None
        self.alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        if(os.path.exists(model_file)):
            try:
                self.staticModel = tf.keras.models.load_model(model_file)
            except Exception as loadFail:
                print(f"Error loading model: {loadFail}")
        else:
            self.train_model()

    def read_sign(self, landmark_coordinates):
        input_coordinates = np.array([landmark_coordinates], dtype = np.float32)

        predict = self.staticModel.predict(input_coordinates, verbose = 0)
        
        predict_index = np.argmax(predict[0])

        return self.alphabet[predict_index]


    def train_model(self):
        x_train = pd.read_csv('train.csv').values.astype('float32') # Extracts 2D array [x_coordinates][y_coordinates]
        y_train = pd.read_csv('trainKey.csv').values.astype('int32') # Extracts 1D array [Answer_Key]

        x_test = pd.read_csv('test.csv').values.astype('float32')
        y_test = pd.read_csv('testKey.csv').values.astype('int32')

        # x_train : x, y coordinates 
        # y_train : A -> 0, B -> 1, C -> 2 
        # x_train[0][0] = (x_coordinate, y_coordinate) -> y_train == 'answer'

        # Model structure uses 3 dense layers: 64 neurons -> 128 -> 26 (26 letters)
        self.staticModel = tf.keras.models.Sequential ([
            tf.keras.layers.Dense(64, input_shape = (42,)),
            tf.keras.layers.Dense(128, activation = 'relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(26)
        ])

        predictions = self.staticModel(x_train[:1]).numpy()

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy( from_logits = True )
        loss_fn(y_train[:1], predictions).numpy()

        self.staticModel.compile(optimizer = 'adam',
                            loss = loss_fn,
                            metrics = ['accuracy'])

        self.staticModel.fit(x_train, y_train, epochs = 45)
        self.staticModel.save('asl_model.h5')

        self.staticModel.evaluate(x_test, y_test, verbose = 2)

        probability_model = tf.keras.Sequential([
            self.staticModel,
            tf.keras.layers.Softmax()
            ])

        probability_model(x_test[:5])




