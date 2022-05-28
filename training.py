from msilib.schema import Class
import tensorflow as tf
import pandas as pd
import numpy as np 


class ModelTrainer:
    
    def __init__(self,dataPath,testPath,labels):
        self.dataPath = dataPath
        self.testPath = testPath
        self.labels = labels
        self.signs = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','DEL','NOTHING','SPACE']

    def loadData(self):
        self.training_data = pd.read_csv(self.dataPath)
        self.training_features = self.training_data.copy()
        self.training_labels = self.training_features.pop(self.labels)
        self.training_features = np.array(self.training_features)

        self.test_data = pd.read_csv(self.testPath)
        self.test_features = self.test_data.copy()
        self.test_labels = self.test_features.pop(self.labels)
        self.test_features = np.array(self.test_features)
    
    def trainNeuralNetwork(self):
        model= tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(60,1)))
        model.add(tf.keras.layers.Dense(units=180,activation=tf.nn.sigmoid))
        model.add(tf.keras.layers.Dense(units=180,activation=tf.nn.sigmoid))
        model.add(tf.keras.layers.Dense(units=29,activation=tf.nn.softmax))
        model.compile(optimizer='adam' , loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(self.training_features, self.training_labels, epochs=1000)#As the number of epochs increases beyond 11,chance of overfitting of the model on training data
        model.save("trained.h5")
        return model
        #TODO add model testing accuracy on test data