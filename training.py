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
        self.training_data = self.training_data.sample(frac = 1)
        self.training_features = self.training_data.copy()
        self.training_labels = self.training_features.pop(self.labels)
        self.training_features = np.array(self.training_features)

        self.test_data = pd.read_csv(self.testPath)
        self.test_data = self.test_data.sample(frac = 1)
        self.test_features = self.test_data.copy()
        self.test_labels = self.test_features.pop(self.labels)
        self.test_features = np.array(self.test_features)

    def createNeuralNetowrkModel(self):
        self.model= tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Flatten(input_shape=(60,1)))
        self.model.add(tf.keras.layers.Dense(units=120,activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(units=120,activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(units=29,activation=tf.nn.softmax))
        self.model.compile(optimizer='adam' , loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def trainNeuralNetwork(self):
        self.history = self.model.fit(self.training_features, self.training_labels, validation_data=(self.test_features,self.test_labels),epochs=50)
        self.model.save("trained.h5")
        results = self.model.evaluate(self.test_features,self.test_labels)
        print("test loss, test acc:", results)
        return self.model

    def evaluate(self):
        results = self.model.evaluate(self.test_features,self.test_labels)
        print(f"test loss: {results[0]}, test acc: {results[1]}", )
