import tensorflow as tf
import pandas as pd
import numpy as np 


test_data = pd.read_csv('test_data.csv')

print(test_data['sign'])

test_features = test_data.copy()
test_labels = test_features.pop('sign')
test_features = np.array(test_features)
'''
test_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64),
    tf.keras.layers.Dense(1)
])
#tf.keras.metrics.Accuracy(name = 'accuracy', dtype = None)
test_model.compile(loss = tf.keras.losses.MeanSquaredError(), 
                   optimizer = tf.optimizers.Adam(),
                   metrics = [tf.keras.metrics.Accuracy()])

test_model.fit(test_features, test_labels, epochs = 10)
print(test_model.predict(test_features[0]))
'''
print(test_labels)
model= tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(63,1)))
model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=30,activation=tf.nn.softmax))
model.compile(optimizer='adam' , loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(test_features, test_labels, epochs=10, batch=50)#As the number of epochs increases beyond 11,chance of overfitting of the model on training data
loss, acc = model.evaluate(np.array(test_features), np.array(test_labels))

#loss , accuracy = model.evaluate(x_test,y_test)
print(acc)
print(loss)
