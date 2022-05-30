from pickle import FALSE, TRUE
from telnetlib import NOP
import tensorflow as tf
import pandas as pd
import numpy as np 
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
import time
from training import ModelTrainer


LOAD = TRUE
signs = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','DEL','NOTHING','SPACE']
np.set_printoptions(threshold=np.inf)
if LOAD == TRUE :
  model = tf.keras.models.load_model("trained.h5")
  trainer = ModelTrainer("training_data.csv","testing_data.csv",'sign')
  trainer.loadData()
  
else: 
    trainer = ModelTrainer("training_data.csv","testing_data.csv",'sign')
    trainer.loadData()
    trainer.createNeuralNetowrkModel()
    model = trainer.trainNeuralNetwork()
    
    plt.plot(trainer.history.history['accuracy'])
    plt.plot(trainer.history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.grid(True)
    plt.legend(['accuracy','validate_accuracy'], loc='lower right')
    plt.savefig("plot.png")
    plt.clf()
    plt.plot(trainer.history.history['loss'])
    plt.plot(trainer.history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.grid(True)
    plt.legend(['loss','val_loss'], loc='upper right')
    plt.savefig("plot2.png")
  
model.evaluate(trainer.test_features,trainer.test_labels)
predictions = model.predict(trainer.test_features)
predictions2 = []
for i in range(len(predictions)):
      predictions2.append(np.argmax(predictions[i]))
 
confusion_matrix = tf.math.confusion_matrix(trainer.test_labels,predictions2)
numpy_confusion = confusion_matrix.numpy()
for i in range(len(numpy_confusion)):
    sum = 0
    all = 0
    for j in range(len(numpy_confusion[i])):
      if not j == i:
        sum += numpy_confusion[i][j]
      all += numpy_confusion[i][j]
    print(f"{signs[i]} : {all-sum}/{all}")


np.savetxt("confsuion.csv",numpy_confusion,delimiter=',',fmt='%s')

    
data = [0]*60

cap = cv2.VideoCapture(0)

prevTime = 0
with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,       #Detection Sensitivity
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand in results.multi_handedness:
          label = hand.classification[0].label
     
      for hand_landmarks in results.multi_hand_landmarks:
       
        for i in range(1,21):
            data[3*(i-1)] = (hand_landmarks.landmark[i].x-hand_landmarks.landmark[0].x)
            data[3*(i-1)+1] = hand_landmarks.landmark[i].y-hand_landmarks.landmark[0].y
            data[3*(i-1)+2] = hand_landmarks.landmark[i].z-hand_landmarks.landmark[0].z
        
        mp_drawing.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
    else:
        data = [0]*60 
    prediction = model.predict(np.array([data]),verbose=0)   
    letter = signs[np.argmax(prediction)]
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    
    cv2.putText(image, f'Predicted: {letter}', (20, 70), cv2.FONT_HERSHEY_DUPLEX, 1, (128, 0, 128), 2)
    cv2.imshow('AI sign language detection', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
    if cv2.getWindowProperty('AI sign language detection',prop_id=cv2.WND_PROP_VISIBLE) <1:
        break
cap.release()
