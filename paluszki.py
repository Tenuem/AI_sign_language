from pickle import FALSE, TRUE
import tensorflow as tf
import pandas as pd
import numpy as np 
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
import time
from training import ModelTrainer
LOAD = TRUE
signs = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','DEL','NOTHING','SPACE']

if LOAD == TRUE :
  model = tf.keras.models.load_model("trained.h5")
  trainer = ModelTrainer("test_data.csv","testing_data.csv",'sign')
  trainer.loadData()
  
else: 
    trainer = ModelTrainer("test_data.csv","testing_data.csv",'sign')
    trainer.loadData()
    trainer.createNeuralNetowrkModel()
    model = trainer.trainNeuralNetwork()


    
data = [0]*60

cap = cv2.VideoCapture(0)

prevTime = 0
with mp_hands.Hands(
    min_detection_confidence=0.5,       #Detection Sensitivity
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      print(results.label)
      for hand_landmarks in results.multi_hand_landmarks:
       # mp_drawing.draw_landmarks(
        #    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        for i in range(1,20):
            data[3*i] = hand_landmarks.landmark[i].x
            data[3*i+1] = hand_landmarks.landmark[i].y
            data[3*i+2] = hand_landmarks.landmark[i].z
        
    else:
        data = [0]*60 
    prediction = model.predict(np.array([data]),verbose=0)   
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    letter = signs[np.argmax(prediction)]
    cv2.putText(image, f'Predicted: {letter}', (20, 70), cv2.FONT_HERSHEY_DUPLEX, 1, (128, 0, 128), 2)
    cv2.imshow('AI sign language detection', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
    if cv2.getWindowProperty('AI sign language detection',prop_id=cv2.WND_PROP_VISIBLE) <1:
        break
cap.release()
