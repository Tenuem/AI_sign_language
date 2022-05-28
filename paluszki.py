import tensorflow as tf
import pandas as pd
import numpy as np 
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
import time


try :
    model = tf.keras.models.load_model("trained.h5")
except:
    test_data = pd.read_csv('test_data.csv')
    test_features = test_data.copy()
    test_labels = test_features.pop('sign')
    test_features = np.array(test_features)
    print(test_labels)
    model= tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(63,1)))
    model.add(tf.keras.layers.Dense(units=180,activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=180,activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=30,activation=tf.nn.softmax))
    model.compile(optimizer='adam' , loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(test_features, test_labels, epochs=50, steps_per_epoch=36,batch_size=18)#As the number of epochs increases beyond 11,chance of overfitting of the model on training data
    model.save("trained.h5")
    test = test_features[0]
    prediction = model.predict(np.array([test]))
    
    print(np.argmax(prediction[0]))
    
#print(test)




data = [0]*63

cap = cv2.VideoCapture(0)

prevTime = 0
with mp_hands.Hands(
    min_detection_confidence=0.5,       #Detection Sensitivity
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
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
      for hand_landmarks in results.multi_hand_landmarks:
       # mp_drawing.draw_landmarks(
        #    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        for i in range(21):
            data[3*i] = hand_landmarks.landmark[i].x
            data[3*i+1] = hand_landmarks.landmark[i].y
            data[3*i+2] = hand_landmarks.landmark[i].z
        
    else:
        data = [0]*63 
    prediction = model.predict(np.array([data]))   
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    if np.argmax(prediction) == 27: letter = 'del'
    elif np.argmax(prediction) == 28: letter = 'nothing'
    elif np.argmax(prediction) == 29: letter = 'space'
    else:
        letter = chr(np.argmax(prediction)+ord('A'))
    cv2.putText(image, f'Predicted: {letter}', (20, 70), cv2.FONT_HERSHEY_DUPLEX, 1, (128, 0, 128), 2)
    cv2.imshow('AI sign language detection', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
    if cv2.getWindowProperty('AI sign language detection',prop_id=cv2.WND_PROP_VISIBLE) <1:
        break
cap.release()
