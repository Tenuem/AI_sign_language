import cv2
import mediapipe as mp
import numpy as np
import os

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

NUMBER_OF_POINTS_IN_HAND = 21
path = 'C:\\Python\\Projects\\paluszki\\datasrata'
# For static images:
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
    #create csv header
    data_file = open('test_data.csv','w')
    data_file.write('sign')
    for i in range(NUMBER_OF_POINTS_IN_HAND):
        data_file.write(',x'+str(i))
        data_file.write(',y'+str(i))
        data_file.write(',z'+str(i))
    data_file.write('\n')
   
    failures = 0
    total = 0
    dir_list = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    for dir in dir_list:
        file_path = path + '\\' + dir
        #print(file_path)
        file_list = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
        #print("Found ", len(file_list), " files ")
        for f in file_list:
            total += 1
            #print(f)
            image = cv2.imread(file_path+'\\'+f)  
            # Convert the BGR image to RGB before processing.
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print handedness and draw hand landmarks on the image.
            #print('Handedness:', results.multi_handedness)
            image_height, image_width, _ = image.shape
            annotated_image = image.copy()
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    #print('hand_landmarks:', hand_landmarks)
                    #print(
                    #    f'Index finger tip coordinates: (',
                    #    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                    #    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                    #)
                    # index of sign
                    if dir == 'del': data_file.write('26')
                    elif dir == 'nothing': data_file.write('27')
                    elif dir == 'space': data_file.write('28')
                    else: data_file.write(str(ord(dir)-ord('A')))
                    # coordinates of points
                    for i in range(NUMBER_OF_POINTS_IN_HAND):
                        data_file.write(',' + str(hand_landmarks.landmark[i].x))
                        data_file.write(','+str( hand_landmarks.landmark[i].y))
                        data_file.write(','+str( hand_landmarks.landmark[i].z))
                    data_file.write('\n')
            elif dir == 'nothing':
                data_file.write('27')
                for i in range(3*NUMBER_OF_POINTS_IN_HAND):
                    data_file.write(',0')
                data_file.write('\n')
            else: 
                print(f'unsuccessful attempt in dir {dir} in file {f}')
                failures += 1
            
    print(f'For {total} attempts {(total-failures)//total * 100}% were successful')
    data_file.close()     
        