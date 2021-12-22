import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp


mp_holistic = mp.solutions.holistic #Holistic Model to make detections
mp_drawing = mp.solutions.drawing_utils #Drawing utilities draw the detections

#Pass frame from OpenCV and the holistic model
def mediapipe_detection(image, model):
    #Color Conversion from one color space to another(Blue Green Red to Red Green Blue)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False 
    #Detection using media pipe
    results = model.process(image)
    image.flags.writeable = True
    #Color Conversion
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results




#Access the webcam which has the value of 0
cap = cv2.VideoCapture(0)

#Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    #Check if the webcam is being accessed
    while cap.isOpened:

        #Read the frame from the webcam
        ret, frame = cap.read()

        #Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)

        #Show the feed with the specified frame title
        cv2.imshow("OpenCV Feed", frame)


        #If q is pressed exit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()