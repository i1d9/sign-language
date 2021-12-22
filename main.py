import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

#Access the webcam which has the value of 0
cap = cv2.VideoCapture(0)

#Check if the webcam is being accessed
while cap.isOpened:

    #Read the frame from the webcam
    ret, frame = cap.read()

    #Show the feed with the specified frame title
    cv2.imshow("OpenCV Feed", frame)


    #If q is pressed exit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()