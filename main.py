import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp


mp_holistic = mp.solutions.holistic  # Holistic Model to make detections
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities draw the detections

# Pass frame from OpenCV and the holistic model
def mediapipe_detection(image, model):
    # Color Conversion from one color space to another(Blue Green Red to Red Green Blue)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    # Detection using media pipe
    results = model.process(image)
    image.flags.writeable = True
    # Color Conversion
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


# Draws detected landmarks connections from the media pipe model
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS
    )
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
    )
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
    )
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
    )


def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
    )
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=1, circle_radius=1),
    )
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
    )
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
    )


# Access the webcam which has the value of 0
cap = cv2.VideoCapture(0)

# Set mediapipe model
with mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) as holistic:
    # Check if the webcam is being accessed
    while cap.isOpened:

        # Read the frame from the webcam
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)

        # Draw landmarks
        draw_styled_landmarks(image, results)

        # Show the feed with the specified frame title
        cv2.imshow("OpenCV Feed", image)

        # If q is pressed exit
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
