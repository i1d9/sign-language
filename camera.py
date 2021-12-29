import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

# Create Functions
# Train view
# Prediction View


class SignCamera:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic  # Holistic Model to make detections
        self.mp_drawing = (
            mp.solutions.drawing_utils
        )  # Drawing utilities draw the detections
        # Access the webcam which has the value of 0
        self.cap = cv2.VideoCapture(0)
        # Path for exported data, numpy arrays
        self.DATA_PATH = os.path.join("MP_DATA")
        # Actions that the code will try to detect
        self.actions = np.array(["hello", "thanks", "iloveyou"])

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
    def draw_landmarks(self, image, results):
        self.mp_drawing.draw_landmarks(
            image, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS
        )
        self.mp_drawing.draw_landmarks(
            image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS
        )
        self.mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS
        )
        self.mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS
        )

    def draw_styled_landmarks(self, image, results):
        self.mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            self.mp_holistic.FACEMESH_CONTOURS,
            self.mp_drawing.DrawingSpec(
                color=(80, 110, 10), thickness=1, circle_radius=1
            ),
            self.mp_drawing.DrawingSpec(
                color=(80, 256, 121), thickness=1, circle_radius=1
            ),
        )
        self.mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            self.mp_holistic.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(
                color=(80, 22, 10), thickness=1, circle_radius=1
            ),
            self.mp_drawing.DrawingSpec(
                color=(80, 44, 121), thickness=1, circle_radius=1
            ),
        )
        self.mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            self.mp_holistic.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(
                color=(121, 22, 76), thickness=2, circle_radius=4
            ),
            self.mp_drawing.DrawingSpec(
                color=(121, 44, 250), thickness=2, circle_radius=2
            ),
        )
        self.mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            self.mp_holistic.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(
                color=(245, 117, 66), thickness=2, circle_radius=4
            ),
            self.mp_drawing.DrawingSpec(
                color=(245, 66, 230), thickness=2, circle_radius=2
            ),
        )

    def extract_keypoints(self, results):
        # Get the values for each landmark and error handle to get zeros if a human feature is not detected
        pose = (
            np.array(
                [
                    [res.x, res.y, res.z, res.visibility]
                    for res in results.pose_landmarks.landmark
                ]
            ).flatten()
            if results.pose_landmarks
            else np.zeros(33 * 4)
        )
        face = (
            np.array(
                [[res.x, res.y, res.z] for res in results.face_landmarks.landmark]
            ).flatten()
            if results.face_landmarks
            else np.zeros(468 * 3)
        )
        lh = (
            np.array(
                [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
            ).flatten()
            if results.left_hand_landmarks
            else np.zeros(21 * 3)
        )
        rh = (
            np.array(
                [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
            ).flatten()
            if results.right_hand_landmarks
            else np.zeros(21 * 3)
        )
        return np.concatenate([pose, face, lh, rh])

    def open_camera_and_save_data_for_training(self, actions):
        # Thirty Videos worth of data
        no_sequences = 30
        # 30fps for each sequence
        sequence_length = 30

        # Makes a directory for each action to be detected with 30 different data points
        for action in actions:
            for sequence in range(no_sequences):
                try:
                    os.makedirs(os.path.join(self.DATA_PATH, action, str(sequence)))
                except:
                    pass
        # Set mediapipe model
        with self.mp_holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as holistic:
            """
            The webcam will be accessible for 30 * number of actions frames
            """

            # Loop through each action
            for action in actions:
                # Loop through each sequence
                for sequence in range(no_sequences):

                    # Loop through each frame(30fps)
                    for frame_num in range(sequence_length):

                        # Read the frame from the webcam
                        ret, frame = self.cap.read()

                        # Make detections
                        image, results = self.mediapipe_detection(frame, holistic)
                        print(results)

                        # Draw landmarks
                        self.draw_styled_landmarks(image, results)

                        if frame_num == 0:
                            cv2.putText(
                                image,
                                "STARTING COLLECTION",
                                (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 255, 0),
                                1,
                                cv2.LINE_AA,
                            )
                            cv2.putText(
                                image,
                                f"Collecting frames for {action} video Number {sequence}",
                                (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 255),
                                1,
                                cv2.LINE_AA,
                            )
                            # Show the feed with the specified frame title
                            cv2.imshow("OpenCV Feed", image)
                            # Wait for two seconds
                            cv2.waitKey(2000)
                        else:
                            cv2.putText(
                                image,
                                f"Collecting frames for {action} video Number {sequence}",
                                (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 255),
                                1,
                                cv2.LINE_AA,
                            )
                            # Show the feed with the specified frame title
                            cv2.imshow("OpenCV Feed", image)

                        keypoints = self.extract_keypoints(results)

                        # Save the keypoints in their respective action folder
                        npy_path = os.path.join(
                            self.DATA_PATH, action, str(sequence), str(frame_num)
                        )
                        np.save(npy_path, keypoints)

                        # If q is pressed exit
                        if cv2.waitKey(10) & 0xFF == ord("q"):
                            break

            self.cap.release()
            cv2.destroyAllWindows()

    def open_camera_and_predict(self, model, actions):
        sentence = []
        threshold = 0.5
        sequence = []

        with self.mp_holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as holistic:
            while self.cap.isOpened():
                ret, frame = self.cap.read()

                # Make predeiction based on the frame
                image, results = self.mediapipe_detection(frame, holistic)
                print(results)

                self.draw_styled_landmarks(image, results)

                keypoints = self.extract_keypoints(results)
                sequence.append(keypoints)

                # Get the last 30 frames from the video feed
                sequence = sequence[-30:]

                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(actions[np.argmax(res)])

                # Compare the highest result to the threshold
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        # Only append if two consequtive actions are different
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

                # If the sentence is greater than 5 only return the last five values
                if len(sentence) > 5:
                    sentence = sentence[-5:]

                cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                cv2.putText(
                    image,
                    " ".join(sentence),
                    (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("OpenCV Feed", image)

                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break

            self.cap.release()
            cv2.destroyAllWindows()
