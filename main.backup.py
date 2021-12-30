import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.model import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score


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


def extract_keypoints(results):
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


# Path for exported data, numpy arrays
DATA_PATH = os.path.join("MP_DATA")

# Actions that the code will try to detect
actions = np.array(["hello", "thanks", "iloveyou"])

# Thirty Videos worth of data
no_sequences = 30
# 30fps for each sequence
sequence_length = 30

# Makes a directory for each action to be detected with 30 different data points
for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass


# Set mediapipe model
with mp_holistic.Holistic(
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
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                print(results)

                # Draw landmarks
                draw_styled_landmarks(image, results)

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

                keypoints = extract_keypoints(results)

                # Save the keypoints in their respective action folder
                npy_path = os.path.join(
                    DATA_PATH, action, str(sequence), str(frame_num)
                )
                np.save(npy_path, keypoints)

                # If q is pressed exit
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break

    cap.release()
    cv2.destroyAllWindows()


"""
Extract saved data for training the model
"""
label_map = {label: num for num, label in enumerate(actions)}
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(
                os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
            )
            window.append(res)

        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# The test partition will be 5% of our saved data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)



log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

#Build the neural network
model = Sequential()
#Add layers
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
#Does not return sequences to the next layer
model.add(LSTM(64, return_sequences=False, activation='relu'))
#Fully connected layers
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
#Returns values with 0-1 with the sum of all values return summing to 1
model.add(Dense(actions.shape[0], activation='softmax'))

#Multiclass classfication model uses categorical_crossentropy
#Binary - Binary_crossentropy
#Regression -Mean Square Error (MSE)
model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['categorical_accuracy'])

#Train the model
model.fit(X_train, y_train, epochs= 2000, callbacks=[tb_callback])

#Save the trained model
model.save('action.h5')

#Load the trained model
model.load_weights('action.h5')


#Model evaluation
yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis =1).tolist()

#The numbers should be within the top left and bottom right range
multilabel_confusion_matrix(ytrue, yhat)

#The higher the number the better
score=accuracy_score(ytrue, yhat)
print(score)

def real_predict():
    sentence = []
    threshold = 0.5
    sequence = []
    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()

            #Make predeiction based on the frame
            image, results = mediapipe_detection(frame, holistic)
            print(results)

            draw_styled_landmarks(image, results)

            keypoints = extract_keypoints(results)
            sequence.append( keypoints)

            #Get the last 30 frames from the video feed
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
            
            #Compare the highest result to the threshold
            if res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    #Only append if two consequtive actions are different 
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])

            #If the sentence is greater than 5 only return the last five values
            if len(sentence) > 5:
                sentence = sentence[-5:]

            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    
            cv2.imshow("OpenCV Feed", image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()