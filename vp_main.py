import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from datetime import datetime
import os

# Initialize Mediapipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Initialize OpenCV Canvas
imgCanvas = np.zeros((720, 1280, 3), dtype=np.uint8)
imgCanvas.fill(255)
colors = [(0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
color_index = 0

# Initialize Finger Tracking Variables
thumb_x, thumb_y = None, None
index_x, index_y = None, None

# Set up hand state flag
hand_full = True

index_thumb_together = False

current_time = datetime.now().strftime("%Y%m%d%H%M%S")
file_name = "painting_" + current_time + ".jpg"

directory = "savepicture"

file_path = os.path.join(directory, file_name)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=1) as hands:

    # create window
    root = tk.Tk()
    root.withdraw()  # hide the main window

    # Start Video Stream
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    while cap.isOpened():

        # Read a Frame from the Video Stream
        success, image = cap.read()
        MediaBoard = np.zeros((720, 1280, 3), dtype=np.uint8)
        MediaBoard.fill(255)
        if not success:
            break

        # Flip the Image Horizontally for a Mirror Effect
        image = cv2.flip(image, 1)

        # Convert the Image from BGR to RGB and Process it with Mediapipe Hands
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Get Hand Landmark Positions
        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # mp_drawing.draw_landmarks(
                #     canvas,
                #     hand_landmarks,
                #     mp_hands.HAND_CONNECTIONS,
                #     mp_drawing_styles.get_default_hand_landmarks_style(),
                #     mp_drawing_styles.get_default_hand_connections_style()
                # )

                # Get the Index and Thumb Finger Landmark Positions
                index_x, index_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image.shape[1]), \
                    int(
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image.shape[0])
                thumb_x, thumb_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image.shape[1]), \
                    int(
                        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image.shape[0])
                # Get Wrist Landmark Position
                wrist_x, wrist_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image.shape[1]), \
                    int(
                        hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image.shape[0])
                # print("index_x and index_y is :" + str(index_x),str(index_y))
                # print("thumb_x and thumb_y is :" + str(thumb_x),str(thumb_y))
                # print("wrist_x and wrist_y is :" + str(wrist_x),str(wrist_y))

                # Draw a Circle on the Index and Thumb Finger Tips
                cv2.circle(image, (index_x, index_y),
                           10, (0, 255, 0), thickness=-1)
                cv2.circle(image, (thumb_x, thumb_y),
                           10, (0, 0, 255), thickness=-1)
                cv2.circle(image, (wrist_x, wrist_y),
                           10, (255, 0, 0), thickness=-1)

                prev_x, prev_y = wrist_x, wrist_y

                # Calculate the Distance between the Index and Thumb Finger Tips
                distance = np.linalg.norm(
                    np.array([index_x, index_y]) - np.array([thumb_x, thumb_y]))
                print("distance between index finger and thumb finger: " + str(distance))


                if distance < 40:
                    index_thumb_together = True
                else:
                    index_thumb_together = False

                # Check  if the index finger and thumb are together
                if index_thumb_together:
                    prev_x, prev_x = wrist_x, wrist_y
                    print("Not drawing")
                else:
                    # Draw a Line from the Previous Finger Position to the Current One
                    if 'prev_x' in locals() and 'prev_y' in locals():
                        thickness = max(1, int(distance/5))

                    cv2.line(imgCanvas, (prev_x, prev_y), (wrist_x,
                             wrist_y), colors[color_index], thickness)

                cv2.line(MediaBoard, (wrist_x, wrist_y), (wrist_x, wrist_y), (0, 0, 255), thickness=10)
                # Update the Previous Finger Position
                prev_x, prev_y = wrist_x, wrist_y

        # imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        # _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        # imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
        # image = cv2.bitwise_and(image,imgInv)
        # image = cv2.bitwise_or(image,imgCanvas)

        # print(image.shape)
        # print(imgInv.shape)

        # Display the Canvas and the Processed Image
        # cv2.imshow("Virtual Painting", imgCanvas)
        cv2.imshow("Camera", image)
        # cv2.imshow('White', imgInv)
        MediaBoard = cv2.bitwise_and(imgCanvas, MediaBoard)
        cv2.imshow("Media Board", MediaBoard)

        # Check for key presses
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            imgCanvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        elif key == ord('s'):
            cv2.imwrite(file_path, MediaBoard)
        elif key == ord('r'):
            color_index = (color_index + 1) % len(colors)

# Release camera and destroy windows
cap.release()
cv2.destroyAllWindows()
