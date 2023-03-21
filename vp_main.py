import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Initialize OpenCV Canvas
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
color_index = 0

# Initialize Finger Tracking Variables
prev_x, prev_y = None, None
thumb_x, thumb_y = None, None
index_x, index_y = None, None

# Set up hand state flag
hand_full = True
hand_full_threshold = 25

index_thumb_together = False

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:

    # Start Video Stream
    cap = cv2.VideoCapture(0)
    while cap.isOpened():

        # Read a Frame from the Video Stream
        success, image = cap.read()
        if not success:
            break

        # Flip the Image Horizontally for a Mirror Effect
        image = cv2.flip(image, 1)

        # Convert the Image from BGR to RGB and Process it with Mediapipe Hands
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Extract Hand Landmark Positions
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Extract the Index and Thumb Finger Landmark Positions
                index_x, index_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image.shape[1]), \
                                   int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image.shape[0])
                thumb_x, thumb_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image.shape[1]), \
                                   int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image.shape[0])
                # print(index_x, index_y)

                # Draw a Circle on the Index and Thumb Finger Tips
                cv2.circle(image, (index_x, index_y), 10, (0, 255, 0), thickness=-1)
                cv2.circle(image, (thumb_x, thumb_y), 10, (0, 0, 255), thickness=-1)

                # Calculate the Distance between the Index and Thumb Finger Tips
                distance = np.linalg.norm(np.array([index_x, index_y]) - np.array([thumb_x, thumb_y]))
                print(distance)

                # Update the Line Thickness based on the Distance between the Fingers
                thickness = max(5, int(distance / 10))

                # Determine if the Index and Thumb Fingers are Closed or Opened
                is_hand_full = distance < 25

                if distance < 35:
                    index_thumb_together = True
                else:
                    index_thumb_together = False

                # Check  if the index finger and thumb are together
                if index_thumb_together:
                    prev_x, prev_x = index_x, index_y
                else:
                    # Draw a Line from the Previous Finger Position to the Current One
                    if 'prev_x' in locals() and 'prev_y' in locals():
                        if is_hand_full:
                            thickness = max(1, thickness - 2)
                        else:   
                            thickness = min(20, thickness + 2)
                    cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), colors[color_index], thickness)

                # Update the Previous Finger Position
                prev_x, prev_y = index_x, index_y

        # Display the Canvas and the Processed Image
        cv2.imshow("Virtual Painting", canvas)
        cv2.imshow("Camera", image)

       

        # Check for key presses
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        elif key == ord('s'):
            cv2.imwrite("painting.jpg", canvas)
        elif key == ord('r'):
            color_index = (color_index + 1) % len(colors)

# Release camera and destroy windows
cap.release()
cv2.destroyAllWindows()
