import cv2
import numpy as np
import mediapipe as mp
import math
import tkinter as tk
import time
from tkinter import messagebox as msgb
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

global xc, yc, ix, iy, operation, toolSet, f1, f2, f3, f4, f5, thumb_y, pinky_y
arr_x = np.zeros(1000)
arr_y = np.zeros(1000)
operation = "draw"
toolClicked = False
ix, iy = -1, -1
count = 0
crop_count = 1
xc = 1920
yc = 1080


def vector_2d_angle(v1, v2):
    v1_x = v1[0]
    v1_y = v1[1]
    v2_x = v2[0]
    v2_y = v2[1]
    try:
        angle_ = math.degrees(math.acos(
            (v1_x*v2_x+v1_y*v2_y)/(((v1_x**2+v1_y**2)**0.5)*((v2_x**2+v2_y**2)**0.5))))
    except:
        angle_ = 180
    return angle_


def hand_angle(hand_):
    angle_list = []
    # thumb 大拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[2][0])),
         (int(hand_[0][1])-int(hand_[2][1]))),
        ((int(hand_[3][0]) - int(hand_[4][0])),
         (int(hand_[3][1]) - int(hand_[4][1])))
    )
    angle_list.append(angle_)
    # index 食指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])-int(hand_[6][0])),
         (int(hand_[0][1]) - int(hand_[6][1]))),
        ((int(hand_[7][0]) - int(hand_[8][0])),
         (int(hand_[7][1]) - int(hand_[8][1])))
    )
    angle_list.append(angle_)
    # middle 中指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[10][0])),
         (int(hand_[0][1]) - int(hand_[10][1]))),
        ((int(hand_[11][0]) - int(hand_[12][0])),
         (int(hand_[11][1]) - int(hand_[12][1])))
    )
    angle_list.append(angle_)
    # ring 無名指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[14][0])),
         (int(hand_[0][1]) - int(hand_[14][1]))),
        ((int(hand_[15][0]) - int(hand_[16][0])),
         (int(hand_[15][1]) - int(hand_[16][1])))
    )
    angle_list.append(angle_)
    # pink 小拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[18][0])),
         (int(hand_[0][1]) - int(hand_[18][1]))),
        ((int(hand_[19][0]) - int(hand_[20][0])),
         (int(hand_[19][1]) - int(hand_[20][1])))
    )
    angle_list.append(angle_)
    return angle_list


def showMessage(message, type='info', timeout=2500):

    root = tk.Tk()
    root.withdraw()
    try:
        root.after(timeout, root.destroy)
        if type == 'info':
            msgb.showinfo('Info', message, master=root)
        elif type == 'warning':
            msgb.showwarning('Warning', message, master=root)
        elif type == 'error':
            msgb.showerror('Error', message, master=root)
    except:
        pass


def hand_pos(finger_angle):
    global f1, f2, f3, f4, f5, crop_count
    f1 = finger_angle[0]   # 大拇指角度
    f2 = finger_angle[1]   # 食指角度
    f3 = finger_angle[2]   # 中指角度
    f4 = finger_angle[3]   # 無名指角度
    f5 = finger_angle[4]   # 小拇指角度
# print(f1,f2,f3,f4,f5)
    if f1 >= 100 and f2 < 50 and f3 < 50 and f4 >= 100 and f5 >= 160:
        # print(thumb_y)
        # print(pinky_y)
        # if thumb_y < pinky_y:
        cv2.imwrite('crop%d.jpg' % crop_count, board)
        showMessage('儲存中請稍後')
        crop_count = crop_count+1
# messagebox.showinfo(title="通知", message="儲存中請稍後")
# time.sleep( 5 )

# if thumb_y > pinky_y:
    if f1 >= 100 and f2 < 40 and f3 >= 100 and f4 >= 100 and f5 >= 160:
        board[:, :, 0].fill(170)
        board[:, :, 1].fill(232)
        board[:, :, 2].fill(238)
        pts = np.array([[960, 50], [470, 540], [960, 1030], [1450, 540]])
        cv2.fillPoly(board, [pts], (0, 0, 255))


def drawing(event, x, y, z, flags):
    global ix, iy, operation, operationIcon, toolClicked, count
    depth = (z-30)//1
    if depth < 10:
        depth = 10
    if depth >= 40:
        depth = 40
    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = x, y
    if (flags == 1):
        if ix == -1 and iy == -1:
            cv2.line(board, (x, y), (x, y), (0, 0, 0), 1+depth)
            ix, iy = x, y
        else:
            cv2.line(board, (ix, iy), (x, y), (0, 223, 255), 1+depth)
            ix, iy = x, y


board = np.ones((1080, 1920, 3), np.uint8)
board[:, :, 0].fill(170)
board[:, :, 1].fill(232)
board[:, :, 2].fill(238)
pts = np.array([[960, 50], [470, 540], [960, 1030], [1450, 540]])
cv2.fillPoly(board, [pts], (0, 0, 255))

cv2.namedWindow("Media Board", cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    "Media Board", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback("Media Board", drawing)

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        max_num_hands=1) as hands:
    while cap.isOpened():
        success, image = cap.read()
        MediaBoard = np.ones((1080, 1920, 3), np.uint8)
        MediaBoard.fill(255)

        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                finger_points = []
                for i in hand_landmarks.landmark:
                    xi = i.x*xc
                    yi = i.y*yc
                    finger_points.append((xi, yi))
                if finger_points:
                    finger_angle = hand_angle(finger_points)
                    text = hand_pos(finger_angle)
            x, y = int(results.multi_hand_landmarks[0].landmark[0].x*xc)+15, int(
                results.multi_hand_landmarks[0].landmark[0].y*yc+15)
            x2, y2 = int(results.multi_hand_landmarks[0].landmark[12].x*xc), int(
                results.multi_hand_landmarks[0].landmark[12].y*yc)
            z = int(results.multi_hand_landmarks[0].landmark[12].z*-1000)
            thumb_y = results.multi_hand_landmarks[0].landmark[4].y
            pinky_y = results.multi_hand_landmarks[0].landmark[20].y
            print(z)

            if f1 < 60 and f2 < 60 and f3 < 60 and f4 < 60 and f5 < 60:
                if count <= 1:
                    arr_x[count] = x
                    arr_y[count] = y
                    count = count + 1
                else:
                    arr_x[count] = x
                    arr_y[count] = y
                    if arr_x[count-2]-500 <= arr_x[count-1] <= arr_x[count]+500 and arr_y[count-2]-500 <= arr_y[count-1] <= arr_y[count]+500:
                        drawing("", x, y, z, 1)
                    else:
                        continue
            else:
                drawing("", x, y, z, 0)
                count = 0

            # mask[x-50:x, x-50:x] = operationIcon
            if operation == "erase":
                cv2.line(MediaBoard, (x, y), (x, y), (255, 0, 0), 20)
            else:
                if z > 60:
                    cv2.line(MediaBoard, (x, y), (x, y), (255, 0, 0), 20)
                else:
                    cv2.line(MediaBoard, (x, y), (x, y), (0, 0, 0), 5)

            cv2.line(image, (x, y), (x, y), (0, 0, 0), 10)
            cv2.line(image, (x2, y2), (x2, y2), (0, 0, 0), 10)
            ix, iy = x, y

        cv2.imshow('Hand Landmarks', image)
        # mask[200:250, 200:250] = operationIcon
        MediaBoard = cv2.bitwise_and(board, MediaBoard)
        # mask = cv2.bitwise_and(image, mask)
        # cv2.imshow('Board', board)
        cv2.imshow('Media Board', MediaBoard)
        if cv2.waitKey(5) & 0xFF == 27:
            break


cap.release()
cv2.destroyAllWindows()
