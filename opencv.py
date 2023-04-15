import cv2

cap = cv2.VideoCapture(0)
while cap.isOpened():

    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    cv2.imshow("Camera", image)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
