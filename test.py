import cv2

img = cv2.VideoCapture(0)

while cv2.waitKey(33) < 0:
    ret, frame = img.read()
    cv2.imshow("Image", frame)
