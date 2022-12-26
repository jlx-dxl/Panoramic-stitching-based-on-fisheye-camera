from detect import Detector
import cv2
detector = Detector()
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
while ret:
    ret, frame = cap.read()
    detector(frame)