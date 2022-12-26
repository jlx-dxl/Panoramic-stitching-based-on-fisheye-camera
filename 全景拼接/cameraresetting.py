import cv2

def cameraresetting(b,c,d):
    s1 = cv2.VideoCapture(b)
    s2 = cv2.VideoCapture(c)
    s3 = cv2.VideoCapture(d)
    return s1,s2,s3







