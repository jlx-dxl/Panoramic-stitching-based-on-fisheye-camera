import cv2

def camerachecking():
    s0 = cv2.VideoCapture(0)
    s1 = cv2.VideoCapture(1)
    s2 = cv2.VideoCapture(2)
    s3 = cv2.VideoCapture(3)

    while 1:
        ret0, frame0 = s0.read()
        cv2.imshow("capture0", frame0)
        if cv2.waitKey(10) & 0xff == ord('q'):
            break
    s0.release()
    cv2.destroyAllWindows()

    while 1:
        ret1, frame1 = s1.read()
        cv2.imshow("capture1", frame1)
        if cv2.waitKey(10) & 0xff == ord('q'):
            break
    s1.release()
    cv2.destroyAllWindows()

    while 1:
        ret2, frame2 = s2.read()
        cv2.imshow("capture2", frame2)
        if cv2.waitKey(10) & 0xff == ord('q'):
            break
    s2.release()
    cv2.destroyAllWindows()

    while 1:
        ret3, frame3 = s3.read()
        cv2.imshow("capture3", frame3)
        if cv2.waitKey(10) & 0xff == ord('q'):
            break
    s3.release()
    cv2.destroyAllWindows()

def cameraresetting(b,c,d):
    s1 = cv2.VideoCapture(b)
    s2 = cv2.VideoCapture(c)
    s3 = cv2.VideoCapture(d)
    return s1,s2,s3







