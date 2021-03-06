#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html#converting-colorspaces
import cv2
import numpy as np

cap = cv2.VideoCapture('casetest.mp4')


while(cap.isOpened()):
    # Take each frame
    ret, frame = cap.read()
    if ret:
    # Convert BGR to HSV
    #if np.shape(cap) != ():
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
        lower_blue = np.array([110,50,50])
        upper_blue = np.array([130,255,255])

    # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame,frame, mask= mask)

    #cv2.imshow('frame',frame)
    #cv2.imshow('mask',mask)
        cv2.imshow('res',res)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
cap.release()
while(1):
    pass
#cv2.destroyAllWindows()