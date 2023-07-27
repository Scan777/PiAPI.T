import cv2
import numpy as np
import glob

cap = cv2.VideoCapture(2)

if cap.isOpened() is True:
    i = 0
    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Преобразовать в серый канал
        cv2.imshow('frame', gray)
        if i % 50 == 0:
            cv2.imwrite('**********' + str(i//50) + '.png', gray)
        if (cv2.waitKey(1) & 0xFF == ord('q')) or (i==10000):
            break
        i += 1
    cap.release()
    cv2.destroyAllWindows()