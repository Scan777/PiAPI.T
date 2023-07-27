import cv2
import math
import numpy as np
import time

def coord(camera):
    cx2=0
    cy2=0
    A=0
    seconds_left = 5
    while seconds_left > 0:
        iSee = False
        controlX = 0.0
        success, frame, depth = camera.get_frame_stream()
        frame_re = frame[146 : 364, 238: 497]
        frame=frame_re
        if success: 
            height, width = frame.shape[0:2]
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            bin1 = cv2.inRange(hsv, (17, 158, 0), (35, 255, 255)) # желтый
            bin2 = cv2.inRange(hsv, (0, 158, 0), (16, 255, 255)) # красный
            bin3 = cv2.inRange(hsv, (36, 158, 0), (52, 255, 255)) # зеленый
            bin4 = cv2.inRange(hsv, (66, 158, 0), (108, 255, 255)) # голубой
            binary = bin1 + bin2+bin3+bin4
            roi = cv2.bitwise_and(frame, frame, mask=binary)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
            if len(contours) != 0:
                maxc = max(contours, key=cv2.contourArea)
                moments = cv2.moments(maxc)
                if moments["m00"] > 20:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                    cx2 = int((344/259)*cx)  # перевод пикселей в мм по оси x
                    cy2 = 294-(int((294/218)*cy)) # перевод пикселей в мм по оси y
                    cx3=238+cx
                    cy3=146+cy
                    depth = 96.3-(820-(depth[cy3, cx3].astype(float)))
                    # определение угла поворота кубика
                    #for cnt in contours:
                            #rect = cv2.minAreaRect(cnt)
                            #box = cv2.boxPoints(rect)
                            #box = np.int0(box)
                            #center = (int(rect[0][0]),int(rect[0][1]))
                            #area = int(rect[1][0]*rect[1][1])
                            #edge1 = np.int0((box[1][0] - box[0][0],box[1][1] - box[0][1]))
                            #edge2 = np.int0((box[2][0] - box[1][0], box[2][1] - box[1][1]))
                            #usedEdge = edge1
                            #if cv2.norm(edge2) > cv2.norm(edge1):
                             #   usedEdge = edge2
                            #reference = (1,0) 
                            #angle = -180+(180.0/math.pi * math.acos((reference[0]*usedEdge[0] + reference[1]*usedEdge[1]) / (cv2.norm(reference)*cv2.norm(usedEdge))))
                            #if angle>5 or angle<85:
                             #   if angle >=90:
                               #     A=90-angle
                    iSee = True
                    controlX = 2 * (cx - width/2) / width
                    cv2.drawContours(frame, maxc, -1, (0, 255, 0), 2)  
                    cv2.line(frame, (cx, 0), (cx, height), (0, 255, 0), 2)  
                    cv2.line(frame, (0, cy), (width, cy), (0, 255, 0), 2) 
            cv2.putText(frame, 'Axes x: {};'.format(cx2), (width - 500, height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, 'Axes y: {}'.format(cy2), (width - 200, height - 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            cv2.waitKey(1)
            seconds_left -= 1
            time.sleep(1)
    cv2.destroyAllWindows()
    return(cx2,cy2,depth)