import cv2
import time
import argparse
from operator import itemgetter
from realsense_camera import *

from ultralytics import YOLO
import supervision as sv
from supervision import Detections
from supervision.geometry.core import Position, Point
import numpy as np

#Сортировка массива по возрастанию
def column(matrix, i):
    return [row[i] for row in matrix]

#Вывод видео потока
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution", 
        default=[640, 480], 
        nargs=2, 
        type=int
    )
    args = parser.parse_args(args=[])
    return args


def Yolo(cap):
    seconds_left = 2
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution
    model = YOLO("finish.pt")
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )
    
    while seconds_left > 0:
        ret, frame, depth = cap.get_frame_stream()
        frame_re = frame[65 : 430, 80: 640]
        frame=frame_re
        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)
        
        coord=(sv.Detections.get_anchor_coordinates(detections, anchor=Position.CENTER)).astype("int")
        if coord.size != 0:
            count_object=sv.Detections.__len__(detections)
            if count_object!=0:
                [detections.confidence[i]>=0.5 for i in range(count_object)]
                coord_center=np.array(coord)    
                y_sort=sorted(coord_center,key=itemgetter(1)) #Координаты [x,y] отсортированные по возрастанию y
                x_sort=sorted(coord_center,key=itemgetter(0)) #Координаты [x,y] отсортированные по возрастанию x
                x_mm=[(((1.3)*x_sort[i][0])).astype(int)  for i in range(count_object)]
                y_mm=[((400-((1.09)*x_sort[i][1]))).astype(int) for i in range(count_object)]
                box=detections.xyxy
                x_coord=column(x_sort,0) #Координаты x отсортированные по возрастанию
                y_coord=column(y_sort,1) #Координаты y отсортированные по возрастанию
                depth = [depth[65+x_sort[i][1], 80+x_sort[i][0]].astype(int) for i in range(count_object)]
                depth_robot=[(-55+(855-depth[i])) for i in range(count_object)]
                labels = [
                    f"{model.model.names[class_id]} {confidence:0.2f}"
                    for _, _, confidence, class_id, _
                    in detections
                ]

                frame = box_annotator.annotate(
                    scene=frame, 
                    detections=detections, 
                    labels=labels
                )

                #Рисование и определение индексов элементов на frame
                for j in range(count_object):
                    #depth = (depth[y_coord, x_coord].astype(float))[j]
                    text=str(x_coord.index(x_coord[j]))
                    text1=str(depth[j])
                    cv2.circle(frame, x_sort[j], 6, (118, 103, 154), 6)
                    cv2.putText(frame, text, x_sort[j]+5, cv2.FONT_HERSHEY_SIMPLEX,
                    1, (118, 103, 154), 2, cv2.LINE_AA)
                    cv2.putText(frame, text1, x_sort[j]+20, cv2.FONT_HERSHEY_SIMPLEX,
                    1, (118, 103, 154), 2, cv2.LINE_AA)
                    j=j+1                
                seconds_left -= 1  
                time.sleep(1)
                cv2.imshow("yolov8", frame)
                cv2.waitKey(1)
                return(x_mm,y_mm, depth_robot, count_object, box)
            else:
                return('объект не найден')
    cv2.destroyAllWindows()
