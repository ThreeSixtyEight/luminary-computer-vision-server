from cvzone.HandTrackingModule import HandDetector
import cv2
from hand_detector import HandsDetector
import mediapipe as mp
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)
handDetector = HandsDetector(min_detection_confidence=0.7)

# Left eye indices list
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
# Right eye indices list
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]

RIGHT_IRIS = [474, 475, 476, 477]
LEFT_IRIS = [469, 470, 471, 472]
L_H_LEFT = [33]  # right eye right most landmark
L_H_RIGHT = [133]  # right eye left most landmark
R_H_LEFT = [362]  # left eye right most landmark
R_H_RIGHT = [263]  # left eye left most landmark

# Euclaidean distance
def euclaideanDistance(point, point1):
    x1, y1 = point.ravel()
    x2, y2 = point1.ravel()
    distance = math.sqrt((x2-x1)**2 + (y2 - y1)**2)
    return distance

def iris_position(iris_center, right_point, left_point):
    center_to_right_dist = euclaideanDistance(iris_center, right_point)
    total_dist = euclaideanDistance(right_point, left_point)
    ratio = center_to_right_dist/total_dist
    iris_position = ''
    
    if ratio <= 0.42:
        iris_position = 'right'
    elif ratio > 0.42 and ratio <= 0.57:
        iris_position = 'center'
    else:
        iris_position = 'left'

    return iris_position, ratio


def drawFaceMesh(image, results, dp=drawing_spec):
    image.flags.writeable = True
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=dp,
                connection_drawing_spec=dp)
        cv2.imshow('MediaPipe FaceMesh', image)
transparent = np.zeros((400, 400, 4), dtype=np.uint8)

while cap.isOpened():
    # Get image frame
    success, frame = cap.read()
    frame.flags.writeable = False
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as face_mesh:
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        
        results = face_mesh.process(rgb_frame)
        #getting width and height or frame
        img_h, img_w = rgb_frame.shape[:2]
        if results.multi_face_landmarks:
            mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            
            # turn center points into np array 
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)
            
            # turn center points into np array 
            cv2.circle(frame, center_left, int(l_radius), (255,0,255), 2)
            cv2.circle(frame, center_right, int(r_radius), (255,0,255), -1)

            # draw eyes
            cv2.polylines(frame, [mesh_points[LEFT_EYE]], True, (0, 255, 0), 2)
            cv2.polylines(frame, [mesh_points[RIGHT_EYE]], True, (0, 255, 0), 2)
            
            # draw eye edge points
            cv2.circle(frame, mesh_points[R_H_RIGHT][0], 3, (255,255,255), -1, cv2.LINE_AA)
            cv2.circle(frame, mesh_points[R_H_LEFT][0], 3, (0,255,255), -1, cv2.LINE_AA)
            
            ip, ratio = iris_position(center_right, mesh_points[R_H_RIGHT][0], mesh_points[R_H_LEFT][0])
            print(ip, ratio)

            mask = np.zeros((img_h, img_w, 4), dtype=np.uint8)
            

            cv2.circle(mask, center_left, int(l_radius), (255,255,255), -1, cv2.LINE_AA)
            cv2.circle(mask, center_right, int(r_radius), (255,255,255), -1, cv2.LINE_AA)
            drawFaceMesh(frame, results)
            # cv2.imshow('MediaPipe Mask', mask)
            # cv2.imshow('MediaPipe FaceMesh', frame)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()
