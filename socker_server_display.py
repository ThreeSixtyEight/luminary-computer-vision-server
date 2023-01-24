import cv2
import mediapipe as mp
import numpy as np
import time
from cvzone.HandTrackingModule import HandDetector

GREEN_COLOR = (0, 255, 0)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2,min_detection_confidence=0.5, min_tracking_confidence=0.5)

mpHands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
drawing_spec_green = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=GREEN_COLOR)
dp = drawing_spec

cap = cv2.VideoCapture(0)
raw_output = [[],[]]
percentage = [[0,0], [0,0]]
output_percentage = [0,0]
landMarkList = []

while cap.isOpened():
    
    value = 50
    success, image = cap.read()
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False
    
    # Get the result
    results = face_mesh.process(image)
    hand_results = mpHands.process(image)
    
    # To improve performance
    image.flags.writeable = True
    
    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    hand_landmarks = hand_results.multi_hand_landmarks
    if hand_landmarks:  # returns None if hand is not found
        
        for lm_idx in range(len(hand_landmarks)):
            hand = hand_landmarks[lm_idx]
            mp_drawing.draw_landmarks(image, hand, mp.solutions.hands.HAND_CONNECTIONS)
            for idx, lm in enumerate(hand.landmark):
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                landMarkList.append([id, x, y])
                if idx == 8:
                    cv2.circle(image, (x, y), 16, (0, 255, 0), -1)
                    if len(hand_landmarks) < 2:
                        raw_output[lm_idx] = [x,y]
                        percentage[lm_idx][0] = int(x/img_w*100) - 50
                        percentage[lm_idx][1] = int(y/img_h*100) - 50
                    else:
                        percentage[0] = [10000,10000]

    cv2.line(image, (0, int(img_h//2)), (img_w, int(img_h//2)), (255, 0, 255), 1)
    cv2.line(image, (int(img_w//2), 0), (int(img_w//2), img_h), (255, 0, 255), 1)
    cv2.circle(image, (int(img_w//2), int(img_h//2)), 4, (255, 255, 255), -1)
    dp = drawing_spec
    if results.multi_face_landmarks :
        for lm_idx in range(len(results.multi_face_landmarks)):
            face_landmark = results.multi_face_landmarks[lm_idx]
            if not hand_landmarks:
                dp = drawing_spec_green
                for idx, lm in enumerate(face_landmark.landmark):
                    if idx == 168:
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        raw_output[lm_idx] = [x,y]
                        percentage[lm_idx][0] = int(x/img_w*100) - 50
                        percentage[lm_idx][1] = int(y/img_h*100) - 50

                        # if percentage[lm_idx] < 0:
                        #     cv2.putText(image, f'{abs(percentage[lm_idx])}', (int(img_w/2),200), cv2.FONT_HERSHEY_SIMPLEX, 4.5, (0,0,255), 3)
                        # else:
                        #     cv2.putText(image, f'{abs(percentage[lm_idx])}', (int(img_w/2),200), cv2.FONT_HERSHEY_SIMPLEX, 4.5, (0,255,0), 3)
                        cv2.circle(image, (x, y), 12, (0, 255, 0), -1)
            
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmark,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=dp,
                connection_drawing_spec=dp)
        
        if not hand_landmarks and len(results.multi_face_landmarks) == 2:
            print(len(results.multi_face_landmarks))
            cv2.line(image, (raw_output[0][0], raw_output[0][1]), (raw_output[1][0], raw_output[1][1]), (255, 255, 255), 3)
            ox = (raw_output[0][0] + raw_output[1][0]) // 2
            oy = (raw_output[0][1] + raw_output[1][1]) // 2
            cv2.circle(image, (ox, oy), 12, (255, 0, 255), -1)
            output_center = [ox, oy]
            output_percentage = [int(ox/img_w*100) - 50, int(oy/img_h*100) - 50]
        else:
            output_percentage = percentage[0]
        
    if hand_landmarks or results.multi_face_landmarks: 
        if output_percentage[0] < 0:
            cv2.putText(image, f'{output_percentage[0]}%', (50, int(img_h/2)-50), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0,0,255), 3)
        else:
            cv2.putText(image, f'{output_percentage[0]}%', (50, int(img_h/2)-50), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0,255,0), 3)
        if output_percentage[1] < 0:
            cv2.putText(image, f'{output_percentage[1]}%', (int(img_w/2)+20, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0,0,255), 3)
        else:
            cv2.putText(image, f'{output_percentage[1]}%', (int(img_w/2)+20, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0,255,0), 3)

        # end = time.time()
        # totalTime = end - start

        # fps = 1 / totalTime
        # #print("FPS: ", fps)

        # cv2.putText(image, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

        # mp_drawing.draw_landmarks(
        #             image=image,
        #             landmark_list=face_landmarks,
        #             connections=mp_face_mesh.FACEMESH_CONTOURS,
        #             landmark_drawing_spec=drawing_spec,
        #             connection_drawing_spec=drawing_spec)


    cv2.imshow('Head Pose Estimation', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break


cap.release()
