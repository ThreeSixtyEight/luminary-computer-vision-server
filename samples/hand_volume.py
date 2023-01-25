import hand_tracking_module as htm
import cv2
import numpy as np
import time
import math
import osascript

cap = cv2.VideoCapture(0)
detector = htm.HandDetector(detectionCon=0.7)

w_cam, h_cam = 1280, 720
target_volume = 0
target_volume_rect = 400
while cap.isOpened():
    # Get image frame
    success, img = cap.read()
    start = time.time()

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    img.flags.writeable = False

    # Find the hand and its landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    
    img.flags.writeable = True

    if len(lmList) > 3:
        fingers = detector.fingersUp()
        finger_count = sum(fingers)
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 10, (255, 255, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 255, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
        cv2.circle(img, (cx, cy), 15, (255, 255, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        print(length, finger_count, fingers)
        if finger_count <= 2 and fingers[0] == 1:
        # if finger_count == 2 and fingers[0] == 1 and fingers[1] == 1:
            target_volume = np.interp(length, [100, 700], [0, 100])
            target_volume_rect = np.interp(length, [100, 700], [400, 150])
            print(f'set volume output volume {target_volume}')
            osascript.osascript(f'set volume output volume {target_volume}')
            if target_volume == 0:
                cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)
            if target_volume == 100:
                cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
    # FPS Timing
    end = time.time()
    totalTime = end - start
    fps = 1 / totalTime

    # Add FPS
    cv2.putText(img, f'Volume: {int(target_volume)}%', (50,500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    if target_volume == 0:
        cv2.rectangle(img, (50, int(target_volume_rect)), (85, 450), (0, 0, 255), cv2.FILLED)
    elif target_volume == 100:
        cv2.rectangle(img, (50, int(target_volume_rect)), (85, 450), (0, 255, 0), cv2.FILLED)
    else:
        cv2.rectangle(img, (50, int(target_volume_rect)), (85, 450), (255, 0, 0), cv2.FILLED)
    
    # Display the frame
    cv2.imshow('Image', img)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()
