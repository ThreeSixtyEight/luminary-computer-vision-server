import cv2
import numpy as np
import pose_module as pm

cap = cv2.VideoCapture(0)
detector = pm.pose_detector()
count = 0
direction = 0
form = 0
feedback = "Fix Form"

alpha = 0.7

while True:
    ret, img = cap.read() #640 x 480
    print(img)
    #Determine dimensions of video - Help with creation of box in Line 43
    width  = cap.get(3)  # float `width`
    height = cap.get(4)  # float `height`
    # print(width, height)
    img_height, img_width= img.shape[:2]
    shapes = np.zeros_like(img, np.uint8)
    out = img.copy()
    img = detector.findPose(img, False)
    lmList = detector.findPosition(out, False)
    
    if len(lmList) != 0:
        
        x1, ankle_y = lmList[27][1], lmList[27][2]
        x2, wrist_y = lmList[15][1], lmList[15][2]

        elbow = detector.findAngle(img, 11, 13, 15)
        shoulder = detector.findAngle(img, 13, 11, 23)
        hip = detector.findAngle(img, 11, 23, 25)
        knee = detector.findAngle(img, 23, 25, 27)
        ankle = detector.findAngle(img, 25, 27, 31)
        
        #Percentage of success of pushup
        per = np.interp(elbow, (90, 160), (0, 100))
        
        #Bar to show Pushup progress
        bar = np.interp(elbow, (90, 160), (380, 50))

        #Check to ensure right form before starting the program
        if elbow > 160 and shoulder > 40 and hip > 160 and wrist_y > ankle_y:
            form = 1
    
        #Check for full range of motion for the pushup
        if form == 1:
            if per == 0:
                if elbow <= 90 and hip > 160:
                    feedback = "Up"
                    if direction == 0:
                        count += 0.5
                        direction = 1
                else:
                    feedback = "Fix Form"
                    
            if per == 100:
                if elbow > 160 and shoulder > 40 and hip > 160:
                    feedback = "Down"
                    if direction == 1:
                        count += 0.5
                        direction = 0
                else:
                    feedback = "Fix Form"
                        # form = 0
                
                    
    
        print(count)

        box_width = 200
        box_margin = 20
        feedback_height = 50

        slider_width = 20
        slider_x = int(width - 100)
        slider_height = 330
        slider_y = 50
        #Draw Bar
        cv2.rectangle(shapes, (slider_x, slider_y), (slider_x+slider_width, slider_y+slider_height), (0, 255, 0), 3)
        if form == 1:
            cv2.rectangle(shapes, (slider_x, int(bar)), (slider_x+slider_width, slider_y+slider_height), (0, 255, 0), cv2.FILLED)
            # cv2.rectangle(shapes, (int(width - 100), int(bar)), (int(width - 80), 380), (0, 255, 0), cv2.FILLED)
            # cv2.putText(shapes, f'{int(per)}%', (565, 430), cv2.FONT_HERSHEY_SIMPLEX, 2,
            #             (255, 0, 0), 2)


        #Pushup counter
        cv2.putText(img, str(int(count)), (70, 265), cv2.FONT_HERSHEY_SIMPLEX, 5,(255, 255, 255), 5)
        cv2.putText(img, 'PUSH UPS', (55, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255, 255, 255), 2)
        cv2.rectangle(shapes, (box_margin, box_margin*2 + feedback_height), (box_margin + box_width, 360), (255, 255, 255), cv2.FILLED)
                    
        
        #Feedback 
        cv2.putText(img, feedback, (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255), 2)
        if feedback == "Fix Form":
            cv2.rectangle(shapes, (box_margin, box_margin), (box_margin + box_width, feedback_height+box_margin), (0, 0, 255), cv2.FILLED)
        else:
            cv2.rectangle(shapes, (box_margin, box_margin), (box_margin + box_width, feedback_height+box_margin), (0, 255, 0), cv2.FILLED)
        

    img_new = cv2.addWeighted(img, alpha, shapes, 1 - alpha, 0)
        
    cv2.imshow('Output', img_new)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
