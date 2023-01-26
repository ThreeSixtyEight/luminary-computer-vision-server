from cvzone.HandTrackingModule import HandDetector
import cv2
from hand_detector import HandsDetector
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=2)

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)
handDetector = HandsDetector(min_detection_confidence=0.7)

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


while True:
    # Get image frame
    success, img = cap.read()
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    img.flags.writeable = False
    
    # Find the hand and its landmarks
    # hands = detector.findHands(img, draw=False)  # without draw
    # hands = detector.findHands(img, draw=False)  # without draw
    hands, img = detector.findHands(img)  # with draw
    count = 0
    fingers = []
    if hands:
        for hand in hands:
            fingers = fingers + detector.fingersUp(hand)
    count = sum(fingers)
    # Display
    cv2.putText(img, str(count), (875, 185), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 25)
    # results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # drawFaceMesh(img, results)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()
