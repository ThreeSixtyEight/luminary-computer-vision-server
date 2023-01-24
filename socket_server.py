from flask import Flask, request,jsonify
from flask_socketio import SocketIO,emit
from flask_cors import CORS
from cvzone.HandTrackingModule import HandDetector
import cv2
import mediapipe as mp
from flask_apscheduler import APScheduler
import datetime
import eventlet

cap = cv2.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh
from engineio.payload import Payload

Payload.max_decode_packets = 50

mpHands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2,min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_face_mesh = mp.solutions.face_mesh

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app,resources={r"/*":{"origins":"*"}})
socketio = SocketIO(app,cors_allowed_origins="*")

SMOOTHNESS = 10
NOSE_POSITION = 28
raw_output = [[],[]]
last_values = []
average_x = 0
average_y = 0

@app.route("/http-call")
def http_call():
    """return JSON with string data as the value"""
    data = {'data':'This text was fetched using an HTTP call to server on render'}
    return jsonify(data)

@socketio.on("connect")
def connected():
    """event listener when client connects to the server"""
    print(request.sid)
    print("client has connected")
    # emit("message",{"data": 100})

@socketio.on('time')
def time_run():
    # print(data)
    global last_values, average_x, average_y
    percentage = 0
    x_percentage = 0
    y_percentage = 0
    value = 50
    tip = [0,0]
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
            for idx, lm in enumerate(hand.landmark):
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                if idx == 8:
                    x_percentage = int(x/img_w*100) - 50
                    y_percentage = int(y/img_h*100) - 50
                    raw_output[lm_idx] = [x_percentage,y_percentage]
                    
                    last_values.append(raw_output[0])

        if len(last_values) > SMOOTHNESS:
            last_values.pop(0)
    
        last_x_values = [coord[0] for coord in last_values]
        last_y_values = [coord[1] for coord in last_values]
        average_x = int(sum(last_x_values) / len(last_x_values))
        average_y = int(sum(last_y_values) / len(last_y_values))

    if results.multi_face_landmarks and not hand_landmarks:
        for lm_idx in range(len(results.multi_face_landmarks)):
            face_landmarks = results.multi_face_landmarks[lm_idx]
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 28:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    x_percentage = int(x/img_w*100) - 50
                    y_percentage = int(y/img_h*100) - 50
                    raw_output[lm_idx] = [x_percentage,y_percentage]

        if len(results.multi_face_landmarks) == 2:
            ox = (raw_output[0][0] + raw_output[1][0]) // 2
            oy = (raw_output[0][1] + raw_output[1][1]) // 2
            last_values.append([ox,oy])
        else:
            last_values.append(raw_output[0])

        if len(last_values) > SMOOTHNESS:
            last_values.pop(0)
    
        last_x_values = [coord[0] for coord in last_values]
        last_y_values = [coord[1] for coord in last_values]
        average_x = int(sum(last_x_values) / len(last_x_values))
        average_y = int(sum(last_y_values) / len(last_y_values))
    # print(average_x,average_y)
    emit("message",{"x": average_x, "y": average_y})
    eventlet.sleep(0)
    

@socketio.on("disconnect")
def disconnected():
    """event listener when client disconnects to the server"""
    print("user disconnected")
    emit("disconnect",f"user {request.sid} disconnected",broadcast=True)



if __name__ == '__main__':
    socketio.run(app,debug=False, port=5001)
