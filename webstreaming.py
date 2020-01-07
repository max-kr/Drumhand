# USAGE
# python webstreaming.py --ip 0.0.0.0 --port 8000

from imutils.video import VideoStream
from flask import Flask, request, jsonify, Response, render_template
from flask_socketio import SocketIO, emit
import threading
import argparse
import imutils
import time
import cv2
from handmotionrecognizer import HandMotionRecognizer

outputFrame = None
prev_fingers = 0

app = Flask(__name__,
            static_url_path='', 
            static_folder='web/static',
            template_folder='web/templates')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

vs = VideoStream(src=0).start()
time.sleep(2.0)

@app.route("/")
def index():
    return render_template("index.html")

def detect_motion(frameCount):
    global vs, outputFrame, prev_fingers

    hr = HandMotionRecognizer(accumWeight=0.1)

    top, right, bottom, left = 10, 350, 225, 590
    num_frames = 0
    calibrated = False

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=700)
        frame = cv2.flip(frame, 1)

        (height, width) = frame.shape[:2]
        roi = frame[top:bottom, right:left]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if num_frames < 30:
            hr.update(gray)
            if num_frames == 1:
                print("[STATUS] please wait! calibrating...")
            elif num_frames == 29:
                print("[STATUS] calibration successfull...")
        else:
            hand = hr.segment(gray)

            if hand is not None:
                (thresholded, segmented) = hand
                cv2.drawContours(frame, [segmented + (right, top)], -1, (0, 0, 255))

                fingers = hr.count(thresholded, segmented)

                cv2.putText(frame, str(fingers), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                if fingers != prev_fingers:
                    prev_fingers = fingers
                    socketio.emit('output', str(prev_fingers))

        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)

        num_frames += 1

        outputFrame = frame.copy()

def generate():
    global outputFrame

    while True:
        (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

        if not flag:
            continue

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            encodedImage.tobytes() + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
        help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
        help="ephemeral port number of the server (1024 to 65535)")

    ap.add_argument("-f", "--frame-count", type=int, default=32,
        help="# of frames used to construct the background model")
    args = vars(ap.parse_args())

    t = threading.Thread(target=detect_motion, args=(
        args["frame_count"],))
    t.daemon = True
    t.start()

    socketio.run(app, host='localhost', port=8000)

vs.stop()