# USAGE
# python webstreaming.py --ip 0.0.0.0 --port 8000

# import the necessary packages
from imutils.video import VideoStream
from flask import Flask, request, jsonify, Response, render_template
from flask_socketio import SocketIO, emit
import threading
import argparse
import imutils
import time
import cv2
from handmotionrecognizer import HandMotionRecognizer

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
prev_fingers = 0

# initialize a flask object
app = Flask(__name__,
            static_url_path='', 
            static_folder='web/static',
            template_folder='web/templates')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# initialize the video stream and allow the camera sensor to
# warmup
vs = VideoStream(src=0).start()
time.sleep(2.0)

@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")

def detect_motion(frameCount):
    # grab global references to the video stream, output frame, and
    # lock variables
    global vs, outputFrame, prev_fingers

    # initialize the motion detector and the total number of frames
    # read thus far
    hr = HandMotionRecognizer(accumWeight=0.1)

    top, right, bottom, left = 10, 350, 225, 590
    num_frames = 0
    calibrated = False

    # loop over frames from the video stream
    while True:
        # read the next frame from the video stream, resize it,
        # convert the frame to grayscale, and blur it
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
                
                #cv2.imshow("Thesholded", thresholded)

        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)

        num_frames += 1

        #cv2.imshow("Video Feed", clone)

        # update the background model and increment the total number
        # of frames read thus far

        outputFrame = frame.copy()

def generate():
    # grab global references to the output frame and lock variables
    global outputFrame

    # loop over frames from the output stream
    while True:
        # encode the frame in JPEG format
        (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

        # ensure the frame was successfully encoded
        if not flag:
            continue

        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            encodedImage.tobytes() + b'\r\n')

@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
        help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
        help="ephemeral port number of the server (1024 to 65535)")

    ap.add_argument("-f", "--frame-count", type=int, default=32,
        help="# of frames used to construct the background model")
    args = vars(ap.parse_args())

    # start a thread that will perform motion detection
    t = threading.Thread(target=detect_motion, args=(
        args["frame_count"],))
    t.daemon = True
    t.start()

    # start the flask app
    #app.run(host=args["ip"], port=args["port"], debug=True, threaded=True, use_reloader=False)

    socketio.run(app, host='localhost', port=8000)
# release the video stream pointer
vs.stop()