from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

path = "./HandSignCNN_0.2Ver3.h5"
model = tf.keras.models.load_model(path)
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

def get_bbox_coordinates(handLadmark, image_shape):
    """ 
    Get bounding box coordinates for a hand landmark.
    Args:
        handLadmark: A HandLandmark object.
        image_shape: A tuple of the form (height, width).
    Returns:
        A tuple of the form (xmin, ymin, xmax, ymax).
    """
    all_x, all_y = [], [] # store all x and y points in list
    for hnd in mp_hands.HandLandmark:
        all_x.append(int(handLadmark.landmark[hnd].x * image_shape[1]) ) # multiply x by image width
        all_y.append(int(handLadmark.landmark[hnd].y * image_shape[0]) ) # multiply y by image height
    
    x_min, y_min, x_max, y_max = min(all_x), min(all_y), max(all_x), max(all_y)
    kw = int(0.1*( max(all_x) - min(all_x)))
    kh = int(0.1*( max(all_y) -  min(all_y)))
   
    return x_min-kw, y_min-kh, x_max+kw, y_max+kh # return as (xmin, ymin, xmax, ymax)


def gen(): 
    cap = cv2.VideoCapture(0)
 
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera image.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue

                    # To improve performance, optionally mark the image as not writeable to
                    # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                w,h,c = image.shape
                results = hands.process(image)

                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:

                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())

                        ## Bouding box:
                        x_min, y_min, x_max, y_max = get_bbox_coordinates(hand_landmarks,(w,h))
                    

                        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                image = cv2.flip(image, 1)
                img_in = image
                cv2.imwrite('demo.jpg', image)
                yield (b'--image\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + open('demo.jpg', 'rb').read() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=image')

@app.route('/predict')
def predict():
    return Response(model.predict(img_in))

if __name__ == '__main__':
    app.run(debug=True)
