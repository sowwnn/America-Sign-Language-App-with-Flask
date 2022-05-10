import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

path = "./HandSignCNN_0.2Ver3.h5"
model = tf.keras.models.load_model(path)

img = np.array([])
cap = cv2.VideoCapture(0)

def to_label(onehot):
  label = [0,1,2,3,4,5,6,7,8,9,'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
  r = [label[i] for i in onehot.argmax(1)]
  return r
  
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

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
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


    ### Landmark:

    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:

        # mp_drawing.draw_landmarks(
        #     image,
        #     hand_landmarks,
        #     mp_hands.HAND_CONNECTIONS,
        #     mp_drawing_styles.get_default_hand_landmarks_style(),
        #     mp_drawing_styles.get_default_hand_connections_style())

        ## Bouding box:
        x_min, y_min, x_max, y_max = get_bbox_coordinates(hand_landmarks,(w,h))
      
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        img = image[x_min:x_max,y_min:y_max].copy()
        if(np.array(img).size != 0):
          img = cv2.resize(img,(224,224))
          img = tf.expand_dims(img, axis=0)
          output = '_'
          t = to_label(model.predict(img))
          if(output!=t):
            output = t
            print(output[0])
            
        # cv2.imshow("Frame", image)
    # Flip the image horizontally for a selfie-view display.

    cv2.imshow('MediaPipe Hands',image)

    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()






