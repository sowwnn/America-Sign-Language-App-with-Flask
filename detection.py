import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

path = "../ASL_ResNet50.h5"
model = tf.keras.models.load_model(path)

img = np.array([])
cap = cv2.VideoCapture(0)

def to_label(onehot):
  label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','del','nothing','space']
  r = [label[i] for i in onehot.argmax(1)]
  return r
with mp_hands.Hands(
    max_num_hands= 1,
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

    if results.hand_rects:
      for hand_rect in results.hand_rects:

        # mp_drawing.draw_landmarks(
        #     image,
        #     hand_landmarks,
        #     mp_hands.HAND_CONNECTIONS,
        #     mp_drawing_styles.get_default_hand_landmarks_style(),
        #     mp_drawing_styles.get_default_hand_connections_style())

        ## Bouding box:
        x_cen = hand_rect.x_center*h
        y_cen = hand_rect.y_center*w
        scale_w = hand_rect.width * h
        scale_h = hand_rect.height * w
      
        # x_min, y_min, x_max, y_max = 
        # print(hand_rect)
      
        cv2.rectangle(image, (int(x_cen - scale_w/2),int( y_cen - scale_h/2)),(int(x_cen + scale_w/2), int(y_cen + scale_h/2)), (0, 255, 0), 2)
        
        img = image[int(y_cen - scale_w/2): int(y_cen + scale_w/2), int(x_cen - scale_h/2): int( x_cen + scale_h/2)]
        if(np.array(img).size != 0):
          cv2.imwrite('crop.jpeg', img)
          img = cv2.resize(img,(224,224))
          img = img/255
          img = tf.expand_dims(img, axis=0)
          t = to_label(model.predict(img))[0]
          print(t)
        
    # Flip the image horizontally for a selfie-view display.
    cv2.rectangle(image, (0,0), (h,40), (255,0,0),-1)
    cv2.putText(image, t, (int(h/2-4),40), fontFace=cv2.FONT_HERSHEY_SIMPLEX , fontScale=1, color=(255,255,255),lineType=cv2.LINE_AA)

    cv2.imshow('MediaPipe Hands',image)

    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()






