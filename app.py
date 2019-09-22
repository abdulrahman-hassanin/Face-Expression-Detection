from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input
import cv2
import numpy as np

# Load Model --------------------------------------------------------
model = load_model('facial_expressions_l2_0.001.h5')



# List of labels
emotion_table = {'0' : 'anger',     # 4953
                 '1' : 'happy',     # 8989
                 '2' : 'sad',       # 6077
                 '3' : 'neutral'}   # 6198

def prediction(gray_image, crop):
  gray_image = cv2.cvtColor(crop[0], cv2.COLOR_BGR2GRAY)
  img = cv2.resize(gray_image,(48 ,48))
  x = np.array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  x = np.reshape(x, (1, 48, 48, 1))
  # print('Input image shape:', x.shape)

  # Prediction
  pred = model.predict(x)
  # print(pred)
  p = np.amax(pred)
  for i in range(len(pred[0])):
    if(p == pred[0][i]):
        i = str(i)
        return emotion_table[i]


# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam. 
cap = cv2.VideoCapture(0)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    crop_img = []
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        crop_img.append(img[y:y+h, x:x+w])
        # Prediction 
        pred = prediction(gray, crop_img)
        cv2.putText(img, pred, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)


    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()



