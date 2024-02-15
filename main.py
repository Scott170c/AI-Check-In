# Reminder to import your own teachable machine model
from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
import pyrebase
import datetime

# config for firebase
config = {
  "apiKey": "insert-here",
  "authDomain": "insert-here",
  "projectId": "insert-here",
  "databaseURL": "insert-here",
  "storageBucket": "insert-here"
}

firebase = pyrebase.initialize_app(config)

auth = firebase.auth()
user = auth.sign_in_with_email_and_password("email", "email password")
user = auth.refresh(user["refreshToken"])

token = user["idToken"]

db = firebase.database()

checked_in = []
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    imageR = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window

    # Make the image a numpy array and reshape it to the models input shape.
    imageR = np.asarray(imageR, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    imageR = (imageR / 127.5) - 1

    # Predicts the model
    prediction = model.predict(imageR)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    # Display confidence, who is being checked, and how many people are already checked
    image = cv2.putText(image, str(np.round(confidence_score * 100))[:-2]+" %", (25, 50), cv2.FONT_HERSHEY_SIMPLEX , 1.5, (0, 0, 0), 1, cv2.LINE_AA) 
    image = cv2.putText(image, class_name[2:] + "   checked: " + str(len(checked_in)), (25, 90), cv2.FONT_HERSHEY_SIMPLEX , 1.5, (0, 0, 0), 1, cv2.LINE_AA) 

    # If not checked check in
    if class_name[2:] not in checked_in and "Nothing" not in class_name[2:] and int(str(np.round(confidence_score * 100))[:-2]) > 65:
        checked_in.append(class_name[2:])
        # add to database
        db.child("members").child(class_name[2:(len(class_name)-1)]).set({"role":"member", "date-checked":str(datetime.datetime.now()).replace(":", "-")}, token)
    cv2.imshow("Club Check-In", image)

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()