import os
import numpy as np
import pickle
import cv2

# MAX PERMITED RESOLUTION FOR HP 14CM-00XXX "640 x 480"
width, height = 1280, 720

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASCADE_CLASSIFIER_DIR = os.path.join(
    BASE_DIR, "data/haarcascades/haarcascade_frontalface_alt2.xml")

# HAAR cascade clasifier xml, choose anyone of cascades/data/___.xml directory
face_cascade = cv2.CascadeClassifier(CASCADE_CLASSIFIER_DIR)
# Implements recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {}

with open("labels.pickle", 'rb') as f:
    original_labels = pickle.load(f)
    # Reverse the original labels
    labels = {v: k for k, v in original_labels.items()}

cap = cv2.VideoCapture(0)


def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)


change_res(width, height)

while (True):
    ret, frame = cap.read()

    # using HAAR to detect faces on the frame
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# C++: void CascadeClassifier::detectMultiScale(const Mat& image, vector<Rect>& objects,
# double scaleFactor=1.1, int minNeighbors=3, int flags=0, Size minSize=Size(), Size maxSize=Size())

# Parameters:
# cascade - Haar classifier cascade (OpenCV 1.x API only). It can be loaded from XML or
# YAML file using Load(). When the cascade is not needed anymore, release it using cvReleaseHaarClassifierCascade(&cascade).
# image - Matrix of the type CV_8U containing an image where objects are detected.
# objects - Vector of rectangles where each rectangle contains the detected object.
# scaleFactor - Parameter specifying how much the image size is reduced at each image scale.
# minNeighbors - Parameter specifying how many neighbors each candidate rectangle should have to retain it.
# flags - Parameter with the same meaning for an old cascade as in the function cvHaarDetectObjects. It is not used for a new cascade.
# minSize - Minimum possible object size. Objects smaller than that are ignored.
# maxSize - Maximum possible object size. Objects larger than that are ignored.

# USING THE FACE CLASSIFIER
    faces = face_cascade.detectMultiScale(
        grayFrame, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        print(x, y, w, h)

        roi_image = grayFrame[y:y+h, x:x+w]
        roi_image_color = frame[y:y+h, x:x+w]

# Implements PEOPLE recognizer
        id_, conf = recognizer.predict(roi_image)
        if conf >= 45:  # sand conf <= 85:
            print(id_)
            print(labels[id_])
            # Put text to rectangle region of interest ROI

            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1,
                        color, stroke, cv2.LINE_AA)

# Save an image of the RECOGNIZED FACE!!
        #img_item = "mi-image.png"
        #cv2.imwrite(img_item, roi_image)

# Drawing a rectangle for region of interest ROI
        color = (255, 0, 0)  # BGR 0 - 255
        stroke = 2

        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
