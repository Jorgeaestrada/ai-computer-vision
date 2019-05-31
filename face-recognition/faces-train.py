import os
import cv2
import pickle
import numpy as np
from PIL import Image

#OS.WALK FOR IMAGE FINDING

#Absolute path and images path for search of image files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(BASE_DIR, "images")

#HAAR cascade clasifier xml, choose anyone of cascades/data/___.xml directory
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

#Find all the directories of the images contained in the root path
for root, dirs, files in os.walk(img_dir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg"):
			path = os.path.join(root, file)
			
			#Labels from directories
			label = os.path.basename(root).replace(" ", "-").lower()
			#print(label, path)

			#Creating training labels
			if not label in label_ids:
				label_ids[label] = current_id
				current_id += 1

			id_ = label_ids[label]
			print(label_ids)

			#y_labels.append(label) #Some number
			#x_train.append(path) #verify this image, turn into a NUMPY array gray
			pil_image = Image.open(path).convert("L") #Grayscale
			#Resize images for training (avoids the loss of accurate)
			size = (550, 550)
			final_image = pil_image.resize(size, Image.ANTIALIAS)

			#Training data into a numpy array
			img_array = np.array(final_image, "uint8")
			#print("Image xD")
			#print(img_array)

			#Region of interest (ROI) in training data
			faces = face_cascade.detectMultiScale(img_array, scaleFactor=1.5, minNeighbors=5)

			for (x, y, w, h) in faces:
				roi_image = img_array[y:y+h, x:x+w]
				x_train.append(roi_image)
				y_labels.append(id_)

#Using PICKLE to save label ID's
with open("labels.pickle", 'wb') as f:
	pickle.dump(label_ids, f)

#Train the OPENCV PEOPLE RECOGNIZER
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")
#Load names from PICKLE