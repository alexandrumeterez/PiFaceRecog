from imutils import paths
import face_recognition
import pickle
import cv2
import os
import numpy as np
IMAGES_FOLDER = "./dataset"

print("Loading images")
image_paths = list(paths.list_images(IMAGES_FOLDER))

known_encodings = []
known_names = []

for (i, image_path) in enumerate(image_paths):
    print("Currently at image: {0}".format(i))
    name = image_path.split(os.path.sep)[-2]
    rgb = face_recognition.load_image_file(image_path)
    boxes = face_recognition.face_locations(np.array(rgb), model = "hog")
    
    encodings = face_recognition.face_encodings(rgb, boxes)
    for encoding in encodings:
        known_encodings.append(encoding)
        known_names.append(name)
print("Serializing encodings")
with open("encodings.pkl", "wb") as f:
    data = {"encodings": known_encodings, "names": known_names}
    f.write(pickle.dumps(data))
    