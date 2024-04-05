#! /usr/bin/env python

# import server_image_handler

import face_recognition
import time

npImage = face_recognition.load_image_file('/code/test.jpeg')
face_bounding_boxes = face_recognition.face_locations(npImage)
print(face_bounding_boxes, npImage.shape)

encoding = face_recognition.face_encodings(npImage, known_face_locations=face_bounding_boxes, num_jitters=200, model='large')
print(time.time() - s)

enc = (json.dumps(list(encoding[0])))