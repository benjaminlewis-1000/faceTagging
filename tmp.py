#! /usr/bin/env python

import face_recognition
import scipy.misc
import matplotlib.pyplot as plt
import time

img = '/home/lewisbp/snippets/face_id/test_imgs/101_2743.JPG'

img = face_recognition.load_image_file(img)


dwn = scipy.misc.imresize(img, 1/3)

# plt.imshow(dwn)
# plt.show()

s=time.time()
loc = face_recognition.face_locations(dwn, number_of_times_to_upsample=2, model='cnn')
# print(loc)
print(time.time() - s)
identity = face_recognition.face_encodings(dwn, known_face_locations=loc, num_jitters=3)
print(time.time() - s)
# print(identity)