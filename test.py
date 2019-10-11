#! /usr/bin/env python


import face_recognition
import cv2
import numpy as np
import matplotlib.pyplot as plt

file = '/home/lewisbp/snippets/face_id/test_imgs/101_2979.JPG'

npImage = face_recognition.load_image_file(file)

h, w, c = npImage.shape
resize_ratio = 5
num_upsamples = 2

resized_chip = cv2.resize(npImage, \
    ( int( w / resize_ratio ), \
      int( h / resize_ratio ) ) )

# top, right, bottom, left
face_locations = face_recognition.face_locations(resized_chip, \
                    number_of_times_to_upsample=num_upsamples,  model='cnn')

for t in face_locations:
    top, right, bottom, left = t

cv2.rectangle(resized_chip, (left, top), (right, bottom), (255, 50, 50), 4)
cv2.rectangle(npImage, (left * resize_ratio, top * resize_ratio), (right * resize_ratio, bottom * resize_ratio), (255, 50, 50), 4)

f_prime = [(top * resize_ratio, right * resize_ratio, bottom * resize_ratio, left * resize_ratio)]

n_jit = 100
identity = face_recognition.face_encodings(resized_chip, known_face_locations=face_locations, num_jitters=n_jit)
identity_prime = face_recognition.face_encodings(npImage, known_face_locations=f_prime, num_jitters=n_jit)

for i in range(-1, 5):
    f_prime_move  = [(top * resize_ratio + i, right * resize_ratio + i, bottom * resize_ratio + i, left * resize_ratio + i)]
    identity_prime_move = face_recognition.face_encodings(npImage, known_face_locations=f_prime_move, num_jitters=n_jit)
    print(np.linalg.norm(identity_prime_move[0] - identity_prime[0]))

plt.imshow(resized_chip)
plt.show(block=False)
plt.figure()
plt.imshow(npImage)
plt.show()



# chip_part = cv_image[top_edge:bottom_edge, left_edge:right_edge]

# Then resize it to fit in the GPU memory, based 
# on the parameters passed to the function.
# height_chip = chip_part.shape[0]
# width_chip = chip_part.shape[1]
# pixels_here = height_chip * width_chip
# resize_ratio = np.sqrt(float(pixels_here) / max_pixels_per_chip)
