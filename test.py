#! /usr/bin/env python


import face_recognition
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imagehash
from PIL import Image
import time


file = '/home/lewisbp/snippets/face_id/test_imgs/101_2979.JPG'

data = cv2.imread(file)

s = time.time()
height, width, chan = data.shape

sq_size = np.min(( height, width) ) // 2

left = width // 2 - sq_size
right = width // 2 + sq_size
top = height // 2 - sq_size
bot = height // 2 + sq_size

square_img = data[top:bot, left:right]
# print(square_img.shape)
print(time.time() - s)

small_px = 16

small = cv2.resize(square_img, (small_px, small_px) )
small_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

mean = np.mean(small_gray)
small_bool = np.where(small_gray > mean, 1, 0)

one_d = small_bool.reshape(-1)

byte_array = np.packbits(one_d)

hash_val = ''.join('{:02x}'.format(x) for x in byte_array)

print(hash_val, time.time() - s)

import xxhash

s = time.time()
x = xxhash.xxh64()
x.update(square_img)
print(x.hexdigest())

print(time.time() - s)