#! /usr/bin/env python

import numpy as np
import face_recognition
import torch
import torchvision.transforms as transforms

def encode_image_with_bounding(npImage, bounding_box, pad_pct = 0.5):
    assert type(npImage) == np.ndarray
    assert type(pad_pct) in [int, float]
    assert type(bounding_box) == list
    assert len(bounding_box) == 1
    assert len(bounding_box[0]) == 4
    assert type(bounding_box[0]) in [tuple, list], f"bounding_box[0] is {type(bounding_box[0])}"

    half_pad = pad_pct / 2

    # Then you pad out the image to match dlib.
    fl = bounding_box[0]
    height = fl[2] - fl[0]
    pad_height = int(half_pad * height)
    width = fl[3] - fl[1]
    pad_width = int(half_pad * width)

    pad_top = np.min((pad_height, fl[0]))
    pad_left = np.min((pad_width, fl[1]))
    pad_bot = pad_top + height
    pad_right = pad_left + width

    bounding_box_adj = [(pad_top, pad_left, pad_bot, pad_right)]

    # bounding_box_adj = [0, 0, fl[2] - fl[0], fl[3] - fl[1]]
    npCrop = npImage[np.max((fl[0] - pad_height, 0)):fl[2] + pad_height, np.max((fl[1] - pad_width, 0)):fl[3] + pad_width]

    cropTensor = torch.Tensor(npCrop)
    cropTensor = cropTensor.permute(2, 0 , 1)
    out = transforms.functional.autocontrast(cropTensor)
    out = out.permute(1, 2, 0)
    npCrop = out.numpy()

    npCrop = npCrop * 255
    npCrop = npCrop.astype(np.uint8)

    encoding = face_recognition.face_encodings(npCrop, known_face_locations=bounding_box_adj, num_jitters=200, model='large')

    return encoding