#! /usr/bin/env python

import face_recognition 
import os
import cv2
from rectangle import Rectangle
import numpy as np
import scipy.misc
import coloredlogs
import random
from tiled_detect import detect_pyramid
import logging
import xmltodict

coloredlogs.install()

params1 = {'upsample': 2, 'height': 600, 'width': 300}
path_to_script = os.path.dirname(os.path.realpath(__file__))
# xmlParamsFile = os.path.join(path_to_script, 'params.xml')

def imageFaceDetect(image_path, parameter_file='parameters.xml'):
    assert isinstance(image_path, str)
    assert os.path.isfile(parameter_file)

    with open(parameter_file, 'r') as fh:
        parameters = xmltodict.parse(fh.read())

    tiled_params = parameters['params']['tiled_detect_params']

    npImage = cv2.imread(image_path)

    faceList = detect_pyramid(npImage, tiled_params)
    print(faceList[0])
    # print(faceList[0].enc_dist(faceList[1]))

    for f in faceList:
        print(f.enc_dist(faceList[0]))

    # return faceList
    # height, width, channels = npImage.shape
    # print(height)
    # print(width)
    # print(np.sqrt(float(height) * width / 4e6))

'''

def processChip(rectangle, photoObject, faceNum, params):

    # Extract the image, path hash, and width and height of the image
    npImage = photoObject.cvImg
    photoPathHash = photoObject.pathHash
    photoWidth = photoObject.photoWidth
    photoHeight = photoObject.photoHeight
    photo_primary_key = photoObject.primaryKey

    # Given the rectangle that defines the face, extract the pixels corresponding
    # to the face and store it in the roi variable.
    centerX = rectangle.centerX
    centerY = rectangle.centerY
    rec_height = rectangle.height
    rec_width = rectangle.width
    top = rectangle.top
    left = rectangle.left
    roi = npImage[top:top+rec_height, left:left+rec_width]

    # The Picasa format for storing faces requires that the center X and Y points
    # of the face ROI as well as the width and height are defined as percentages of
    # the width and height of the image. 
    centerX_scale = float(centerX) / float(photoWidth)
    centerY_scale = float(centerY) / float(photoHeight)
    rec_width_scale = float(rec_width) / float(photoWidth)
    rec_height_scale = float(rec_height) / float(photoHeight)

    # Build up the path name of where the image chip will be stored. 
    stored_prefix = params['chipSaveParams']['storedChipPrefix']
    chip_file_name = '{}_{}_{}.jpg'.format(stored_prefix, str(photoPathHash), faceNum)
    print(chip_file_name)

    # Put the image in the right directory and save it with OpenCV imwrite. 
    chip_save_name = os.path.join(params['chipSaveParams']['chipSaveDir'], chip_file_name)
    # assert not os.isfile(chip_save_name)
    cv.imwrite(chip_save_name , roi)

    # Warn that we are assigning a random vector to the face, for now. 
    logging.warn('Random face vector for stored faces.')
    face_vector = np.random.rand(128)

    # Build up the dict we need to store this in the database. 
    faceDBStruct = dict()
    faceDBStruct['assigned_name'] = None
    faceDBStruct['photo_primary_key'] = photo_primary_key
    faceDBStruct['center_X_scaled'] = centerX_scale
    faceDBStruct['center_Y_scaled'] = centerY_scale
    faceDBStruct['width_scale'] = rec_width_scale
    faceDBStruct['height_scale'] = rec_height_scale
    faceDBStruct['chip_save_name'] = chip_save_name
    faceDBStruct['face_vector'] = face_vector
    faceDBStruct['best_guesses'] = None

    faceNum += 1

    return faceNum, faceDBStruct

def imageFaceDetect(photoObject, faceNum, params):

    photoPrimaryKey = photoObject.primaryKey
    storedFaces = photoObject.faces
    photoWidth = photoObject.photoWidth
    photoHeight = photoObject.photoHeight

    num_detected_faces = np.random.random_integers(1, 6)

    faces = []

    logging.warn("imageFaceDetect is not implemented yet! This is for trial only.")

    for i in range(num_detected_faces):
        face_vec = np.random.rand(128)
        height = photoHeight / 10
        width = photoWidth / 10
        centerX = np.random.random_integers(width, photoWidth - width)
        centerY = np.random.random_integers(height, photoHeight - height)

        roi_rectangle = Rectangle(height, width, centerX=centerX, centerY=centerY)

        faceNum, faceDBStruct = processChip(roi_rectangle, photoObject, faceNum, params)

        logging.warn('Best_guesses and chip_saving are not implemented here.')
        faces.append(faceDBStruct)

    return faces

def classifyFaces(faceVector):
    logging.warn("classifyFaces currently returns a random top-5 name vector")
    possibles = ['Larry', 'Moe', 'Curly', 'Joey', 'Freddy', 'Liz', 'Joan', 'Chelsea', 'Barb', 'Kathy', \
    'Ben', 'Jess', 'Jude', 'Karen', 'Leia', 'Luke', 'Anakin', 'Padme']
    random.shuffle(possibles)
    return possibles[0:5]

def imageFaceDetect(image_path):
    assert isinstance(image_path, str)
    npImage = cv2.imread(image_path)
    height, width, channels = npImage.shape
    print(height)
    print(width)
    print(np.sqrt(float(height) * width / 4e6))

    # print(resizeFactor)
    # print(downsizedArray.shape)
    print("CHECK THE ORIENTATION LOADING")
    maxSize = 5e5 # pixels

    # Either downsize or upsize the image, based on
    # the total size of the image. This helps get 
    # a constant image size. 
    if height * width > maxSize:
        resizeFactor = 1.0 / np.sqrt(float(height) * width / maxSize)
        downsizedArray = cv2.resize(npImage, None, fx=resizeFactor, fy=resizeFactor, interpolation=cv2.INTER_CUBIC)
        print(downsizedArray.shape)
        face_recognition.face_locations(downsizedArray, number_of_times_to_upsample=2, model='cnn') 
    else:
        # Upsize...
        resizeFactor = 1.0 / np.sqrt(float(height) * width / maxSize)
        downsizedArray = scipy.misc.imresize(npImage, resizeFactor, interp='bilinear', mode=None)        
        face_recognition.face_locations(downsizedArray, number_of_times_to_upsample=2, model='cnn') 
    
    areas = []
    loc = face_recognition.face_locations(downsizedArray, model='cnn') 
    locBig = []
    # print(loc)
    for face in loc:
        # print(face)
        top = face[0]
        right = face[1]
        bottom = face[2]
        left = face[3]
        width = right - left
        height = bottom - top


        upsize = 1 / resizeFactor
        centerX = int(left + width / 2.0) * upsize
        centerY = int(top + height / 2.0) * upsize
        upWidth = width * upsize
        upHeight = height * upsize

        upleft = int(centerX - 0.5 * upWidth)
        upright = int(centerX + 0.5* upWidth)
        uptop = int(centerY - 0.5 * upHeight)
        upbottom=int(centerY+ 0.5 * upHeight)

        upLoc = (uptop, upright, upbottom, upleft)
        locBig.append(upLoc)

        rect = Rectangle(height=height, width=width, leftEdge = left, topEdge = top)
        rect.resize( 1 / resizeFactor )
        # rect.expand(0.2, 0.2)
        areas.append(rect)

        assert abs(rect.topLeft.x - left) < 2
        assert abs(rect.topLeft.y - top) < 2

    identity = face_recognition.face_encodings(downsizedArray, known_face_locations=locBig, num_jitters=3)
    # print(identity)
    assert len(identity) == len(locBig)
    assert len(identity) == len(areas)

    return zip(areas, identity)

# # image = face_recognition.load_image_file('/home/lewis/test_imgs/DSC_9857.JPG')
# # # loc = face_recognition.face_locations(image, model='cnn')

# if __name__ == '__main__':
# # # print(loc)
#     from photoLoader import photo
#     file = '/home/lewis/test_imgs/2018-03-23 21.20.20-9.jpeg'

#     # image = face_recognition.load_image_file(file)
#     # areas = imageFaceDetect(image)       
#     # cvImg = cv2.imread(file)
#     # for rect in areas:
#     #     rect.drawOnPhoto(cvImg, colorTriple=(0,255,0))


#     # img = cv2.resize(cvImg, (cvImg.shape[1] / 6, cvImg.shape[0]/ 6))
#     # cv2.imshow('img',img)
#     # cv2.waitKey(0)
#     # exit()

#             #/home/lewis/test_imgs/2018-03-29 09.50.44-8.jpeg
#         # /home/lewis/test_imgs/DSC_9839.JPG -- small face
#         # /home/lewis/test_imgs/DSC_9836.JPG -- overlapping faces
#         # /home/lewis/test_imgs/2018-03-29 09.50.44-3.jpeg - small faces 
#         # /home/lewis/test_imgs/2018-03-26 07.21.44.jpeg - side face


#     files = ['/home/lewis/test_imgs/2018-03-29 09.50.44-8.jpeg',
#         '/home/lewis/test_imgs/DSC_9839.JPG',
#         '/home/lewis/test_imgs/DSC_9836.JPG',
#         '/home/lewis/test_imgs/2018-03-29 09.50.44-3.jpeg',
#         '/home/lewis/test_imgs/2018-03-26 07.21.44.jpeg']

#     for root, dirname, files in os.walk(test_photo_dir, topdown=False):

#         # print(files[0])
#         # print(dirname)
#         for fname in files:
#             this_img = os.path.join(root,fname)
#     # if True:
#             print(this_img)
#             # this_img = fname
#             cvImg = cv2.imread(this_img)
#             ph = photo(this_img, xmlParamsFile)
#             if os.path.exists('/tmp/{}.JPG'.format(ph.hash)):
#                 continue
#             stored_faces = ph.extract_stored_faces()
#             print(len(stored_faces))
#             for j in range(len(stored_faces)):
#                 rect = stored_faces[j]['rectangle']
#                 rect.drawOnPhoto(cvImg, colorTriple=(0,255,0))
                
#             # image = face_recognition.load_image_file(this_img)
#             image = np.asarray(cvImg)
#             face_locations = imageFaceDetect(image)  #face_recognition.face_locations(image, model='cnn')
#             # face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=2, model='hog')
#             print(face_locations)
#             for rect in face_locations:
#                 # top = face[0]
#                 # right = face[1]
#                 # bottom = face[2]
#                 # left = face[3]
#                 # width = right - left
#                 # height = bottom - top
#                 # rect = Rectangle(height=height, width=width, leftEdge = left, topEdge = top)
#                 rect[0].drawOnPhoto(cvImg)

#             # encodings = face_recognition.face_encodings(image)
#             # for i in encodings:
#             #     print(i)


#             img = cv2.resize(cvImg, (cvImg.shape[1] / 6, cvImg.shape[0]/ 6))
#             cv2.imwrite('/tmp/{}.JPG'.format(ph.hash), img)
#             # cv2.imshow('img',img)
#             # cv2.waitKey(0)


if __name__ == "__main__":
    for root, dirs, files in os.walk(test_photo_dir):
        for f in files:
            print(os.path.join(root, f))

'''