#! /usr/bin/env python

# Face tagging test

import os
import face_recognition
from PIL import Image
from time import sleep
import cv2
# import pyexiv2
# from pyPicasaFaceXMP import picasaXMPFaceReader as pfr
from rectangle import Rectangle as rec
import re
import binascii
import io
import xml
import xmltodict
import collections
from rectangle import Rectangle
import matplotlib.pyplot as plt

# def toHex(s):
#     lst = []
#     for ch in s:
#         hv = hex(ord(ch)).replace('0x', '')
#         if len(hv) == 1:
#             hv = '0'+hv
#         lst.append(hv)
    
#     return reduce(lambda x,y:x+y, lst)



# This function will extract the XMP Bag Tag and Picasa faces
def Get_XMP_Faces(file, test=False):

    file_data = None

    if isinstance(file, str):
        try:
            # attempt to open the file as binary
            file_as_binary = open(file,'rb')
            # if it opened try to read the file
            file_data = file_as_binary.read()
            # close the file afterward done
            file_as_binary.close()
        except:
            # if we sell the open the file abort
            return False, None

    elif isinstance(file, io.BytesIO):
        # getvalue, not read
        file_data = file.getvalue()

    else:
        return False, None

    if file_data is None:
        return False, None


    image = face_recognition.load_image_file(file)
 
    # using the file data, attempt to locate the starting XMP XML Bag tag
    # print(type(file_data))

    # start_string = '<rdf:Bag'

    xmp_start = 0
    xmp_end = 0
    bag_tags = []
    file_data = repr(file_data)
    while xmp_start > -1:
        file_data = file_data[xmp_end:]
        xmp_start = (file_data).find('<rdf:Bag')
        # also try and locate the ending XMP XML Bag tag
        xmp_end = (file_data).find('</rdf:Bag') + 10
        xmp_data = (file_data)[xmp_start:xmp_end ]
        bag_tags.append(xmp_data)


    persons = []
    for bag in bag_tags:
        try:
            bag_data = xmltodict.parse(bag)
            if 'rdf:Bag' in bag_data.keys() and 'rdf:li' in bag_data['rdf:Bag'].keys():
                bag_data = bag_data['rdf:Bag']['rdf:li']
                # print((bag_data))

                if isinstance(bag_data, collections.OrderedDict):
                    persons.append(get_person_data(bag_data))
                elif len(bag_data) and isinstance(bag_data[0], collections.OrderedDict):
                    for person_num in range(len(bag_data)):
                        persons.append(get_person_data(bag_data[person_num]))

                def get_person_data(person_dict):
                    person_data = person_dict['rdf:Description']
                    assert '@mwg-rs:Name' in person_data.keys()
                    assert '@mwg-rs:Type' in person_data.keys()
                    assert 'mwg-rs:Area' in person_data.keys()
                    name = person_data['@mwg-rs:Name']
                    assert person_data['@mwg-rs:Type'] == 'Face'
                    area = person_data['mwg-rs:Area']
                    area_x = area['@stArea:x']
                    area_y = area['@stArea:y']
                    area_w = area['@stArea:w']
                    area_h = area['@stArea:h']
                    return {'Name': name, 'Area_x': area_x, 'Area_y': area_y, 'Area_w': area_w, 'Area_h': area_h}
        except xml.parsers.expat.ExpatError:
            pass

    # X and Y are locations in the middle of the face. 
    # image = face_recognition.load_image_file(file)
    img_height, img_width, _ = image.shape

    for p_num in range(len(persons)-1, -1, -1):
        left = float(persons[p_num]['Area_x'])
        top = float(persons[p_num]['Area_y']) 
        height = float(persons[p_num]['Area_h']) 
        width = float(persons[p_num]['Area_w'])         

        right = left + width / 2
        bottom = top + height / 2
        left = left - width / 2
        top = top - height / 2

        # print(left, right, top, bottom)

        left = int(left * img_width)
        right = int(right * img_width)
        top = int(top * img_height)
        bottom = int(bottom * img_height)

        height = int(height * img_height)
        width = int(width * img_width)

        # persons[p_num]['left'] = left
        # persons[p_num]['right'] = right
        # persons[p_num]['top'] = top
        # persons[p_num]['bottom'] = bottom
        # persons[p_num]['height'] = height
        # persons[p_num]['width'] = width

        bounding_rectangle = Rectangle(height, width, leftEdge=left, topEdge=top)
        persons[p_num]['bounding_rectangle'] = bounding_rectangle

        persons[p_num].pop('Area_x')
        persons[p_num].pop('Area_y')
        persons[p_num].pop('Area_h')
        persons[p_num].pop('Area_w')

        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 5)

 
    # if nothing is found abort
    if len(bag_tags) == 0:
    # if xmp_bag == "":
        return False, None
 
    #  if we found something, return tag information
    return True, persons


    # success, persons = process_image(file_data, image)
    # return success, persons

# def Get_XMP_Faces_byteIO(file_string, file_image, bytesi, test=False):

#     print(bytesi.getvalue()[:5000])
#     print(file_string[:5000])

#     file_data = bytesi.getvalue()

#     # if the file is empty abort
#     if file_data is None:
#         return False, None

#     success, persons = process_image(file_data, file_image)
#     return success, persons

'''

train_dir = '/home/benjamin/Desktop/photos_for_test/train'
test_dir = '/home/benjamin/Desktop/photos_for_test/test'

for root, dirs, files in os.walk(train_dir):
    for name in files:
        fullPath = os.path.join(root, name)
        print fullPath

        metadata = pyexiv2.ImageMetadata(fullPath)
        metadata.read()
        picasaMetadata = pfr.Imagedata(fullPath)
        xmpFaces = pfr.XMPFace(picasaMetadata).getFaces()
        # print xmpFaces

        # # Load as a numpy array
        image = face_recognition.load_image_file(fullPath)
        ocvImage = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # ocvImage = cv2.resize(ocvImage, (800, 600) )
        for faces in xmpFaces:
            personName = faces[4]

            left = faces[0]
            top = faces[1]
            width = faces[2]
            height = faces[3]
            right = left + width
            bottom = top + height
            locRect = rec(left, top, right, bottom)

            cv2.rectangle(ocvImage, (left, top), (right, bottom), (255, 0, 0), 5)
        # print ocvImage.shape

        # print image.shape
        ocvImage = cv2.resize(ocvImage, ( ocvImage.shape[1] / 10, ocvImage.shape[0] / 10 ) )
        print ocvImage.shape
        face_locations = face_recognition.face_locations(ocvImage, number_of_times_to_upsample=3,  model='cnn')
        # print face_locations

        for location in face_locations:
            top, right, bottom, left = location
            locRect = rec(left, top, right, bottom)
            face_image = image[top:bottom, left:right]
            cv2.rectangle(ocvImage, (left, top), (right, bottom), (0, 255, 0), 5)
            # pil_image = Image.fromarray(face_image)
            # pil_image.show()



        # Convert to OpenCV colors, get a resized window, and show image. 
        cv2.namedWindow('Resized Window', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Resized Window', 800, 600)
        cv2.imshow('Resized Window', ocvImage)
        cv2.waitKey(0)
        # sleep(1)
        # cv2.destroyAllWindows()

            # sleep(0.3)
            # pil_image.close()

'''