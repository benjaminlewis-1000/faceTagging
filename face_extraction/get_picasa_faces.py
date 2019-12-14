#! /usr/bin/env python

# Face tagging test
import os
import sys

# PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(PARENT_DIR)
# sys.path.append(PARENT_DIR)

from .rectangle import Rectangle
from PIL import Image
from time import sleep
import binascii
import collections
import cv2
import face_recognition
import io
import matplotlib.pyplot as plt
import re
import xml
import xmltodict

# This function will extract the XMP Bag Tag from the header of 
# a JPG file. This is where the now-defunct Picasa program, by 
# Google, stored face information. This function supports both 
# paths to files as well as BytesIO in-memory files. 
def Get_XMP_Faces(file, test=False):

    file_data = None

    # Read the file's data as a text string, stored in the 
    # variable file_data
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
        raise ValueError('Input to Get_XMP_Faces must be a file path or a BytesIO object.')

    if file_data is None:
        return False, None

    image = face_recognition.load_image_file(file)
    # image = cv2.imread(file)
 
    # using the string of the file header,
    # locate the starting XMP XML Bag tag. It begins with 
    # the XML tag '<rdf:Bag'. 

    xmp_start = 0
    xmp_end = 0
    bag_tags = []
    file_data = repr(file_data)
    while xmp_start > -1:
        file_data = file_data[xmp_end:]
        xmp_start = (file_data).find('<rdf:Bag')
        # also try and locate the ending XMP XML Bag tag
        xmp_end = (file_data).find('</rdf:Bag') + 10 # 10 being the length of the tag. 
        # Cut out the contents of the tag and append to the 
        # array of bag tags. 
        xmp_data = (file_data)[xmp_start:xmp_end ]
        bag_tags.append(xmp_data)

    # Now we process the XML. 
    persons = []
    for bag in bag_tags:
        try:
            bag_data = xmltodict.parse(bag)
            if 'rdf:Bag' in bag_data.keys() and 'rdf:li' in bag_data['rdf:Bag'].keys():
                # These are the sub-tags that should be there. 
                bag_data = bag_data['rdf:Bag']['rdf:li']

                # This is from looking through the tags and figuring
                # out which fields are which. This function
                # returns the tagged name, the x and y location
                # of the top-left point, and the width and 
                # height of the face location. 
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

                # Two options, both have the person data. 
                if isinstance(bag_data, collections.OrderedDict):
                    persons.append(get_person_data(bag_data))
                elif len(bag_data) and isinstance(bag_data[0], collections.OrderedDict):
                    for person_num in range(len(bag_data)):
                        persons.append(get_person_data(bag_data[person_num]))

        except xml.parsers.expat.ExpatError:
            pass

    # X and Y are locations in the middle of the face. 
    img_height, img_width, _ = image.shape

    # Reverse parsing. We process the list of persons
    # *again* to turn the tags into Rectangle objects
    # and put that in the list that will be returned.
    # The intermediate data is then popped from the 
    # dictionary. 
    for p_num in range(len(persons)-1, -1, -1):
        left = float(persons[p_num]['Area_x'])
        top = float(persons[p_num]['Area_y']) 
        height = float(persons[p_num]['Area_h']) 
        width = float(persons[p_num]['Area_w'])         

        right = left + width / 2
        bottom = top + height / 2
        left = left - width / 2
        top = top - height / 2

        left = int(left * img_width)
        right = int(right * img_width)
        top = int(top * img_height)
        bottom = int(bottom * img_height)

        height = int(height * img_height)
        width = int(width * img_width)

        bounding_rectangle = Rectangle(height, width, leftEdge=left, topEdge=top)
        persons[p_num]['bounding_rectangle'] = bounding_rectangle

        persons[p_num].pop('Area_x')
        persons[p_num].pop('Area_y')
        persons[p_num].pop('Area_h')
        persons[p_num].pop('Area_w')

        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 5)

    if test:
        plt.imshow(image)
        plt.show()

 
    # if nothing is found, return None
    if len(bag_tags) == 0:
        return True, None
 
    #  if we found something, return tag information
    return True, persons

if __name__ == "__main__":
    print(Get_XMP_Faces(os.path.join('/home/benjamin/gitRepos/test_imgs', '1.JPG'), True))