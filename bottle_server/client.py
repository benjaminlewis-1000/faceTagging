
# This will read the *entire* image file as a data_string and send it over flask.

import os
import sys

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(PARENT_DIR)
sys.path.append(PARENT_DIR)

# import flask
import xmltodict
import base64
import requests
import json
import numpy as np
import cv2
import hashlib
import face_extraction
# from rectangle import Point, Rectangle

with open('my_pic.jpg', 'rb') as imageFile:
    data_str = base64.b64encode(imageFile.read())

# Checksum on binary data
checksum = hashlib.md5(data_str)
checksum = checksum.hexdigest()
# print(checksum)

# Convert bytes to string
data_str = data_str.decode('utf-8')
# data_str += '3'

addr = 'http://localhost:5000'
test_url = addr + '/api/test_fullfile'

# prepare headers for http request
content_type = 'text'
headers = {'content-type': content_type}

payload={
    'base64_file': data_str,
    'checksum': checksum
}

# print(data_str[0:100])
payload = json.dumps(payload)

# send http request with image and receive response
try:
    response = requests.post(test_url, data=payload, headers=headers)
    # decode response
    retval = json.loads(response.text)
except requests.exceptions.ConnectionError:
    print("Oh well")
    exit()




def decode_object(o):
 
    if '__Point__' in o:
 
        a = face_extraction.Point(0, 0)
 
        a.__dict__.update(o['__Point__'])
 
        return a
 
    elif '__Rectangle__' in o:
 
        a = face_extraction.Rectangle(10, 10, centerX = 5, centerY = 5)
 
        a.__dict__.update(o['__Rectangle__'])
 
        return a

    elif '__FaceRect__' in o:

        a = face_extraction.FaceRect(None, None, None, None)
        a.__dict__.update(o['__FaceRect__'])
        a.encoding = np.asarray(a.encoding)
        a.image = np.asarray(a.image)

        return a
 
    elif '__datetime__' in o:
 
        return datetime.strptime(o['__datetime__'], '%Y-%m-%dT%H:%M:%S')        
 
    return o

if not retval['success']:
    print(retval['message'])
else:
    xmp_data = json.loads(retval['xmp_data'], object_hook = decode_object) 
    for idx in range(len(xmp_data)):
        # print(xmp_data[idx]['__FaceRect__'].keys())# .encoding = np.asarray(xmp_data[idx].encoding)
        print(type(xmp_data[idx].encoding))
        print(type(xmp_data[idx].image))
        print(xmp_data[idx].image.shape)
        print(xmp_data[idx])


    parameter_file=os.path.join(PARENT_DIR, 'parameters.xml')
    with open(parameter_file, 'r') as fh:
        parameters = xmltodict.parse(fh.read())

    matched_faces, _, _ = face_extraction.extract_faces_from_image('my_pic.jpg', parameters)

    print(matched_faces)
    assert matched_faces == xmp_data