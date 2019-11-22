
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
from client_ip_discover import find_external_server
# from rectangle import Point, Rectangle


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

def image_for_network(filename):

    with open(filename, 'rb') as imageFile:
        data_str = base64.b64encode(imageFile.read())

    # Checksum on binary data
    checksum = hashlib.md5(data_str)
    checksum = checksum.hexdigest()

    data_str = data_str.decode('utf-8')

    # prepare headers for http request
    content_type = 'text'
    headers = {'content-type': content_type}

    payload={
        'base64_file': data_str,
        'checksum': checksum
    }
    
    payload = json.dumps(payload)

    return payload, headers

def main(filename):
    ext_ip = find_external_server()

    with open(os.path.join(PARENT_DIR, 'parameters.xml')) as p:
        config = xmltodict.parse(p.read())
    port_flask = int(config['params']['server']['port_flask'])
    port_ip_disc = int(config['params']['server']['port_ip_disc'])

    process_local = True
    if not ext_ip:
        process_local = True
    else:
        payload, headers = image_for_network(filename)
        addr = f'http://{ext_ip}:{port_flask}'
        alive_url = addr + '/api/alive'
        try:
            response = requests.post(alive_url, data=payload, headers=headers)
            # decode response
            retval = json.loads(response.text)
            if response.status_code == 200:
                process_local = False

        except requests.exceptions.ConnectionError:
            process_local = True

    if process_local:
        matched_faces, _, _ = face_extraction.extract_faces_from_image(filename, config)
    else:
        addr = f'http://{ext_ip}:{port_flask}'
        face_extract_url = addr + '/api/face_extract'

        # send http request with image and receive response
        try:
            response = requests.post(face_extract_url, data=payload, headers=headers)
            # decode response
            retval = json.loads(response.text)


            if not retval['success']:
                print("No success: ", retval['message'])
            else:
                matched_faces = json.loads(retval['xmp_data'], object_hook = decode_object) 

        except requests.exceptions.ConnectionError:
            matched_faces, _, _ = face_extraction.extract_faces_from_image(filename, config)

    return matched_faces

mf = main('my_pic.jpg')
# print(mf)

test = False
if test:
    with open(os.path.join(PARENT_DIR, 'parameters.xml')) as p:
        config = xmltodict.parse(p.read())
    mf2, _, _ = face_extraction.extract_faces_from_image('my_pic.jpg', config)

    for f in range(len(mf)):
        f1 = mf[f]
        f2 = mf[f]
        # print(f1)
        assert np.mean(f1.encoding - f2.encoding) == 0
        assert f1.name == f2.name
        assert f1.rectangle == f2.rectangle
    print("Test done!")





#     parameter_file=os.path.join(PARENT_DIR, 'parameters.xml')
#     with open(parameter_file, 'r') as fh:
#         parameters = xmltodict.parse(fh.read())

#     matched_faces, _, _ = face_extraction.extract_faces_from_image('my_pic.jpg', parameters)

#     print(matched_faces)
#     assert matched_faces == xmp_data