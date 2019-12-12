
# This will read the *entire* image file as a data_string and send it over flask.

import os
import sys

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# print(PARENT_DIR)
sys.path.append(PARENT_DIR)
sys.path.append(THIS_DIR)

# import flask
import xmltodict
import base64
import requests
import json
import numpy as np
import cv2
import hashlib
import face_extraction
import matplotlib.pyplot as plt
from client_ip_discover import find_external_server
import logging
import coloredlogs

# from rectangle import Point, Rectangle
# reload(logging)
# logging.basicConfig(level=logging.DEBUG)
# logging.debug('Helo!')
logger = logging.getLogger('my.logger')
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)
coloredlogs.install()
logger.debug("Hello!")
# logger.basicConfig(level=logging.DEBUG)
# logging.setLevel(logging.DEBUG)

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
        a.square_face = np.asarray(a.square_face)

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

def face_extract_client(filename):
    ext_ip = find_external_server()

    with open(os.path.join(PARENT_DIR, 'parameters.xml')) as p:
        config = xmltodict.parse(p.read())
    port_image_handle = int(config['params']['ports']['server_port_image_handle'])
    port_ip_disc = int(config['params']['ports']['server_port_ip_disc'])
    connect_timeout = int(config['params']['timeout']['connect_timeout'])
    read_timeout = int(config['params']['timeout']['read_timeout'])

    process_local = True
    if not ext_ip:
        logger.error('GPU server not available to extract faces from {}'.format(filename))
        process_local = True
    else:
        payload, headers = image_for_network(filename)
        addr = f'http://{ext_ip}:{port_image_handle}'
        alive_url = addr + '/api/alive'
        logger.debug("Alive url is {}".format(alive_url))
        try:
            # Since 'alive' expects no payload, it will throw 
            # an error if it receives one. So this is proper.
            response = requests.get(alive_url, timeout=(connect_timeout, read_timeout))
            # decode response
            retval = json.loads(response.text)
            if not retval['server_supports_cuda']:
                logger.error("Server does not support CUDA: processing locally.")
            if response.status_code == 200:
                process_local = not retval['server_supports_cuda'] 

        except requests.exceptions.ConnectionError as ce:
            print(ce)
            logger.error("Connection error for API -- will process locally")
            process_local = True

    if process_local:
        logger.warning("Processing locally!")
        matched_faces, _, _, elapsed_time = face_extraction.extract_faces_from_image(filename, config)
    else:
        addr = f'http://{ext_ip}:{port_image_handle}'
        face_extract_url = addr + '/api/face_extract'
        logger.debug("Using GPU, address is {}".format(face_extract_url))

        # send http request with image and receive response
        try:
            response = requests.post(face_extract_url, data=payload, headers=headers, timeout=(connect_timeout, read_timeout))
            # decode response
            try:
                retval = json.loads(response.text)
            except json.decoder.JSONDecodeError as jde:
                if '500 Internal Server Error' in response.text:
                    logger.critical("Your server face extract code is broken! It broke on filename {}".format(filename))
                    raise IOError('Your server face extract code is broken. Fix it! It broke on filename {}'.format(filename))
                else:
                    print("Text response is : {}".format(response.text))
                    print(jde)
                    # raise(jde)
                    return
            # retval = json.loads(response.text)
            elapsed_time = retval['elapsed_time']

            if not retval['success']:
                print("No success: ", retval['message'])
            else:
                matched_faces = json.loads(retval['xmp_data'], object_hook = decode_object)

            logger.debug('GPU server **was** used to extract faces from {}'.format(filename))

        except requests.exceptions.ConnectionError:
            logger.error('GPU server could not connect in face extraction.')
            matched_faces, _, _, elapsed_time = face_extraction.extract_faces_from_image(filename, config)
        except requests.exceptions.ReadTimeout:
            logger.error('GPU server timed out when face extracting {}'.format(filename))
            matched_faces, _, _, elapsed_time = face_extraction.extract_faces_from_image(filename, config)

    logger.debug('Elapsed time to extract faces from {} was {}'.format(filename, elapsed_time))
    return matched_faces

if __name__ == "__main__":
    mf = face_extract_client('my_pic.jpg')
    logger.debug(mf)

    # for m in mf:
    #     print(m.square_face.shape)
    #     plt.imshow(m.square_face)
    #     plt.show()

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
