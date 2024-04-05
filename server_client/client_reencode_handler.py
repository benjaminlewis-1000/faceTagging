#! /usr/bin/env python

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
import matplotlib.pyplot as plt
import client_ip_discover # import find_external_server
import logging
from PIL import Image, ExifTags


def image_and_box_for_network(filename, bounding_box):

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
        'checksum': checksum,
        'bounding_box': bounding_box
    }
    
    payload = json.dumps(payload)

    return payload, headers


def face_encoding_client(filename, bounding_box, server_ip_finder, logger=None):

#    server_there = server_ip_finder.check_ip()
#    if server_there:
#        ext_ip = server_ip_finder.server_ip
#    else:
#        server_ip_finder.find_external_server()
#        ext_ip = None

    ext_ip = server_ip_finder.server_ip
    with open(os.path.join(PARENT_DIR, 'parameters.xml')) as p:
        config = xmltodict.parse(p.read())
    port_image_handle = int(config['params']['ports']['server_port_image_handle'])
    port_ip_disc = int(config['params']['ports']['server_port_ip_disc'])
    connect_timeout = int(config['params']['timeout']['connect_timeout'])
    read_timeout = int(config['params']['timeout']['read_timeout'])

    # Get orientation data

    process_local = True
    if not ext_ip:
        # logger.error('GPU server not available to extract faces from {}'.format(filename))
        process_local = True
        print("Processing locally")
    else:
        payload, headers = image_and_box_for_network(filename, bounding_box)
        addr = f'http://{ext_ip}:{port_image_handle}'
        alive_url = addr + '/api/alive'
        # logger.debug("Alive url is {}".format(alive_url))
        try:
            # Since 'alive' expects no payload, it will throw 
            # an error if it receives one. So this is proper.
            response = requests.get(alive_url, timeout=(connect_timeout, read_timeout))
            # decode response
            retval = json.loads(response.text)
            if not retval['server_supports_cuda']:
                print("Server does not support CUDA: processing locally.")
            if response.status_code == 200:
                process_local = not retval['server_supports_cuda'] 

        except requests.exceptions.ConnectionError as ce:
            print(ce)
            print("Connection error for API -- will process locally")
            process_local = True

    addr = f'http://{ext_ip}:{port_image_handle}'
    face_extract_url = addr + '/api/face_reencode'

    # send http request with image and receive response
    try:
        response = requests.post(face_extract_url, data=payload, headers=headers, timeout=(connect_timeout, read_timeout))
        # decode response
        try:
            retval = json.loads(response.text)
        except json.decoder.JSONDecodeError as jde:
            if '500 Internal Server Error' in response.text:
                print("Your server face extract code is broken! It broke on filename {}".format(filename))
                raise IOError(f'Your server face extract code is broken. Fix it! It broke on filename {filename}. \nError text: {response.text}')
            else:
                print(f"File is: {filename}. Text response is : {response.text[:300]}")
                print(jde)
                # raise(jde)
                return

        if not retval['success']:
            print("No success: ", retval['message'])
        else:
            encoding = json.loads(retval['encoding'])


    except:
        print(ce)
        print('GPU server could not connect in face extraction.')
        raise ce


    return encoding

if __name__ == '__main__':

    # With top-to-bottom as Y and left-to-right as X, 
    # face_location is described as (top Y, left X, bottom Y, right X)
    face_location = [(173, 1350, 2098, 2899)]
    # Image chip would then be:
    # img[fl[0]:fl[2], fl[1]:fl[3]]
    source_image_file = '/photos/Completed/Pictures_finished/Family Pictures/2017/December/_DSC0927.JPG'

    client_ip = client_ip_discover.server_finder()
    # print(clie)
    # # client_ip = '192.168.1.146'
    # if 'IN_DOCKER' in os.environ.keys() and os.environ['IN_DOCKER']:
    #     mf = face_extract_client(os.path.join('/test_imgs_filepopulate/', 'has_face_tags.jpg'), client_ip)

    print(face_encoding_client(source_image_file, face_location, client_ip ))
