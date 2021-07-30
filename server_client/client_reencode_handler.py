#! /usr/bin/env python

import os
import sys
import dlib

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
import client_ip_discover # import find_external_server
import logging
import coloredlogs
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


def face_extract_client(filename, bounding_box, server_ip_finder, logger=None):

    if logger is None:
        logger = logging.getLogger('__main__')
        logger.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        logger.addHandler(ch)
        coloredlogs.install()

    server_there = server_ip_finder.check_ip()
    if server_there:
        ext_ip = server_ip_finder.server_ip
    else:
        server_ip_finder.find_external_server()
        ext_ip = None

    with open(os.path.join(PARENT_DIR, 'parameters.xml')) as p:
        config = xmltodict.parse(p.read())
    port_image_handle = int(config['params']['ports']['server_port_image_handle'])
    port_ip_disc = int(config['params']['ports']['server_port_ip_disc'])
    connect_timeout = int(config['params']['timeout']['connect_timeout'])
    read_timeout = int(config['params']['timeout']['read_timeout'])

    # Get orientation data

    process_local = True
    if not ext_ip:
        logger.error('GPU server not available to extract faces from {}'.format(filename))
        process_local = True
    else:
        payload, headers = image_and_box_for_network(filename, bounding_box)
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
        if not dlib.DLIB_USE_CUDA:
            raise IOError("No GPU available")
        else:
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
                    raise IOError(f'Your server face extract code is broken. Fix it! It broke on filename {filename}. \nError text: {response.text}')
                else:
                    print(f"File is: {filename}. Text response is : {response.text[:300]}")
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

        except requests.exceptions.ConnectionError as ce:
            print(ce)
            logger.error('GPU server could not connect in face extraction.')
            raise ce
        except requests.exceptions.ReadTimeout as ce:
            logger.error('GPU server timed out when face extracting {}'.format(filename))
            raise ce

    logger.debug('Elapsed time to extract face encoding from {} was {}'.format(filename, elapsed_time))

    print(matched_faces)

    # for face_num in range(len(matched_faces)):
    #     # matched_faces[face_num].reconstruct_square_face(filename)
    #     matched_faces[face_num].reconstruct_nonrect_face(filename)

    return matched_faces