
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
import client_ip_discover # import find_external_server
import logging
import coloredlogs
from PIL import Image, ExifTags

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
        if a.encoding is not None:
            a.encoding = np.asarray(a.encoding)
        # if a.image is not None:
        #     a.image = np.asarray(a.image)
        # else:
        #     logger.critical("Returned face did not have a regular image.")
        # if a.square_face is not None:
        #     a.square_face = np.asarray(a.square_face)
        # else:
        #     logger.critical("Returned face did not have a regular image.")

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

def face_extract_client(filename, server_ip_finder):

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

        except requests.exceptions.ConnectionError:
            logger.error('GPU server could not connect in face extraction.')
            matched_faces, _, _, elapsed_time = face_extraction.extract_faces_from_image(filename, config)
        except requests.exceptions.ReadTimeout:
            logger.error('GPU server timed out when face extracting {}'.format(filename))
            matched_faces, _, _, elapsed_time = face_extraction.extract_faces_from_image(filename, config)

    logger.debug('Elapsed time to extract faces from {} was {}'.format(filename, elapsed_time))

    if len(matched_faces) > 0: 


        image = Image.open(filename)
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break

        exif=dict(image._getexif().items())

        for face_num in range(len(matched_faces)):
            matched_faces[face_num].reconstruct_square_face(filename)
            matched_faces[face_num].reconstruct_nonrect_face(filename)
            # plt.imshow(matched_faces[face_num].square_face)
            # print(matched_faces[face_num].square_face.shape)
            # plt.show()
            # plt.imshow(matched_faces[face_num].face_image_nonrect)
            # print(matched_faces[face_num].face_image_nonrect.shape)
            # plt.show()

            mfi = matched_faces[face_num].square_face
            rect = matched_faces[face_num].rectangle
            top_i = rect.top
            left_i = rect.left
            r_height = rect.height
            r_width = rect.width
            im_height = image.height
            im_width = image.width
            imm = cv2.imread(filename)

            if exif[orientation] == 3:
                # Rotate 180
                matched_faces[face_num].square_face=np.rot90(mfi, 2) # image.rotate(180, expand=True)
                top = im_height - top_i - 1 - r_height
                left = im_width - left_i - 1 - r_width
            elif exif[orientation] == 6:
                # Rotate right
                matched_faces[face_num].square_face=np.rot90(mfi, 3) # image.rotate(270, expand=True)

                left = im_height - top_i - r_height
                top = left_i
                # right = top
                # top = left
                # left = right - r_width
                # left = im_height - left - 1
                # top = im_width - top - 1
                wid = r_width
                r_width = r_height
                r_height = wid
            elif exif[orientation] == 8:
                # Rotate left
                matched_faces[face_num].square_face=np.rot90(mfi, 1) # image.rotate(90, expand=True)
                top = im_width - left_i - r_width
                left = top_i
                wid = r_width
                r_width = r_height
                r_height = wid
            else:
                top = top_i
                left = left_i
                
            matched_faces[face_num].rectangle = face_extraction.rectangle.Rectangle(height=int(r_height), \
                width=int(r_width), topEdge=int(top), leftEdge=int(left))

    return matched_faces

if __name__ == "__main__":
    # mf = face_extract_client('my_pic.jpg')

    client_ip = client_ip_discover.server_finder()
    if 'IN_DOCKER' in os.environ.keys() and os.environ['IN_DOCKER']:
        mf = face_extract_client(os.path.join('/test_imgs_filepopulate/', 'has_face_tags.jpg'), client_ip)
    else:
        file = '/mnt/NAS/Photos/Pictures_In_Progress/2020/Erica Post-mission visit/DSC_4551.JPG'
        file = '/mnt/NAS/Photos/Pictures_In_Progress/2019/Baltimore Trip/DSC_1245.JPG'

        # mf = face_extract_client(os.path.join('/home/benjamin/gitRepos/test_imgs', '1.JPG'), client_ip)
        # mf = face_extract_client('/home/benjamin/Desktop/DSC_1209.JPG', client_ip)
        mf = face_extract_client(file, client_ip)
        import matplotlib.pyplot as plt
        # plt.imshow(mf[0].square_face)
        # plt.show()
        # plt.imshow(mf[1].square_face)
        # plt.show()
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for i in range(len(mf)):
            r = mf[i].rectangle
            print(r.top, r.left, r.bottom, r.right)
            cv2.rectangle(img, (r.left, r.top), (r.right, r.bottom), (255, 255, 130), 18)
        plt.imshow(img)
        plt.show()
    logger.debug(mf)

    # out = {'8': '/mnt/NAS/Photos/Pictures_In_Progress/2020/Erica Post-mission visit/DSC_4551.JPG', \
    #        '6': '/mnt/NAS/Photos/Pictures_In_Progress/2020/Erica Post-mission visit/20200225_170413.jpg', \
    #        '3': '/mnt/NAS/Photos/Pictures_In_Progress/2020/Erica Post-mission visit/2020-02-21 20.52.00.jpg'}

    # for key in ['3', '6', '8']:
    #     m3 = face_extract_client(out[key], client_ip)
    #     img = cv2.imread(out[key])
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     logger.debug(m3)
    #     for i in range(len(m3)):
    #         r = m3[i].rectangle
    #         print(r.top, r.left, r.bottom, r.right)
    #         cv2.rectangle(img, (r.left, r.top), (r.right, r.bottom), (255, 255, 130), 18)
    #     plt.imshow(img)
    #     plt.show()

    # for m in mf:
    #     print(m.square_face.shape)
    #     plt.imshow(m.square_face)
    #     plt.show()

    # test = False
    # if test:

    #     with open(os.path.join(PARENT_DIR, 'parameters.xml')) as p:
    #         config = xmltodict.parse(p.read())
    #     mf2, _, _ = face_extraction.extract_faces_from_image('my_pic.jpg', config)

    #     for f in range(len(mf)):
    #         f1 = mf[f]
    #         f2 = mf[f]
    #         # print(f1)
    #         assert np.mean(f1.encoding - f2.encoding) == 0
    #         assert f1.name == f2.name
    #         assert f1.rectangle == f2.rectangle
    #     print("Test done!")





#     parameter_file=os.path.join(PARENT_DIR, 'parameters.xml')
#     with open(parameter_file, 'r') as fh:
#         parameters = xmltodict.parse(fh.read())

#     matched_faces, _, _ = face_extraction.extract_faces_from_image('my_pic.jpg', parameters)

#     print(matched_faces)
#     assert matched_faces == xmp_data
