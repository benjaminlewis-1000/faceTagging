#! /usr/bin/env python

import os
import torch
from torchvision import transforms
import face_extraction
from face_extraction.encode_image_with_bounding import encode_image_with_bounding
import sys

# Put this file in crontab -e with the line (for initial system startup)
# @reboot (<PATH_TO_GUNICORN>/gunicorn -b 0.0.0.0:5000 -w 1 --chdir <PATH_TO_THIS_GIT_REPO> server_client.server_image_handler:app ) &

### Put the following script in your /lib/systemd/system-sleep directory, with modifications to get the right directory. 
'''
# **** This script is used to kill the service on sleep. However, there are issues
# **** with re-starting it in sudo space (I think due to GPU), so we have to do something
# **** else on restart
case $1/$2 in
  pre/*)
    echo "Going to $2..."
    killall libinput-debug-events
    for procID in ` ps aux | grep server_image_handler | grep -v 'grep' | awk '{print $2}'`; do
        kill $procID
    done
    sleep .2
    ;;
  post/*)
    echo "Waking up from $2..."
    # Place your post suspend (resume) commands here, or `exit 0` if no post suspend action required
    ;;
esac

### Put the following script in /etc/systemd/system/facerec.service, again modifying paths as needed:
### (Source : https://unix.stackexchange.com/questions/152039/how-to-run-a-user-script-after-systemd-wakeup 
### ==> https://unix.stackexchange.com/a/492497 for specific answer)

[Unit]
Description=Run face_recognition
After=suspend.target hibernate.target hybrid-sleep.target suspend-then-hibernate.target

[Service]
ExecStart=/home/benjamin/.local/bin/gunicorn -b 0.0.0.0:5000 -w 1 --chdir /home/benjamin/gitRepos/faceTagging server_client.server_image_handler:app
User=benjamin

[Install]
WantedBy=suspend.target hibernate.target hybrid-sleep.target suspend-then-hibernate.target

### Then, enable it in the system by using ``sudo systemctl enable facerec``.

'''

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)


try:
    from .server_ip_discover import ip_responder
except:
    from server_ip_discover import ip_responder

from flask import Flask, request, Response, session
from flask_session import Session

# Required on Jetson now for OpenCV to work.
os.environ['OPENBLAS_CORETYPE'] = 'ARMV8'

from PIL import Image, ExifTags
import base64
import cv2
import face_extraction
import face_recognition
import hashlib
import io
import json
import time
import matplotlib.pyplot as plt
import jsonpickle
import logging
import numpy as np
import xmltodict


# Initialize the Flask application
app = Flask(__name__)
# SESSION_TYPE = 'redis'
app.config.from_object(__name__)
# Session(app)
# app.secret_key = os.urandom(24)
# app.config.update(SECRET_KEY=os.urandom(24))
# # sess = session()
# app.config.from_object(__name__)
try:
    import dlib
    using_cuda = dlib.DLIB_USE_CUDA
except AttributeError:
    using_cuda = False
except ImportError:
    using_cuda = False



def open_and_rotate_image(source_image_file):

    try:
        npImage = face_recognition.load_image_file(source_image_file)
    except Image.DecompressionBombError:

        if type(source_image_file) == type('string'):
            npImage = cv2.imread(source_image_file)
        elif type(source_image_file) == io.BytesIO:
            file_bytes = np.asarray(bytearray(source_image_file.getvalue()), dtype=np.uint8)
            npImage = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            this_pix = npImage.shape[0] * npImage.shape[1]
            if this_pix < 5 * Image.MAX_IMAGE_PIXELS:
                # No problem, just a really huge image. 
                Image.MAX_IMAGE_PIXELS = this_pix + 1
                npImage = face_recognition.load_image_file(source_image_file)
            else:
                # Could actually be a decompression bomb attack, IDK, and not
                # just a huge image. 
                return None
            # print(npImage.shape[0] * npImage.shape[1], Image.MAX_IMAGE_PIXELS)
    image_exif = Image.open(source_image_file)

    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation]=='Orientation':
            break

    if 'items' in dir(image_exif._getexif()):
        exif=dict(image_exif._getexif().items())
    else:
        exif = {}

    if orientation in exif.keys():
        if exif[orientation] == 3:
            # Rotate 180
            npImage = cv2.rotate(npImage, cv2.ROTATE_180)
        elif exif[orientation] == 6:
            # Rotate right -- 270
            npImage = cv2.rotate(npImage, cv2.ROTATE_90_CLOCKWISE)
        elif exif[orientation] == 8:
            # Rotate left -- 90 
            npImage = cv2.rotate(npImage, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return npImage

# Source of JSON encoder: https://code.tutsplus.com/tutorials/serialization-and-deserialization-of-python-objects-part-1--cms-26183
class CustomEncoder(json.JSONEncoder):

    def default(self, o):
        return {'__{}__'.format(o.__class__.__name__): o.__dict__}

@app.route('/api/alive', methods=['GET', 'POST'])
def alive():
    r = request

    # build a response dict to send back to client
    response = {'message': 'alive', 'server_supports_cuda': using_cuda}

    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


# route http posts to this method
@app.route('/api/face_reencode', methods=['POST'])
def face_reencode():
    r = request
    print("Face reencode")

    if not request.content_type == 'text':
        raise ValueError("Posted data must be text.")


    data = json.loads(r.data)
    file_data = data['base64_file']
    bounding_box = data['bounding_box']
    # Convert the string to bytes. We know that the string 
    # is a base64 string encoded using utf-8.
    file_data = file_data.encode('utf-8')
    # Get the hex checksum from the payload
    checksum_data = data['checksum']
    # Compute the md5 checksum of the binary string. It should
    # match that of the checksum payload. 
    loc_checksum = hashlib.md5(file_data)
    loc_checksum = loc_checksum.hexdigest()

    # If checksums don't match, send information back to client. 
    if checksum_data != loc_checksum:
        response = {'success': False, 'message': 'Bad image -- does not match the checksum.' } 

        # # encode response using jsonpickle
        response_pickled = jsonpickle.encode(response)
        return Response(response=response_pickled, status=200, mimetype="application/json")

    # Shove that file data into a BytesIO object that can
    # be read using PIL. The key to getting string data back
    # from this IO object is to use getvalue, not read.

    dt = base64.b64decode(file_data)

    file = io.BytesIO(dt)

    parameter_file=os.path.join(PARENT_DIR, 'parameters.xml')
    with open(parameter_file, 'r') as fh:
        parameters = xmltodict.parse(fh.read())
        
    print("extracting image encoding")

    npImage = open_and_rotate_image(file)

    s = time.time()
    encoding = encode_image_with_bounding(npImage = npImage, bounding_box=bounding_box, pad_pct = 0.25)
    print(time.time() - s)

    enc = (json.dumps(list(encoding[0])))

    # # build a response dict to send back to client
    response = {'success': True, 'message': 'image received and processed', \
                'encoding': enc} 

    # # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)
    
    return Response(response=response_pickled, status=200, mimetype="application/json")

# route http posts to this method
@app.route('/api/face_extract', methods=['POST'])
def face_extract():
    r = request

    # print(session)
    # if 'starttime' in session:
    #     old_start = session['starttime']
    #     print(f"Elapsed since last session is {time.time() - old_start}")
    # session['starttime'] = time.time()
    # print(session['starttime'])

    if not request.content_type == 'text':
        raise ValueError("Posted data must be text.")

    # Load the dict from the request. It should have a base64_file and
    # a checksum field.
    data = json.loads(r.data)
    file_data = data['base64_file']
    # Convert the string to bytes. We know that the string 
    # is a base64 string encoded using utf-8.
    file_data = file_data.encode('utf-8')
    # Get the hex checksum from the payload
    checksum_data = data['checksum']

    # Compute the md5 checksum of the binary string. It should
    # match that of the checksum payload. 
    loc_checksum = hashlib.md5(file_data)
    loc_checksum = loc_checksum.hexdigest()

    # If checksums don't match, send information back to client. 
    if checksum_data != loc_checksum:
        response = {'success': False, 'message': 'Bad image -- does not match the checksum.' } 

        # # encode response using jsonpickle
        response_pickled = jsonpickle.encode(response)
        return Response(response=response_pickled, status=200, mimetype="application/json")

    # Shove that file data into a BytesIO object that can
    # be read using PIL. The key to getting string data back
    # from this IO object is to use getvalue, not read.

    dt = base64.b64decode(file_data)

    # Retrieve the XMP faces. 
    file = io.BytesIO(dt)
    success, xmp_data = face_extraction.Get_XMP_Faces(file)
    assert success, 'XMP extraction failed'

    parameter_file=os.path.join(PARENT_DIR, 'parameters.xml')
    with open(parameter_file, 'r') as fh:
        parameters = xmltodict.parse(fh.read())
        
    print("extracting")
    matched_faces, _, _, elapsed_time = face_extraction.extract_faces_from_image(file, parameters, xmp_data)

    for idx in range(len(matched_faces)):
        encoding = matched_faces[idx].encoding
        if encoding is not None:
            matched_faces[idx].encoding = encoding.tolist()
        mm = matched_faces[idx]
        # image = matched_faces[idx].face_image_nonrect
        # if image is not None: # Shouldn't happen any other way...
        #     matched_faces[idx].image = image.tolist()
        #     logging.debug("Reg image OK")
        # else:
        #     logging.error("Your face extractor returned no image. This shouldn't happen.")
        #     return
        matched_faces[idx].face_image_nonrect = None
        
        # square_face = matched_faces[idx].square_face
        # if square_face is not None:
        #     logging.debug("Squre image OK")
        #     matched_faces[idx].square_face = square_face.tolist()
        # else:
        #     logging.error("Your face extractor returned no square image. This shouldn't happen.")
        #     return
        matched_faces[idx].square_face = None

    enc = (json.dumps(matched_faces, cls=CustomEncoder))

    # # build a response dict to send back to client
    response = {'success': True, 'message': 'image received and processed', \
                'xmp_data': enc, 'elapsed_time': elapsed_time } 

    # # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)
    
    return Response(response=response_pickled, status=200, mimetype="application/json")


if __name__ == "__main__":
    print("Main")
    # start flask app
    with open(os.path.join(PARENT_DIR, 'parameters.xml')) as p:
        config = xmltodict.parse(p.read())

    port = int(config['params']['ports']['server_port_image_handle'])

    import threading

    ip_thread = threading.Thread(target = ip_responder)
    ip_thread.start()
    print(ip_thread)
    # Do NOT join the thread -- it will cause the while True
    # to block.
    # sess.init_app(app)
    app.run(host="0.0.0.0", port=port)
