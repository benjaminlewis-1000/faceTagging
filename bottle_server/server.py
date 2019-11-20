#! /usr/bin/env python

import os
import sys

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(PARENT_DIR)
sys.path.append(PARENT_DIR)


from flask import Flask, request, Response
import jsonpickle
import numpy as np
import json
import cv2
import io
import base64
import face_extraction
# from get_picasa_faces import Get_XMP_Faces
# from rectangle import Point, Rectangle
import hashlib
from PIL import Image
import face_recognition
import xmltodict

# Initialize the Flask application
app = Flask(__name__)

# Source of JSON encoder: https://code.tutsplus.com/tutorials/serialization-and-deserialization-of-python-objects-part-1--cms-26183
class CustomEncoder(json.JSONEncoder):

    def default(self, o):
        return {'__{}__'.format(o.__class__.__name__): o.__dict__}

# def decode_object(o):
 
#     if '__Point__' in o:
 
#         a = face_extraction.Point(0, 0)
 
#         a.__dict__.update(o['__Point__'])
 
#         return a
 
#     elif '__Rectangle__' in o:
 
#         a = face_extraction.Rectangle(10, 10, centerX = 5, centerY = 5)
 
#         a.__dict__.update(o['__Rectangle__'])
 
#         return a
 
#     return o


# route http posts to this method
@app.route('/api/test', methods=['POST'])
def test():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # do some fancy processing here....

    # build a response dict to send back to client
    response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0])
                }
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")

# route http posts to this method
@app.route('/api/test_fullfile', methods=['POST'])
def test_fullfile():
    r = request

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
    file = io.BytesIO(dt)

    # Open the image as a numpy image for face recognition. 
    image = face_recognition.load_image_file(file)

    # Retrieve the XMP faces. 
    file = io.BytesIO(dt)
    xmp_data = face_extraction.Get_XMP_Faces(file)


    parameter_file=os.path.join(PARENT_DIR, 'parameters.xml')
    with open(parameter_file, 'r') as fh:
        parameters = xmltodict.parse(fh.read())

    matched_faces, _, _ = face_extraction.extract_faces_from_image(file, parameters)
    matched_faces == matched_faces
    for idx in range(len(matched_faces)):
        encoding = matched_faces[idx].encoding
        matched_faces[idx].encoding = encoding.tolist()
        image = matched_faces[idx].image
        matched_faces[idx].image = image.tolist()

    xmp_data = matched_faces

    enc = (json.dumps(xmp_data, cls=CustomEncoder))

    # # build a response dict to send back to client
    response = {'success': True, 'message': 'image received and processed', 'xmp_data': enc } #{}x{}'.format(img.shape[1], img.shape[0])}
    # # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)
    # print(response_pickled)

    return Response(response=response_pickled, status=200, mimetype="application/json")


if __name__ == "__main__":
    # start flask app
    app.run(host="0.0.0.0", port=5000)