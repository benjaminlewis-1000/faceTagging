from flask import Flask, request, Response
import jsonpickle
import numpy as np
import json
import cv2
import base64
from get_picasa_faces import Get_XMP_Faces
from rectangle import Point, Rectangle
import hashlib

# Initialize the Flask application
app = Flask(__name__)

# Source of JSON encoder: https://code.tutsplus.com/tutorials/serialization-and-deserialization-of-python-objects-part-1--cms-26183
class CustomEncoder(json.JSONEncoder):
     def default(self, o):
         return {'__{}__'.format(o.__class__.__name__): o.__dict__}

def decode_object(o):
 
    if '__Point__' in o:
 
        a = Point(0, 0)
 
        a.__dict__.update(o['__Point__'])
 
        return a
 
    elif '__Rectangle__' in o:
 
        a = Rectangle(10, 10, centerX = 5, centerY = 5)
 
        a.__dict__.update(o['__Rectangle__'])
 
        return a
 
    elif '__datetime__' in o:
 
        return datetime.strptime(o['__datetime__'], '%Y-%m-%dT%H:%M:%S')        
 
    return o


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
    # 
    if not request.content_type == 'text':
        raise ValueError("Posted data must be text.")
    # print((r.data)) # nparr = np.fromstring(r.data, np.uint8)


    data = json.loads(r.data)
    file_data = data['base64_file']
    file_data = file_data.encode('utf-8')
    checksum_data = data['checksum']

    loc_checksum = hashlib.md5(file_data)
    loc_checksum = loc_checksum.hexdigest()

    if checksum_data != loc_checksum:

        # # build a response dict to send back to client
        response = {'success': False, 'message': 'Bad image -- does not match the checksum.' } #{}x{}'.format(img.shape[1], img.shape[0])}
        # # encode response using jsonpickle
        response_pickled = jsonpickle.encode(response)
        return Response(response=response_pickled, status=200, mimetype="application/json")

    file = open('tmp.jpg', 'wb')
    file.write(base64.b64decode(file_data))
    file.close() 
    # # decode image
    # # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    xmp_data = Get_XMP_Faces('tmp.jpg')
    # print(xmp_data)
    enc = (json.dumps(xmp_data, cls=CustomEncoder))
    # print("------------------------")
    # print(json.loads(enc, object_hook = decode_object))

    # # do some fancy processing here....

    # # build a response dict to send back to client
    response = {'success': True, 'message': 'image received and processed', 'xmp_data': enc } #{}x{}'.format(img.shape[1], img.shape[0])}
    # # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)
    # print(response_pickled)

    return Response(response=response_pickled, status=200, mimetype="application/json")


if __name__ == "__main__":
    # start flask app
    app.run(host="0.0.0.0", port=5000)