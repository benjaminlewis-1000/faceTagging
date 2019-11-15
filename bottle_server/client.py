
# This will read the *entire* image file as a data_string and send it over flask.

# import flask
import base64
import requests
import json
import cv2
import hashlib
from rectangle import Point, Rectangle

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

if not retval['success']:
    print(retval['message'])
else:
    print(json.loads(retval['xmp_data'], object_hook = decode_object) )