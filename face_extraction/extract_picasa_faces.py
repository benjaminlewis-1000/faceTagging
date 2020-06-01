
if __name__ == "__main__":
    from rectangle import Rectangle 
else:
    from .rectangle import Rectangle 
from PIL import Image, ExifTags
from time import sleep
import binascii
import collections
import cv2
import base64
from base64 import b64decode
import face_recognition
import io
import matplotlib.pyplot as plt
import numpy as np
import re
import xml
import xmltodict
from libxmp.utils import file_to_dict
from libxmp import XMPFiles, consts
from tempfile import NamedTemporaryFile



def extractFaces(file):

    if isinstance(file, str):
        with open(file, 'rb') as fh:
            data = fh.read()
    else:
        data = file.read()
        f = NamedTemporaryFile()
        f.write(data)
        file = f.name

    try:
        xmpfile = XMPFiles( file_path=file, open_forupdate=True )
        xmp = xmpfile.get_xmp()
    except Exception as e:
        xmp = None

    # Only look in the xmp metadata tags. 
    # Keeps us from finding crazy coincidences
    # in the file. I mean, it can still happen,
    # but less likely.
#    header = re.match(b'.*?\xff\xc0', data, re.DOTALL).group(0)
    data = re.findall('<x:xmpmeta.*?</x:xmpmeta>', str(data))
    # header_xmp = re.findall('<x:xmpmeta.*</x:xmpmeta>', str(header))
    # assert len(data) == len(header_xmp)
#    assert len(header) > 0
    if len(data) == 0:
        return []
    else:
        data = data[0]
    names = re.findall('.{7}:name=', str(data), re.I)
    # Exclude an Adobe color thing 
    names = [x for x in names if 'crs:Name' not in x]
    for n in names:
        if 'mwg-rs' not in n:
            raise ValueError("Not as expected'")
    number_of_names = len(names)
    # print(number_of_names)
    print(f"File {file} has {number_of_names} people.")

    if xmp is not None:
        xmp_dict = file_to_dict(file)

    if xmp is None or len(xmp_dict) == 0:
        # This is where I get into a number of caveats that I built up as the code failed. 
        xmp_data_all = re.findall('<x:xmpmeta.*?/x:xmpmeta>', str(data), re.I)
        print("TYPE xmp data is ", type(xmp_data_all))
        if len(xmp_data_all) == 0:
            assert number_of_names == 0
            return []
        
        person_data = []
        for xmp_data in xmp_data_all:
        # xmp_data = xmp_data[0]
            # print(xmp_data)
            xmp_data = xmltodict.parse(xmp_data)
            xmp_data = xmp_data['x:xmpmeta']['rdf:RDF']
            if 'rdf:Description' in xmp_data.keys():
                xmp_data = xmp_data['rdf:Description']
            else:
                # assert number_of_names == 0
                # return []
                continue
            if 'mwg-rs:Regions' in xmp_data.keys():
                face_keys = xmp_data['mwg-rs:Regions']['mwg-rs:RegionList']['rdf:Bag']['rdf:li']
            else:
                # assert number_of_names == 0
                # return []
                continue

            if type(face_keys) == list:
                for i in range(len(face_keys)):
                    region = face_keys[i]['rdf:Description']
                    name = region['@mwg-rs:Name']
                    area = region['mwg-rs:Area'] 
                    area_x = area['@stArea:x']
                    area_y = area['@stArea:y']
                    area_w = area['@stArea:w']
                    area_h = area['@stArea:h']
                    data = {'Name': name, 'Area_x': area_x, 'Area_y': area_y, 'Area_w': area_w, 'Area_h': area_h}
                    person_data.append(data)
            elif type(face_keys) == collections.OrderedDict:
                region = face_keys['rdf:Description']
                name = region['@mwg-rs:Name']
                area = region['mwg-rs:Area'] 
                area_x = area['@stArea:x']
                area_y = area['@stArea:y']
                area_w = area['@stArea:w']
                area_h = area['@stArea:h']
                data = {'Name': name, 'Area_x': area_x, 'Area_y': area_y, 'Area_w': area_w, 'Area_h': area_h}
                person_data.append(data)


        assert len(person_data) == number_of_names
        return person_data

    else: 

        regions = xmp_dict.get('http://www.metadataworkinggroup.com/schemas/regions/')

        if regions is None:
            assert number_of_names == 0, f'Should have found {number_of_names} names.'
            return []

        this_regionlist = [x for x in regions if 'RegionList[1]' in x[0]]

        person_data = []
        rl_num = 2
        while len(this_regionlist) > 0:
            name = None
            area_x = None
            area_y = None
            area_h = None
            area_w = None

            for i in range(len(this_regionlist)):
                if this_regionlist[i][0].find('Name') != -1:
                    name = this_regionlist[i][1]
                if this_regionlist[i][0].find('mwg-rs:Area/stArea:x') != -1:
                    area_x = this_regionlist[i][1]
                if this_regionlist[i][0].find('mwg-rs:Area/stArea:y') != -1:
                    area_y = this_regionlist[i][1]
                if this_regionlist[i][0].find('mwg-rs:Area/stArea:w') != -1:
                    area_w = this_regionlist[i][1]
                if this_regionlist[i][0].find('mwg-rs:Area/stArea:h') != -1:
                    area_h = this_regionlist[i][1]

            try:
                if name is not None:
                    data = {'Name': name, 'Area_x': area_x, 'Area_y': area_y, 'Area_w': area_w, 'Area_h': area_h}
                    person_data.append(data)
            except UnboundLocalError:
                pass

            this_regionlist = [x for x in regions if f'RegionList[{rl_num}]' in x[0]]
            rl_num += 1

        assert len(person_data) == number_of_names, f'Only {len(person_data)} of {number_of_names} found'

        return person_data

# This function will extract the XMP Bag Tag from the header of 
# a JPG file. This is where the now-defunct Picasa program, by 
# Google, stored face information. This function supports both 
# paths to files as well as BytesIO in-memory files. 
def Get_XMP_Faces(file, test=False):

    persons = extractFaces(file)
    print(persons)

    if type(file) == type('string'):
        image = cv2.imread(file)
    elif type(file) == io.BytesIO:
        file_bytes = np.asarray(bytearray(file.getvalue()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    image_exif = Image.open(file)

    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation]=='Orientation':
            break

    if 'items' in dir(image_exif._getexif()):
        exif=dict(image_exif._getexif().items())
    else:
        exif = {}


    if orientation in exif.keys():
        if exif[orientation] in [6, 8]:
            img_width, img_height, _ = image.shape
        else:
            img_height, img_width, _ = image.shape
    plt.imshow(image)
    plt.show()
    # X and Y are locations in the middle of the face. 

    print(img_height, img_width)

    # Reverse parsing. We process the list of persons
    # *again* to turn the tags into Rectangle objects
    # and put that in the list that will be returned.
    # The intermediate data is then popped from the 
    # dictionary. 
    for p_num in range(len(persons)-1, -1, -1):
        left = float(persons[p_num]['Area_x'])
        top = float(persons[p_num]['Area_y']) 
        # print(left, top)
        height = float(persons[p_num]['Area_h']) 
        width = float(persons[p_num]['Area_w'])         

        right = left + width / 2
        bottom = top + height / 2
        left = left - width / 2
        top = top - height / 2

        left = int(left * img_width)
        right = int(right * img_width)
        top = int(top * img_height)
        bottom = int(bottom * img_height)

        height = int(height * img_height)
        width = int(width * img_width)

        bounding_rectangle = Rectangle(height, width, leftEdge=left, topEdge=top)
        persons[p_num]['bounding_rectangle'] = bounding_rectangle

        # Turning the list into a list of rectangles.
        persons[p_num].pop('Area_x')
        persons[p_num].pop('Area_y')
        persons[p_num].pop('Area_h')
        persons[p_num].pop('Area_w')
  
    #  if we found something, return tag information
    print("At the end: ", persons)
    return True, persons


if __name__ == "__main__":


    file = '/mnt/NAS/Photos/tmp_pic/DSC_1303.JPG'
    file = '/mnt/NAS/Photos/Pictures_In_Progress/Adam Mission/Adam mission book/landscape/Lewis Reunion 2012 (34).JPG'
    file = '/mnt/NAS/Photos/Completed/Pictures_finished/2016/Utah/baker_reunion (3).jpg'
    file = '/mnt/NAS/Photos/Completed/Pictures_finished/Family Pictures/2017/Mom Phone/1479419708717.jpg'

    d = Get_XMP_Faces(file)
    print(d)

    with open(file, 'rb') as imageFile:
#        data_str = base64.b64encode(imageFile.read())
        data_str = io.BytesIO(imageFile.read())
        dd = Get_XMP_Faces(data_str)
        print(dd)
