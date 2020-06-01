#! /usr/bin/env python

import os
import cv2 
from libxmp.utils import file_to_dict
from libxmp import XMPFiles, consts
import re
import collections
from queue import Queue
import threading
import time
import xmltodict
import signal

file = '/mnt/NAS/Photos/tmp_pic/DSC_1303.JPG'
file = '/mnt/NAS/Photos/tmp_pic/c/12-10-07 158.jpg'
file = '/mnt/NAS/Photos/tmp_pic/c/2-29-08 016.jpg'
file = '/mnt/NAS/Photos/tmp_pic/c/pict0929.jpg'
file = '/mnt/NAS/Photos/tmp_pic/e/20180915_215726.jpg'
file = '/mnt/NAS/Photos/tmp_pic/b/377611_10151954743470296_1763500400_n.jpg'
file = '/mnt/NAS/Photos/Pictures_In_Progress/2019/John Land Party/DSC_3215.JPG'
file = '/mnt/NAS/Photos/Pictures_In_Progress/2019/Matt and Rachel Wedding/Debbie_matt_rachel_wedding/IMG_1311 2.JPG'
file = '/mnt/NAS/Photos/Pictures_In_Progress/Erica Mission/zips/northernlights.jpg'
file = '/mnt/NAS/Photos/Pictures_In_Progress/preprocess/from_dropbox/2019-02-22 20.55.09.jpg'
file = '/mnt/NAS/Photos/Completed/Pictures_finished/Family Pictures/2017/Mom Phone/1482109274897.jpg'
file = '/mnt/NAS/Photos/Completed/Pictures_finished/Family Pictures/Old House, Nick Eagle Project/DSCN0203.JPG'
file = '/mnt/NAS/Photos/Pictures_In_Progress/2019/Baltimore Trip/2019-04-18 19.25.31-1.jpg'
file = "/mnt/NAS/Photos/Completed/Pictures_finished/Emily's Pictures/3-30-2009/DSC08050.JPG"
file = '/mnt/NAS/Photos/Pictures_In_Progress/2019/Utah Trip Summer/2019-10-02 18.18.52.jpg'


def extractFaces(file):

    try:
        xmpfile = XMPFiles( file_path=file, open_forupdate=True )
        xmp = xmpfile.get_xmp()
    except Exception as e:
        xmp = None

    with open(file, 'rb') as fh:
        data = fh.read()
        # Only look in the xmp metadata tags. 
        # Keeps us from finding crazy coincidences
        # in the file. I mean, it can still happen,
        # but less likely.
        header = re.match(b'.*?\xff\xc0', data, re.DOTALL).group(0)
        data = re.findall('<x:xmpmeta.*</x:xmpmeta>', str(data))
        # header_xmp = re.findall('<x:xmpmeta.*</x:xmpmeta>', str(header))
        # assert len(data) == len(header_xmp)
        # print(header)
        # print()
        assert len(header) > 0
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

data = Get_XMP_Faces(file)

fls = Queue()
for root, dirs, files in os.walk('/mnt/NAS/Photos'):
    for num, f in enumerate(files):
        if f.lower().endswith(('.jpg', '.jpeg')):
            fname = os.path.join(root, f)
            fls.put(fname)
            # print(f"{num}/{len(files)},  {fname}")

data = Get_XMP_Faces(fname)
signal = threading.Event()

def process_data(threadName):
    # print(threadName)
    while not fls.empty():
        fname = fls.get()
        print(f'{fls.qsize()} left to do')
        # print(fname)
        s = time.time()
        try:
            data = Get_XMP_Faces(fname)
        except :
            print(f"Error on {fname}")
            stop_threads=True
            signal.set()
            break      
        # print(time.time() - s)      
        if signal.is_set():
            break            

num_todo = 759000
# num_todo = 78900
while fls.qsize() > num_todo:
    f = fls.get()

threads = []
for t in range(100):
    t = threading.Thread(target=process_data, args=(t,) )
    threads.append(t)
    t.start()

# threading.start_new_thread(process_data, 1)
