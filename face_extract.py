#! /usr/bin/env python

import face_recognition 
import os
import cv2
from rectangle import Rectangle
import numpy as np
import scipy.misc
import itertools
import coloredlogs
import random
import time
import tiled_detect
import logging
import face_rect
import xmltodict
import matplotlib.pyplot as plt
import get_picasa_faces

coloredlogs.install()

params1 = {'upsample': 2, 'height': 600, 'width': 300}
path_to_script = os.path.dirname(os.path.realpath(__file__))

def imageFaceDetect(image_path, parameter_file='parameters.xml'):
    assert isinstance(image_path, str)
    assert os.path.isfile(parameter_file)

    with open(parameter_file, 'r') as fh:
        parameters = xmltodict.parse(fh.read())

    ml_detected_faces, tagged_faces = extract_faces_from_image(image_path, parameters)

def extract_faces_from_image(image_path, parameters):
    assert isinstance(image_path, str)
    assert os.path.isfile(image_path)
    assert 'params' in parameters.keys()
    assert 'tiled_detect_params' in parameters['params']

    tiled_params = parameters['params']['tiled_detect_params']

    npImage = face_recognition.load_image_file(image_path)

    ml_detected_faces = tiled_detect.detect_pyramid(npImage, tiled_params)

    success_faces, tagged_faces = get_picasa_faces.Get_XMP_Faces(image_path)

    assert success_faces, 'Picasa face extraction failed.'

    return ml_detected_faces, tagged_faces

def associate_detections_and_tags(image_path, detected_faces, tagged_faces, disp_photo=False):

    if len(detected_faces) > 0:
        for df in detected_faces:
            assert isinstance(df, face_rect.FaceRect)

    if len(tagged_faces) > 0: 
        for tf in tagged_faces:
            assert isinstance(tf, dict)
            assert 'Name' in tf.keys()
            assert 'bounding_rectangle' in tf.keys()

    image = face_recognition.load_image_file(
        image_path)
    pristine_image = image.copy()

    if disp_photo:
        draw_image = image.copy()
        for df in detected_faces:
            df.rectangle.drawOnPhoto(draw_image)
        for tf in tagged_faces:
            tf['bounding_rectangle'].drawOnPhoto(draw_image, colorTriple=(0,255, 0))

    # Cluster the detected faces based on IOU. 
    clusters = []

    remaining_clusters = detected_faces.copy()
    assert isinstance(remaining_clusters, list)

    max_iter = 5000
    iter_num = 0
    while len(remaining_clusters) > 0: 
        # Break infinite loops
        iter_num += 1
        assert iter_num < max_iter
        
        current_face = remaining_clusters[0]
        current_cluster = [current_face]
        for cmp_idx in range(len(remaining_clusters) - 1, 0, -1): 
            # For all the others, reverse order
            compare_face = remaining_clusters[cmp_idx]
            iou = current_face.rectangle.IOU(compare_face.rectangle)
            if iou > 0: 
                current_cluster.append(compare_face)
                remaining_clusters.pop(cmp_idx)

        remaining_clusters.pop(0)
        current_cluster = tuple(current_cluster)
        clusters.append(current_cluster)
        # print("Current cluster length: ", len(current_cluster))

    # Clusters have *any* IOU
    # print(len(clusters))

    contained_inside_rect_thresh = 0.8
    enc_dist_thresh = 0.4

    deconflicted_detections = []
    for each_cluster in clusters:
        if len(each_cluster) == 1:
            # print("cluster len 1, no merges")
            deconflicted_detections.append(each_cluster[0])
        else:
            merged = False
            cluster_cp = list(each_cluster)

            # Complex -- find all the overlaps with the first
            # thing in the cluster. Remove that first
            # element and all overlap elements from the cluster.
            # Repeat until the cluster candidates are empty. 
            # The tricky thing is the indexing. As usual.
            iter_num = 0 
            while len(cluster_cp) > 0: 
                # Break infinite loops
                iter_num += 1
                assert iter_num < max_iter
                # First thing in the cluster
                current_root = cluster_cp[0]
                
                # Get the value of merges for all
                # items in the cluster. The first
                # element is a merge with itself, which
                # will be true, of course. 
                merges = [x.test_merge(current_root) for x in cluster_cp]
                # print(merges)

                # If it merges with anything else, then 
                # proceed to merge. 
                if np.any(merges[1:]):
                    # Start at the end of the list and 
                    # work backward. 
                    for m_idx in range(len(cluster_cp) - 1, 0, -1):
                        if merges[m_idx]:
                            # If the given index is mergable, 
                            # then merge it with the current
                            # root and pop it out. 
                            current_root = current_root.merge_with(cluster_cp[m_idx], pristine_image)
                            cluster_cp.pop(m_idx)

                # Put the current root in the deconflicted
                # list. Then pop it from the cluster. 
                # print(len(cluster_cp))
                deconflicted_detections.append(current_root)
                cluster_cp.pop(0)

    if disp_photo:
        for d in deconflicted_detections:
            # print(type(d))
            d.rectangle.drawOnPhoto(draw_image, colorTriple=(0, 0, 255))

    matched_face_rects = []

    def join_faces(tag_face, det_face=None):
        # This is used either to turn a tagged face into
        # a FaceRect, or to merge a detected and a 
        # tagged face.
        assert isinstance(tag_face, dict)
        assert 'bounding_rectangle' in tag_face.keys()
        assert 'Name' in tag_face.keys()
        assert tag_face['Name'] is not None
        if det_face is not None:
            assert isinstance(det_face, face_rect.FaceRect)

        if det_face is not None:
            rect_intersect = tag_face['bounding_rectangle'].intersect(det_face.rectangle)
            # This will be 1 if the tagged is in the 
            # detected area (100%), smaller if not
            tag_in_det = rect_intersect / tag_face['bounding_rectangle'].area
            # This will be 1 if the detection is in the
            # tag area (100%), smaller if not. 
            det_in_tag = rect_intersect / det_face.rectangle.area

            # Here, we want to use the bigger, more
            # relevant rectangle. So if the detected
            # one is smaller than the tagged one,
            # use the tagged face; otherwise, vice versa.
            if det_in_tag > tag_in_det:
                rect = tag_face['bounding_rectangle']
            else:
                rect = det_face.rectangle

            detection_level = det_face.detection_level
            encoding = det_face.encoding

        else:
            rect = tag_face['bounding_rectangle']
            encoding = None
            detection_level = -1

        name = tag_face['Name']

        face_image = pristine_image[rect.top:rect.bottom, rect.left:rect.right]

        fr = face_rect.FaceRect(rect, face_image, detection_level, encoding, name)

        return fr


    # Three cases:
    # 1 - Detection with no Picasa Face
    # 2 - Picasa face with no detection
    # 3 - Picasa face with overlapping detection(s).

    # Case 1:  Detection with no picasa face
    print("Case 1 todo -- increase image chip size detection")
    num_det = len(deconflicted_detections)
    for d_num in range(num_det - 1, -1, -1):
        # print(d_num)
        det = deconflicted_detections[d_num]
        have_match = False
        for tag in tagged_faces:
            if tag['bounding_rectangle'].IOU(det.rectangle) > 0:
                have_match = True
                break

        if not have_match:
            matched_face_rects.append(det)
            # print("Popping ", d_num, len(deconflicted_detections))
            deconflicted_detections.pop(d_num)


    # We should have removed detections that have no
    # picasa match from the deconflicted_detections,
    # so we check that here.

    # Assertions for case 1 
    assert len(set(deconflicted_detections).intersection(set(matched_face_rects) ) ) == 0
    for match in matched_face_rects:
        for tag in tagged_faces:
            assert(tag['bounding_rectangle'].IOU(match.rectangle)) == 0

    # Case 2: Tagged faces with no detections
    num_tags = len(tagged_faces)
    for t_num in range(num_tags - 1, -1, -1):
        tagged = tagged_faces[t_num]
        have_match = False
        for det in deconflicted_detections:
            if tagged['bounding_rectangle'].IOU(det.rectangle) > 0:
                have_match = True
                break

        if not have_match:
            tag_rect = join_faces(tagged)
            matched_face_rects.append(tag_rect)
            tagged_faces.pop(t_num)

    # Assertions for case 2
    for m in matched_face_rects:
        assert isinstance(m, face_rect.FaceRect)
    assert len(set(deconflicted_detections).intersection(set(matched_face_rects) ) ) == 0
    # Assert that everything else has IOUs. 
    for t in tagged_faces:
        overlap = False
        for d in deconflicted_detections:
            if d.rectangle.IOU(t['bounding_rectangle']) > 0:
                overlap = True
        assert overlap


    if disp_photo:
        for df in matched_face_rects:
            rr = df.rectangle
            rr.left = rr.left - 10
            rr.top = rr.top - 10
            rr.width = rr.width + 20
            rr.height = rr.height + 20
            df.rectangle.drawOnPhoto(draw_image,colorTriple=(235,189,52))


    # Case 3. Every picasa tag left has at least some 
    # IOU relation with at least one detected tag that
    # is left.

    detection_tag_clusters = []

    for tag in tagged_faces:
        det_list = []
        for det in deconflicted_detections:
            if tag['bounding_rectangle'].IOU(det.rectangle) > 0:
                det_list.append(det)
        tag_cluster = (tag, det_list)
        detection_tag_clusters.append(tag_cluster)

    if disp_photo:
        plt.figure()
        plt.imshow(draw_image)
        plt.show()

    for cluster in detection_tag_clusters:
        tag, det_list = cluster
        ious = [tag['bounding_rectangle'].IOU(x.rectangle) for x in det_list]
        intersect_thresh = 0.5
        norm_intersections = [tag['bounding_rectangle'].intersect(x.rectangle) / min(x.rectangle.area, tag['bounding_rectangle'].area) for x in det_list]
        int_over_thresh = [x > intersect_thresh for x in norm_intersections]

        # print(len(int_over_thresh))
        # print(norm_intersections)

        # Sub-case 1: only one intersection at all
        if len(det_list) == 1:
            if int_over_thresh[0] > 0.3:
                joint = join_faces(tag, det_list[0])
                matched_face_rects.append(joint)
                deconflicted_detections.remove(det_list[0])
        else:
            # Sub-case 2: only one intersection over
            # the threshold
            if int_over_thresh.count(True) == 1:
                idx = int_over_thresh.index(True)
                isect_face = det_list[idx]
                joint = join_faces(tag, isect_face)
                matched_face_rects.append(joint)
                # Have to consider that the other
                # overlaps may not belong to any
                # tagged face. 
                deconflicted_detections.remove(isect_face)
            # Sub-case 3: multiple normalized intersections
            # that are over the threshold level.
            # Get the one that is closest to the center
            # of the tag and call the other one a 
            # separate face. 
            else:
                distances = [tag['bounding_rectangle'].distance(x.rectangle)[0] for x in det_list]
                idx = np.argmin(distances)
                closest_face = det_list.pop(idx)
                # for oth_norm in norm_intersections:
                #     assert oth_norm < max_norm / 2
                joint = join_faces(tag, closest_face)
                deconflicted_detections.remove(closest_face)
                matched_face_rects.append(joint)
                for m in det_list:
                    matched_face_rects.append(m)
                    deconflicted_detections.remove(m)

    if disp_photo:
        for df in matched_face_rects:
            rr = df.rectangle
            rr.left = rr.left - 20
            rr.top = rr.top - 20
            rr.width = rr.width + 40
            rr.height = rr.height + 40
            df.rectangle.drawOnPhoto(draw_image,colorTriple=(245, 66, 203))

    for others in deconflicted_detections:
        for matched in matched_face_rects:
            assert matched.rectangle.IOU(others.rectangle) < intersect_thresh
        matched_face_rects.append(others)
        if disp_photo:
            others.rectangle.drawOnPhoto(draw_image, colorTriple=(100, 200, 150))

    if disp_photo:
        plt.figure()
        plt.imshow(draw_image)
        plt.show()

    # Not always true that one is completely
    # inside of the other
    
    # Things we do know: two faces can't be co-located. 

    # Same person should have similar detection
    # encodings in the same picture.

    # I think it's ideal to focus on the largest (?) 
    # decection/one that comes from the whole image,
    # if available. 

'''

def processChip(rectangle, photoObject, faceNum, params):

    # Extract the image, path hash, and width and height of the image
    npImage = photoObject.cvImg
    photoPathHash = photoObject.pathHash
    photoWidth = photoObject.photoWidth
    photoHeight = photoObject.photoHeight
    photo_primary_key = photoObject.primaryKey

    # Given the rectangle that defines the face, extract the pixels corresponding
    # to the face and store it in the roi variable.
    centerX = rectangle.centerX
    centerY = rectangle.centerY
    rec_height = rectangle.height
    rec_width = rectangle.width
    top = rectangle.top
    left = rectangle.left
    roi = npImage[top:top+rec_height, left:left+rec_width]

    # The Picasa format for storing faces requires that the center X and Y points
    # of the face ROI as well as the width and height are defined as percentages of
    # the width and height of the image. 
    centerX_scale = float(centerX) / float(photoWidth)
    centerY_scale = float(centerY) / float(photoHeight)
    rec_width_scale = float(rec_width) / float(photoWidth)
    rec_height_scale = float(rec_height) / float(photoHeight)

    # Build up the path name of where the image chip will be stored. 
    stored_prefix = params['chipSaveParams']['storedChipPrefix']
    chip_file_name = '{}_{}_{}.jpg'.format(stored_prefix, str(photoPathHash), faceNum)
    print(chip_file_name)

    # Put the image in the right directory and save it with OpenCV imwrite. 
    chip_save_name = os.path.join(params['chipSaveParams']['chipSaveDir'], chip_file_name)
    # assert not os.isfile(chip_save_name)
    cv.imwrite(chip_save_name , roi)

    # Warn that we are assigning a random vector to the face, for now. 
    logging.warn('Random face vector for stored faces.')
    face_vector = np.random.rand(128)

    # Build up the dict we need to store this in the database. 
    faceDBStruct = dict()
    faceDBStruct['assigned_name'] = None
    faceDBStruct['photo_primary_key'] = photo_primary_key
    faceDBStruct['center_X_scaled'] = centerX_scale
    faceDBStruct['center_Y_scaled'] = centerY_scale
    faceDBStruct['width_scale'] = rec_width_scale
    faceDBStruct['height_scale'] = rec_height_scale
    faceDBStruct['chip_save_name'] = chip_save_name
    faceDBStruct['face_vector'] = face_vector
    faceDBStruct['best_guesses'] = None

    faceNum += 1

    return faceNum, faceDBStruct

def imageFaceDetect(photoObject, faceNum, params):

    photoPrimaryKey = photoObject.primaryKey
    storedFaces = photoObject.faces
    photoWidth = photoObject.photoWidth
    photoHeight = photoObject.photoHeight

    num_detected_faces = np.random.random_integers(1, 6)

    faces = []

    logging.warn("imageFaceDetect is not implemented yet! This is for trial only.")

    for i in range(num_detected_faces):
        face_vec = np.random.rand(128)
        height = photoHeight / 10
        width = photoWidth / 10
        centerX = np.random.random_integers(width, photoWidth - width)
        centerY = np.random.random_integers(height, photoHeight - height)

        roi_rectangle = Rectangle(height, width, centerX=centerX, centerY=centerY)

        faceNum, faceDBStruct = processChip(roi_rectangle, photoObject, faceNum, params)

        logging.warn('Best_guesses and chip_saving are not implemented here.')
        faces.append(faceDBStruct)

    return faces

def classifyFaces(faceVector):
    logging.warn("classifyFaces currently returns a random top-5 name vector")
    possibles = ['Larry', 'Moe', 'Curly', 'Joey', 'Freddy', 'Liz', 'Joan', 'Chelsea', 'Barb', 'Kathy', \
    'Ben', 'Jess', 'Jude', 'Karen', 'Leia', 'Luke', 'Anakin', 'Padme']
    random.shuffle(possibles)
    return possibles[0:5]

def imageFaceDetect(image_path):
    assert isinstance(image_path, str)
    npImage = cv2.imread(image_path)
    height, width, channels = npImage.shape
    print(height)
    print(width)
    print(np.sqrt(float(height) * width / 4e6))

    # print(resizeFactor)
    # print(downsizedArray.shape)
    print("CHECK THE ORIENTATION LOADING")
    maxSize = 5e5 # pixels

    # Either downsize or upsize the image, based on
    # the total size of the image. This helps get 
    # a constant image size. 
    if height * width > maxSize:
        resizeFactor = 1.0 / np.sqrt(float(height) * width / maxSize)
        downsizedArray = cv2.resize(npImage, None, fx=resizeFactor, fy=resizeFactor, interpolation=cv2.INTER_CUBIC)
        print(downsizedArray.shape)
        face_recognition.face_locations(downsizedArray, number_of_times_to_upsample=2, model='cnn') 
    else:
        # Upsize...
        resizeFactor = 1.0 / np.sqrt(float(height) * width / maxSize)
        downsizedArray = scipy.misc.imresize(npImage, resizeFactor, interp='bilinear', mode=None)        
        face_recognition.face_locations(downsizedArray, number_of_times_to_upsample=2, model='cnn') 
    
    areas = []
    loc = face_recognition.face_locations(downsizedArray, model='cnn') 
    locBig = []
    # print(loc)
    for face in loc:
        # print(face)
        top = face[0]
        right = face[1]
        bottom = face[2]
        left = face[3]
        width = right - left
        height = bottom - top


        upsize = 1 / resizeFactor
        centerX = int(left + width / 2.0) * upsize
        centerY = int(top + height / 2.0) * upsize
        upWidth = width * upsize
        upHeight = height * upsize

        upleft = int(centerX - 0.5 * upWidth)
        upright = int(centerX + 0.5* upWidth)
        uptop = int(centerY - 0.5 * upHeight)
        upbottom=int(centerY+ 0.5 * upHeight)

        upLoc = (uptop, upright, upbottom, upleft)
        locBig.append(upLoc)

        rect = Rectangle(height=height, width=width, leftEdge = left, topEdge = top)
        rect.resize( 1 / resizeFactor )
        # rect.expand(0.2, 0.2)
        areas.append(rect)

        assert abs(rect.topLeft.x - left) < 2
        assert abs(rect.topLeft.y - top) < 2

    identity = face_recognition.face_encodings(downsizedArray, known_face_locations=locBig, num_jitters=3)
    # print(identity)
    assert len(identity) == len(locBig)
    assert len(identity) == len(areas)

    return zip(areas, identity)

# # image = face_recognition.load_image_file('/home/lewis/test_imgs/DSC_9857.JPG')
# # # loc = face_recognition.face_locations(image, model='cnn')

# if __name__ == '__main__':
# # # print(loc)
#     from photoLoader import photo
#     file = '/home/lewis/test_imgs/2018-03-23 21.20.20-9.jpeg'

#     # image = face_recognition.load_image_file(file)
#     # areas = imageFaceDetect(image)       
#     # cvImg = cv2.imread(file)
#     # for rect in areas:
#     #     rect.drawOnPhoto(cvImg, colorTriple=(0,255,0))


#     # img = cv2.resize(cvImg, (cvImg.shape[1] / 6, cvImg.shape[0]/ 6))
#     # cv2.imshow('img',img)
#     # cv2.waitKey(0)
#     # exit()

#             #/home/lewis/test_imgs/2018-03-29 09.50.44-8.jpeg
#         # /home/lewis/test_imgs/DSC_9839.JPG -- small face
#         # /home/lewis/test_imgs/DSC_9836.JPG -- overlapping faces
#         # /home/lewis/test_imgs/2018-03-29 09.50.44-3.jpeg - small faces 
#         # /home/lewis/test_imgs/2018-03-26 07.21.44.jpeg - side face


#     files = ['/home/lewis/test_imgs/2018-03-29 09.50.44-8.jpeg',
#         '/home/lewis/test_imgs/DSC_9839.JPG',
#         '/home/lewis/test_imgs/DSC_9836.JPG',
#         '/home/lewis/test_imgs/2018-03-29 09.50.44-3.jpeg',
#         '/home/lewis/test_imgs/2018-03-26 07.21.44.jpeg']

#     for root, dirname, files in os.walk(test_photo_dir, topdown=False):

#         # print(files[0])
#         # print(dirname)
#         for fname in files:
#             this_img = os.path.join(root,fname)
#     # if True:
#             print(this_img)
#             # this_img = fname
#             cvImg = cv2.imread(this_img)
#             ph = photo(this_img, xmlParamsFile)
#             if os.path.exists('/tmp/{}.JPG'.format(ph.hash)):
#                 continue
#             stored_faces = ph.extract_stored_faces()
#             print(len(stored_faces))
#             for j in range(len(stored_faces)):
#                 rect = stored_faces[j]['rectangle']
#                 rect.drawOnPhoto(cvImg, colorTriple=(0,255,0))
                
#             # image = face_recognition.load_image_file(this_img)
#             image = np.asarray(cvImg)
#             face_locations = imageFaceDetect(image)  #face_recognition.face_locations(image, model='cnn')
#             # face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=2, model='hog')
#             print(face_locations)
#             for rect in face_locations:
#                 # top = face[0]
#                 # right = face[1]
#                 # bottom = face[2]
#                 # left = face[3]
#                 # width = right - left
#                 # height = bottom - top
#                 # rect = Rectangle(height=height, width=width, leftEdge = left, topEdge = top)
#                 rect[0].drawOnPhoto(cvImg)

#             # encodings = face_recognition.face_encodings(image)
#             # for i in encodings:
#             #     print(i)


#             img = cv2.resize(cvImg, (cvImg.shape[1] / 6, cvImg.shape[0]/ 6))
#             cv2.imwrite('/tmp/{}.JPG'.format(ph.hash), img)
#             # cv2.imshow('img',img)
#             # cv2.waitKey(0)


if __name__ == "__main__":
    for root, dirs, files in os.walk(test_photo_dir):
        for f in files:
            print(os.path.join(root, f))

'''