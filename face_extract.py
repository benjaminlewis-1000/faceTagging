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

def join_faces(pristine_image, tag_face, det_face=None):
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
    pristine_image = face_recognition.load_image_file(
        image_path)

    if disp_photo:
        draw_image = image.copy()
        for df in detected_faces:
            df.rectangle.drawOnPhoto(draw_image)
        for tf in tagged_faces:
            tf['bounding_rectangle'].drawOnPhoto(draw_image, colorTriple=(0,255, 0))

    merged_detections = _merge_detected_faces(detected_faces, pristine_image)

    if disp_photo:
        for d in merged_detections:
            d.rectangle.drawOnPhoto(draw_image, colorTriple=(0, 0, 255))

    fully_matched = _merge_detections_with_tags(merged_detections, tagged_faces, pristine_image)

    if disp_photo:
        for df in fully_matched:
            rr = df.rectangle.copy()
            rr.left = rr.left - 10
            rr.top = rr.top - 10
            rr.width = rr.width + 20
            rr.height = rr.height + 20
            df.rectangle.drawOnPhoto(draw_image,colorTriple=(235,189,52))

            font                   = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (rr.left, rr.bottom)
            fontScale              = 2
            fontColor              = (255,255,255)
            lineType               = 2


            if df.name is not None:
                name = df.name
            else:
                name = "None"

            cv2.putText(draw_image,name, 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)

        plt.figure()
        plt.imshow(draw_image)
        plt.show()
        plt.figure()
        plt.imshow(pristine_image)
        plt.show()

    for df in fully_matched:
        r = df.rectangle
        cut_rectangle = pristine_image[r.top:r.bottom, r.left:r.right]
        assert(df.image.shape == cut_rectangle.shape)
        df.image = cut_rectangle
    
    return fully_matched

def _merge_detections_with_tags(merged_detections, tagged_faces, pristine_image):

    matched_face_rects = []

    # Three cases:
    # 1 - Detection with no Picasa Face
    # 2 - Picasa face with no detection
    # 3 - Picasa face with overlapping detection(s).

    # Case 1:  Detection with no picasa face
    num_det = len(merged_detections)
    for d_num in range(num_det - 1, -1, -1):
        # print(d_num)
        det = merged_detections[d_num]
        have_match = False
        for tag in tagged_faces:
            iou = tag['bounding_rectangle'].IOU(det.rectangle)
            if iou > 0:
                have_match = True
                break

        if not have_match:
            # We should have removed detections that 
            # have no picasa match from the 
            # merged_detections, so we check that here.
            # Note -- these assertions are only true
            # before we expand the rectangle. 
            for tag in tagged_faces:
                assert(tag['bounding_rectangle'].IOU(det.rectangle)) == 0

            # Expand the rectangle, since chips
            # from face_detections tend to be very
            # tight around facial features. 
            det.rectangle.expand()
            r = det.rectangle
            det.image = pristine_image[r.top:r.bottom, r.left:r.right]
            matched_face_rects.append(det)
            merged_detections.pop(d_num)

    # Assertions for case 1 
    assert len(set(merged_detections).intersection(set(matched_face_rects) ) ) == 0

    # Case 2: Tagged faces with no detections
    num_tags = len(tagged_faces)
    for t_num in range(num_tags - 1, -1, -1):
        tagged = tagged_faces[t_num]
        have_match = False
        for det in merged_detections:
            if tagged['bounding_rectangle'].IOU(det.rectangle) > 0:
                have_match = True
                break

        if not have_match:
            tag_rect = join_faces(pristine_image, tagged)
            matched_face_rects.append(tag_rect)
            tagged_faces.pop(t_num)

    # Assertions for case 2
    for m in matched_face_rects:
        assert isinstance(m, face_rect.FaceRect)
    assert len(set(merged_detections).intersection(set(matched_face_rects) ) ) == 0
    # Assert that everything else has IOUs. 
    for t in tagged_faces:
        overlap = False
        for d in merged_detections:
            if d.rectangle.IOU(t['bounding_rectangle']) > 0:
                overlap = True
        assert overlap




    # Case 3. Every picasa tag left has at least some 
    # IOU relation with at least one detected tag that
    # is left.

    detection_tag_clusters = []

    for tag in tagged_faces:
        det_list = []
        for det in merged_detections:
            if tag['bounding_rectangle'].IOU(det.rectangle) > 0:
                det_list.append(det)
        tag_cluster = (tag, det_list)
        detection_tag_clusters.append(tag_cluster)

        for i in range(len(det_list)):
            for j in range(len(det_list)):
                if i != j:
                    assert det_list[i].rectangle != det_list[j].rectangle


    touched = [False] * len(merged_detections)

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
                joint = join_faces(pristine_image, tag, det_list[0])
                matched_face_rects.append(joint)
                deconflict_idx = merged_detections.index(det_list[0])
                touched[deconflict_idx] = True
        else:
            # Sub-case 2: only one intersection over
            # the threshold
            if int_over_thresh.count(True) == 1:
                idx = int_over_thresh.index(True)
                isect_face = det_list[idx]
                joint = join_faces(pristine_image, tag, isect_face)
                matched_face_rects.append(joint)
                # Have to consider that the other
                # overlaps may not belong to any
                # tagged face. 
                deconflict_idx = merged_detections.index(isect_face)
                touched[deconflict_idx] = True

            # Sub-case 3: multiple normalized intersections
            # that are over the threshold level.
            # Get the one that is closest to the center
            # of the tag and call the other one a 
            # separate face. 
            else:
                # Compute distances to each overlapping
                # face 
                distances = [tag['bounding_rectangle'].distance(x.rectangle)[0] for x in det_list]
                idx = np.argmin(distances)
                closest_face = det_list.pop(idx)
                joint = join_faces(pristine_image, tag, closest_face)
                deconflict_idx = merged_detections.index(closest_face)
                touched[deconflict_idx] = True

                matched_face_rects.append(joint)
                for m in det_list:
                    deconflict_idx = merged_detections.index(m)
                    touched[deconflict_idx] = True

    # if disp_photo:
    #     for df in matched_face_rects:
    #         rr = df.rectangle
    #         rr.left = rr.left - 20
    #         rr.top = rr.top - 20
    #         rr.width = rr.width + 40
    #         rr.height = rr.height + 40
    #         df.rectangle.drawOnPhoto(draw_image,colorTriple=(245, 66, 203))

    for idx in range(len(merged_detections)):
        if not touched[idx]:
            untouch_rect = merged_detections[idx]
        else:
            continue
        # for matched in matched_face_rects:
        #     assert matched.rectangle.IOU(untouch_rect.rectangle) < intersect_thresh
        matched_face_rects.append(untouch_rect)
        # if disp_photo:
        #     untouch_rect.rectangle.drawOnPhoto(draw_image, colorTriple=(100, 200, 150))

    return matched_face_rects

def _merge_detected_faces(detection_list, pristine_image):
    clusters = []

    touched = [False] * len(detection_list)

    for j in range(len(detection_list)):
        if touched[j]:
            continue
        touched[j] = True
        current_face = detection_list[j]
        current_cluster = [current_face]
        for k in range(len(detection_list)):
            if k != j:
                cmp_face = detection_list[k]
        
                iou = current_face.rectangle.IOU(cmp_face.rectangle)
                if iou > 0: 
                    touched[k] = True
                    current_cluster.append(cmp_face)

        xs = [x.rectangle.left for x in current_cluster]
        current_cluster = tuple(current_cluster)
        clusters.append(current_cluster)

    # Clusters have *any* IOU
    # print(len(clusters))

    contained_inside_rect_thresh = 0.8
    enc_dist_thresh = 0.4

    max_iter = 500

    deconflicted_detections = []
    for each_cluster in clusters:

        for i in range(len(each_cluster)):
            for j in range(len(each_cluster)):
                if i != j:
                    assert each_cluster[i].rectangle != each_cluster[j].rectangle

        if len(each_cluster) == 1:
            deconflicted_detections.append(each_cluster[0])
        else:
            cluster_cp = list(each_cluster)

            # This is complex -- find all the overlaps with the first
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
                current_root = cluster_cp.pop(0)
                
                # Get the value of merges for all
                # items in the cluster. The first
                # element is a merge with itself, which
                # will be true, of course. 
                merges = [x.test_merge(current_root) for x in cluster_cp]

                # If it merges with anything else, then 
                # proceed to merge. 
                if np.any(merges):
                    # Start at the end of the list and 
                    # work backward. 
                    for m_idx in range(len(cluster_cp) - 1, -1, -1):
                        if merges[m_idx]:
                            # If the given index is mergable, 
                            # then merge it with the current
                            # root and pop it out. 
                            to_merge = cluster_cp.pop(m_idx)
                            current_root = current_root.merge_with(to_merge, pristine_image)

                # Check if this rectangle is already in 
                # the list. This is an artifact of my 
                # clustering. 
                is_in_list = False
                for det in deconflicted_detections:
                    if det.rectangle == current_root.rectangle:
                        is_in_list = True
                # Put the current root in the deconflicted list
                if not is_in_list:
                    deconflicted_detections.append(current_root)

    for i in range(len(deconflicted_detections)):
        for j in range(len(deconflicted_detections)):
            if i != j:
                assert deconflicted_detections[i].rectangle != deconflicted_detections[j].rectangle

    return deconflicted_detections