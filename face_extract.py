#! /usr/bin/env python

import face_recognition 
import os
import cv2
import numpy as np
import scipy.misc
import itertools
import face_extraction
import coloredlogs
import random
import time
import logging
import io
import xmltodict
import matplotlib.pyplot as plt

coloredlogs.install()

params1 = {'upsample': 2, 'height': 600, 'width': 300}
path_to_script = os.path.dirname(os.path.realpath(__file__))

# def detect_and_merge_faces(image_path, parameter_file='parameters.xml'):
#     assert isinstance(image_path, str)
#     assert os.path.isfile(parameter_file)

#     with open(parameter_file, 'r') as fh:
#         parameters = xmltodict.parse(fh.read())

#     ml_detected_faces, tagged_faces = extract_faces_from_image(image_path, parameters)

#     matched = associate_detections_and_tags(image_path, ml_detected_faces, tagged_faces, disp_photo=False, test=test_bigface)


def extract_faces_from_image(image_path, parameters):
    assert isinstance(image_path, str) or isinstance(image_path, io.BytesIO)
    if isinstance(image_path, str):
        assert os.path.isfile(image_path)

    assert 'params' in parameters.keys()
    assert 'tiled_detect_params' in parameters['params']

    tiled_params = parameters['params']['tiled_detect_params']

    npImage = face_recognition.load_image_file(image_path)

    success_faces, tagged_faces = face_extraction.Get_XMP_Faces(image_path)

    ml_detected_faces = face_extraction.detect_pyramid(npImage, tiled_params)

    assert success_faces, 'Picasa face extraction failed.'

    matched_faces = associate_detections_and_tags(image_path, ml_detected_faces, tagged_faces, disp_photo=False, test=False)

    return ml_detected_faces, tagged_faces, matched_faces

def join_faces(pristine_image, tag_face, det_face=None):
    # This is used either to turn a tagged face into
    # a FaceRect, or to merge a detected and a 
    # tagged face.
    assert isinstance(tag_face, dict)
    assert 'bounding_rectangle' in tag_face.keys()
    assert 'Name' in tag_face.keys()
    assert tag_face['Name'] is not None
    if det_face is not None:
        assert isinstance(det_face, face_extraction.FaceRect)

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

    fr = face_extraction.FaceRect(rect, face_image, detection_level, encoding, name)

    return fr

def associate_detections_and_tags(image_path, detected_faces, tagged_faces, disp_photo=False, test=False):

    if len(detected_faces) > 0:
        for df in detected_faces:
            assert isinstance(df, face_extraction.FaceRect)

    if len(tagged_faces) > 0: 
        for tf in tagged_faces:
            assert isinstance(tf, dict)
            assert 'Name' in tf.keys()
            assert 'bounding_rectangle' in tf.keys()


    image = face_recognition.load_image_file(
        image_path)
    pristine_image = face_recognition.load_image_file(
        image_path)

    if test:
        huge_face = {'Name': "nada", 'bounding_rectangle': face_extraction.Rectangle(int(image.shape[0] * .95), int(image.shape[1] * .95), centerX = image.shape[1]// 2, centerY  = image.shape[0] // 2)}
        tagged_faces.append(huge_face)

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
            # Expand the rectangle +- 10 pixels in each direction
            rr.left = rr.left - 10
            rr.top = rr.top - 10
            rr.width = rr.width + 20
            rr.height = rr.height + 20
            df.rectangle.drawOnPhoto(draw_image,colorTriple=(235,189,52))

            font                   = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (rr.left, rr.bottom)
            fontScale              = 2
            fontColor              = (255,255,255)
            lineType               = 4


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

    for df in fully_matched:
        r = df.rectangle
        cut_rectangle = pristine_image[r.top:r.bottom, r.left:r.right]
        assert(df.image.shape == cut_rectangle.shape)
        df.image = cut_rectangle
    
    return fully_matched

def _merge_detected_faces(list_of_detects, pristine_image):


    def cluster_and_merge(detection_list, pristine_image):
        # This is a recursive algorithm. It takes a list of 
        # detections that haven't been merged together and 
        # merges them. It has two parts -- group detections 
        # into clusters that have *any* IOU, and then determine
        # which rectangles in those clusters belong together. 

        clusters = []
        deconflicted_detections = []
        touched = [False] * len(detection_list)

        for j in range(len(detection_list)):
            # We use the 'touched' vector to determine
            # if a rectangle has been included in a group
            # already. It's fine for a rectangle to be in 
            # multiple groups, but multiple groups should 
            # not be the same sets of rectangles. 
            if touched[j]:
                continue
            # Set to touched, and get the current face
            # to compare to. However, we don't remove it 
            # from the list of comparisons. For example,
            # two different root faces can touch the 
            # same rectangle without touching each other,
            # so the same rectangle can be in two clusters. 
            touched[j] = True
            current_face = detection_list[j]
            current_cluster = [current_face]
            # Comparison. If IOU > 0 for any other rectangle,
            # add it to the current cluster. Set that 
            # rectangle to be touched so it doesn't in
            # turn become a root rectangle. We assume that 
            # it will either be joined to another rectangle
            # or it will not and pass to the next iteration.
            for k in range(len(detection_list)):
                if k != j:
                    cmp_face = detection_list[k]
            
                    iou = current_face.rectangle.IOU(cmp_face.rectangle)
                    if iou > 0: 
                        touched[k] = True
                        current_cluster.append(cmp_face)
            current_cluster = tuple(current_cluster)
            clusters.append(current_cluster)

        # Assert that our clusters aren't the same. I tested
        # this assertion, it works. 
        assert len(set(clusters)) == len(clusters)

        # Keep track of rectangles that have been merged together.
        merged_in = [False] * len(detection_list)

        for each_cluster in clusters:
            # More assertions. 
            for i in range(len(each_cluster)):
                for j in range(len(each_cluster)):
                    if i != j:
                        assert each_cluster[i].rectangle != each_cluster[j].rectangle

            if len(each_cluster) == 1:
                # Cluster all by itself -- easy. Append to
                # deconflicted, call it merged, move on. 
                deconflicted_detections.append(each_cluster[0])
                idx = detection_list.index(each_cluster[0])
                merged_in[idx] = True
            else:
                cluster_cp = list(each_cluster)

                # Get the first thing in the cluster, and mark
                # it as merged in. 
                current_root = cluster_cp.pop(0)
                idx = detection_list.index(current_root)
                merged_in[idx] = True

                # Get the mergable calculation for all
                # items in the cluster. The item being
                # compared has already been removed from
                # cluster_cp.
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
                            # root and pop it out. Also
                            # mark it as merged in.
                            to_merge = cluster_cp.pop(m_idx)
                            idx = detection_list.index(to_merge)
                            merged_in[idx] = True
                            current_root = current_root.merge_with(to_merge, pristine_image)

                # Check if this rectangle is already in 
                # the list of merged detections. This shouldn't
                # be necessary, but it's a good check.
                is_in_list = False
                for det in deconflicted_detections:
                    if det.rectangle == current_root.rectangle:
                        is_in_list = True
                # Put the current root in the deconflicted list
                if not is_in_list:
                    deconflicted_detections.append(current_root)

        # Find things that aren't merged yet (not enough overlap
        # with a root) and put them in another list that 
        # will be passed to the next iteration of this 
        # function. 
        yet_unmerged = []
        for t in range(len(merged_in)):
            tch = merged_in[t]
            if not tch:
                yet_unmerged.append(detection_list[t])

        return deconflicted_detections, yet_unmerged

    deconflicted_detections = []

    iters = 0
    while(len(list_of_detects) > 0):
        iters += 1 
        assert iters < 50

        # Repeated call of cluster_and_merge until
        # there are no more to merge. 
        sub_deconf, list_of_detects = cluster_and_merge(list_of_detects, pristine_image)
        # Add to the master list. 
        deconflicted_detections += sub_deconf

    # Assertions.
    for i in range(len(deconflicted_detections)):
        for j in range(len(deconflicted_detections)):
            if i != j:
                assert deconflicted_detections[i].rectangle != deconflicted_detections[j].rectangle

    return deconflicted_detections


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
        assert isinstance(m, face_extraction.FaceRect)
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


    merged = [False] * len(merged_detections)

    area_cmp_thresh = 5

    for cluster in detection_tag_clusters:
        tag, det_list = cluster
        # print(tag)
        ious = [tag['bounding_rectangle'].IOU(x.rectangle) for x in det_list]
        intersect_thresh = 0.5
        norm_intersections = [tag['bounding_rectangle'].intersect(x.rectangle) / min(x.rectangle.area, tag['bounding_rectangle'].area) for x in det_list]
        int_over_thresh = [x > intersect_thresh for x in norm_intersections]

        # print(len(int_over_thresh))
        # print(norm_intersections)

        # Sub-case 1: only one intersection at all
        if len(det_list) == 1:
            area_cmp = tag['bounding_rectangle'].area / det_list[0].rectangle.area
            if area_cmp > area_cmp_thresh:
                matched_face_rects.append(join_faces(pristine_image, tag))
                continue

            if int_over_thresh[0] > 0.3:
                joint = join_faces(pristine_image, tag, det_list[0])
                matched_face_rects.append(joint)
                deconflict_idx = merged_detections.index(det_list[0])
                merged[deconflict_idx] = True
        else:
            # Sub-case 2: only one intersection over
            # the threshold
            if int_over_thresh.count(True) == 1:
                idx = int_over_thresh.index(True)
                isect_face = det_list[idx]

                area_cmp = tag['bounding_rectangle'].area / det_list[idx].rectangle.area
                if area_cmp > area_cmp_thresh:
                    matched_face_rects.append(join_faces(pristine_image, tag))
                    continue

                joint = join_faces(pristine_image, tag, isect_face)
                matched_face_rects.append(joint)
                # Have to consider that the other
                # overlaps may not belong to any
                # tagged face. 
                deconflict_idx = merged_detections.index(isect_face)
                merged[deconflict_idx] = True

            # Sub-case 3: multiple normalized intersections
            # that are over the threshold level.
            # Get the one that is closest to the center
            # of the tag and call the other one a 
            # separate face. 
            else:
                # print("Here")
                # Reject tagged faces that are huge
                area_cmp = [tag['bounding_rectangle'].area / x.rectangle.area for x in det_list]
                # Compute distances to each overlapping
                # face 
                distances = [tag['bounding_rectangle'].distance(x.rectangle)[0] for x in det_list]
                idx = np.argmin(distances)
                if area_cmp[idx] > area_cmp_thresh:
                    matched_face_rects.append(join_faces(pristine_image, tag))
                    continue

                closest_face = det_list.pop(idx)
                joint = join_faces(pristine_image, tag, closest_face)
                deconflict_idx = merged_detections.index(closest_face)
                merged[deconflict_idx] = True

                matched_face_rects.append(joint)
                # for m in det_list:
                #     deconflict_idx = merged_detections.index(m)
                #     touched[deconflict_idx] = True

    for idx in range(len(merged_detections)):
        if not merged[idx]:
            untouch_rect = merged_detections[idx]
            matched_face_rects.append(untouch_rect)

    return matched_face_rects
