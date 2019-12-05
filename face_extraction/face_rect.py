#! /usr/bin/env python

from .rectangle import Rectangle
import numpy as np
import hashlib

class FaceRect:
    def __init__(self, rectangle, face_image, detection_level, encoding = None, name=None, square_face = None):
        self.rectangle = rectangle
        self.encoding = encoding
        self.name = name
        self.image = face_image
        self.detection_level = detection_level
        self.square_face = square_face

    def __eq__(self, otherFace):
        return self.rectangle == otherFace.rectangle

    def __hash__(self):
        m = hashlib.md5()
        if self.name is not None:
            m.update(self.name.encode('utf-8'))
        if self.rectangle is not None:
            m.update(repr(self.rectangle).encode('utf-8'))
        if self.detection_level is not None:
            m.update(str(self.detection_level).encode('utf-8'))

        return int(m.hexdigest(), 16)

    def __str__(self):
        return "rectangle = {}, name = {}, encoding = {}...".\
            format(self.rectangle, self.name, \
                self.encoding[0:5])

    def __repr__(self):
        return "rectangle = {}, name = {}, encoding = {}...".\
            format(self.rectangle, self.name, \
                self.encoding[0:5])

    def __assert_cmp_to_face__(self, face):
        assert isinstance(face, FaceRect), \
          'encoding distance must be called on another FaceRect object.'
        assert self.encoding is not None, 'Encoding on both FaceRect objects must not be none.'
        assert face.encoding is not None, 'Encoding on both FaceRect objects must not be none.'
        assert len(face.encoding) == len(self.encoding), 'Length of both encodings must be equal.'

    def enc_dist(self, face):
        self.__assert_cmp_to_face__(face)

        distance = np.linalg.norm(self.encoding - face.encoding)
        # distance = np.mean(np.abs(self.encoding - face.encoding))
        return distance

    def test_merge(self, face, dist_thresh = 0.5, enc_thresh = 0.4, intersect_thresh = 0.8 ):
        # If the IOU is large, one rectangle is completely inside the other, or if the encoding is similar, then they are the same. 

        self.__assert_cmp_to_face__(face)

        enc_dist = self.enc_dist(face)
        intersect_area = self.rectangle.intersect(face.rectangle)
        this_inside_other = intersect_area / self.rectangle.area
        other_inside_this = intersect_area / face.rectangle.area
        insideness = max(this_inside_other, other_inside_this)
        # If one of the rectangles is basically the entire photo,
        # then reject it as a merge possibility. 
        too_big_insideness = min(this_inside_other, other_inside_this)
        # Calculate absolute and normalized distances between the 
        # rectangles. 
        dist, norm_dist = self.rectangle.distance(face.rectangle)

        # If one rectangle is less than 4% the size of the other
        # rectangle, then we will reject it. 
        if too_big_insideness < 0.04:
            return False

        if enc_dist < enc_thresh:
            enc_score = 1
        else:
            # We want this to taper off pretty quickly.
            enc_score = (1 - abs(enc_dist - enc_thresh) ) ** 5

        if insideness > intersect_thresh:
            intersect_score = 1
        else:
            intersect_score = (1 - abs(insideness - intersect_thresh) ) ** 1.5

        if norm_dist < dist_thresh:
            dist_score = 1
        else:
            dist_score = (1 - abs(norm_dist - dist_thresh) ) ** 5

        mergeable = False
        # Hard-and-fast true indicators: 
        if intersect_score > 0.98:
            mergeable = True
        elif enc_score > 0.95:
            mergeable = True
        elif dist_score > 0.98:
            mergeable = True
        else:
            total_score = enc_score + intersect_score + dist_score
            if total_score > 2:
                mergeable = True
        # print(enc_score, intersect_score, dist_score)

        return mergeable

    def merge_with(self, face, np_image):
        if self.detection_level < face.detection_level:
            encoding = self.encoding
        elif face.detection_level < self.detection_level:
            encoding = face.encoding
        else:
            # Not sure this is best idea... 
            if self.rectangle.area > face.rectangle.area:
                encoding = self.encoding
            else:
                encoding = face.encoding

        if self.name == face.name:
            newname = self.name
        elif self.name == None:
            newname = face.name
        else:
            newname = self.name

        detection_level = min(self.detection_level, face.detection_level)

        new_rect_top = min(self.rectangle.top, face.rectangle.top)
        new_rect_bottom = max(self.rectangle.bottom, face.rectangle.bottom)
        new_rect_left = min(self.rectangle.left, face.rectangle.left)
        new_rect_right = max(self.rectangle.right, face.rectangle.right)
        new_width = abs(new_rect_left - new_rect_right)
        new_height = abs(new_rect_top - new_rect_bottom)

        # print(new_rect_left, self.rectangle.left, face.rectangle.left)
        # print(new_rect_right, self.rectangle.right, face.rectangle.right)

        new_rect = Rectangle(new_height, new_width, leftEdge=new_rect_left, topEdge=new_rect_top)

        face_image = np_image[new_rect_top:new_rect_bottom, new_rect_left:new_rect_right]

        merge_rect = FaceRect(rectangle=new_rect, face_image=face_image, detection_level=detection_level, encoding = encoding, name=newname)

        return merge_rect
        # Need a new face rectangle...

            # TBD
        
        # We want a graceful fall-off. If < enc_thresh, then 100% for that, and so on.