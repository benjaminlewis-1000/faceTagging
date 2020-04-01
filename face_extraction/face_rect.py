#! /usr/bin/env python

import numpy as np
import hashlib
import face_recognition
from PIL import Image, ExifTags
import cv2
import matplotlib.pyplot as plt
try:
    from .rectangle import Rectangle
except ImportError:
    from rectangle import Rectangle

class FaceRect:
    def __init__(self, rectangle, face_image, detection_level, encoding = None, name=None, square_face = None):
        self.rectangle = rectangle
        self.encoding = encoding
        self.name = name
        self.face_image_nonrect = face_image
        self.detection_level = detection_level
        self.square_face = square_face

        self.square_top = None

    def __rotate_chip__(self, file_path, image_chip):

        image = Image.open(file_path)
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break

        if 'items' in dir(image._getexif()):
            exif=dict(image._getexif().items())
        else:
            return image_chip

        if not orientation in exif.keys():
            return image_chip            

        if exif[orientation] == 3:
            # Rotate 180
            image_chip=np.rot90(image_chip, 2) # image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            # Rotate right
            image_chip=np.rot90(image_chip, 3) # image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            # Rotate left
            image_chip=np.rot90(image_chip, 1) # image.rotate(90, expand=True)

        return image_chip

    # def reconstruct_square_face(self, pristine_img_path):
    #     if self.square_top is None:
    #         return
    #     pristine_img = face_recognition.load_image_file(pristine_img_path)
    #     pristine_img = self.__rotate_chip__(pristine_img_path, pristine_img)
    #     square_img = pristine_img[self.square_top:self.square_bot, self.square_left:self.square_right]

    #     self.square_face = square_img # self.__rotate_chip__(pristine_img_path, square_img)


    def reconstruct_nonrect_face(self, img_path):
        img = cv2.imread(img_path)
        pristine_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # pristine_img = face_recognition.load_image_file(pristine_img_path)
        # pristine_img = self.__rotate_chip__(pristine_img_path, pristine_img)
        r = self.rectangle
        # Center X and Y got screwed up somewhere, so fixing that.
        r.centerX = (r.right + r.left) // 2
        r.centerY = (r.bottom + r.top) // 2

        self.face_image_nonrect = pristine_img[r.top:r.bottom, r.left:r.right]

        self.add_square_face(pristine_img)

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
        if self.encoding is not None:
            enc_fragment = "{}...".format(self.encoding[:5])
        else:
            enc_fragment = '<no encoding>'
        if self.face_image_nonrect is not None:
            img_size = self.face_image_nonrect.shape
        else:
            img_size = "N/A"

        if self.square_face is not None:
            sq_size = self.square_face.shape
        else:
            sq_size = "N/A"
        return "rectangle = {}, name = {}, encoding = {}, img_size = {}, sq_img_size = {}".\
            format(self.rectangle, self.name, enc_fragment, img_size, sq_size)

    def __repr__(self):
        if self.encoding is not None:
            enc_fragment = "{}...".format(self.encoding[:5])
        else:
            enc_fragment = '<no encoding>'
        if self.face_image_nonrect is not None:
            img_size = self.face_image_nonrect.shape
        else:
            img_size = "N/A"

        if self.square_face is not None:
            sq_size = self.square_face.shape
        else:
            sq_size = "N/A"
        return "rectangle = {}, name = {}, encoding = {}, img_size = {}, sq_img_size = {}".\
            format(self.rectangle, self.name, enc_fragment, img_size, sq_size)

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

    def add_square_face(self, pristine_img):
        
        im_h, im_w, _ = pristine_img.shape
        rect = self.rectangle

        # Double-checking centers -- they got screwed up somewhere. 
        # Center X and Y got screwed up somewhere, so fixing that.
        rect.centerX = (rect.right + rect.left) // 2
        rect.centerY = (rect.bottom + rect.top) // 2

        # Get a square face image as well. 
        square_size = np.max(( rect.width, rect.height ))
        if square_size > im_w:
            square_size = im_w
        if square_size > im_h:
            square_size = im_h

        half_size = square_size // 2

        square_left = rect.centerX - half_size
        square_right = rect.centerX + half_size
        if square_left < 0:
            square_left = 0
            square_right = square_size
        elif square_right > im_w:
            square_right = im_w
            square_left = square_right - square_size

        square_top = rect.centerY - half_size
        square_bot = rect.centerY + half_size
        if square_top < 0:
            square_top = 0
            square_bot = square_size
        elif square_bot > im_h:
            square_bot = im_h
            square_top = square_bot - square_size

        self.square_top = int(square_top)
        self.square_bot = int(square_bot)
        self.square_left = int(square_left)
        self.square_right = int(square_right)

        square_img = pristine_img[self.square_top:self.square_bot, self.square_left:self.square_right]

        sq_h, sq_w, ch = square_img.shape
        assert ch == 3
        assert np.abs(sq_h - sq_w) <= 2
        assert sq_h > 0

        self.square_face = square_img