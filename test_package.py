#! /usr/bin/env python

from rectangle import Rectangle, rectangleError
import unittest
import face_extract
import os
import numpy as np
import xmltodict
import get_picasa_faces
import re
import base64
import io
import pickle
# import classify_faces

path_to_script = os.path.dirname(os.path.realpath(__file__))

class rectangleTester(unittest.TestCase):
    def setUp(self):
        pass

    def test_make_rect(self):
        a = Rectangle(50, 40, centerX=5, centerY=5)
        b = Rectangle(50, 40, leftEdge=3, topEdge=6)

    def test_missing_args(self):
        with self.assertRaises(rectangleError):
            a = Rectangle(50, 40)
        with self.assertRaises(rectangleError):
            a = Rectangle(50, 40, centerX = 5)
        with self.assertRaises(rectangleError):
            a = Rectangle(50, 40, centerX=5, centerY=5, leftEdge=3)
        with self.assertRaises(rectangleError):
            a = Rectangle(50, 40, centerX=5, centerY=5, leftEdge=3, topEdge=6)
        with self.assertRaises(rectangleError):
            a = Rectangle(50, 40, leftEdge=3)


    def test_intersect(self):
        r1 = Rectangle(20, 20, centerX = 30, centerY = 30)
        r2 = Rectangle(20, 20, centerX = 35, centerY = 35)
        ovd = r1.intersect(r2)
        self.assertEqual(ovd, 225)

    def test_iou(self):
        r1 = Rectangle(20, 20, centerX = 30, centerY = 30)
        r2 = Rectangle(20, 20, centerX = 40, centerY = 30)

        self.assertNotEqual(r1.IOU(r2), 1)
        self.assertNotEqual(r1.IOU(r2), 0)
        self.assertEqual(r1.IOU(r1), 1)
        self.assertEqual(r2.IOU(r2), 1)


        r1 = Rectangle(20, 20, centerX=10, centerY = 10)
        r2 = Rectangle(20, 20, centerX=20, centerY = 10)
        self.assertEqual(r1.IOU(r2), 1/3)

        r2 = Rectangle(20, 20, centerX=50, centerY=20)
        self.assertEqual(r1.IOU(r2), 0)

    def test_resizing(self):
        a = Rectangle(50, 43, leftEdge=10, topEdge = 15)
        # print(a)
        a_size = 50 * 43
        a.resize(2)
        self.assertEqual(4*a_size, a.height*a.width)
        a.resize(0.5)
        self.assertEqual(a_size, a.height*a.width)
        with self.assertRaises(ValueError):
            a.resize(0)

    # def test_distance(self):
    #     raise NotImplementedError

    # def test_merge(self):
    #     r1 = Rectangle(20, 20, centerX = 30, centerY = 30)
    #     r2 = Rectangle(20, 20, centerX = 35, centerY = 35)
    #     # print(a)
    #     merged = r1.mergeWith(r2)
    #     print(merged)
    #     self.assertEqual(merged, Rectangle(25, 25, leftEdge = 20, topEdge = 20))



class FaceExtractTester(unittest.TestCase):
    def setUp(self):
        self.test_photo_dir = os.path.join(path_to_script, 'test_imgs')
        self.photos_list = []
        for root, dirs, files in os.walk(self.test_photo_dir):
            for f in files:
                if f.lower().endswith(('.jpeg', '.jpg')):
                    self.photos_list.append(os.path.join(root, f))


        parameter_file='parameters.xml'
        with open(parameter_file, 'r') as fh:
            self.parameters = xmltodict.parse(fh.read())


    def __image_preprocess__(self, photo_file):
        return photo_file, photo_file

    def test_one_photo_facedetect(self):
        # for photo in self.photos_list:
        for photo in self.photos_list[1:]:
            print(photo)
            photo, filename = self.__image_preprocess__(photo)
            ml_faces, tagged_faces = face_extract.extract_faces_from_image(photo, self.parameters)
            if len(ml_faces) > 0:
                break
        # Assert that we at least are getting one image
        # with a detected face. 
        self.assertTrue(len(ml_faces) > 0)

    def test_extract_and_group_faces(self, redo=False):

        # for photo in self.photos_list:
        problems = [41, 65, 77, 79, 80, 83, 96, 100, 151]

        all_matches = []
        for p in range(len(self.photos_list)):
        # for p in problems:
            photo = self.photos_list[p]
            photo, filename = self.__image_preprocess__(photo)

            print(p, filename)
            out_file = re.sub('.(jpg|JPEG|JPG|jpeg)$', '.pkl', filename)
            if not os.path.isfile(out_file) or redo:
                ml_faces, tagged_faces = face_extract.extract_faces_from_image(photo, self.parameters)
                assert ml_faces is not None
                assert tagged_faces is not None
                with open(out_file, 'wb') as fh:
                    pickle.dump([ml_faces, tagged_faces], fh)
            else:
                with open(out_file, 'rb') as fh:
                    ml_faces, tagged_faces = pickle.load(fh)

            test_bigface = False
            num_faces_file = re.sub('.(jpg|JPEG|JPG|jpeg)$', '_numface.pkl', filename)
            matched = face_extract.associate_detections_and_tags(photo, ml_faces, tagged_faces, disp_photo=False, test=test_bigface)

            all_matches += matched

            if test_bigface:
                pass
            else:

                if not os.path.isfile(num_faces_file):
                    with open(num_faces_file, 'wb') as fh:
                        pickle.dump([len(matched)], fh)
                else:
                    with open(num_faces_file, 'rb') as fh:
                        expected_num_faces = pickle.load(fh)[0]
                        assert expected_num_faces == len(matched)

    '''
    def test_classification(self):

        face_list_file = os.path.join(self.test_photo_dir, 'class_list.pkl')

        if not os.path.isfile(face_list_file):
            all_matches = []
            for photo in self.photos_list:
                print(photo)
                out_file = re.sub('.(jpg|JPEG|JPG|jpeg)$', '.pkl', photo)
                if not os.path.isfile(out_file):
                    ml_faces, tagged_faces = face_extract.extract_faces_from_image(photo, self.parameters)
                    assert ml_faces is not None
                    assert tagged_faces is not None
                    with open(out_file, 'wb') as fh:
                        pickle.dump([ml_faces, tagged_faces], fh)
                else:
                    with open(out_file, 'rb') as fh:
                        ml_faces, tagged_faces = pickle.load(fh)

                matched = face_extract.associate_detections_and_tags(photo, ml_faces, tagged_faces)

                all_matches += matched

            with open(face_list_file, 'wb') as fh:
                pickle.dump([all_matches], fh)

        else:
            with open(face_list_file, 'rb') as fh:
                all_matches = pickle.load(fh)[0]

        classify_faces.sort_common_faces(all_matches, num_inst_thresh = 10)
    '''

    def test_get_xmp(self):
        for photo in self.photos_list:
            photo, filename = self.__image_preprocess__(photo)
            success, saved_faces = get_picasa_faces.Get_XMP_Faces(photo)
            self.assertTrue(success)

    # def test_get_face_rotations(self):
    #     raise NotImplementedError('Should do this on GPU.')

    # Need to test XMP when image is rotated... 


class FaceExtractTesterByteIO(FaceExtractTester):

    def __image_preprocess__(self, photo_file):
        with open(photo_file, 'rb') as imageFile:
            data_str = base64.b64encode(imageFile.read())

        data = base64.b64decode(data_str)
        file = io.BytesIO(data)

        return file, photo_file


if __name__ == '__main__':
    unittest.main()