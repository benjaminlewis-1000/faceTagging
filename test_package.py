#! /usr/bin/env python

from rectangle import Rectangle, rectangleError
import unittest
from face_extract import imageFaceDetect
import os
import get_picasa_faces

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

        self.assertNotEqual(r1.intersectOverUnion(r2), 1)
        self.assertNotEqual(r1.intersectOverUnion(r2), 0)
        self.assertEqual(r1.intersectOverUnion(r1), 1)
        self.assertEqual(r2.intersectOverUnion(r2), 1)


        r1 = Rectangle(20, 20, centerX=10, centerY = 10)
        r2 = Rectangle(20, 20, centerX=20, centerY = 10)
        self.assertEqual(r1.intersectOverUnion(r2), 1/3)

        r2 = Rectangle(20, 20, centerX=50, centerY=20)
        self.assertEqual(r1.intersectOverUnion(r2), 0)

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

    # def test_merge(self):
    #     r1 = Rectangle(20, 20, centerX = 30, centerY = 30)
    #     r2 = Rectangle(20, 20, centerX = 35, centerY = 35)
    #     # print(a)
    #     merged = r1.mergeWith(r2)
    #     print(merged)
    #     self.assertEqual(merged, Rectangle(25, 25, leftEdge = 20, topEdge = 20))

import re
import pickle


class FaceExtractTester(unittest.TestCase):
    def setUp(self):
        self.test_photo_dir = os.path.join(path_to_script, 'test_imgs')
        self.photos_list = []
        for root, dirs, files in os.walk(self.test_photo_dir):
            for f in files:
                if f.lower().endswith(('.jpeg', '.jpg')):
                    self.photos_list.append(os.path.join(root, f))

    # def test_one_photo_facedetect(self):
    #     # for photo in self.photos_list:
    #     print(self.photos_list[2])
    #     ml_faces, tagged_faces = imageFaceDetect(self.photos_list[2])

    def test_extract_and_do_something(self, redo=False):
        for photo in self.photos_list:
            out_file = re.sub('.(jpg|JPEG|JPG|jpeg)$', '.pkl', photo)
            if not os.path.isfile(out_file) or redo:
                ml_faces, tagged_faces = imageFaceDetect(photo)
                assert ml_faces is not None
                assert tagged_faces is not None
                with open(out_file, 'wb') as fh:
                    pickle.dump([ml_faces, tagged_faces], fh)
            else:
                with open(out_file, 'rb') as fh:
                    ml_faces, tagged_faces = pickle.load(fh)
                    # print(ml_faces)
                    # print(tagged_faces)

    def test_get_xmp(self):
        for photo in self.photos_list:
            success, saved_faces = get_picasa_faces.Get_XMP_Faces(photo)
            self.assertTrue(success)

    # Need to test XMP when image is rotated... 

if __name__ == '__main__':
    unittest.main()