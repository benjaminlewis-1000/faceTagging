#! /usr/bin/env python

import math
import cv2 as cv
import os
import numpy as np
import xmltodict

path_to_script = os.path.dirname(os.path.realpath(__file__))

class rectangleError(Exception):
    pass

class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def isLeftOf(self, inputPoint):
        return self.x <= inputPoint.x

    def isAbove(self, inputPoint):
        return self.y <= inputPoint.y

    def horizAlign(self, inputPoint):
        return self.y == inputPoint.y

    def vertAlign(self, inputPoint):
        return self.x == inputPoint.x

class Rectangle():

    def __init__(self, height, width, **kwargs): 

        self.centerX = kwargs['centerX'] if 'centerX' in kwargs else None
        self.centerY = kwargs['centerY'] if 'centerY' in kwargs else None
        self.leftEdge = kwargs['leftEdge'] if 'leftEdge' in kwargs else None
        self.topEdge = kwargs['topEdge'] if 'topEdge' in kwargs else None
        self.width = width
        self.height = height
        
        if self.centerX is not None and self.centerY is not None and self.leftEdge is not None and self.topEdge is not None:
            raise rectangleError('A center point or a bottom point must be defined.')
        if (self.centerX is not None) != (self.centerY is not None):
            raise rectangleError('centerX and centerY must both be defined in init of Rectangle class.')
        if (self.leftEdge is not None) != (self.topEdge is not None):
            raise rectangleError('leftEdge and topEdge must both be defined in init of Rectangle class.')
        if (self.centerX is not None) == (self.topEdge is not None):
            raise rectangleError('A bottom point and a center point may not both be defined.')

        halfWidthLeft = math.floor(width / 2.0)
        halfWidthRight = math.ceil(width / 2.0)
        halfHeightUp = math.floor(height / 2.0)
        halfHeightDown = math.ceil(height / 2.0)

        assert halfWidthRight + halfWidthLeft == width, 'Didn''t do math right on width.'
        assert halfHeightUp + halfHeightDown == height, 'Didn''t do math right on height.'

        if self.centerX: # We have a center point
            self.centerPoint = Point(self.centerX, self.centerY)
            self.bottomLeft = Point(self.centerX - halfWidthLeft, self.centerY + halfHeightUp)
            self.bottomRight = Point(self.centerX + halfWidthRight, self.centerY + halfHeightUp)
            self.topLeft = Point(self.centerX - halfWidthLeft, self.centerY - halfHeightDown)
            self.topRight = Point(self.centerX + halfWidthRight, self.centerY - halfHeightDown)
        else:
            self.topLeft = Point(self.leftEdge, self.topEdge)
            self.centerPoint = Point(self.leftEdge + halfWidthLeft, self.topEdge + halfHeightDown)
            self.bottomLeft = Point(self.leftEdge, self.topEdge + self.height)
            self.bottomRight = Point(self.leftEdge + self.width, self.topEdge + self.height)
            self.topRight = Point(self.leftEdge + self.width, self.topEdge)

        self.right = int(self.topRight.x)
        self.left = int(self.topLeft.x)
        self.top = int(self.topLeft.y)
        self.bottom = int(self.bottomLeft.y)
        self.centerX = int(self.centerPoint.x)
        self.centerY = int(self.centerPoint.y)

        assert self.topLeft.vertAlign(self.bottomLeft)
        assert self.topRight.vertAlign(self.bottomRight)

        assert self.topLeft.horizAlign(self.topRight)
        assert self.bottomLeft.horizAlign(self.bottomRight)

        assert self.topLeft.isLeftOf(self.topRight)
        assert self.topLeft.isAbove(self.bottomLeft)

        assert self.centerPoint.isLeftOf(self.topRight)
        assert self.topLeft.isLeftOf(self.centerPoint)

        assert self.centerPoint.isAbove(self.bottomRight)
        assert self.topLeft.isAbove(self.centerPoint)

        self.area = width * height

    def resize(self, resizeFactor):

        if resizeFactor == 0:
            raise ValueError('Resize value is 0')
        self.width = self.width * resizeFactor
        self.height = self.height * resizeFactor
        self.centerX = int(self.centerX * resizeFactor)
        self.centerY = int(self.centerY * resizeFactor)

        self.left = int(self.centerX - 0.5 * self.width)
        self.right = int(self.centerX + 0.5* self.width)
        self.top = int(self.centerY - 0.5 * self.height)
        self.bottom=int(self.centerY+ 0.5 * self.height)

    def intersect(self, otherRectangle):
        # Find the number of pixels that overlap between two rectangles. 

        otherRight = otherRectangle.right
        otherLeft = otherRectangle.left
        otherTop = otherRectangle.top
        otherBottom = otherRectangle.bottom

        x_overlap = max(0, min(self.right, otherRight) - max(self.left, otherLeft));
        y_overlap = max(0, min(self.bottom, otherBottom) - max(self.top, otherTop));

        overlapArea = x_overlap * y_overlap;

        return overlapArea

        # r1_percent = overlapArea / self.area
        # r2_percent = overlapArea / otherRectangle.area

        # overlap_dict = dict()
        # overlap_dict['area'] = overlapArea
        # overlap_dict['r1_percent'] = r1_percent
        # overlap_dict['r2_percent'] = r2_percent
        # return overlap_dict

    def union(self, otherRectangle):
        return self.area + otherRectangle.area - self.intersect(otherRectangle)

    def IOU(self, otherRectangle):
        return float(self.intersect(otherRectangle)) / self.union(otherRectangle)

    def drawOnPhoto(self, cvImg, colorTriple=(255,0,0), lineWidth=8):
        x = int(self.left)
        y = int(self.top)
        w = int(self.width)
        h = int(self.height)
        cv.rectangle(cvImg,(x,y),(x+w,y+h),colorTriple, lineWidth)

    # def mergeWith(self, otherRectangle):
    #     ovd = self.intersect(otherRectangle)

    #     mergeDict = dict

    #     if ovd['r1_percent'] > 0 and ovd['r2_percent'] > 0:
    #         leftmost = min(self.left, otherRectangle.left)
    #         rightmost = max(self.right, otherRectangle.right)
    #         topmost = min(self.top, otherRectangle.top)
    #         bottommost = max(self.bottom, otherRectangle.bottom)

    #         height = bottommost - topmost
    #         width = rightmost - leftmost

    #         newRect = Rectangle(height, width, leftEdge = leftmost, topEdge = topmost)
    #         return newRect

    #     else:
    #         return None

    def distance(self, other_rect):
        x_dist = self.centerX - other_rect.centerX
        y_dist = self.centerY - other_rect.centerY
        dist = np.sqrt(x_dist ** 2 + y_dist ** 2)

        max_x_size = max(self.width, other_rect.width)
        max_y_size = max(self.height, other_rect.height)

        # This is very much an approximation.
        norm_dist = np.sqrt( (x_dist / max_x_size) ** 2 + (y_dist / max_y_size) ** 2 )

        return dist, norm_dist

    def expand(self, percentVertical = 0.15, percentHorizontal = 0.15):

        self.width = self.width * (1 + percentHorizontal)
        self.height = self.height * (1 + percentVertical)

        self.left = int(self.centerX - 0.5 * self.width)
        self.right = int(self.centerX + 0.5* self.width)
        self.top = int(self.centerY - 0.5 * self.height)
        self.bottom=int(self.centerY+ 0.5 * self.height)

    def __lt__(self, other):
        return self.area < other.area

    # def __eq__(self, other):
    #     return self.area == other.area

    def __gt__(self, other):
        return self.area > other.area

    def __repr__(self):
        return "Rectangle: \n  Height: {}\n  Width: {}\n  Top-left: {} x, {} y\n"\
            .format(int(self.height), int(self.width), int(self.left), int(self.top))

if __name__ == "__main__":
    passed = True
    try:
        a = Rectangle(50, 40, centerX=5, centerY=5)
        print(type(a))
    except rectangleError as e:
        passed = False
    try:
        Rectangle(50, 40)
        passed = False
    except rectangleError as e:
        pass
    try:
        Rectangle(50, 40, centerX=5)
        passed = False
    except rectangleError as e:
        pass
    try:
        Rectangle(50, 40, centerX=5, centerY=5, leftEdge=3)
        passed = False
    except rectangleError as e:
        pass
    try:
        Rectangle(50, 40, centerX=5, centerY=5, leftEdge=3, topEdge=6)
        passed = False
    except rectangleError as e:
        pass
    try:
        Rectangle(50, 40, leftEdge=3, topEdge=6)
    except rectangleError as e:
        passed = False
    try:
        Rectangle(50, 40, leftEdge=3)
        passed = False
    except rectangleError as e:
        pass

    r1 = Rectangle(20, 20, centerX = 30, centerY = 30)
    r2 = Rectangle(20, 20, centerX = 35, centerY = 35)
    ovd = r1.intersect(r2)
    assert ovd == 225
    print(ovd)
    print(r1.union(r2))

    print("IOU: " + str(r1.intersectOverUnion(r2)))

    a = Rectangle(50, 43, leftEdge=10, topEdge = 15)
    print(a)
    a.resize(2)
    print(a)
    a.resize(0.5)
    print(a)
    # merged = r1.mergeWith(r2)
    # print(merged)
    # assert merged == Rectangle(25, 25, leftEdge = 20, topEdge = 20)

    if passed:
        print("All unit tests have passed!")
    else:
        print("A unit test failed. Check this.")