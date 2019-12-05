#! /usr/bin/env python

import cv2
import numpy as np
import itertools
from .rectangle import Rectangle 
from .face_rect import FaceRect
import time
import face_recognition

# For my particular GPU, I'm finding that I can upsize by 3x for a 280 * 420 image,
# or by 2x for a 350 * 530 image. 
# Experiments TBD on CPU

# params1 = {'upsample': 2, 'height': 600, 'width': 800}
# params2 = {'upsample': 3, 'height': 280, 'width': 420}


def split_range(range_len, num_parts):
    # Given a range, split it into a number of approximately equal parts. 
    # The list will be of length num_parts + 1, 
    # for ease of indexing. 

    avg_len = float(range_len) / num_parts

    val_list = []
    for i in range(num_parts):
        val_list.append( int( i * avg_len ) )

    val_list.append(range_len)

    return val_list

def detect_pyramid(cv_image, parameters):

    # Do a pyramidal detection on an image using the face_recognition
    # library. I found that my (very old) GPU didn't handle the large images 
    # very well, and downscaling tends to miss small faces. So I do 
    # a multi-part detection pyramid - detect at the top level, then
    # split into a 3x3 grid and detect at each of those levels. The 
    # faces are then fused between detects in another script. 

    assert isinstance(cv_image, np.ndarray)
    assert 'upsample' in parameters.keys()
    assert 'height' in parameters.keys()
    assert 'width' in parameters.keys()

    start_time = time.time()
    max_pixels_per_chip = int(parameters['height']) * int(parameters['width'])
    num_upsamples = int(parameters['upsample'])

    # Downsample the image so that it has the number of 
    # pixels defined by height and width in the parameters.
    # This is dynamic to preserve aspect ratio. This should
    # make it possible to fit in the GPU and we can get the biggest,
    # hardest-to-miss faces.
    height = cv_image.shape[0]
    width = cv_image.shape[1]

    num_pixels = height * width
    num_chips = float(num_pixels / max_pixels_per_chip)
    num_iters = np.sqrt(num_chips)

    num_faces = 0
    
    faceList = []

    # A measure of how much we want the sub-images to overlap,
    # where 1 is 100% overlap.
    pct_exp = 0.06

    # Cut the image in a 3x3 grid, using split_range. Then we will
    # expand these even cuts slightly on top of each other to catch
    # faces that are on the border between the grids. 
    for cuts in [1, 3]: 
        # Get lists of left/right and top/bottom indices. 
        width_parts = split_range(width, cuts)
        height_parts = split_range(height, cuts)

        # Expansion of the borders by a few percent. 
        width_x_percent = int(pct_exp * width / cuts )
        height_x_percent = int(pct_exp * height / cuts )

        for leftIdx in range(len(width_parts) - 1):
            for topIdx in range(len(height_parts) - 1):

                # Get the top/bottom, left/right of each
                # grid. 
                left_edge_0 = width_parts[leftIdx]
                right_edge_0 = width_parts[leftIdx + 1]
                top_edge_0 = height_parts[topIdx]
                bottom_edge_0 = height_parts[topIdx + 1]

                # Since the faces may be split on an edge,
                # put in a pct_exp% overlap between tiles.
                # Also have logic for only going to the 
                # edge of the image. 
                left_edge = max(0, left_edge_0 - width_x_percent)
                top_edge = max(0, top_edge_0 - height_x_percent)
                right_edge = min(width, right_edge_0 + width_x_percent)
                bottom_edge = min(height, bottom_edge_0 + height_x_percent)

                assert left_edge < right_edge
                assert top_edge < bottom_edge

                assert (bottom_edge - top_edge) <= int((bottom_edge_0 - top_edge_0) * (1 + pct_exp * 2) + 1)
                assert (right_edge - left_edge) <= int((right_edge_0 - left_edge_0) * (1 + pct_exp * 2) + 1)

                # Cut out the chip. 
                chip_part = cv_image[top_edge:bottom_edge, left_edge:right_edge]

                # Then resize it to fit in the GPU memory, based 
                # on the parameters passed to the function.
                height_chip = chip_part.shape[0]
                width_chip = chip_part.shape[1]
                pixels_here = height_chip * width_chip
                resize_ratio = np.sqrt(float(pixels_here) / max_pixels_per_chip)

                resized_chip = cv2.resize(chip_part, \
                    ( int( width_chip / resize_ratio ), \
                      int( height_chip / resize_ratio ) ) )

                # Detect the locations of the faces in a given chip
                # using face_recognition's CNN model. 
                face_locations = face_recognition.face_locations(resized_chip, \
                    number_of_times_to_upsample=num_upsamples,  model='cnn')

                num_faces += len(face_locations)

                # Iterate over the detecte faces
                for index in range(len(face_locations)):
                    # Get the locations of the face from the
                    # small, resized chip. These indices will
                    # need to be scaled back up for proper 
                    # identification.
                    top_chip, right_chip, bottom_chip, left_chip = face_locations[index]

                    # While our rectangle class does have a 
                    # resize method, it wouldn't appropriately
                    # account for the shift on sub-images. 
                    # So we need to build our own. This will
                    # get the locations of the chip in the larger
                    # original image. 
                    top_scaled = int(top_chip * resize_ratio + top_edge)
                    bottom_scaled = int(bottom_chip * resize_ratio + top_edge)
                    left_scaled = int(left_chip * resize_ratio + left_edge)
                    right_scaled = int(right_chip * resize_ratio + left_edge)

                    height_face = int(np.abs(bottom_scaled - top_scaled))
                    width_face = int(np.abs(right_scaled - left_scaled))

                    face_loc_rescaled = [(top_scaled, right_scaled, bottom_scaled, left_scaled)]

                    # Get the encoding on the upscaled image 
                    # using the upsampled face bounding boxes 
                    encoding = face_recognition.face_encodings(cv_image, known_face_locations=face_loc_rescaled, num_jitters=10)
                    assert len(encoding) == 1
                    encoding = encoding[0]

                    # Draw a rectangle on the image if desired. 
                    # cv2.rectangle(cv_image, (left_scaled, top_scaled), (right_scaled, bottom_scaled), (0, 255, 0), 5)
                    # pil_image = Image.fromarray(face_image)
                    # pil_image.show()

                    face_img = cv_image[top_scaled:bottom_scaled, left_scaled:right_scaled]

                    face_loc_rect = Rectangle(height_face, width_face, leftEdge = left_scaled, topEdge = top_scaled)

                    face = FaceRect(rectangle = face_loc_rect, face_image = face_img, encoding = encoding, name=None, detection_level = cuts)
                    # Append the face to the list. No effort to de-duplicate
                    # has been made yet -- that's in another script. 
                    faceList.append(face)

    elapsed_time = time.time() - start_time
    print("Elapsed time is : " + str( elapsed_time ) )

    # Drawing function 
    # for eachFace in list(set(faceList)):
    #     r = eachFace.rectangle
    #     left = r.left
    #     right = r.right
    #     top = r.top
    #     bottom = r.bottom
    #     cv2.rectangle(cv_image, (left, top), (right, bottom), (255, 0, 0), 5)
        
    # Convert to OpenCV colors, get a resized window, and show image. 
#    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
#    cv2.namedWindow('Resized Window', cv2.WINDOW_NORMAL)
#    cv2.resizeWindow('Resized Window', 800, 600)
#    cv2.imshow('Resized Window', cv_image)
#    cv2.waitKey(0)

    return faceList, elapsed_time



if __name__ == "__main__":
    path = '/home/benjamin/Desktop/photos_for_test/train/B+J-36wedding.jpg'

