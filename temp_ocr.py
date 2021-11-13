# import the necessary packages
import numpy as np
import argparse
import imutils
import glob
import cv2

# load the image image, convert it to grayscale, and detect edges
template = cv2.imread('target.PNG')
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = cv2.Canny(template, 50, 200)
(tH, tW) = template.shape[:2]
cv2.imshow("Template", template)
# loop over the images to find the template in
image = cv2.imread('target_test.PNG')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
found = None
# loop over the scales of the image
for scale in np.linspace(0.2, 1.0, 20)[::-1]:
    # resize the image according to the scale, and keep track
    # of the ratio of the resizing
    resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
    r = gray.shape[1] / float(resized.shape[1])
    # if the resized image is smaller than the template, then break
    # from the loop
    if resized.shape[0] < tH or resized.shape[1] < tW:
        break
    # detect edges in the resized, grayscale image and apply template
    # matching to find the template in the image
    edged = cv2.Canny(resized, 50, 200)
    result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
    # check to see if the iteration should be visualized
    # if we have found a new maximum correlation value, then update
    # the bookkeeping variable
    if found is None or maxVal > found[0]:
        found = (maxVal, maxLoc, r)
    # unpack the bookkeeping variable and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
    # draw a bounding box around the detected result and display the image
    print(startX, startY)
    cv2.rectangle(image, (int(startX), int(startY)), (int(endX), int(endY)), (0, 0, 255), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys
import pytesseract

pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\User\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

THRESHOLD = 50  # possible values from 0 to 255

WIDTH = 210.0
HEIGHT = 297.0
ALPHA = 0.2
RESOLUTION = 200
R = ALPHA * WIDTH
DELTA = WIDTH / RESOLUTION
R_POINT_SPACE = R / DELTA

print(R, DELTA, R_POINT_SPACE)


def convert_to_white_and_black_image(fname):
    img_rgb = cv.imread(fname)
    grayImage = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
    return grayImage


def check_point(loc, image):
    for i in range(int(loc[0] - R_POINT_SPACE), int(loc[0] + R_POINT_SPACE)):
        for j in range(int(loc[1] - R_POINT_SPACE), int(loc[1] + R_POINT_SPACE)):
            if (0 <= i < len(image)) and (0 <= j < len(image[0])):
                print(image[i][j])
                if (image[i][j] > THRESHOLD):
                    return False
            else:
                return False
    return True


def find_black_spots(image):
    x = list()
    y = list()
    for i in range(RESOLUTION):
        # print(i)
        for j in range(int(RESOLUTION * HEIGHT / WIDTH)):
            if (check_point((int(i / RESOLUTION * WIDTH), int(j / RESOLUTION * WIDTH)), image)):
                x.append(int(i / RESOLUTION * WIDTH))
                y.append(int(j / RESOLUTION * WIDTH))
    return x, y


def crop_image(fname, coords):
    img = cv.imread(fname)
    mxy = max(coords[0][1], coords[1][1])
    mny = min(coords[0][1], coords[1][1])
    mxx = max(coords[0][0], coords[1][0])
    mnx = min(coords[0][0], coords[1][0])
    crop_img = img[mny:mxy, mnx:mxx]
    print(mnx, mny, mxx, mxy)
    cv.imwrite('cropped.png', crop_img)


def extract_text():
    s = pytesseract.image_to_string('cropped.png', lang='rus')
    return s


image = convert_to_white_and_black_image(input('name: '))

x, y = find_black_spots(image)
#print(x, y)
plt.scatter(x, y)
