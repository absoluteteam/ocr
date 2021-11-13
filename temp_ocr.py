import cv2 as cv
import numpy as np
import os
import pytesseract
pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\User\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
def template_match_get_coords(fname):
    img_rgb = cv.imread(fname)
    result = cv.matchTemplate(img_rgb, cv.imread('target.PNG'), cv.TM_SQDIFF)

    # the get the best match fast use this:
    (min_x, max_y, minloc, maxloc) = cv.minMaxLoc(result)
    (x, y) = minloc

    # get all the matches:
    result2 = np.reshape(result, result.shape[0] * result.shape[1])
    sort = np.argsort(result2)
    (y1, x1) = np.unravel_index(sort[0], result.shape)  # best match
    (y2, x2) = np.unravel_index(sort[1], result.shape)  # second best match
    return ((x1,y1),(x2,y2))

def crop_image(fname,coords):
    img = cv.imread(fname)
    mxy = max(coords[0][1],coords[1][1])
    mny = min(coords[0][1],coords[1][1])
    mxx = max(coords[0][0], coords[1][0])
    mnx = min(coords[0][0], coords[1][0])
    crop_img = img[mny:mxy, mnx:mxx]
    print(mnx,mny,mxx,mxy)
    cv.imwrite('cropped.png', crop_img)

def extract_text():
    s = pytesseract.image_to_string('cropped.png',lang='rus')
    return s

crop_image(os.argv[1],template_match_get_coords(os.argv[1]))
s = extract_text()
print(s)



