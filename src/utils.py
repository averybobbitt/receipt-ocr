from os import PathLike

import cv2
import imutils
import pytesseract
from imutils.perspective import four_point_transform
from numpy import ndarray


def perform_ocr(img: str | PathLike | ndarray):
    # input must be a file path or image
    if isinstance(img, str) or isinstance(img, PathLike):
        img_orig = cv2.imread(img)
    elif isinstance(img, ndarray):
        img_orig = cv2.imdecode(img, cv2.IMREAD_COLOR)
    else:
        raise TypeError

    # normalize image
    image = img_orig.copy()
    image = imutils.resize(image, width=500)
    ratio = img_orig.shape[1] / float(image.shape[1])

    # convert the image to grayscale, blur it slightly, and then apply edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # find contours in the edge map and sort them by size in descending order
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # initialize a contour that corresponds to the receipt outline
    receipt_cnt = None

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points, then we can assume we have found the outline of the receipt
        if len(approx) == 4:
            receipt_cnt = approx
            break

    # if the receipt contour is empty then our script could not find the outline and we should be notified
    if receipt_cnt is None:
        raise Exception("Could not find receipt outline. Try debugging your edge detection and contour steps.")

    # apply a four-point perspective transform to the *original* image to get a top-down bird's-eye view of the receipt
    receipt = four_point_transform(img_orig, receipt_cnt.reshape(4, 2) * ratio)

    # apply OCR to the receipt image by assuming column data, ensuring the text is *concatenated across the row*
    # (additionally, for your own images you may need to apply additional processing to clean up the image, including
    # resizing, thresholding, etc.)
    text = pytesseract.image_to_string(cv2.cvtColor(receipt, cv2.COLOR_BGR2RGB), config="--psm 6")

    return text
