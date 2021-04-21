import os
import re
import cv2
import math
import imghdr
import argparse
import numpy as np
import pytesseract
from scipy import ndimage


def resize_img(img, height=None, width=None):
    """
    Resizes the image to a given height or width keeping the aspect ratio.
    If both are given, it prioritizes height. If height (or width) given is
    greater than original, the image remains unchanged.

    Parameters:
    img (ndarray): Image to resize.
    height (int): Expected height of resized image.
    width (int): Expected width of resized image.

    Returns:
    ndarray: Resized image.
    """
    img_h, img_w, _ = img.shape
    ratio = img_w / img_h

    if height is not None and height < img_h:
        new_size = (int(height * ratio), height)
    elif width is not None and width < img_w:
        new_size = (width, int(width / ratio))
    else:
        return img

    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def approximate_contour(contour, accuracy_alpha=0.1):
    """
    Approximates contour with a curve or polygon.

    Parameters:
    contour (ndarray): Contour to approximate.
    accuracy_alpha (float, optional): Parameter to calculate approximation
        accuracy. Higher values produces more flexible approximations
        (default is 0.1).

    Returns:
    ndarray: Contour approximation.
    """
    perimeter = cv2.arcLength(contour, True)
    # Here accuracy is a 'high' value to be more flexible and find contour of
    # curved or folded receipts.
    accuracy = accuracy_alpha * perimeter
    return cv2.approxPolyDP(contour, accuracy, True)


def is_considered_rectangle(approx):
    """
    Check if the approximation is too deviated from a rectangle.
    It is considered rectangle when polygon diagonals difference is less than
    2 times.

    Parameters:
    approx (ndarray): Contour approximation to analyse.

    Returns:
    bool: Is the approximation considered a rectangle?
    """
    reshaped = approx.reshape(4, 2)
    if len(set(map(tuple, reshaped))) == 4:
        diag_1 = math.dist(reshaped[0], reshaped[2])
        diag_2 = math.dist(reshaped[1], reshaped[3])
        diag_diff = min([diag_1, diag_2]) / max([diag_1, diag_2])
        # To filter cases when a 4-point contour is found but at least 2 points
        # are almost the same. Useful for situations when the receipt contour
        # is not entirely inside de image.
        if diag_diff >= 0.5:
            return True
    return False


def get_receipt_contour(contours, img, min_area_covered=0.1):
    """
    Select the biggest rectangular contour that covers at least 'area_covered'
    of the original image. It is considered the receipt contour.

    Parameters:
    contours (list): List of receipt candidate contours.
    img (ndarray): Image to check area covered by contours.
    min_area_covered (float, optional): Indicates the minimum area (interpreted
        as percentage) that should be covered by a contour in order to be
        considered. e.g. 0.1 indicates that the contour should cover at least
        10% of the image (default is 0.1).

    Returns:
    ndarray or None: Receipt contour candidate (if found).
    """
    for c in contours:
        approx = approximate_contour(c)
        # In some cases, contours found have 4 points but they are actually a
        # line or a triangle with repeated points. With this, only polygons
        # with 4 different points are considered. This 'if' condition takes
        # advantage of lazy evaluation given that 'is_considered_rectangle'
        # assumes the approximation has 4 points.
        if len(approx) == 4 and is_considered_rectangle(approx):
            _, _, c_w, c_h = cv2.boundingRect(approx)
            contour_area = c_w * c_h
            i_h, i_w = img.shape
            img_area = i_w * i_h
            cover_area = contour_area / img_area
            # This is useful for cases when the receipt contour is not found,
            # but small contours are detected. With this, they are filtered.
            # A possible problem here is when the picture is taken far from the
            # receipt. If the resolution is big enough, a zoom can be made and
            # process that portion of the image.
            if cover_area >= 0.1:
                return approx


def get_adaptive_threshold(img):
    """
    Applies adaptive threshold to the given image.

    Parameters:
    img (ndarray): Image where adaptive threshold will be applied.

    Returns:
    ndarray: Image with adaptive threshold applied.
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blockSize intensifies black color and C reduces noise (also black).
    thresh_img = cv2.adaptiveThreshold(
        gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=21, C=10)
    return thresh_img


def image_is_upside_down(img):
    """
    Checks if the given image is upside down applying Orientation and Script
    Detection (OSD).

    Parameters:
    img (ndarray): Image where OSD will be applied. It is recommended to have
        thresholding applied to the image.

    Returns:
    bool: Is the image upside down?
    """
    osd = pytesseract.image_to_osd(np.array(img))
    orientation = int(re.search(r'(?<=Rotate: )\d+', osd).group(0))
    if orientation not in [0, 180]:
        print("WARNING: Image is not vertically aligned.")
    return orientation != 0


def align_image_by_text(img, rotation_step=2):
    """
    Rotates (2D) the image to vertically align the receipt only analysing text.
    It is time consuming given that it rotates the image until it founds the
    rotation containing the maximum text boxes quantity.

    Parameters
    ----------
    img (ndarray): Array containing the image to align.
    rotation_step (int, optional): Indicates the angle difference (in degrees)
        between one alignment attempt and the next one. (default is 2).

    Returns
    -------
    ndarray: Aligned image if possible, else the original image is
        returned.
    """
    thresh_receipt = get_adaptive_threshold(img)

    # Count text boxes detected while rotating the image.
    # Hypothesis: maximum text boxes are given when the image is vertically or
    # horizontally aligned.
    q_boxes = list()
    for tmp_angle in range(0, 360, rotation_step):
        base = ndimage.rotate(thresh_receipt, tmp_angle, cval=0)
        d = pytesseract.image_to_data(base, output_type='dict')
        q_boxes.append([len(d['level']), tmp_angle])
    angle = sorted(q_boxes, reverse=True)[0][1]

    # In some cases, more boxes are detected when image is aligned horizontally
    # and those cases are fixed comparing the amount of vertical and horizontal
    # boxes.
    base = ndimage.rotate(thresh_receipt, angle, cval=0)
    d = pytesseract.image_to_data(base, output_type='dict')
    n_boxes = len(d['level'])
    horizontal = sum([d['width'][i] >= d['height'][i] for i in range(n_boxes)])
    vertical = n_boxes - horizontal
    # If there are more vertical boxes than horizontal boxes, it means the
    # image is horizontally aligned.
    if vertical > horizontal:
        angle += 90

    # At this point, the image should be vertically aligned, that is why
    # image_is_upside_down makes sense.
    rotated = ndimage.rotate(img, angle, cval=0)
    thresh_receipt = get_adaptive_threshold(rotated)
    if image_is_upside_down(thresh_receipt):
        angle += 180

    aligned_img = ndimage.rotate(img, angle, cval=0)
    return aligned_img


def align_image(img):
    """
    Rotates (2D) the image to vertically align the receipt analysing contour
    and/or text.

    Parameters
    ----------
    img (ndarray): Array containing the image to align.

    Returns
    -------
    ndarray: Aligned image if possible, else the original image is returned.
    """
    # Copy image to keep original unchanged
    image = img.copy()

    # Reduce image size because it is better for finding contours.
    # In some cases this default height is big and then contours are not found.
    # A loop trying different height values can be implemented.
    image = resize_img(image, height=400)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate thresholds for edges detection.
    high_thresh, _ = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.33 * high_thresh

    # Blur and dilate help to smooth contours.
    # Good for cases when the receipt has small extra paper usually in corners.
    image = cv2.GaussianBlur(image, (5, 5), 0)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    image = cv2.dilate(image, rectKernel)

    # Erode to avoid near objects to be fusioned with receipt.
    image = cv2.erode(image, rectKernel)

    # Look for contours. Not rectangular shapes, incomplete contours and object
    # overlap is covered later. Besides those cases, shadows can be a problem.
    # Steps to remove shadow (e.g. dilate, blur, normalization) can be applied.
    edged = cv2.Canny(image, low_thresh, high_thresh)
    contours, _ = cv2.findContours(
        edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Assumption: Receipt is in the first 5 bigger contours and it is the
    # biggest rectangular contour. This is done to filter different objects.
    bigger_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    receipt_contour = get_receipt_contour(bigger_contours, image)
    if receipt_contour is None:
        print("WARNING: No receipt found. It will try alignment by text.")
        try:
            # In many cases, the receipt contour is not completely inside the
            # image, or there are objects over the receipt (e.g. a hand or
            # fingers grabbbing it). In these cases, some text can be
            # interpreted to align image (if enough image resolution).
            # This is also useful for the cases when the receipt contour can
            # not be distinguised from the background.
            aligned_img = align_image_by_text(img)
            return aligned_img
        except Exception:
            # Sometimes the contour and the text approach might not work.
            # The next step can be to detect straight lines to match them as
            # the sides of the receipt and align the image given the slope of
            # the detected sides.
            print("WARNING: Could not find receipt nor text."
                  " Returning original image.")
            return img

    # A straight line is fit to the receipt contour to find its slope given
    # that if using 'cv2.minAreaRect' the angle obtained could be the same
    # on different cases and it does not contemplates contour height nor width.
    # In some cases when there is a remarkable receipt perspective, the line is
    # not perpendicular to the receipt contour. Forcing it to be perpendicular
    # to the top and the bottom of the rectangle may be a better option for
    # those special cases.
    [vx, vy, x, y] = cv2.fitLine(receipt_contour, cv2.DIST_L2, 0, 0.01, 0.01)
    slope = -1 * (vx / vy)
    angle = np.degrees(math.atan(slope))

    # Resize contour to work with original image. This is better to determine
    # if the temporal aligned image is upside down.
    ratio_h = img.shape[0] / image.shape[0]
    ratio_w = img.shape[1] / image.shape[1]
    receipt_contour[:, :, 0] = receipt_contour[:, :, 0] * ratio_w
    receipt_contour[:, :, 1] = receipt_contour[:, :,  1] * ratio_h

    # Create an aligned mask to only get the aligned receipt.
    mask = np.zeros_like(img)
    cv2.drawContours(mask, [receipt_contour], 0, (255, 255, 255), -1)
    aligned_mask = ndimage.rotate(mask, angle, cval=0)
    aligned_img = ndimage.rotate(img, angle, cval=0)

    # Crop the aligned receipt from the aligned image using the created mask.
    (y, x, _) = np.where(aligned_mask == 255)
    (top_y, top_x) = (np.min(y), np.min(x))
    (bottom_y, bottom_x) = (np.max(y), np.max(x))
    aligned_receipt = aligned_img[top_y:bottom_y + 1, top_x:bottom_x + 1]

    # In most of the cases, the text was not correctly detected.
    # With this adaptive threshold, the text is shown cleaner.
    thresh_receipt = get_adaptive_threshold(aligned_receipt)

    # This Orientation and Script Detection (OSD) is applied for cases when the
    # receipt is aligned upside down.
    try:
        if image_is_upside_down(thresh_receipt):
            angle += 180
            aligned_img = ndimage.rotate(img, angle, cval=0)
    except Exception:
        print("WARNING: Could not detect text. "
              "Image may be aligned upside down.")

    return aligned_img


def main(input_path, output_path=None):
    """
    Main process to align an image. It also checks if the input file is valid
    and store result in a file if output path is given.

    Parameters:
    input_path (str): Input file path to attempt alignment.
    output_path (str, optional): Destination file of the aligned image
        (default is None).

    Returns:
    ndarray: Aligned image.
    """
    # For cases when the file does not exist or it is not an image.
    if os.path.isfile(input_path) and imghdr.what(input_path) is not None:
        img_or = cv2.imread(input_path)
        try:
            if output_path:
                if os.path.isfile(output_path):
                    print("WARNING: Output file will be overwritten.")

                directory, _ = os.path.split(output_path)
                if os.path.isdir(directory):
                    aligned_image = align_image(img_or)
                    cv2.imwrite(output_path, aligned_image)
                else:
                    print("ERROR: Output folder does not exist.")
            else:
                aligned_image = align_image(img_or)
                return aligned_image
        except Exception:
            print("ERROR: Could not align image.")
    else:
        print("ERROR: {} is not a valid file.".format(input_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Vertically align a given image according to the receipt.')
    parser.add_argument(
        '-i', '--input', type=str, required=True,
        help='Input file path to attempt alignment.')
    parser.add_argument(
        '-o', '--output', type=str, required=True,
        help=('Destination file of the aligned image. '
              'If the destination folder is the current one, start with ./'))

    args = parser.parse_args()
    main(args.input, args.output)
