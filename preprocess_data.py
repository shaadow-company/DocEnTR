import cv2
import argparse
import imutils
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
import os

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("-i", "--image", required=True, help="Path to the image file", default=None)
argument_parser.add_argument("-f", "--folder", required=False, help="Folder of teh image to process", default=None)
argument_parser.add_argument("---iphone", required=False,type=bool, action="stroe_true")

args = argument_parser.parse_args()

def calculate_corners(contour_points):
    """
    -----------
    Description
    -----------
    Finds the four points tu center the image
    
    ------
    Return
    ------
    Corners of the new image
    """
    corners = np.zeros((4,2), dtype='float32')
    sum_points = contour_points.sum(axis=1)
    corners[0] = contour_points[np.argmin(sum_points)]
    corners[2] = contour_points[np.argmax(sum_points)]
    diff_points = np.diff(contour_points, axis=1)
    corners[1] = contour_points[np.argmin(diff_points)]
    corners[3] = contour_points[np.argmax(diff_points)]
    return corners

def transform_perspective(input_image, contour_points):
    """
    -----------
    Description
    -----------
    Transform the image with the corners calculated in the calculate_corners function

    --------
    Return
    ------
    Image with perspective corrected
    """
    corners = calculate_corners(contour_points)
    (top_left, top_right, bottom_right, bottom_left) = corners

    width_top = np.linalg.norm(bottom_right - bottom_left)
    width_bottom = np.linalg.norm(top_right - top_left)
    max_width = max(int(width_top), int(width_bottom))

    height_left = np.linalg.norm(top_right - bottom_right)
    height_right = np.linalg.norm(top_left - bottom_left)
    max_height = max(int(height_left), int(height_right))

    destination = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(corners, destination)
    output_image = cv2.warpPerspective(input_image, matrix, (max_width, max_height))

    return output_image

def process_heic_images():
    register_heif_opener()
    path = args.folder
    files = os.listdir(path)
    for file in files:
        file_path = os.path.join(path, file)
        im = Image.open(file_path)
        im.save(file, "png")
    


def process_image():

    original_image = cv2.imread(args.image)
    image_ratio = original_image.shape[0] / 500.0
    resized_image = imutils.resize(original_image, height=500)

    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edge_detected = cv2.Canny(blurred_image, 75, 200)

    # STEP 1: Edge Detection
    print("STEP 1: Edge Detection")
    cv2.imshow("Edges", edge_detected)

    contours = cv2.findContours(edge_detected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    detected_screen = None
    for contour in sorted_contours:
        contour_perimeter = cv2.arcLength(contour, True)
        approximated_contour = cv2.approxPolyDP(contour, 0.02 * contour_perimeter, True)

        if len(approximated_contour) == 4:
            detected_screen = approximated_contour
            break

    if detected_screen is not None:
        # STEP 2: Finding Boundary
        print("STEP 2: Finding Boundary")
        cv2.drawContours(resized_image, [detected_screen], -1, (0, 255, 0), 2)
        cv2.imshow("Boundary", resized_image)

        transformed_image = transform_perspective(original_image, detected_screen.reshape(4, 2) * image_ratio)
        grayscale_transformed = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY)

        # STEP 3: Apply Perspective Transform
        print("STEP 3: Apply Perspective Transform")
        cv2.imshow("Scanned Image", transformed_image)
        cv2.imwrite('scanned_document.jpg', transformed_image)
        cv2.waitKey(0)
    else:
        print("No valid screen contour found.")

    cv2.destroyAllWindows()