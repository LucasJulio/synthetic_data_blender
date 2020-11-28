import cv2 as cv
from glob import glob
import os

DATASETS_MAIN_PATH = os.path.expanduser("./datasets/")
SELECTED_DATASET = "Arduino_3q"

# TODO: beware of "_whole" directories!
if __name__ == '__main__':
    main_path = os.path.join(DATASETS_MAIN_PATH, SELECTED_DATASET)

    whole_input_imgs_paths = glob(os.path.join(main_path, "inputs_whole/*"))
    divided_input_imgs_folder = os.path.join(main_path, "inputs/")
    for path in whole_input_imgs_paths:
        img = cv.imread(path)

        # Get middle lines
        v_mid = int(img.shape[0]/2)
        h_mid = int(img.shape[1]/2)

        # Get cropped images
        img_q1 = img[:v_mid, :h_mid]
        img_q2 = img[v_mid:, :h_mid]
        img_q3 = img[:v_mid, h_mid:]
        img_q4 = img[v_mid:, h_mid:]

        previous_name = "".join(path.split("/")[-1][:-4])  # without ".png" at the end

        # Write each divided image
        cv.imwrite(divided_input_imgs_folder + previous_name + "_q1.png", img_q1)
        cv.imwrite(divided_input_imgs_folder + previous_name + "_q2.png", img_q2)
        cv.imwrite(divided_input_imgs_folder + previous_name + "_q3.png", img_q3)
        cv.imwrite(divided_input_imgs_folder + previous_name + "_q4.png", img_q4)

    whole_map_imgs_paths = glob(os.path.join(main_path, "maps_whole/*"))
    divided_map_imgs_folder = os.path.join(main_path, "maps/")
    for path in whole_map_imgs_paths:
        img = cv.imread(path)

        # Get middle lines
        v_mid = int(img.shape[0]/2)
        h_mid = int(img.shape[1]/2)

        # Get cropped images
        img_q1 = img[:v_mid, :h_mid]
        img_q2 = img[v_mid:, :h_mid]
        img_q3 = img[:v_mid, h_mid:]
        img_q4 = img[v_mid:, h_mid:]

        previous_name = "".join(path.split("/")[-1][:-4])  # without ".png" at the end

        # Write each divided image
        cv.imwrite(divided_map_imgs_folder + previous_name + "_q1.png", img_q1)
        cv.imwrite(divided_map_imgs_folder + previous_name + "_q2.png", img_q2)
        cv.imwrite(divided_map_imgs_folder + previous_name + "_q3.png", img_q3)
        cv.imwrite(divided_map_imgs_folder + previous_name + "_q4.png", img_q4)
