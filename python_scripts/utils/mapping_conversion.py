import cv2 as cv
import numpy as np
from glob import glob
import os

DATASETS_MAIN_PATH = os.path.expanduser("~/blender_experiments/datasets/")
SELECTED_DATASET = "sample_dataset"
NUMBER_OF_CLASSES = 3


def convert_segmentation_to_mapping(img):
    inter = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    m_img = np.zeros_like(inter).astype(np.uint8)
    val = 0
    step = 255//NUMBER_OF_CLASSES
    low_bound = np.clip(val, 0, 255)
    up_bound = np.clip(low_bound + step, 1, 255)
    for i in range(1, NUMBER_OF_CLASSES + 1):
        m_img[np.logical_and(inter >= low_bound, inter < up_bound)] = i
        val = val + step
        low_bound = np.clip(val, 1, 255)
        up_bound = np.clip(low_bound + step, 1, 255)

    return m_img


if __name__ == '__main__':
    main_path = os.path.join(DATASETS_MAIN_PATH, SELECTED_DATASET)
    visible_segmentation_imgs_paths = glob(os.path.join(main_path, "visible_maps/*"))
    converted_map_imgs_paths = os.path.join(main_path, "maps/")
    for path in visible_segmentation_imgs_paths:
        vis_map_img = cv.imread(path)
        map_img = convert_segmentation_to_mapping(vis_map_img)
        map_path = converted_map_imgs_paths + "".join(path.split("/")[-1][1:])  # drops the "v" from prefix
        cv.imwrite(map_path, map_img)
