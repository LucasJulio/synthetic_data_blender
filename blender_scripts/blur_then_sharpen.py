import numpy as np
from cv2 import GaussianBlur, imread, imwrite
from glob import glob
import os

DATASETS_MAIN_PATH = os.path.expanduser("./datasets/")
SELECTED_DATASET = "Arduino_3q3_bs"


def unsharp_mask(image, kernel_size=(7, 7), sigma=1.5, amount=1.5, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def blur_then_sharpen(im, iters):
    for _ in range(iters):
        im = GaussianBlur(im, (7, 7), 1.5)
    for _ in range(iters):
        im = unsharp_mask(im)
    im = GaussianBlur(im, (3, 3), 1.5)
    return im


if __name__ == '__main__':
    main_path = os.path.join(DATASETS_MAIN_PATH, SELECTED_DATASET)
    input_imgs_paths = glob(os.path.join(main_path, "inputs_raw/*"))
    transformed_imgs_folder = os.path.join(main_path, "inputs/")
    for path in input_imgs_paths:
        img = imread(path)
        img_transformed = blur_then_sharpen(img, iters=4)
        previous_name = "".join(path.split("/")[-1][:-4])  # without ".png" at the end
        imwrite(transformed_imgs_folder + previous_name + "_bs.png", img_transformed)
