import tensorflow as tf
import cv2 as cv
import numpy as np
from PIL import Image
from glob import glob
from time import sleep

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

root_img_directory = "/home/ribeiro-desktop/POLI/TCC/blender_experiments/images_for_testing/macro/"
img_files_paths = glob(root_img_directory + "*")
model = tf.keras.models.load_model("/home/ribeiro-desktop/POLI/TCC/blender_experiments/neural_networks/logs/11-20--21-09",
                                   compile=False)


def normalize_input(input_image):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image


def create_mask(p_mask):
    p_mask = tf.argmax(p_mask, axis=-1)
    p_mask = p_mask[..., tf.newaxis]
    return p_mask[0]


def inference_on_image(im, classifier_model):
    target_size = max(classifier_model.layers[0].input_shape)
    im = Image.fromarray(im)
    im = im.resize((target_size[1], target_size[2]), Image.ANTIALIAS)
    np_img = np.asarray(im)

    img_batch = np.expand_dims(np_img, axis=0)
    pre_processed_input = normalize_input(img_batch)[:, :, :, :3]
    p_mask = create_mask(classifier_model.predict(pre_processed_input))
    return p_mask


while True:
    for img_path in img_files_paths:
        frame = cv.imread(img_path)

        # Our operations on the frame come here
        img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        pred_mask = np.asarray(inference_on_image(img_rgb, model))

        # Convert to Hue channel for HSV visualization
        hue_pred_mask = pred_mask.astype(np.uint8) * (180//17)
        saturation_value = np.ones_like(pred_mask) * 255
        saturation_value = saturation_value.astype(np.uint8)
        """
        PIXEL_VALUE ---------- CORRESPONDING_CLASS 
        000---------------------0_Background
        015---------------------1_Substrate
        030---------------------2_Header_Female
        045---------------------3_Header_Male
        060---------------------4_Main_Microcontroller
        075---------------------5_USB_Connector
        090---------------------6_DC_Jack
        105---------------------7_Button
        120---------------------8_Voltage_Regulator
        135---------------------9_SMD_CDR
        150---------------------10_SMD_LED
        165---------------------11_SO_Transistor
        180---------------------12_Crystal_Oscillator
        195---------------------13_Electrolytic_Capacitor
        210---------------------14_USB_Controller
        225---------------------15_IC
        240---------------------16_Polyfuse
        """

        pred_mask_display = cv.merge((hue_pred_mask, saturation_value, saturation_value)).astype(np.uint8)
        pred_mask_display = cv.cvtColor(pred_mask_display, cv.COLOR_HSV2RGB)
        img_resized = cv.resize(frame, (896, 896))
        pred_mask_display = cv.resize(pred_mask_display, (896, 896))
        display = np.hstack([img_resized, pred_mask_display])
        # Display the resulting frame
        cv.imshow('frame', display)
        sleep(4)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
