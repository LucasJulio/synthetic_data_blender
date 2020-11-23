import tensorflow as tf
import cv2
import numpy as np
from PIL import Image


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


cap = cv2.VideoCapture(0)

# TODO: specify model when running
model_path = "/home/ribeiro-desktop/POLI/TCC/blender_experiments/neural_networks/logs/11-23--12-50"
model = tf.keras.models.load_model(model_path, compile=False)


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
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

    pred_mask_display = cv2.merge((hue_pred_mask, saturation_value, saturation_value)).astype(np.uint8)
    pred_mask_display = cv2.cvtColor(pred_mask_display, cv2.COLOR_HSV2RGB)
    img_resized = cv2.resize(frame, (896, 896))
    pred_mask_display = cv2.resize(pred_mask_display, (896, 896))
    display = np.hstack([img_resized, pred_mask_display])
    # Display the resulting frame
    cv2.imshow('frame', display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
