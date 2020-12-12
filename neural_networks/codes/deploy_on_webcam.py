import os
import time
import tensorflow as tf
import cv2 as cv
import numpy as np
from PIL import Image
from neural_networks.codes.utils import blur_then_sharpen


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


cap = cv.VideoCapture(0)
cam_width = 1920
cam_height = 1080
cap.set(cv.CAP_PROP_FRAME_WIDTH, cam_width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, cam_height)

input_width = 896
input_height = 896

display_size = (448, 448)

def crop_img_to_center(img, target_width, target_height):
    return img[((cam_height//2) - target_height//2): ((cam_height//2) + target_height//2),
               ((cam_width//2) - target_width//2): ((cam_width//2) + target_width//2)]


os.system("v4l2-ctl -c focus_auto=0")
os.system("v4l2-ctl -c focus_absolute=24")
os.system("v4l2-ctl -c zoom_absolute=100")
os.system("v4l2-ctl -c backlight_compensation=0")
os.system("v4l2-ctl -c exposure_auto=3")
os.system("v4l2-ctl -c gain=0")

# model_path = "/home/ribeiro-desktop/POLI/TCC/blender_experiments/neural_networks/codes/Models/" \
#              "Trained/6_best/20201207-213732_acc_0.9990__ce_0.0031"  # Great results!
model_path = "/home/ribeiro-desktop/POLI/TCC/blender_experiments/neural_networks/codes/Models/" \
             "Trained/7_best/20201210-110420_acc_0.9968__ce_0.0092"
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

BLUR_THEN_SHARPEN_ITERS = 4
true_mask = cv.imread("/home/ribeiro-desktop/POLI/TCC/blender_experiments/neural_networks/codes/true_mask.png",
                      flags=cv.IMREAD_GRAYSCALE)
true_mask_resized = cv.resize(np.expand_dims(true_mask, axis=-1), (448, 448))
hue_true_mask = true_mask.astype(np.uint8) * (180//17)
saturation_value = np.ones_like(hue_true_mask) * 255
saturation_value = saturation_value.astype(np.uint8)
true_mask_display = cv.merge((hue_true_mask, saturation_value, saturation_value)).astype(np.uint8)
true_mask_display = cv.cvtColor(true_mask_display, cv.COLOR_HSV2RGB)
true_mask_display_resized = cv.resize(true_mask_display, display_size)
# i = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = crop_img_to_center(frame, target_width=input_width, target_height=input_height)

    # Our operations on the frame come here
    img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    if BLUR_THEN_SHARPEN_ITERS > 0:
        img_rgb_preproc = blur_then_sharpen(img_rgb, iters=BLUR_THEN_SHARPEN_ITERS)
    else:
        img_rgb_preproc = img_rgb

    pred_mask = np.asarray(inference_on_image(img_rgb_preproc, model))
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
    img_resized = cv.resize(frame, display_size)
    pred_mask_display_resized = cv.resize(pred_mask_display, display_size)
    # img_rgb_preproc_resized = cv.resize(img_rgb_preproc, display_size)
    img_mask_diff = cv.subtract(true_mask_resized, pred_mask.astype(np.uint8))
    img_resized_added = cv.addWeighted(img_resized, 0.7, true_mask_display_resized, 0.3, 0)
    display_1 = np.hstack([img_resized_added, true_mask_display_resized])
    img_mask_diff_resized = cv.resize(img_mask_diff, display_size)
    img_mask_diff_resized = cv.threshold(img_mask_diff_resized, 1, 255, type=cv.THRESH_BINARY)[1]
    img_mask_diff_resized_merged = cv.merge((img_mask_diff_resized, img_mask_diff_resized, img_mask_diff_resized))
    display_2 = np.hstack([img_mask_diff_resized_merged, pred_mask_display_resized])
    display = np.vstack([display_1, display_2])
    # display = frame
    # Display the resulting frame
    cv.imshow('frame', display)
    BLUR_THEN_SHARPEN_ITERS += 1
    BLUR_THEN_SHARPEN_ITERS %= 5
    time.sleep(0.1)
    # filename = "input_%s.png" % str(i)
    # cv.imwrite(filename, frame)
    # filename = "output_%s.png" % str(i)
    # cv.imwrite(filename, pred_mask)
    # i += 1
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
