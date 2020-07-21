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
model = tf.keras.models.load_model("/home/ribeiro-desktop/blender_experiments/neural_networks/models/augmentation_v1-BSF")


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
    pred_mask = pred_mask.astype(np.uint8)*84
    pred_mask_display = cv2.merge((pred_mask, pred_mask, pred_mask)).astype(np.uint8)
    img_resized = cv2.resize(frame, (448, 448))
    display = np.hstack([img_resized, pred_mask_display])
    # Display the resulting frame
    cv2.imshow('frame', display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
