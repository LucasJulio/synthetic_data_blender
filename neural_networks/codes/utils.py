import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow_addons.image import gaussian_filter2d


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer,
                                               use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                      kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


"""
ARCHITECTURES
"""


def Custom(output_channels):
    inputs = tf.keras.layers.Input(shape=[512, 512, 3])

    down_stack = [
        downsample(32, 4, apply_batchnorm=False),  # (bs, 256, 256, 64)
        downsample(32, 4),  # (bs, 128, 128, 64)
        downsample(64, 4),  # (bs, 64, 64, 128)
        downsample(128, 4),  # (bs, 32, 32, 256)
        downsample(256, 4),  # (bs, 16, 16, 512)
        downsample(256, 4),  # (bs, 8, 8, 512)
        downsample(256, 4),  # (bs, 4, 4, 512)
        downsample(512, 4),  # (bs, 2, 2, 512)
        downsample(512, 4),  # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(256, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(256, 4),  # (bs, 16, 16, 1024)
        upsample(256, 4),  # (bs, 32, 32, 512)
        upsample(128, 4),  # (bs, 64, 64, 256)
        upsample(64, 4),  # (bs, 128, 128, 128)
        upsample(32, 4),  # (bs, 256, 256, 64)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(output_channels, 4, strides=2, padding='same', kernel_initializer=initializer
                                           , activation='tanh')  # (bs, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


"""
UTILS
"""


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask


@tf.function
def augment_data(input_img, input_mask):
    """
    :param input_img: input tensor to be augmented
    :param input_mask: just the same mask
    :return: tensor with the following possible augmentations: brightness changes, contrast changes,
    HUE channel changes
    """
    # Brightness
    # tf.random.uniform(()) > 0.5:
    input_img = tf.image.random_brightness(input_img, 0.2)

    # Contrast
    # tf.random.uniform(()) > 0.5:
    input_img = tf.image.random_contrast(input_img, 0.7, 1.3)

    # HUE mess
    # if tf.random.uniform(()) > 0.5:
    input_img = tf.image.random_hue(input_img, 0.4)

    if tf.random.uniform(()) > 0.5:
        input_img = gaussian_filter2d(input_img)

    augmented_img = input_img

    return augmented_img, input_mask


@tf.function
def load_image_train(datapoint):
    # input_image = tf.image.resize_with_crop_or_pad(datapoint['image'], 540, 540)  # eliminates unnecessary borders
    # input_mask = tf.image.resize_with_crop_or_pad(datapoint['segmentation_mask'], 540, 540)
    # input_image = tf.image.resize(input_image, (448, 448))
    # input_mask = tf.image.resize(input_mask, (448, 448))

    input_image = tf.image.resize(datapoint['image'], (512, 512))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (512, 512))

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def load_image_test(datapoint):
    # input_image = tf.image.resize_with_crop_or_pad(datapoint['image'], 540, 540)  # eliminates unnecessary borders
    # input_mask = tf.image.resize_with_crop_or_pad(datapoint['segmentation_mask'], 540, 540)
    # input_image = tf.image.resize(input_image, (448, 448))
    # input_mask = tf.image.resize(input_mask, (448, 448))

    input_image = tf.image.resize(datapoint['image'], (512, 512))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (512, 512))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def normalize_input(input_image):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image


def create_mask(p_mask):
    p_mask = tf.argmax(p_mask, axis=-1)
    p_mask = p_mask[..., tf.newaxis]
    return p_mask[0]


def inference_on_image(image_file_path, classifier_model):
    target_size = max(classifier_model.layers[0].input_shape)
    im = Image.open(image_file_path)
    im = im.resize((target_size[1], target_size[2]), Image.ANTIALIAS)
    np_img = np.asarray(im)

    img_batch = np.expand_dims(np_img, axis=0)
    pre_processed_input = normalize_input(img_batch)[:, :, :, :3]
    pred_mask = create_mask(classifier_model.predict(pre_processed_input))
    return np_img, pred_mask
