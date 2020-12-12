import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import GaussianNoise, Input, Conv2DTranspose, Concatenate, Add
from tensorflow.keras.applications import MobileNetV2
from tensorflow_examples.models.pix2pix import pix2pix
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow_addons.image import gaussian_filter2d
from cv2 import GaussianBlur


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


def create_unet_model(output_channels, freeze_percentage, noise_stdev, base_model_name="MobileNetV2"):
    if base_model_name == "MobileNetV2":
        base_model = MobileNetV2(input_shape=[448, 448, 3], include_top=False)

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',
        'block_3_expand_relu',
        'block_6_expand_relu',
        'block_13_expand_relu',
        'block_16_project',
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]
    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
    # down_stack.trainable = False
    # Freeze first layers
    for l_idx, layer in enumerate(down_stack.layers):
        if l_idx < np.ceil((freeze_percentage / 100) * len(down_stack.layers)) or \
                layer.__class__.__name__ == 'BatchNormalization':
            if layer.__class__.__name__ not in ["LeakyReLU", "Dense", "Activation"]:
                layer.trainable = False
        else:
            layer.trainable = True
    up_stack = [
        pix2pix.upsample(512, 3, apply_dropout=True),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),  # 32x32 -> 64x64
    ]

    inputs = Input(shape=[448, 448, 3])
    x = GaussianNoise(stddev=noise_stdev)(inputs)  # TODO: this should be a hyperparameter
    x_i = x

    # Downsampling through the model
    skips = down_stack(x_i)
    x_i = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x_i = up(x_i)
        concat = Concatenate()
        x_i = concat([x_i, skip])

    # These are the last layers of the model
    last = Conv2DTranspose(output_channels, 3, strides=2, padding='same')  # 64x64 -> 128x128
    x_i = last(x_i)
    return Model(inputs=inputs, outputs=x_i)


def custom_model(output_channels):
    inputs = Input(shape=[448, 448, 3])

    down_stack = [
        downsample(32, 4, apply_batchnorm=True),  # (bs, 256, 256, 64)
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
    last = Conv2DTranspose(output_channels, 4, strides=2, padding='same', kernel_initializer=initializer
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
        x = Concatenate()([x, skip])

    x = last(x)

    return Model(inputs=inputs, outputs=x)


def normalize(input_image, input_mask):
    """
    Normalizes input image. Should be used in image before feeding it to the neural network
    """
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask


def normalize_input(input_image):
    """
    Same as 'normalize', but works for inputs only (without target_mask)
    """
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image


def create_augmentation_function(hp_brightness_range,
                                 hp_contrast_min_range,
                                 hp_contrast_max_range,
                                 hp_hue_range,
                                 hp_gaussian_blur_probs):
    """
    Wrapper for 'augment_data' function
    """
    @tf.function
    def augment_data(input_img, input_mask,
                     brightness_range=hp_brightness_range,
                     contrast_min_range=hp_contrast_min_range,
                     contrast_max_range=hp_contrast_max_range,
                     hue_range=hp_hue_range,
                     gaussian_blur_probs=hp_gaussian_blur_probs,
                     ):
        """
        Performs data augmentation transformations on the images. If necessary, does the same to the corresponding masks
        """
        input_img = tf.image.random_brightness(input_img, brightness_range)
        input_img = tf.image.random_contrast(input_img, contrast_min_range, contrast_max_range)
        input_img = tf.image.random_hue(input_img, hue_range)

        if tf.random.uniform(()) > gaussian_blur_probs:
            input_img = gaussian_filter2d(image=input_img, filter_shape=[5, 5])
            if tf.random.uniform(()) > gaussian_blur_probs:
                input_img = gaussian_filter2d(image=input_img, filter_shape=[5, 5])
                if tf.random.uniform(()) > gaussian_blur_probs:
                    input_img = gaussian_filter2d(image=input_img, filter_shape=[5, 5])
        augmented_img = input_img

        # TODO: random crops of reasonable size; the same area must be extracted from input and mask

        return augmented_img, input_mask
    return augment_data


@tf.function
def load_image_train(datapoint):
    # input_image = tf.image.resize_with_crop_or_pad(datapoint['image'], 540, 540)  # eliminates unnecessary borders
    # input_mask = tf.image.resize_with_crop_or_pad(datapoint['segmentation_mask'], 540, 540)
    # input_image = tf.image.resize(datapoint['image'], (512, 512))
    # input_mask = tf.image.resize(datapoint['segmentation_mask'], (512, 512))
    input_image = tf.image.resize(datapoint['image'], (448, 448))
    input_mask = tf.math.round(tf.image.resize(datapoint['segmentation_mask'], (448, 448)))

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def load_image_not_train(datapoint):
    # input_image = tf.image.resize_with_crop_or_pad(datapoint['image'], 540, 540)  # eliminates unnecessary borders
    # input_mask = tf.image.resize_with_crop_or_pad(datapoint['segmentation_mask'], 540, 540)
    # input_image = tf.image.resize(datapoint['image'], (512, 512))
    # input_mask = tf.image.resize(datapoint['segmentation_mask'], (512, 512))
    input_image = tf.image.resize(datapoint['image'], (448, 448))
    input_mask = tf.math.round(tf.image.resize(datapoint['segmentation_mask'], (448, 448)))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


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


def configure_gpus():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            # print(e)
            pass


def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ['Imagem de entrada', 'Rótulos verdadeiros', 'Rótulos preditos']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def show_predictions(model, ds, num=1):
    for im, msk in ds.take(num):
        prediction = model.predict(im[tf.newaxis, ...])
        pred_mask = create_mask(prediction)
        display([im[0], msk[0], pred_mask])


# TODO: Replace this with tensorboard images
class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        show_predictions()
        print('\nSample Prediction after epoch {}\n'.format(epoch+1))


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