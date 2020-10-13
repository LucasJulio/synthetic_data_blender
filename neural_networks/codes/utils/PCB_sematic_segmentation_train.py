import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow_examples.models.pix2pix import pix2pix
from neural_networks.codes.utils import PCB_dataset
from PIL import Image


DATASETS_MAIN_PATH = os.path.expanduser("/datasets/")
SELECTED_DATASET = "sample_dataset"


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


dataset_builder = PCB_dataset.PCB()
dataset_builder.download_and_prepare()
dataset = dataset_builder.as_dataset()


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
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
    if tf.random.uniform(()) > 0.2:
        input_img = tf.image.random_brightness(input_img, 0.4)

    # Contrast
    if tf.random.uniform(()) > 0.2:
        input_img = tf.image.random_contrast(input_img, 0.8, 1.2)

    # HUE mess
    if tf.random.uniform(()) > 0.2:
        input_img = tf.image.random_hue(input_img, 0.05)

    augmented_img = input_img

    return augmented_img, input_mask


@tf.function
def load_image_train(datapoint):
    input_image = tf.image.resize_with_crop_or_pad(datapoint['image'], 540, 540)  # eliminates unnecessary borders
    input_mask = tf.image.resize_with_crop_or_pad(datapoint['segmentation_mask'], 540, 540)
    input_image = tf.image.resize(input_image, (448, 448))
    input_mask = tf.image.resize(input_mask, (448, 448))

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def load_image_test(datapoint):
    input_image = tf.image.resize_with_crop_or_pad(datapoint['image'], 540, 540)  # eliminates unnecessary borders
    input_mask = tf.image.resize_with_crop_or_pad(datapoint['segmentation_mask'], 540, 540)
    input_image = tf.image.resize(input_image, (448, 448))
    input_mask = tf.image.resize(input_mask, (448, 448))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


TRAIN_LENGTH = 3000
BATCH_SIZE = 16
BUFFER_SIZE = 3000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test = dataset['test'].map(load_image_test)

# TODO: confirm that augmentation worked
train_dataset = train.cache().map(augment_data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


for image, mask in train.take(1):
    sample_image, sample_mask = image, mask
    display([sample_image, sample_mask])

OUTPUT_CHANNELS = 4

# TODO: try to change this
base_model = tf.keras.applications.MobileNetV2(input_shape=[448, 448, 3], include_top=False)

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

down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]


def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=[448, 448, 3])
    x_i = inputs

    # Downsampling through the model
    skips = down_stack(x_i)
    x_i = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x_i = up(x_i)
        concat = tf.keras.layers.Concatenate()
        x_i = concat([x_i, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(output_channels, 3, strides=2, padding='same')  # 64x64 -> 128x128
    x_i = last(x_i)
    return tf.keras.Model(inputs=inputs, outputs=x_i)


model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer=Adam(learning_rate=0.001, amsgrad=False),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


def create_mask(p_mask):
    p_mask = tf.argmax(p_mask, axis=-1)
    p_mask = p_mask[..., tf.newaxis]
    return p_mask[0]


def show_predictions(ds=None, num=1):
    if ds:
        for im, msk in ds.take(num):
            pred_mask = model.predict(im[tf.newaxis, ...])
            display([im[0], msk[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask, create_mask(model.predict(sample_image[tf.newaxis, ...]))])


show_predictions()


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        show_predictions()
        print('\nSample Prediction after epoch {}\n'.format(epoch+1))


checkpoint = ModelCheckpoint(filepath="/home/ribeiro-desktop/blender_experiments/neural_networks/logs/current_run",
                             save_best_only=True)

VALIDATION_LENGTH = 1000
EPOCHS = 40
VAL_SUBSPLITS = 10
VALIDATION_STEPS = VALIDATION_LENGTH//BATCH_SIZE//VAL_SUBSPLITS

model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_dataset,
                          callbacks=[DisplayCallback(), checkpoint])

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(EPOCHS)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()

# Load best model from current run
best = tf.keras.models.load_model("/home/ribeiro-desktop/blender_experiments/neural_networks/logs/current_run")

for img, mask in test.take(1):
    test_image, test_mask = img, mask
    predicted_mask = create_mask(best.predict(test_image[tf.newaxis, ...]))
    display([test_image, test_mask, predicted_mask])


print("Train finished. Making some inferences")


def normalize_input(input_image):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image


def inference_on_image(image_file_path, classifier_model):
    target_size = max(classifier_model.layers[0].input_shape)
    im = Image.open(image_file_path)
    im = im.resize((target_size[1], target_size[2]), Image.ANTIALIAS)
    np_img = np.asarray(im)

    img_batch = np.expand_dims(np_img, axis=0)
    pre_processed_input = normalize_input(img_batch)[:, :, :, :3]
    pred_mask = create_mask(classifier_model.predict(pre_processed_input))
    display([np_img, pred_mask])


for i in range(1, 31):
    inference_on_image("/home/ribeiro-desktop/Pictures/Webcam/ri_" + str(i).zfill(7) + ".jpg", best)
    inference_on_image("/home/ribeiro-desktop/Pictures/Webcam/ri_" + str(i).zfill(7) + ".jpg", model)
