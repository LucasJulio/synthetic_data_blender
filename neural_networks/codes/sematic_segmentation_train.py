import os
import gc
from datetime import datetime
import tensorflow as tf
from .utils import Custom, load_image_train, load_image_test, augment_data, create_mask
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow_examples.models.pix2pix import pix2pix
from .PCB_dataset import PCB
from focal_loss import SparseCategoricalFocalLoss

DATASETS_MAIN_PATH = os.path.expanduser("/datasets/")
SELECTED_DATASET = "sample_dataset"
RUN_LOG_PATH = "/home/ribeiro-desktop/POLI/TCC/blender_experiments/neural_networks/logs/" \
               + datetime.now().strftime("%m-%d--%H-%M")


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
print("tf.__version__: ", tf.__version__)

# TODO: tensorboard for hyperparameters, metrics and images

dataset_builder = PCB()
dataset_builder.download_and_prepare()
dataset = dataset_builder.as_dataset()

tf.keras.backend.clear_session()
gc.collect()
tf.compat.v1.get_default_graph()._py_funcs_used_in_graph = []


TRAIN_LENGTH = 6000
BATCH_SIZE = 16
BUFFER_SIZE = 512
STEPS_PER_EPOCH = (TRAIN_LENGTH // BATCH_SIZE)  # TODO: investigate memory leak

train = dataset['train'].map(load_image_train)  # , num_parallel_calls=tf.data.experimental.AUTOTUNE)
test = dataset['test'].map(load_image_test)

# TODO: confirm that augmentation worked
train_dataset = train.map(augment_data).shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)  # .cache("./tf_data.cache")
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


OUTPUT_CHANNELS = 17

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


# TODO: try freezing batch normalization layers
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

    # These are the last layers of the model
    last = tf.keras.layers.Conv2DTranspose(output_channels, 3, strides=2, padding='same')  # 64x64 -> 128x128
    x_i = last(x_i)
    return tf.keras.Model(inputs=inputs, outputs=x_i)


model = unet_model(OUTPUT_CHANNELS)
# model = Custom(OUTPUT_CHANNELS)  # TODO: try this again


# loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
model.compile(optimizer=Adam(learning_rate=0.001, amsgrad=False),
              loss=SparseCategoricalFocalLoss(gamma=8, from_logits=True),
              metrics=['accuracy'])


def show_predictions(ds=None, num=1):
    if ds:
        for im, msk in ds.take(num):
            pred_mask = model.predict(im[tf.newaxis, ...])
            display([im[0], msk[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask, create_mask(model.predict(sample_image[tf.newaxis, ...]))])


show_predictions()


# TODO: replace this with tensorboard images
class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        show_predictions()
        print('\nSample Prediction after epoch {}\n'.format(epoch+1))


checkpoint = ModelCheckpoint(filepath=RUN_LOG_PATH, save_best_only=True)

early_stopping = EarlyStopping(patience=5)

VALIDATION_LENGTH = 2000
EPOCHS = 50
VAL_SUBSPLITS = 10
VALIDATION_STEPS = VALIDATION_LENGTH//BATCH_SIZE//VAL_SUBSPLITS

with tf.device('/device:gpu:0'):
    model.fit(train_dataset, epochs=EPOCHS,
              steps_per_epoch=STEPS_PER_EPOCH,
              validation_steps=VALIDATION_STEPS,
              validation_data=test_dataset,
              callbacks=[DisplayCallback(), checkpoint, early_stopping]
              )


# Load best model from current run
best = tf.keras.models.load_model(RUN_LOG_PATH)

for img, mask in test.take(1):
    test_image, test_mask = img, mask
    predicted_mask = create_mask(best.predict(test_image[tf.newaxis, ...]))
    display([test_image, test_mask, predicted_mask])
