import os
import gc
import tensorflow as tf
from .utils import configure_gpus
configure_gpus()

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from focal_loss import SparseCategoricalFocalLoss
from datetime import datetime
from argparse import ArgumentParser
from .PCB_dataset import PCB
from .utils import load_image_train, load_image_not_train, create_augmentation_function,\
    normalize_input, create_mask, custom_model, create_unet_model
from .hyperparameters_and_metrics import FREEZE_FACTOR, LEARNING_RATE, LOSS_FUNCTION, BRIGHTNESS_RANGE,\
    CONTRAST_MIN_RANGE, CONTRAST_MAX_RANGE, HUE_RANGE, GAUSSIAN_BLUR_PROBS, NOISE_STDEV,\
    ACCURACY, CROSSENTROPY, PRECISION, RECALL
from tensorboard.plugins.hparams import api as hp


BUFFER_SIZE = 128
OUTPUT_CHANNELS = 17
MODEL_OUTPUT_PATH = 'Models/Trained/'


# TODO: tensorboard for hyperparameters, metrics and images
def train_and_evaluate(idx, epochs, batch_size, train_length, validation_length, should_evaluate_model, logdir):
    # Preventing RAM issues
    tf.keras.backend.clear_session()
    gc.collect()
    tf.compat.v1.get_default_graph()._py_funcs_used_in_graph = []

    # Get timestamp back:
    timestamp = logdir.split("/")[-1]

    # Write hyperparameters and metrics to tensorboard
    with tf.summary.create_file_writer(logdir + '/hparam_tuning').as_default():
        hp.hparams_config(
            hparams=[
                FREEZE_FACTOR,
                LEARNING_RATE,
                BRIGHTNESS_RANGE,
                CONTRAST_MIN_RANGE,
                CONTRAST_MAX_RANGE,
                HUE_RANGE,
                GAUSSIAN_BLUR_PROBS,
                LOSS_FUNCTION,
                NOISE_STDEV,
            ],
            metrics=[
                hp.Metric(ACCURACY.name, display_name=ACCURACY.name),
                hp.Metric(CROSSENTROPY.name, display_name=CROSSENTROPY.name),
            ],
        )

    # Prepare data
    dataset_builder = PCB("Arduino_3q3_bs")
    dataset_builder.download_and_prepare()
    dataset = dataset_builder.as_dataset()
    train_data = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # We're aware that this is actually validation data, as propper test data uses real images in our application
    validation_data = dataset['test'].map(load_image_not_train)

    # TODO: maybe use .cache("./tf_data.cache")
    data_augmentation_function = create_augmentation_function(hparams[BRIGHTNESS_RANGE],
                                                              hparams[CONTRAST_MIN_RANGE],
                                                              hparams[CONTRAST_MAX_RANGE],
                                                              hparams[HUE_RANGE],
                                                              hparams[GAUSSIAN_BLUR_PROBS],
                                                              )
    train_dataset = train_data.map(data_augmentation_function).shuffle(BUFFER_SIZE).repeat().batch(batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    validation_dataset = validation_data.batch(batch_size)

    # TODO: try different feature extractors
    # Define model to be used
    model = create_unet_model(output_channels=OUTPUT_CHANNELS,
                              freeze_percentage=hparams[FREEZE_FACTOR],
                              noise_stdev=hparams[NOISE_STDEV])

    if hparams[LOSS_FUNCTION] == 'Perda Focal':
        loss_function = SparseCategoricalFocalLoss(gamma=8, from_logits=True, name="Perda_focal")
    elif hparams[LOSS_FUNCTION] == 'Entropia Cruzada':
        loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name="Entropia_cruzada"),

    model.compile(optimizer=Adam(learning_rate=hparams[LEARNING_RATE]),
                  loss=loss_function,
                  metrics=[ACCURACY, CROSSENTROPY],  # , PRECISION, RECALL],  # TODO: solve 'ValueError: Shapes (None, 448, 448, 17) and (None, 448, 448, 1) are incompatible'
                  )

    # Hyperparameters writer
    hp_writer = tf.summary.create_file_writer(logdir + '/hparam_tuning')

    # Callbacks
    current_model_output_path = MODEL_OUTPUT_PATH + str(idx) + '_train/' + timestamp
    os.makedirs(current_model_output_path, exist_ok=True)
    checkpoint = ModelCheckpoint(filepath=current_model_output_path, save_best_only=True, save_weights_only=True)
    early_stopping = EarlyStopping(patience=7)
    reduce_lr_on_plateau = ReduceLROnPlateau(patience=4, cooldown=5, min_delta=1e-5, factor=0.5)
    tensorboard = TensorBoard(log_dir=logdir, histogram_freq=1, profile_batch=0)
    hparams_callback = hp.KerasCallback(writer=hp_writer, hparams=hparams, trial_id=timestamp)

    # Show some training images samples in Tensorboard
    img_file_writer = tf.summary.create_file_writer(logdir + str("/images"))
    train_data_sample = train_dataset.take(16)

    with img_file_writer.as_default():
        for im, msk in train_data_sample:
            tf.summary.image("Entradas de treino", im, max_outputs=16, step=0,
                             description="Dados de entrada utilizados para treinamento, sujeitos a augmentation")
            tf.summary.image("Máscaras de treino", msk / 16, max_outputs=16, step=0,
                             description="Dados de rotulação utilizados para treinamento, sujeitos a augmentation")

    #TODO: imgfix
    # Show some validation images samples in Tensorboard
    # validation_data_batch = validation_data.batch(batch_size)
    # val_image_sample = validation_data_batch[0] / 255
    # with img_file_writer.as_default():
    #     tf.summary.image("Exemplos de validação", val_image_sample, max_outputs=16, step=0,
    #                      description="Dados utilizados para validação, não sujeitos a augmentation"))

    val_steps = validation_length//batch_size
    steps_per_epoch = (train_length // batch_size)
    model.fit(train_dataset,
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_data=validation_dataset,
              validation_steps=val_steps,
              callbacks=[
                         checkpoint,
                         early_stopping,
                         reduce_lr_on_plateau,
                         tensorboard,
                         hparams_callback,
                         ]
              )

    # Load best model weights configuration, according to validation data score
    model.load_weights(current_model_output_path)

    if should_evaluate_model:  # TODO: make evaluations over REAL (not synthetic), labeled data
        """
        ev_loss, ev_accuracy, ev_precision, ev_recall = model.evaluate(test_data, steps=(test_length // batch_size))

        # Send data to 'HParams' and 'Scalar' dashboards in Tensorboard
        hp_file_writer = tf.summary.create_file_writer(logdir + str("/test"))
        with hp_file_writer.as_default():
            tf.summary.scalar("Teste_" + ACCURACY, ev_accuracy, step=1)
            tf.summary.scalar("Teste_" + PRECISION, ev_precision, step=1)
            tf.summary.scalar("Teste_" + RECALL, ev_recall, step=1)
        """
        pass

    # TODO: this
    # Show some corresponding true label samples in Tensorboard

    #TODO: imgfix
    # Show some predictions in Tensorboard
    # pre_processed_input = normalize_input(validation_data_batch)[:, :, :, :3]
    # pred_masks = create_mask(model.predict(pre_processed_input))
    # with img_file_writer.as_default():
    #     tf.summary.image("Exemplos de dados de validação", pred_masks, max_outputs=16, step=0)

    _, validation_accuracy, validation_crossentropy = model.evaluate(validation_dataset)

    with hp_writer.as_default():
        hp.hparams(hparams)  # record the values used in this trial
        tf.summary.scalar(ACCURACY.name, validation_accuracy, step=1)
        tf.summary.scalar(CROSSENTROPY.name, validation_crossentropy, step=1)

    # Saves best model
    model.save(current_model_output_path.replace("_train", "_best") + "_acc_%.4f__ce_%.4f" % (validation_accuracy, validation_crossentropy))


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('-ev', '--evaluate', action='store_false', default=True,
                    help='Evaluate model using test data? Default: True')
    ap.add_argument('-i', '--idx', type=int, default=0,
                    help='Index. Use this index to train different models')
    ap.add_argument('-e', '--epochs', type=int, default=50,
                    help='Number of epochs. Default: 50')
    ap.add_argument('-b', '--batch_size', type=int, default=16,
                    help='Batch size. Default: 16')
    ap.add_argument('-tl', '--train_length', type=int, default=3000,
                    help='Number of training examples (NOT INCLUDING VALIDATION SAMPLES).'
                         ' Should match dataset used in \"anotations\\list.txt\"')
    ap.add_argument('-vl', '--validation_length', type=int, default=1000,
                    help='Number of validation examples. Should match dataset used in \"anotations\\list.txt\"')
    args = ap.parse_args()

    for _ in range(50):
        # Get a random hyperparameter configuration
        lr = LEARNING_RATE.domain.sample_uniform()
        fp = FREEZE_FACTOR.domain.sample_uniform()
        lf = LOSS_FUNCTION.domain.sample_uniform()
        br = BRIGHTNESS_RANGE.domain.sample_uniform()
        cmir = CONTRAST_MIN_RANGE.domain.sample_uniform()
        cmar = CONTRAST_MAX_RANGE.domain.sample_uniform()
        hr = HUE_RANGE.domain.sample_uniform()
        gbp = GAUSSIAN_BLUR_PROBS.domain.sample_uniform()
        nstd = NOISE_STDEV.domain.sample_uniform()

        hparams = {
            LEARNING_RATE: lr,
            FREEZE_FACTOR: fp,
            LOSS_FUNCTION: lf,
            BRIGHTNESS_RANGE: br,
            CONTRAST_MIN_RANGE: cmir,
            CONTRAST_MAX_RANGE: cmar,
            HUE_RANGE: hr,
            GAUSSIAN_BLUR_PROBS: gbp,
            NOISE_STDEV: nstd,
        }

        log_directory = "logs/" + str(args.idx) + "/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        train_and_evaluate(args.idx,
                           args.epochs,
                           args.batch_size,
                           args.train_length,
                           args.validation_length,
                           args.evaluate,
                           logdir=log_directory)
