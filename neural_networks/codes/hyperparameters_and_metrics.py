from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.metrics import SparseCategoricalAccuracy, Precision, Recall


# Hyperparameters
FREEZE_FACTOR = hp.HParam('Percentual de congelamento', hp.RealInterval(50.0, 100.0))
LEARNING_RATE = hp.HParam('Taxa de aprendizado', hp.RealInterval(0.0001, 0.01))
BRIGHTNESS_RANGE = hp.HParam('Variação de Brilho', hp.RealInterval(0.0, 0.3))
CONTRAST_MIN_RANGE = hp.HParam('Variação Mínima de Contraste', hp.RealInterval(0.6, 1.0))
CONTRAST_MAX_RANGE = hp.HParam('Variação Máxima de Contraste', hp.RealInterval(1.0, 1.4))
HUE_RANGE = hp.HParam('Variação de Matiz', hp.RealInterval(0.0, 0.4))
GAUSSIAN_BLUR_PROBS = hp.HParam('Probabilidades de Borramento', hp.RealInterval(0.0, 0.6))
LOSS_FUNCTION = hp.HParam('Função de perda', hp.Discrete(['sparse_categorical_focal_loss',
                                                          'sparse_categorical_crossentropy']))

# Metrics
ACCURACY = SparseCategoricalAccuracy(name='Acurácia')
PRECISION = Precision(name='Precisão')
RECALL = Recall(name='Revocação')
