from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.metrics import SparseCategoricalAccuracy, SparseCategoricalCrossentropy, Precision, Recall


# Hyperparameters
FREEZE_FACTOR = hp.HParam('Percentual de congelamento', hp.RealInterval(80.0, 100.0))
LEARNING_RATE = hp.HParam('Taxa de aprendizado', hp.RealInterval(0.0001, 0.01))
BRIGHTNESS_RANGE = hp.HParam('Variação de brilho', hp.RealInterval(0.0, 0.3))
CONTRAST_MIN_RANGE = hp.HParam('Variação mínima de contraste', hp.RealInterval(0.6, 1.0))
CONTRAST_MAX_RANGE = hp.HParam('Variação máxima de contraste', hp.RealInterval(1.0, 1.4))
HUE_RANGE = hp.HParam('Variação de matiz', hp.RealInterval(0.0, 0.4))
GAUSSIAN_BLUR_PROBS = hp.HParam('Probabilidades de borramento', hp.RealInterval(0.0, 0.4))
NOISE_STDEV = hp.HParam('Desvio-padrão de ruído gaussiano', hp.RealInterval(0.001, 0.1))
LOSS_FUNCTION = hp.HParam('Função de perda', hp.Discrete(['Perda Focal',
                                                          'Entropia Cruzada']))

# Metrics
ACCURACY = SparseCategoricalAccuracy(name='Acurácia')
CROSSENTROPY = SparseCategoricalCrossentropy(name='Entropia Cruzada', from_logits=True)
PRECISION = Precision(name='Precisão')
RECALL = Recall(name='Revocação')
