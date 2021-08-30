import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import os
import numpy as np

from keras_example_cifar10_resnet.resnet import resnet_v1
from keras_example_cifar10_resnet.cifar10 import load_cifar10
from bias import add_bias

base_path = 'Results/Models'

scenarios = [
    {'bias': 'unbiased'}
] + [
    {'bias': 'size_bias', 'class_name': 'cat', 'error_percentage': error_percentage} for error_percentage in [30, 60, 90]
] + [
    {'bias': 'size_bias', 'class_name': 'ship', 'error_percentage': 60}
] + [
    {'bias': 'gaussian_noise_bias', 'class_name': 'cat', 'error_percentage': 60}
] + [
    {'bias': 'mislabeling_bias', 'error_percentage': error_percentage} for error_percentage in [30, 60, 80]
]

(x_train, y_train), (x_test, y_test) = load_cifar10()
input_shape = np.shape(x_test)[1:]
batch_size = 128
epochs = 200


def lr_schedule(epoch):
    # from keras example
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def create_callbacks(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
        factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(
        save_dir, 'best.h5'), verbose=2, save_best_only=True, save_weights_only=True, monitor='val_accuracy')
    logger = tf.keras.callbacks.CSVLogger(
        os.path.join(save_dir, 'training.log'), append=True)

    return [logger, checkpoint, lr_reducer, lr_scheduler]


for bias_args in scenarios:
    model_name = '_'.join([str(arg) for arg in list(bias_args.values())])

    model = resnet_v1(input_shape=input_shape, depth=20)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(
        learning_rate=lr_schedule(0)), metrics=['accuracy'])

    save_dir = os.path.join(base_path, model_name)
    callbacks = create_callbacks(save_dir=save_dir)

    x, y = add_bias(x_train, y_train, **bias_args)

    model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_data=(
        x_test, y_test), shuffle=True, callbacks=callbacks, verbose=1)
    model.save_weights(os.path.join(save_dir, 'latest.h5'))
