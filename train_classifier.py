"""
This module trains a classifier using a pretrained CPC encoder as feature extractor.
"""

from os.path import join, basename, dirname, exists
import keras
import os

from classifier_model import network_classifier
from data_generator import NCEGenerator
from prepare_data import augment_images_mnist


def train_classifier(input_dir, encoder_path, epochs, batch_size, output_dir, code_size,
                     lr=1e-3, train_step_multiplier=1.0, val_step_multiplier=1.0):

    """
    This function initializes and trains a digit classifier using a pretrained CPC model as feature extractor.

    :param input_dir: path to directory containing numpy training data (see NCEGenerator).
    :param encoder_path: path to pretrained Keras CPC encoder.
    :param epochs: number of times that the entire dataset will be used during training.
    :param batch_size: number of samples in the mini-batch.
    :param output_dir: directory to store the trained model.
    :param code_size: length of the embedding vector used in CPC.
    :param lr: learning rate.
    :param train_step_multiplier: percentage of training samples used in each epoch.
    :param val_step_multiplier: percentage of validation samples used in each epoch.
    :return: nothing.
    """

    # Output dir
    if not exists(output_dir):
        os.makedirs(output_dir)

    # Prepare data
    training_data = NCEGenerator(
        x_path=join(input_dir, 'training_x.npy'),
        y_path=join(input_dir, 'training_y.npy'),
        batch_size=batch_size,
        n_classes=10,
        n_negatives=0,
        augment_image_fn=augment_images_mnist,
        augment_crop_fn=None
    )
    validation_data = NCEGenerator(
        x_path=join(input_dir, 'validation_x.npy'),
        y_path=join(input_dir, 'validation_y.npy'),
        batch_size=batch_size,
        n_classes=10,
        n_negatives=0,
        augment_image_fn=None,
        augment_crop_fn=None
    )

    # Prepares the model
    model = network_classifier(
        encoder_path=encoder_path,
        crop_shape=(16, 16, 3),
        n_crops=7,
        code_size=code_size,
        lr=lr,
        n_classes=10
    )

    # Callbacks
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1/3, patience=2, min_lr=1e-5),
        keras.callbacks.CSVLogger(filename=join(output_dir, 'history.csv'), separator=',', append=True),
        keras.callbacks.ModelCheckpoint(filepath=join(output_dir, 'checkpoint.h5'), monitor='val_loss', save_best_only=True, mode='min'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, mode='min')
    ]

    # Trains the model
    model.fit_generator(
        generator=training_data,
        steps_per_epoch=int(len(training_data) * train_step_multiplier),
        validation_data=validation_data,
        validation_steps=int(len(validation_data) * val_step_multiplier),
        epochs=epochs,
        verbose=1,
        callbacks=callbacks
    )


if __name__ == '__main__':

    train_classifier(
        input_dir=join('.', 'resources', 'data'),
        encoder_path=join('.', 'resources', 'cpc_model', 'encoder_model.h5'),
        epochs=10,
        batch_size=32,
        output_dir=join('.', 'resources', 'classifier_model'),
        code_size=32,
        lr=1e-3,
        train_step_multiplier=1.0,
        val_step_multiplier=1.0
    )