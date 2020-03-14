"""
Main module training the CPC model.
"""

from os.path import join, basename, dirname, exists
import keras
import os
from data_generator import NCEGenerator
from cpc_model import network_cpc, get_custom_objects_cpc
from prepare_data import augment_images_mnist, augment_crops_mnist


def train_cpc(input_dir, epochs, batch_size, output_dir, code_size, lr=1e-3, train_step_multiplier=1.0,
              val_step_multiplier=1.0):
    """
    This function initializes and trains an instance of the contrastive-predictive-coding model for images.

    :param input_dir: path to directory containing numpy training data (see NCEGenerator).
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
        n_negatives=19,
        augment_image_fn=augment_images_mnist,
        augment_crop_fn=augment_crops_mnist,
        label_dim_mul=(15 * 7) * 2,
        neg_same_ratio=0.75
    )
    validation_data = NCEGenerator(
        x_path=join(input_dir, 'validation_x.npy'),
        y_path=join(input_dir, 'validation_y.npy'),
        batch_size=batch_size,
        n_classes=10,
        n_negatives=19,
        augment_image_fn=augment_images_mnist,
        augment_crop_fn=augment_crops_mnist,
        label_dim_mul=(15 * 7) * 2,
        neg_same_ratio=0.75
    )

    # Prepares the model
    model = network_cpc(
        crop_shape=(16, 16, 3),
        n_crops=7,
        code_size=code_size,
        learning_rate=lr,
        ks=5,
        n_neg=19,
        pred_dir=2
    )

    # Stores architecture in disk
    with open(join(output_dir, 'architecture.json'), 'w') as f:
        f.write(model.to_json(sort_keys=True, indent=4))

    # Callbacks
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1/3, patience=2, min_lr=1e-5),
        keras.callbacks.CSVLogger(filename=join(output_dir, 'history.csv'), separator=',', append=True),
    ]

    # Trains the model
    model.fit_generator(
        generator=training_data,
        steps_per_epoch=int(len(training_data) * train_step_multiplier),
        validation_data=validation_data,
        validation_steps=int(len(validation_data) * val_step_multiplier),
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
    )

    # Saves the model
    model.save(join(output_dir, 'cpc_model.h5'))

    # Saves the encoder
    encoder = model.layers[2].layer
    encoder.save(join(output_dir, 'encoder_model.h5'))


if __name__ == '__main__':

    train_cpc(
        input_dir=join('.', 'resources', 'data'),
        epochs=5,
        batch_size=8,
        output_dir=join('.', 'resources', 'cpc_model'),
        code_size=32,
        lr=1e-3,
        train_step_multiplier=0.1,
        val_step_multiplier=0.05
    )
