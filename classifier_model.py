"""
This module implements the image classifier model that uses a pretrained CPC to extract features.
"""

import keras

from cpc_model import network_encoder, get_custom_objects_cpc


def network_classifier(encoder_path, crop_shape, n_crops, code_size, lr, n_classes):
    """
    Builds a Keras model that make predictions of image crops using a pretrained CPC encoder to extract features.

    :param encoder_path: path to pretrained CPC encoder model.
    :param crop_shape: size of the image crops/patches (16, 16, 3).
    :param n_crops: resulting number of image crops (for example 7 for a 7x7 grid of crops).
    :param code_size: length of embedding vector.
    :param lr: optimizer's learning rate during training.
    :param n_classes: number of possible predicted classes.
    :return: compiled Keras model.
    """

    if encoder_path is not None:
        print('Reading encoder from disk and freezing weights.', flush=True)
        encoder_model = keras.models.load_model(encoder_path, custom_objects=get_custom_objects_cpc())
        encoder_model.trainable = False
        for l in encoder_model.layers:
            l.trainable = False
    else:
        encoder_input = keras.layers.Input(crop_shape)
        encoder_output = network_encoder(encoder_input, code_size)
        encoder_model = keras.models.Model(encoder_input, encoder_output, name='encoder')
        encoder_model.summary()

    # Crops feature extraction
    x_input = keras.layers.Input((n_crops, n_crops) + crop_shape)
    x = keras.layers.Reshape((n_crops * n_crops, ) + crop_shape)(x_input)
    x = keras.layers.TimeDistributed(encoder_model)(x)
    x = keras.layers.Reshape((n_crops, n_crops, code_size))(x)

    # # Define the classifier
    # x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x) #
    # x = LayerNormalization()(x)
    # x = keras.layers.LeakyReLU()(x)
    # x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='linear')(x) #
    # x = LayerNormalization()(x)
    # x = keras.layers.LeakyReLU()(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(units=code_size, activation='linear')(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(units=n_classes, activation='softmax')(x)

    # Model
    model = keras.models.Model(inputs=x_input, outputs=x)

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(lr=lr),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    model.summary()

    return model

