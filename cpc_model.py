"""
This module implements the contrastive-predictive-coding model for image processing.
"""

import keras
import tensorflow as tf
from keras import backend as K
import numpy as np
import math


def network_cpc(crop_shape, n_crops, code_size, learning_rate, ks, n_neg, pred_dir):
    """
    Main function defining the Contrastrive-Predictive-Coding model.

    Note about prediction offset (ks): the current implementation predicts up to 5 rows below. If we start computing
    context from the top row, we have to predict at least 2 rows below (1 row free). This is because image patches
    overlap with each other, therefore, a central patch contains pixels from the row below it. If we would predict
    context embeddings for the row just below, the information will flow directly from the central patch and the
    encoder will not learn anything useful. Therefore, if we have 7 rows, we can only predict up to 5. For example,
    starting from row 0, we can predict one of the following groups:
        Offset-2-row: [2,3,4,5,6]
        Offset-3-row: [3,4,5,6]
        Offset-4-row: [3,4,5]
        Offset-5-row: [3,4]
        Offset-6-row: [4]

    :param crop_shape: size of the image crops/patches (16, 16, 3).
    :param n_crops: resulting number of image crops (for example 7 for a 7x7 grid of crops).
    :param code_size: length of embedding vector.
    :param learning_rate: optimizer's learning rate during training.
    :param ks: number of prediction offsets to use. Warning: current implementation works with ks=5 only.
    :param n_neg: number of negative samples to compare with (crop-wise).
    :param pred_dir: number of prediction directions (top-bottom, bottom-top, left-right, right-left).
    :return: Keras model compiled for training.
    """

    # Define encoder model (maps an image crop into an embedding vector)
    encoder_input = keras.layers.Input(crop_shape)
    encoder_output = network_encoder(encoder_input, code_size)
    encoder_model = keras.models.Model(encoder_input, encoder_output, name='encoder')
    encoder_model.summary()

    # Crops feature extraction (apply encoder to image crops)
    x_input = keras.layers.Input((n_crops, n_crops) + crop_shape)
    x = keras.layers.Reshape((n_crops * n_crops, ) + crop_shape)(x_input)
    x = keras.layers.TimeDistributed(encoder_model)(x)
    x_encoded = keras.layers.Reshape((n_crops, n_crops, code_size))(x)

    # Negative crops feature extraction (apply encoder to negative samples)
    neg_input = keras.layers.Input((n_crops, n_crops, n_neg) + crop_shape)
    neg_x = keras.layers.Reshape((n_crops * n_crops * n_neg, ) + crop_shape)(neg_input)
    neg_x = keras.layers.TimeDistributed(encoder_model)(neg_x)
    neg_encoded = keras.layers.Reshape((n_crops, n_crops, n_neg, code_size))(neg_x)

    # Compute context and prediction for each direction
    x_pred_list = []
    for i in range(pred_dir):
        # Context: applies masked convolutions to the encoded image following a particular propagation direction
        x_context = network_context(x_encoded, code_size, pred_dir=i)

        # Prediction: maps context vectors into prediction vectors depending on the row offset
        x_pred = network_prediction(x_context, code_size, ks=ks)
        x_pred_list.append(x_pred)

    # Stack multiple directions
    x_pred = StackLayer()(x_pred_list)

    # Add CPC loss (returns probabilities of
    probs = CPCLayer()([x_encoded, neg_encoded, x_pred])

    # Model
    cpc_model = keras.models.Model(inputs=[x_input, neg_input], outputs=probs)

    # Compile model
    cpc_model.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate),
        loss=categorical_crossentropy_custom,
        metrics=[categorical_accuracy_custom]
    )
    cpc_model.summary()

    return cpc_model


class CPCLayer(keras.layers.Layer):

    """
    Computes the dot product between true and predicted embedding vectors.
    """

    def __init__(self, **kwargs):
        super(CPCLayer, self).__init__(**kwargs)

    def call(self, inputs):

        # Inputs
        # x_encoded dims [batch, rows, cols, code]
        # neg_encoded dims [batch, rows, cols, n_samples, code]
        # x_pred dims [batch, pred_dir, pred_offset, rows, cols, code]
        x_encoded, neg_encoded, x_pred = inputs

        # Concatenate positive and negative samples
        x_encoded = K.expand_dims(x_encoded, axis=-2)
        x_encoded = K.concatenate([x_encoded, neg_encoded], axis=-2)

        # Align embeddings
        dots = self.align_embeddings(x_pred, x_encoded)

        # Probs (-1 axis to select one sample among the negatives+correct images)
        probs = K.softmax(dots)

        return probs

    def align_embeddings(self, x_pred, x_encoded):
        """
        Aligns predicted and groundtruth embedded vectors. Predictions need to be compared (dot product) with embeddings
        in the right row. For example for top-to-bottom direction and offset of 2 rows (k=0), predictions in row 0-to-5
        are compared with encoded embeddings from row 2-to-7. If offset of 3 rows (k=1), predictions in row 0-to-4
        are compared with encoded embeddings from row 3-to-7.

        :param x_pred: predictions with dimensions [batch, pred_dir, pred_offset, rows, cols, code].
        :param x_encoded: groundtruth embeddings with dimensions [batch, rows, cols, n_samples, code].
        :return: dot products between all vectors with last dimension corresponding to "n_samples".
        """

        batch, pred_dir, pred_offset, rows, cols, code = K.int_shape(x_pred)
        dots = []
        for i in range(pred_dir):
            for k in range(pred_offset):

                # Alignment
                if i == 0:
                    pred, x = self.top_to_bottom(x_pred[:, i, :, :, :, :], x_encoded, k)
                elif i == 1:
                    pred, x = self.bottom_to_top(x_pred[:, i, :, :, :, :], x_encoded, k)
                elif i == 2:
                    pred, x = self.left_to_right(x_pred[:, i, :, :, :, :], x_encoded, k)
                elif i == 3:
                    pred, x = self.right_to_left(x_pred[:, i, :, :, :, :], x_encoded, k)
                else:
                    raise NotImplementedError('Invalid prediction orientation')

                # Dimensions
                x_shape = K.int_shape(x)
                n_rows = x_shape[1]
                n_cols = x_shape[2]
                n_samples = x_shape[3]
                n_code = x_shape[4]

                # Repeat preds
                pred = K.stack([pred] * n_samples, axis=3)

                # Dot product
                dot_product = K.sum(x * pred, axis=-1)
                dot_product = K.reshape(dot_product, (-1, n_rows * n_cols, n_samples))
                dots.append(dot_product)

        # Compute total loss
        dots = K.concatenate(dots, axis=-2)  # concat along ijkp

        return dots

    def top_to_bottom(self, x_pred, x_encoded, k):

        if k == 0:
            pred = x_pred[:, k, 0:5, :, :]
            x = x_encoded[:, 2:7, :, :, :]
        elif k == 1:
            pred = x_pred[:, k, 0:4, :, :]
            x = x_encoded[:, 3:7, :, :, :]
        elif k == 2:
            pred = x_pred[:, k, 0:3, :, :]
            x = x_encoded[:, 4:7, :, :, :]
        elif k == 3:
            pred = x_pred[:, k, 0:2, :, :]
            x = x_encoded[:, 5:7, :, :, :]
        elif k == 4:
            pred = x_pred[:, k, 0:1, :, :]
            x = x_encoded[:, 6:7, :, :, :]
        else:
            raise NotImplementedError('Invalid prediction offset k')

        return pred, x

    def bottom_to_top(self, x_pred, x_encoded, k):

        if k == 0:
            pred = x_pred[:, k, 2:7, :, :]
            x = x_encoded[:, 0:5, :, :, :]
        elif k == 1:
            pred = x_pred[:, k, 3:7, :, :]
            x = x_encoded[:, 0:4, :, :, :]
        elif k == 2:
            pred = x_pred[:, k, 4:7, :, :]
            x = x_encoded[:, 0:3, :, :, :]
        elif k == 3:
            pred = x_pred[:, k, 5:7, :, :]
            x = x_encoded[:, 0:2, :, :, :]
        elif k == 4:
            pred = x_pred[:, k, 6:7, :, :]
            x = x_encoded[:, 0:1, :, :, :]
        else:
            raise NotImplementedError('Invalid prediction offset k')

        return pred, x

    def left_to_right(self, x_pred, x_encoded, k):

        if k == 0:
            pred = x_pred[:, k, :, 0:5, :]
            x = x_encoded[:, :, 2:7, :, :]
        elif k == 1:
            pred = x_pred[:, k, :, 0:4, :]
            x = x_encoded[:, :, 3:7, :, :]
        elif k == 2:
            pred = x_pred[:, k, :, 0:3, :]
            x = x_encoded[:, :, 4:7, :, :]
        elif k == 3:
            pred = x_pred[:, k, :, 0:2, :]
            x = x_encoded[:, :, 5:7, :, :]
        elif k == 4:
            pred = x_pred[:, k, :, 0:1, :]
            x = x_encoded[:, :, 6:7, :, :]
        else:
            raise NotImplementedError('Invalid prediction offset k')

        return pred, x

    def right_to_left(self, x_pred, x_encoded, k):

        if k == 0:
            pred = x_pred[:, k, :, 2:7, :]
            x = x_encoded[:, :, 0:5, :, :]
        elif k == 1:
            pred = x_pred[:, k, :, 3:7, :]
            x = x_encoded[:, :, 0:4, :, :]
        elif k == 2:
            pred = x_pred[:, k, :, 4:7, :]
            x = x_encoded[:, :, 0:3, :, :]
        elif k == 3:
            pred = x_pred[:, k, :, 5:7, :]
            x = x_encoded[:, :, 0:2, :, :]
        elif k == 4:
            pred = x_pred[:, k, :, 6:7, :]
            x = x_encoded[:, :, 0:1, :, :]
        else:
            raise NotImplementedError('Invalid prediction offset k')

        return pred, x

    def compute_output_shape(self, input_shape):
        batch, rows, cols, n_samples, code = input_shape[1]
        batch, pred_dir, pred_offset, rows, cols, code = input_shape[2]
        return (batch, (1+2+3+4+5)*7*pred_dir, n_samples + 1)


def network_encoder(x, code_size):
    """
    Defines the neural network that maps image patches to embeddings.

    :param x: Keras input layer representing an image patch.
    :param code_size: length of latent vector.
    :return: Keras dense layer representing the embedded image vector.
    """

    x = keras.layers.Conv2D(filters=code_size//4, kernel_size=3, strides=2, activation='linear', padding='same')(x) # 8
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=code_size//4, kernel_size=3, strides=1, activation='linear', padding='same')(x) # 8
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=code_size//2, kernel_size=3, strides=2, activation='linear', padding='same')(x) # 4
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=code_size//2, kernel_size=3, strides=1, activation='linear', padding='same')(x) # 4
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=code_size//4*3, kernel_size=3, strides=2, activation='linear', padding='same')(x) # 2
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=code_size//4*3, kernel_size=3, strides=1, activation='linear', padding='same')(x) # 2
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=code_size, activation='linear')(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(units=code_size, activation='linear')(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(units=code_size, activation='linear', use_bias=False)(x)

    return x


def network_context(x, code_size, pred_dir):

    """
    This function applies masked convolutions to the encoded image following a particular propagation direction.

    :param x: input Keras layer.
    :param code_size: number of filters used in the masked convolutions.
    :param pred_dir: a number from 0 to 3 indicating the direction of the signal propagation (0: top-to-bottom,
    1: bottom-to-top, 2: left-to-right, 3: right-to-left).
    :return: output Keras layer.
    """

    # WARNING: do not use a receptive field larger than 3x3 (filter size or multiple layers)
    # Otherwise, information from target patches will leak and semantics will be ignored

    pd_map = {0: 'tb', 1: 'bt', 2: 'lr', 3: 'rl'}

    x = MaskedConvolution2D(mask_orientation=pd_map[pred_dir], filters=code_size, kernel_size=3, strides=1, activation='linear', padding='same')(x)  # rf: 3x3
    x = keras.layers.LeakyReLU()(x)
    x = MaskedConvolution2D(mask_orientation=pd_map[pred_dir], filters=code_size, kernel_size=3, strides=1, activation='linear', padding='same')(x)  # rf: 5x5
    x = keras.layers.LeakyReLU()(x)
    x = MaskedConvolution2D(mask_orientation=pd_map[pred_dir], filters=code_size, kernel_size=3, strides=1, activation='linear', padding='same')(x)  # rf: 7x7
    x = keras.layers.LeakyReLU()(x)
    x = MaskedConvolution2D(mask_orientation=pd_map[pred_dir], filters=code_size, kernel_size=3, strides=1, activation='linear', padding='same')(x)  # rf: 9x9
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=code_size, kernel_size=1, strides=1, activation='linear', padding='same')(x)

    return x


class MaskedConvolution2D(keras.layers.Conv2D):

    def __init__(self, *args, mask_orientation='tb', **kwargs):
        """
        Performs masked convolution by zeroing certain elements of the convolution filter. Since the convolution
        is always performed top to bottom, the image is rotated before and after to support other directions.

        :param args: arguments of Keras Conv2D layer.
        :param mask_orientation: propagation direction ('tb': top-to-botomm, 'bt': bottom-to-top, 'lr': left-to-right,
        'rl': right-to-left).
        :param kwargs: arguments of Keras Conv2D layer.
        """
        super().__init__(*args, **kwargs)
        self.mask_orientation = mask_orientation
        self.mask = None

    def build(self, input_shape):
        super().build(input_shape)

        # Create empty mask
        self.mask = np.zeros(self.kernel.shape.as_list())
        filter_size = self.mask.shape[0]
        filter_center = math.ceil(filter_size / 2)

        # Activate top rows
        self.mask[:filter_center, ...] = 1

        # Convert the numpy mask into a tensor mask.
        self.mask = K.variable(self.mask)

    def call(self, x, mask=None):

        # Rotate image according to orientation
        if self.mask_orientation == 'tb':
            pass
        elif self.mask_orientation == 'bt':
            x = K.map_fn(lambda l: tf.image.rot90(l, k=2), x)
        elif self.mask_orientation == 'lr':
            x = K.map_fn(lambda l: tf.image.rot90(l, k=3), x)
        elif self.mask_orientation == 'rl':
            x = K.map_fn(lambda l: tf.image.rot90(l, k=1), x)

        # Convolve
        outputs = K.conv2d(
            x,
            self.kernel * self.mask,  # masked kernel
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )

        # Restore image rotation according to orientation
        if self.mask_orientation == 'tb':
            pass
        elif self.mask_orientation == 'bt':
            outputs = K.map_fn(lambda l: tf.image.rot90(l, k=2), outputs)
        elif self.mask_orientation == 'lr':
            outputs = K.map_fn(lambda l: tf.image.rot90(l, k=1), outputs)
        elif self.mask_orientation == 'rl':
            outputs = K.map_fn(lambda l: tf.image.rot90(l, k=3), outputs)

        # Add bias
        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        # Add activation
        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def get_config(self):
        # Add the mask orientation property to the config.
        return dict(list(super().get_config().items()) + list({'mask_orientation': self.mask_orientation}.items()))


def network_prediction(context, code_size, ks=5):
    """
    This network makes predictions from the context vectors depending on the offset between context and target
    prediction (there are 5 offsets). For example, for the first offset, it predicts the vectors located 2 rows
    below the context row. For the second offset, it predicts 3 rows below, etc. Each of the 5 predictions is
    performed with a simple linear dense layer.

    Think of each of these networks as a "semantic mapping agent": starting from the same context, each network
    projects how the future should look like depending on the offset (how "2-row-offset-below" look like,
    "3-row-offset-below", etc.).

    :param context: input layer representing the context of the image (crop embeddings after masked convolution).
    :param code_size: length of prediction embeddings.
    :param ks: number of offset networks to use. Warning: current implementation only works with ks=5.
    :return: stack of prediction layers.
    """

    context_shape = K.int_shape(context)
    context = keras.layers.Reshape((-1, context_shape[-1]))(context)
    outputs = []
    for k in range(ks):

        # Predict
        x = keras.layers.TimeDistributed(keras.layers.Dense(units=code_size, activation="linear"))(context)
        x = keras.layers.Reshape(context_shape[1:-1] + (code_size, ))(x)
        outputs.append(x)

    # Stack
    output = StackLayer()(outputs)

    return output


class StackLayer(keras.layers.Layer):
    """
    Stacks a list of layers into one layer.
    """

    def __init__(self, **kwargs):
        super(StackLayer, self).__init__(**kwargs)

    def call(self, inputs):

        # Stack
        # if not isinstance(inputs, list):
        if len(inputs) == 1:
            output = keras.layers.Lambda(lambda x: K.expand_dims(x, axis=1))(inputs[0])
        else:
            output = keras.layers.Lambda(lambda x: K.stack(x, axis=1))(inputs)

        return output

    def compute_output_shape(self, input_shape):

        if isinstance(input_shape, list):
            return (input_shape[0][0], ) + (len(input_shape), ) + input_shape[0][1:]
        else:
            return (input_shape[0], 1, ) + input_shape[1:]


def get_custom_objects_cpc():
    """
    Function used to retrieve the custom objects required to load the CPC Keras model from disk.

    :return: dictionary with custom objects.
    """

    return {
        'CPCLayer': CPCLayer,
        'categorical_crossentropy_custom': categorical_crossentropy_custom,
        'categorical_accuracy_custom': categorical_accuracy_custom,
        'MaskedConvolution2D': MaskedConvolution2D,
        'StackLayer': StackLayer
    }


def categorical_crossentropy_custom(y_true, y_pred):
    """
    Customized categorical cross-entropy loss. Required to reshape predictions and groundtruth.
    """

    # Reshape
    y_true = K.reshape(y_true, (-1, K.int_shape(y_pred)[-1]))
    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))

    # Loss
    loss = keras.losses.categorical_crossentropy(y_true, y_pred)
    loss = K.mean(loss, axis=0, keepdims=True)  # avoids error: input and output batch size is different

    return loss


def categorical_accuracy_custom(y_true, y_pred):
    """
    Customized categorical accuracy. Required to reshape predictions and groundtruth.
    """

    # Reshape
    y_true = K.reshape(y_true, (-1, K.int_shape(y_pred)[-1]))
    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))

    # Acc
    acc = keras.metrics.categorical_accuracy(y_true, y_pred)
    acc = K.mean(acc, axis=0, keepdims=True)  # avoids error: input and output batch size is different

    return acc


if __name__ == '__main__':

    cpc_model = network_cpc(
        crop_shape=(16, 16, 3),
        n_crops=7,
        code_size=128,
        learning_rate=1e-3,
        ks=5,
        n_neg=19,
        pred_dir=2
    )

