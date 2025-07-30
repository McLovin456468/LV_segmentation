from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, Dropout, Concatenate, Activation, ReLU

def encoder_block(x, filters, use_batchnorm=True):
    x = Conv2D(filters, (3, 3), strides=2, padding='same',
               kernel_initializer=HeNormal(), kernel_regularizer=l2(1e-4))(x)
    if use_batchnorm:
        x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def decoder_block(x, skip, filters, use_dropout=False):
    x = Conv2DTranspose(filters, (3, 3), strides=2, padding='same',
                        kernel_initializer=HeNormal(), kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    if use_dropout:
        x = Dropout(0.3)(x)
    x = Concatenate()([x, skip])
    x = ReLU()(x)
    return x

def build_light_unet(input_shape=(256, 256, 1)):
    inputs = Input(shape=input_shape)

    e1 = encoder_block(inputs, 32, use_batchnorm=False)
    e2 = encoder_block(e1, 64)
    e3 = encoder_block(e2, 128)
    e4 = encoder_block(e3, 256)

    b = Conv2D(512, (3, 3), strides=2, padding='same',
               kernel_initializer=HeNormal(), kernel_regularizer=l2(1e-4))(e4)
    b = ReLU()(b)
    b = Dropout(0.5)(b)

    d1 = decoder_block(b, e4, 256)
    d2 = decoder_block(d1, e3, 128)
    d3 = decoder_block(d2, e2, 64)
    d4 = decoder_block(d3, e1, 32)

    outputs = Conv2DTranspose(1, (3, 3), strides=2, padding='same',
                              kernel_initializer=HeNormal())(d4)
    outputs = Activation('sigmoid')(outputs)

    return Model(inputs, outputs)

