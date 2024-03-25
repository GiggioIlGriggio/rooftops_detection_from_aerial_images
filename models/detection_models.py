import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from keras_unet_collection import models, base
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Dropout, Activation, UpSampling2D, GlobalMaxPooling2D, multiply
from tensorflow.keras.backend import max

from models.losses import get_loss


def get_model(config):
    if config.model_name == "unet":
        model = build_unet(input_shape=(config.crop_size, config.crop_size, config.num_channels))
    if config.model_name == "attunet":
        model = build_attunet(input_shape=(config.crop_size,config.crop_size,config.num_channels), n_ch=32, L=3)
    if config.model_name == "unet_collection":
        model = models.att_unet_2d((config.crops_size, config.crops_size, config.num_channels), filter_num=[64, 128, 256, 512], n_labels=1,
                           stack_num_down=2, stack_num_up=2, activation='ReLU',
                           atten_activation='ReLU', attention='add', output_activation='Sigmoid',
                           batch_norm=True, pool=False, unpool=False,
                           backbone=config.backbone, weights=config.pretrained,
                           freeze_backbone=config.freeze_backbone, freeze_batch_norm=False,
                           name='attunet')
    if config.model_name == "unet_3plus_2d":
        model = build_unet3plu2d((config.crop_size, config.crop_size, config.num_channels))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=get_loss(config.loss),
              metrics=[tf.keras.metrics.BinaryAccuracy()])
    
    if config.checkpoint:
        print(f"Importing weights at checkpoint {config.checkpoint}...")
        model.load_weights(config.checkpoint)

    return model

def build_unet3plu2d(input_shape):
    name = 'unet3plus'
    activation = 'ReLU'
    filter_num_down = [32, 64, 128, 256, 512]
    filter_num_skip = [32, 32, 32, 32]
    filter_num_aggregate = 160

    stack_num_down = 2
    stack_num_up = 1
    n_labels = 1

    # `unet_3plus_2d_base` accepts an input tensor
    # and produces output tensors from different upsampling levels
    # ---------------------------------------- #
    input_tensor = keras.layers.Input(input_shape)
    # base architecture
    X_decoder = base.unet_3plus_2d_base(
        input_tensor, filter_num_down, filter_num_skip, filter_num_aggregate,
        stack_num_down=stack_num_down, stack_num_up=stack_num_up, activation=activation,
        batch_norm=True, pool=True, unpool=True, backbone=None, name=name)

    # allocating deep supervision tensors
    OUT_stack = []
    # reverse indexing `X_decoder`, so smaller tensors have larger list indices
    X_decoder = X_decoder[::-1]

    # deep supervision outputs
    for i in range(1, len(X_decoder)):
        # 3-by-3 conv2d --> upsampling --> sigmoid output activation
        pool_size = 2**(i)
        X = Conv2D(n_labels, 3, padding='same', name='{}_output_conv1_{}'.format(name, i-1))(X_decoder[i])

        X = UpSampling2D((pool_size, pool_size), interpolation='bilinear',
                        name='{}_output_sup{}'.format(name, i-1))(X)

        X = Activation('sigmoid', name='{}_output_sup{}_activation'.format(name, i-1))(X)
        # collecting deep supervision tensors
        OUT_stack.append(X)

    # the final output (without extra upsampling)
    # 3-by-3 conv2d --> sigmoid output activation
    X = Conv2D(n_labels, 3, padding='same', name='{}_output_final'.format(name))(X_decoder[0])
    X = Activation('sigmoid', name='{}_output_final_activation'.format(name))(X)
    # collecting final output tensors
    OUT_stack.append(X)

    # Classification-guided Module (CGM)
    # ---------------------------------------- #
    # dropout --> 1-by-1 conv2d --> global-maxpooling --> sigmoid
    X_CGM = X_decoder[-1]
    X_CGM = Dropout(rate=0.1)(X_CGM)
    X_CGM = Conv2D(filter_num_skip[-1], 1, padding='same')(X_CGM)
    X_CGM = GlobalMaxPooling2D()(X_CGM)
    X_CGM = Activation('sigmoid')(X_CGM)

    CGM_mask = max(X_CGM, axis=-1) # <----- This value could be trained with "none-organ image"

    for i in range(len(OUT_stack)):
        if i < len(OUT_stack)-1:
            # deep-supervision
            OUT_stack[i] = multiply([OUT_stack[i], CGM_mask], name='{}_output_sup{}_CGM'.format(name, i))
        else:
            # final output
            OUT_stack[i] = multiply([OUT_stack[i], CGM_mask], name='{}_output_final_CGM'.format(name))
    model = keras.models.Model([input_tensor,], OUT_stack)
    return model

############### UNET
def build_unet(input_shape):
    """Return a classic UNET model

    For our task, the purpose is the following. It takes in input a crop image
    (C, C, 3) and it returns the per-pixel binary predictions (C, C, 1).

    These binary predictions are per-pixel scores: the higher the value of a
    pixel, the higher the probability of being a positive pixel (i.e. pixel 
    belonging to a green area). 

    Parameters
    ----------
    input_shape : tuple of int
        Shape of the input crop image, i.e. (C, C, 3)

    Returns
    -------
    tensorflow.keras.Model
        UNET model
    """
    inputs = keras.Input(shape=input_shape)

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = keras.layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256, 512, 1024]:
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Activation("relu")(x)
        x = keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = keras.layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [1024, 512, 256, 128, 64, 32]:
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.UpSampling2D(2)(x)

        # Project residual
        residual = keras.layers.UpSampling2D(2)(previous_block_activation)
        residual = keras.layers.Conv2D(filters, 1, padding="same")(residual)
        x = keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    #outputs = keras.layers.Conv2D(1, 3, activation="sigmoid", padding="same")(
    #    x
    #)
    outputs = keras.layers.Conv2D(1, 3, padding="same")(
        x
    )
    # IMPORTANT: last layer must be `Activation` with `dtype='float32'`.
    # This because we are using mixed precision.
    # https://www.tensorflow.org/guide/mixed_precision
    outputs = keras.layers.Activation('sigmoid', dtype='float32')(outputs)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model



############### ATTENTION UNET
def conv_block(x, filter_size, size, dropout, batch_norm=True):
    conv = keras.layers.Conv2D(size, (filter_size, filter_size), padding="same")(x)
    if batch_norm is True:
        conv = keras.layers.BatchNormalization(axis=3)(conv)
    conv = keras.layers.Activation("relu")(conv)

    conv = keras.layers.Conv2D(size, (filter_size, filter_size), padding="same")(conv)
    if batch_norm is True:
        conv = keras.layers.BatchNormalization(axis=3)(conv)
    conv = keras.layers.Activation("relu")(conv)

    if dropout > 0:
        conv = keras.layers.Dropout(dropout)(conv)

    return conv

def repeat_elem(tensor, rep):
    # lambda function to repeat Repeats the elements of a tensor along an axis
    #by a factor of rep.
    # If tensor has shape (None, 256,256,3), lambda will return a tensor of shape
    #(None, 256,256,6), if specified axis=3 and rep=2.

     return keras.layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                          arguments={'repnum': rep})(tensor)

def res_conv_block(x, size, filter_size= 3, dropout= 0, batch_norm=True):
    '''
    Residual convolutional layer.
    Two variants....
    Either put activation function before the addition with shortcut
    or after the addition (which would be as proposed in the original resNet).

    1. conv - BN - Activation - conv - BN - Activation
                                          - shortcut  - BN - shortcut+BN

    2. conv - BN - Activation - conv - BN
                                     - shortcut  - BN - shortcut+BN - Activation

    Check fig 4 in https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf
    '''

    conv = keras.layers.Conv2D(size, (filter_size, filter_size), padding='same')(x)
    if batch_norm is True:
        conv = keras.layers.BatchNormalization(axis=3)(conv)
    conv = keras.layers.Activation('relu')(conv)

    conv = keras.layers.Conv2D(size, (filter_size, filter_size), padding='same')(conv)
    if batch_norm is True:
        conv = keras.layers.BatchNormalization(axis=3)(conv)
    #conv = keras.layers.Activation('relu')(conv)    #Activation before addition with shortcut
    if dropout > 0:
        conv = keras.layers.Dropout(dropout)(conv)

    shortcut = keras.layers.Conv2D(size, kernel_size=(1, 1), padding='same')(x)
    if batch_norm is True:
        shortcut = keras.layers.BatchNormalization(axis=3)(shortcut)

    res_path = keras.layers.add([shortcut, conv])
    res_path = keras.layers.Activation('relu')(res_path)    #Activation after addition with shortcut (Original residual block)
    return res_path

def attention_block(x, gating, inter_shape):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)

# Getting the x signal to the same shape as the gating signal
    theta_x = keras.layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
    shape_theta_x = K.int_shape(theta_x)

# Getting the gating signal to the same number of filters as the inter_shape
    phi_g = keras.layers.Conv2D(inter_shape, (1, 1), padding='same')(gating)
    upsample_g = keras.layers.Conv2DTranspose(inter_shape, (3, 3),
                                 strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                 padding='same')(phi_g)  # 16

    concat_xg = keras.layers.add([upsample_g, theta_x])
    act_xg = keras.layers.Activation('relu')(concat_xg)
    psi = keras.layers.Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = keras.layers.Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = keras.layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

    upsample_psi = repeat_elem(upsample_psi, shape_x[3])

    y = keras.layers.multiply([upsample_psi, x])

    result = keras.layers.Conv2D(shape_x[3], (1, 1), padding='same')(y)
    result_bn = keras.layers.BatchNormalization()(result)
    return result_bn

def gating_signal(input, out_size, batch_norm=True):
    """
    resize the down layer feature map into the same dimension as the up layer feature map
    using 1x1 conv
    :return: the gating feature map with the same dimension of the up layer feature map
    """
    x = keras.layers.Conv2D(out_size, (1, 1), padding='same')(input)
    if batch_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    return x


def build_attunet(input_shape, n_ch=32, L=3):
    """Return an attention UNET model

    For our task, the purpose is the following. It takes in input a crop image
    (C, C, 3) and it returns the per-pixel binary predictions (C, C, 1).

    These binary predictions are per-pixel scores: the higher the value of a
    pixel, the higher the probability of being a positive pixel (i.e. pixel 
    belonging to a green area). 

    Parameters
    ----------
    input_shape : tuple of int
        Shape of the input crop image, i.e. (C, C, 3)
    n_ch : int
        Number of channels at the beginning of the first model floor
    L : int
        Number of floors

    Returns
    -------
    tensorflow.keras.Model
        Attention UNET model
    """

    # L = number of floors

    inputs = keras.layers.Input(shape=input_shape)

    x= inputs

    # DOWN
    bLayers = []
    upLayers = []
    for l in range(L):
        x = res_conv_block(x, size= n_ch)
        bLayers.append(x)
        x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)
        n_ch = n_ch * 2

    x = res_conv_block(x, size= n_ch)
    upLayers.append(x)

    # UP
    for l in range(L):
        # Reduce the channels
        n_ch = n_ch // 2
        x = gating_signal(x, n_ch)
        a = attention_block(bLayers.pop(-1), x, n_ch)
        #u = keras.layers.UpSampling2D(size=(2, 2), data_format="channels_last")(upLayers.pop(-1))      #Upsampling
        u = keras.layers.Conv2DTranspose(n_ch, 3, 2, padding='same')(upLayers.pop(-1))                  #Convolutional upsapling layer
        u = keras.layers.concatenate([a, u], axis=3)
        x = res_conv_block(u, size= n_ch)
        upLayers.append(x)

    # Image of the first segmentation
    conv1 = keras.layers.Conv2D(1, kernel_size=(1,1))(x)
    conv1 = keras.layers.BatchNormalization(axis=3)(conv1)
    outputs = keras.layers.Activation('sigmoid', dtype='float32')(conv1)

    # Subtraction to get the second image
    #conv2 = 2*inputs - conv1

    # Output
    #result = keras.layers.concatenate([conv2,conv1], axis= 2)

    return keras.models.Model(inputs, outputs)