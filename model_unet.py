from typing import Tuple, List, Text, Dict, Any

if __name__ == '__main__':
    import os
    #os.environ['KERAS_BACKEND'] = 'theano'
    #os.environ["THEANO_FLAGS"] = "exception_verbosity=high,optimizer=None,device=cpu"
    from keras.backend import set_image_data_format, set_floatx
    set_floatx('float16')
    # set_image_data_format('channels_first') # theano
    set_image_data_format("channels_last") # tensorflow

from keras.models import Model
from keras.layers import Input
from keras.layers.core import Activation, Dropout, Reshape
from keras.layers.merge import Concatenate, Multiply
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.backend import floatx




def create_enc_dec(input_tensor: Input, filters: int, mode: str, ker_init: str="glorot_uniform", neck: List[bool]=[True]) -> Any:
    # enc
    x =                       Conv2D(         filters*1, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=ker_init)( input_tensor )       ; e1 = x
    x = BatchNormalization()( Conv2D(         filters*2, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=ker_init)( LeakyReLU(0.2)(x) ) ); e2 = x
    x = BatchNormalization()( Conv2D(         filters*4, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=ker_init)( LeakyReLU(0.2)(x) ) ); e3 = x
    x = BatchNormalization()( Conv2D(         filters*8, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=ker_init)( LeakyReLU(0.2)(x) ) ); e4 = x
    x = BatchNormalization()( Conv2D(         filters*8, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=ker_init)( LeakyReLU(0.2)(x) ) ); e5 = x
    x = BatchNormalization()( Conv2D(         filters*8, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=ker_init)( LeakyReLU(0.2)(x) ) ); e6 = x
    x = BatchNormalization()( Conv2D(         filters*8, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=ker_init)( LeakyReLU(0.2)(x) ) ); e7 = x
    x =                       Conv2D(         filters*8, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=ker_init)( LeakyReLU(0.2)(x) )  ; e8 = x
    # dec
    outputs = [] # type: List[Any]
    for unet in neck:
        x = e8
        x = BatchNormalization()( Conv2DTranspose(filters*8, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=ker_init)( Activation("relu")(x) ) ); x = Concatenate()([Dropout(0.5)(x), e7]) if unet else Dropout(0.5)(x)
        x = BatchNormalization()( Conv2DTranspose(filters*8, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=ker_init)( Activation("relu")(x) ) ); x = Concatenate()([Dropout(0.5)(x), e6]) if unet else Dropout(0.5)(x)
        x = BatchNormalization()( Conv2DTranspose(filters*8, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=ker_init)( Activation("relu")(x) ) ); x = Concatenate()([Dropout(0.5)(x), e5]) if unet else Dropout(0.5)(x)
        x = BatchNormalization()( Conv2DTranspose(filters*8, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=ker_init)( Activation("relu")(x) ) ); x = Concatenate()([x, e4]) if unet else x
        x = BatchNormalization()( Conv2DTranspose(filters*4, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=ker_init)( Activation("relu")(x) ) ); x = Concatenate()([x, e3]) if unet else x
        x = BatchNormalization()( Conv2DTranspose(filters*2, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=ker_init)( Activation("relu")(x) ) ); x = Concatenate()([x, e2]) if unet else x
        x = BatchNormalization()( Conv2DTranspose(filters*1, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=ker_init)( Activation("relu")(x) ) ); x = Concatenate()([x, e1]) if unet else x
        outputs.append(x)
    if len(neck) == 1:
        return outputs[0]
    else:
        return outputs

def create_unet(in_shape: Tuple[int,int,int], filters: int, mode: str, ker_init: str="glorot_uniform", train=True) -> Model:
    '''
    reference models
    * https://github.com/phillipi/pix2pix/blob/master/models.lua#L47
    * https://github.com/tdeboissiere/DeepLearningImplementations/blob/master/pix2pix/src/model/models.py#L317
    '''

    # input1
    input_tensor = Input(shape=in_shape, dtype=floatx()) # type: Input

    if mode == "hydra":
        outputs = create_enc_dec(input_tensor, filters, mode, ker_init, neck=[True, False])
        assert len(outputs) == 2

        # binarize
        x = outputs[0]
        x = Conv2DTranspose(1, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=ker_init)( Activation("relu")(x) )
        x = Activation("sigmoid")(x)
        outputs[0] = Reshape((in_shape[0], in_shape[1]), name="output1")(x)

        # heatmap
        x = outputs[1]
        x = Conv2DTranspose(filters, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=ker_init)( Activation("relu")(x) )
        x = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding="same", kernel_initializer=ker_init)( Activation("relu")(x) )
        x = Activation("relu")(x)
        outputs[1] = Reshape((in_shape[0], in_shape[1]), name="output2")(x)

        if train == True:
            hydra = Model(inputs=[input_tensor], outputs=outputs)
        else:
            hydra = Model(inputs=[input_tensor], outputs=[Multiply()(outputs)])
        
        return hydra

    # 1st stage
    x = create_enc_dec(input_tensor, filters, mode, ker_init)

    if mode == "multistage":
        # output1
        x = Conv2DTranspose(1, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=ker_init)( Activation("relu")(x) )
        x = Activation("sigmoid")(x)
        output1 = Reshape((in_shape[0], in_shape[1]), name="output1")(x)
        
        # input2
        input_tensor2 = Concatenate()([input_tensor, x])

        # 2nd stage
        x = create_enc_dec(input_tensor2, filters, mode, ker_init)
        
        # output2
        x = Conv2DTranspose(filters, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=ker_init)( Activation("relu")(x) )
        x = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding="same", kernel_initializer=ker_init)( Activation("relu")(x) )
        x = Activation("relu")(x)
        output2 = Reshape((in_shape[0], in_shape[1]), name="output2")(x)

        unet = Model(inputs=[input_tensor], outputs=[output1, output2])

        return unet

    if mode == "heatmap" or mode == "integrated":
        x = Conv2DTranspose(filters, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=ker_init)( Activation("relu")(x) )
        x = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding="same", kernel_initializer=ker_init)( Activation("relu")(x) )
        x = Activation("relu")(x)
    elif mode == "binarize":
        x = Conv2DTranspose(1, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=ker_init)( Activation("relu")(x) )
        x = Activation("sigmoid")(x)
    else:
        raise Exception("unknown mode: "+mode)

    output = Reshape((in_shape[0], in_shape[1]))(x)

    unet = Model(inputs=[input_tensor], outputs=[output])
    
    return unet


if __name__ == '__main__':
    from keras.utils import plot_model
    unet = create_unet((256, 256, 3), 64, "binarize")
    unet.summary()
    unet.save_weights("unet.hdf5")
    with open('unet.json', 'w') as f: f.write(unet.to_json())

    plot_model(unet, to_file='unet.png', show_shapes=True, show_layer_names=True)
    
    exit()
