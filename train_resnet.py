from typing import Tuple, List, Text, Dict, Any, Iterator, Union, Sized, Callable, cast
from datetime import datetime
import argparse
import sys
sys.path.append("/usr/local/Cellar/opencv3/3.2.0/lib/python3.5/site-packages/") # mac opencv path
import cv2
import numpy as np
np.random.seed(2017) # for reproducibility
import os
#os.environ['KERAS_BACKEND'] = 'theano'
#os.environ["THEANO_FLAGS"] = "exception_verbosity=high,optimizer=None,device=cpu"
#os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=cpu,floatX=float32,optimizer=fast_compile'
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.backend import set_image_data_format, set_floatx, floatx
# keras.backend.backend()
# keras.backend.set_epsilon(1e-07)
# keras.backend.epsilon()
#set_floatx('float16')
# keras.backend.floatx()
# set_image_data_format('channels_first') # theano
set_image_data_format("channels_last")
# keras.backend.image_data_format()
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, Callback, TensorBoard
from keras.optimizers import SGD, Adam
from keras.backend import tensorflow_backend
import keras.backend as K

from chainer.iterators import MultiprocessIterator, SerialIterator
from chainer.dataset.dataset_mixin import DatasetMixin

from model_unet import create_unet
from mscoco import CamVid, convert_to_keras_batch, CamVidCrowd


from matplotlib import pyplot as plt
import skimage.io as io
import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Activation, Dropout, Reshape
from keras.layers.merge import Concatenate, Multiply
from keras.layers.convolutional import Conv2D, Conv2DTranspose, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.backend import floatx
import keras.backend as K
from typing import Tuple









from keras.applications.resnet50 import ResNet50
def create_resunet(in_shape: Tuple[int,int,int], mode: str="relu", ker_init: str="glorot_uniform") -> Model:
    input_tensor = Input(shape=(512, 512, 3)) # power of 2 >= 256
    x = ZeroPadding2D(padding=(1,1))(input_tensor)

    model = resnet = ResNet50(include_top=False,
        #weights='imagenet',
        weights=None,
        input_tensor=x, input_shape=None)
    #for i, layer in enumerate(model.layers): layer.trainable = False
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy') # freeze
    
    '''
    resnet.layers[4].output # activation_1(256, 256, 64)
    resnet.layers[17].output # activation_4(127, 127, 256)
    resnet.layers[27].output # activation_7(127, 127, 256)
    resnet.layers[37].output # activation_10(127, 127, 256)
    resnet.layers[49].output # activation_13(64, 64, 512)
    resnet.layers[59].output # activation_16(64, 64, 512)
    resnet.layers[69].output # activation_19(64, 64, 512)
    resnet.layers[79].output # activation_22(64, 64, 512)
    resnet.layers[91].output # activation_25(32, 32, 1024)
    resnet.layers[101].output # activation_28(32, 32, 1024)
    resnet.layers[111].output # activation_31(32, 32, 1024)
    resnet.layers[121].output # activation_34(32, 32, 1024)
    resnet.layers[131].output # activation_37(32, 32, 1024)
    resnet.layers[141].output # activation_40(32, 32, 1024)
    resnet.layers[153].output # activation_40(16, 16, 2048)
    resnet.layers[163].output # activation_46(16, 16, 2048)
    '''
    
    x = model.layers[173].output # actiavtion_49(16, 16, 2048)

    #print(x.shape) 

    x = Activation('relu')(BatchNormalization()(Conv2DTranspose(filters=512, kernel_size=(4, 4), strides=(2,2), padding="same", kernel_initializer=ker_init)(x)))
    x = Concatenate()([Dropout(0.5)(x), resnet.layers[141].output]) # activation_40

    #print(x.shape) 

    x = Activation('relu')(BatchNormalization()(Conv2DTranspose(filters=512, kernel_size=(4, 4), strides=(2,2), padding="same", kernel_initializer=ker_init)(x)))
    x = Concatenate()([Dropout(0.5)(x), resnet.layers[79].output]) # activation_22

    #print(x.shape) 

    x = Activation('relu')(BatchNormalization()(Conv2DTranspose(filters=32, kernel_size=(4, 4), strides=(2,2), padding="same", kernel_initializer=ker_init)(x)))
    x = Concatenate()([Dropout(0.5)(x), resnet.layers[37].output]) # activation_10

    #print(x.shape) 

    x = Activation(mode)(Conv2DTranspose(1, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=ker_init)( x ))
    x = Reshape((int(in_shape[0]/2), int(in_shape[1]/2)), name="output1")(x)
    
    output_tensor = x
    
    model = Model(inputs=[input_tensor], outputs=[output_tensor])
    return model



def dice_coef(y_true: K.tf.Tensor, y_pred: K.tf.Tensor) -> K.tf.Tensor:
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return (2. * intersection + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)

def dice_coef_loss(y_true: K.tf.Tensor, y_pred: K.tf.Tensor) -> K.tf.Tensor:
    return -dice_coef(y_true, y_pred)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='U-net trainer from mscoco')

    name = "data/"
    name += datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    name += "_" + floatx()
    name += "_resnet-mask"

    print("name: ", name)

    batch_size = 8
    resize_shape = (512, 512)

    train = CamVidCrowd("data/annotations/instances_train2014.json", "data/train2014/", resize_shape)
    valid = CamVidCrowd("data/annotations/instances_val2014.json", "data/val2014/", resize_shape)


    train_iter = convert_to_keras_batch(
        MultiprocessIterator(
            train,
            batch_size=batch_size,
            n_processes=12,
            n_prefetch=120,
            shared_mem=1000*1000*5
        )
    ) # type: Iterator[Tuple[np.ndarray, np.ndarray]]

    valid_iter = convert_to_keras_batch(
        MultiprocessIterator(
            valid,
            batch_size=batch_size,
            #repeat=False,
            shuffle=False,
            n_processes=12,
            n_prefetch=120,
            shared_mem=1000*1000*5
        )
    ) # type: Iterator[Tuple[np.ndarray, np.ndarray]]


    
    old_session = tensorflow_backend.get_session()

    with K.tf.Graph().as_default():
        session = K.tf.Session("")
        tensorflow_backend.set_session(session)
        tensorflow_backend.set_learning_phase(1)
        

        loss = dice_coef_loss
        metrics = [dice_coef]
        filename = "_weights.epoch{epoch:04d}-val_loss{val_loss:.2f}-val_dice_coef{val_dice_coef:.2f}.hdf5"
        optimizer = Adam(lr=0.00001)
        model = create_resunet((512, 512, 3), mode="sigmoid")
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        callbacks = [] # type: List[Callback]

        callbacks.append(ModelCheckpoint(
            name+filename,
            verbose=1,
            save_best_only=False,
            save_weights_only=True,
            period=1,
        ))

        callbacks.append(TensorBoard(
            log_dir=name+'_log',
            histogram_freq=1,
            write_graph=False,
            write_images=False,
        ))

        hist = model.fit_generator(
            generator=train_iter,
            steps_per_epoch=int(len(cast(Sized, train))/batch_size),
            epochs=200,
            verbose=1,
            callbacks=callbacks,
            validation_data=valid_iter,
            validation_steps=60,
        )

        model.save_weights(name+'_weight_final.hdf5')
        with open(name+'_history.json', 'w') as f: f.write(repr(hist.history))

        tensorflow_backend.set_session(old_session)



