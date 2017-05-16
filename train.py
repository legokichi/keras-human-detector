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
from mscoco import CamVid, convert_to_keras_batch

def dice_coef(y_true: K.tf.Tensor, y_pred: K.tf.Tensor) -> K.tf.Tensor:
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return (2. * intersection + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)

def dice_coef_loss(y_true: K.tf.Tensor, y_pred: K.tf.Tensor) -> K.tf.Tensor:
    return -dice_coef(y_true, y_pred)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='U-net trainer from mscoco')
    parser.add_argument("--epochs",  action='store', type=int, default=100, help='epochs')
    parser.add_argument("--resume",  action='store', type=str, default="", help='*_weights.hdf5')
    parser.add_argument("--initial_epoch", action='store', type=int, default=0, help='initial_epoch')
    parser.add_argument("--ker_init", action='store', type=str, default="glorot_uniform", help='conv2D kernel initializer')
    parser.add_argument("--lr", action='store', type=float, default=0.0001, help='learning late')
    parser.add_argument("--optimizer", action='store', type=str, default="adam", help='adam|nesterov')
    parser.add_argument("--filters", action='store', type=int, default=64, help='32|64|128')
    parser.add_argument("--batch_size", action='store', type=int, default=8, help='batch_size')
    parser.add_argument("--dir", action='store', type=str, default="./", help='mscoco dir')
    parser.add_argument("--data_aug", action='store_true', help='use data augmentation')
    parser.add_argument("--shape", action='store', type=int, default=256, help='input size width & height (power of 2)')
    parser.add_argument("--mode", action='store', type=str, default="heatmap", help='heatmap | binarize')

    args = parser.parse_args()

    name = args.dir + "/"
    name += datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    name += "_" + floatx()
    name += "_" + args.mode
    name += "_fil" + str(args.filters)
    name += "_" + args.optimizer
    name += "_lr" + str(args.lr)
    name += "_" + args.ker_init
    name += "_shape" + str(args.shape) + "x" + str(args.shape)
    name += "_batch_size" + str(args.batch_size)
    if args.data_aug: name += "_data_aug"

    print("name: ", name)

    resize_shape = (args.shape, args.shape)

    train = CamVid(args.mode, args.dir+"/annotations/person_keypoints_train2014.json", args.dir+"/annotations/instances_train2014.json", args.dir+"/train2014/", resize_shape, data_aug=True) # type: DatasetMixin
    valid = CamVid(args.mode, args.dir+"/annotations/person_keypoints_val2014.json",   args.dir+"/annotations/instances_val2014.json",   args.dir+"/val2014/",   resize_shape) # type: DatasetMixin

    print("train:", len(train))
    print("valid:", len(valid))

    train_iter = convert_to_keras_batch(
        MultiprocessIterator(
            train,
            batch_size=args.batch_size,
            n_processes=12,
            n_prefetch=120,
            shared_mem=1000*1000*5
        )
    ) # type: Iterator[Tuple[np.ndarray, np.ndarray]]

    valid_iter = convert_to_keras_batch(
        MultiprocessIterator(
            valid,
            batch_size=args.batch_size,
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
        

        if args.mode == "heatmap":
            input_shape = (resize_shape[0], resize_shape[1], 4)
            loss = "mean_squared_error" # type: Any
            metrics = ["accuracy"] # type: Any
            filename = "_weights.epoch{epoch:04d}-val_loss{val_loss:.2f}-val_acc{val_acc:.2f}.hdf5"
        elif args.mode == "binarize":
            input_shape = (resize_shape[0], resize_shape[1], 3)
            loss = dice_coef_loss
            metrics = [dice_coef]
            filename = "_weights.epoch{epoch:04d}-val_loss{val_loss:.2f}-val_dice_coef{val_dice_coef:.2f}.hdf5"
        elif args.mode == "integrated":
            input_shape = (resize_shape[0], resize_shape[1], 3)
            loss = "mean_squared_error"
            metrics = ["accuracy"]
            filename = "_weights.epoch{epoch:04d}-val_loss{val_loss:.2f}-val_acc{val_acc:.2f}.hdf5"
            train.DISTANCE_SCALE = 8.0
            val.DISTANCE_SCALE = 8.0
        else:
            raise Exception("unknown mode: "+args.mode)
        
        model = create_unet(input_shape, args.filters, args.mode, args.ker_init)

        if args.optimizer == "nesterov":
            optimizer = SGD(lr=args.lr, momentum=0.9, decay=0.0005, nesterov=True)
        else:
            optimizer = Adam(lr=args.lr)#, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=0.0)

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

        if len(args.resume) > 0:
            model.load_weights(args.resume)

        with open(name+'_model.json', 'w') as f: f.write(model.to_json())

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
            steps_per_epoch=int(len(cast(Sized, train))/args.batch_size),
            epochs=args.epochs,
            verbose=1,
            callbacks=callbacks,
            validation_data=valid_iter,
            validation_steps=60,
            initial_epoch=args.initial_epoch,
        )

        model.save_weights(name+'_weight_final.hdf5')
        with open(name+'_history.json', 'w') as f: f.write(repr(hist.history))

    tensorflow_backend.set_session(old_session)

    
