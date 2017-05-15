# keras-human-detector

## install

```
git clone --recursive https://github.com/legokichi/keras-human-detector.git
pyenv shell anaconda3-4.1.1
sudo apt-get install graphviz
conda install theano pygpu
pip install tensorflow-gpu
pip install keras
pip install mypy
pip install pydot_ng
pip install imgaug
```

## type check

```
mypy --ignore-missing-imports train.py 
```

## show model

```
python model_unet.py
```

## train

```
source download_mscoco.sh
env CUDA_VISIBLE_DEVICES=0 python train.py --data_aug
tensorboard --port=8888 --logdir=log
jupyter notebook --ip=0.0.0.0
```

### resume

```
env CUDA_VISIBLE_DEVICES=0 python train.py --initial_epoch=5 --resume=2017-04-17-08-29-19_weights.epoch0005.hdf5 
```

## predict

use `predict.ipynb` or Web Server

### Web Server

#### setup

```
pip install gunicorn flask
```

#### on gnicorn

```
gunicorn -w 4  -b 0.0.0.0:8888  server:app
```

#### on flask

```
env FLASK_APP=server.py flask run --host=0.0.0.0 --port 8888
```

## model

![unet](https://raw.githubusercontent.com/legokichi/keras-human-detector/master/unet.png)

