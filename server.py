from typing import Tuple, List, Text, Dict, Any, Iterator
import sys
sys.path.append("/usr/local/Cellar/opencv3/3.2.0/lib/python3.5/site-packages/") # mac opencv path
import cv2
import numpy as np
import time
from flask import Flask, request, json, send_file
from imgaug import augmenters as iaa
import keras.backend as  K
from keras.models import model_from_json
import skimage.io as io
import skimage.color as color
from model_unet import create_unet

folder = "./data/"

app = Flask(__name__, static_url_path='')


with K.tf.device('/cpu:0'):
    model = create_unet((512, 512, 3), 64, "binarize")
    #model.load_weights("./data/2017-04-28-11-11-28_fil64_adam_lr0.0001_glorot_uniform_dice_coef_weights.epoch0081-val_loss-0.75-val_dice_coef0.75.hdf5") # ancient
    model.load_weights("./data/2017-05-17-07-58-54_float32_binarize_fil64_adam_lr0.0001_glorot_uniform_shape512x512_batch_size8_weights.epoch0046-val_loss-0.77-val_dice_coef0.77.hdf5") # 復元モデル
    model2 = create_unet((512, 512, 4), 64, "heatmap")
    model2.load_weights("./data/2017-05-12-06-02-02_fil64_adam_lr0.0001_glorot_uniform_shape256x256_learn_head_data_aug_mean_squared_error_weights.epoch0081-val_loss129.64-val_acc0.71.hdf5")
    
    #model3 = create_unet((512, 512, 3), 64, "hydra")
    #model3.load_weights("./data/2017-05-17-07-36-40_float32_hydra_fil64_adam_lr0.0001_glorot_uniform_shape512x512_batch_size8_data_aug_weights.epoch0048-val_loss-0.77.hdf5")


@app.route('/')
def root():
    return app.send_static_file('index.html')

@app.route('/post', methods=['POST'])
def upload_file():
    if not request.method == 'POST':
        print("post it")
        return app.response_class(status=400)

    files = request.files.getlist("files")

    _files = []
    for (i, file) in enumerate(files):
        name = ("/tmp/img%d.img" % i)
        file.save(name)
        _files.append(name)

    if len(_files) == 0:
        print("no file")
        return app.response_class(status=400)
    
    filename = _files[0]

    print("processing...", _files)
    start = time.time()

    img = io.imread(filename)
    img = cv2.resize(img, (512, 512))
    if img.shape[2] == 4:
        print("drop alpha channel")
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    seq_noise = iaa.Sequential([
        iaa.ContrastNormalization(1.),
    ]) # type: iaa.Sequential
    img  = seq_noise.augment_images(img)

    img = np.expand_dims(img, axis=0)

    with K.tf.device('/cpu:0'):
        
        output1 = model.predict(img)
        
        #output3,output4 = model3.predict(img)
        output = output1# + output3
        output[output>1]=1
        
        img2 = np.expand_dims(np.dstack((img[0], output[0])), axis=0)
        output2 = model2.predict(img2)


    # postprocessing
    th1 = (output2[0]/output2[0].max()*255).astype("uint8") # to uint8
    th1[th1 < 128] = 0 # threashold or use FFT 
    labelnum, labelimg, contours, GoCs = cv2.connectedComponentsWithStats(th1)

    alpha = (output[0]/output[0].max()*255).astype("uint8") # to uint8
    alpha[alpha<64] = 64
    img3 = np.expand_dims(np.dstack((img[0], alpha)), axis=0) # alpha


    clips = [] # type: List[Tuple[int, int, int, int]]
    for label in range(labelnum):
        x,y,w,h,size = contours[label]
        if w == 512 and h == 512: continue
        if size < 5*5: continue
        if th1[y:y+h,x:x+w].max() < 192: continue

        clips.append((x,y,w,h))
        img3[0] = cv2.rectangle(img3[0], (x,y), (x+w,y+h), (0,255,0,255), 1)

    
    elapsed = time.time() - start
    print(elapsed, "sec, ", filename)

    if True:
        # return png
        filename += ".png"
        io.imsave(filename, img3[0].astype("uint8"))
        res = send_file(filename, mimetype='image/png')
    else:
        # return json
        res = app.response_class(
            response=json.dumps(clips),
            status=200,
            mimetype='application/json'
        )
        h = res.headers
        h['Access-Control-Allow-Origin'] = "*"
        h['Access-Control-Allow-Methods'] = "POST"
        h['Access-Control-Max-Age'] = "21600"

    return res





