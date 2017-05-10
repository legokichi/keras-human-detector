from typing import Tuple, List, Text, Dict, Any, Iterator, Union, Sized, Callable
import sys
import os
import numpy as np
sys.path.append("/usr/local/Cellar/opencv3/3.2.0/lib/python3.5/site-packages/") # mac opencv path
import cv2
import skimage.io as io

from imgaug import augmenters as iaa
sys.path.append("./coco/PythonAPI/")

from pycocotools.coco import COCO
from pycocotools import mask as coco_mask

from chainer.iterators import MultiprocessIterator, SerialIterator
from chainer.dataset.dataset_mixin import DatasetMixin


def check(coco: COCO, info: dict, drop_crowd=False, drop_small=False, need_head=False, need_body=False)-> bool:
    '''
    mscoco の画像から使えそうなものを判定する
    '''
    anns = coco.loadAnns(coco.getAnnIds(imgIds=[info['id']], iscrowd=0)) # type: List[dict]
    
    centers = [(ann["bbox"][0]+ann["bbox"][2]/2, ann["bbox"][1]+ann["bbox"][3]/2) for ann in anns]

    for i, ann in enumerate(anns):
        
        # drop if people are overlapping
        if drop_crowd:
            x, y, w, h = ann["bbox"]
            for j, (cx, cy) in enumerate(centers):
                if i == j: continue
                if x < cx < x+w and y < cy < y+h: return False

        # drop this person if parts number is too low
        if ann["num_keypoints"] < 5: return False

        # drop if segmentation area is too small
        if drop_small and ann["area"] < 64*64: return False
        
        keys = ann["keypoints"] # type: List[int]

        # drop if no head
        if not( # 2 is visible
            need_head
            and keys[0*3+2] == 2 # nose
            or keys[1*3+2] == 2 # l-eye
            or keys[2*3+2] == 2 # r-eye
            or keys[3*3+2] == 2 # l-ear
            or keys[4*3+2] == 2 # r-ear
        ): return False

        # drop if no body
        if (
            need_body
            and not(keys[5*3+2] == 2 or keys[6*3+2] == 2) # shoulder
            or not(keys[7*3+2] == 2 or keys[8*3+2] == 2) # elbow
            or not(keys[11*3+2] == 2 or keys[12*3+2] == 2) # llium
        ): return False

    return True

def load_image(info: dict, dir: str) -> np.ndarray :
    if(dir != None): img = io.imread(dir + info['file_name']) # type: np.ndarray
    else: img = io.imread(info['coco_url'])
    if    img.ndim == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif  img.ndim == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def create_mask(coco: COCO, info: dict) -> np.ndarray:
    anns = coco.loadAnns(coco.getAnnIds(imgIds=[info['id']], iscrowd=False)) # type: List[dict]
    w, h = info["width"], info["height"] # type: Tuple[int, int]
    mask_all = np.zeros((h, w), np.uint8) # type: np.ndarray

    for ann in anns:
        rles = coco_mask.frPyObjects(ann["segmentation"], h, w) # type: List[dict]
        for rle in rles:
            mask = coco_mask.decode(rle) # type: np.ndarray
            mask[mask > 0] = 255
            mask_all += mask

    return mask_all

class CamVid(DatasetMixin):
    def __init__(self, json_path: str, img_path: str, resize_shape: Tuple[int, int]=None, use_data_check: bool=False, data_aug: bool=False):
        self.data_aug = data_aug
        self.img_path = img_path
        self.resize_shape = resize_shape # type: Tuple[int, int]
        self.coco = COCO(json_path) # type: COCO
        coco = self.coco
        infos = coco.loadImgs(coco.getImgIds(catIds=coco.getCatIds(catNms=['person']))) # type: List[dict]
        self.infos = infos # type: List[dict]
        if use_data_check:
            self.infos = [info for info in infos if check(coco, info)]
        self.seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            #iaa.Crop(px=((0, 50), (0, 50), (0, 50), (0, 50))), # crop images from each side by 0 to 16px (randomly chosen)
            #iaa.Affine(
            #    rotate=(-10, 10), # rotate by -45 to +45 degrees
            #    shear=(-4, 4), # shear by -16 to +16 degrees
            #    translate_px={"x": (-16, 16), "y": (-16, 16)}, # translate by -16 to +16 pixels (per axis)
            #    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            #),
        ]) # type: iaa.Sequential
        self.seq_noise = iaa.Sequential([
            iaa.GaussianBlur(sigma=(0, 0.5)),
            iaa.AdditiveGaussianNoise(scale=(0., 0.1*255), per_channel=0.5),
            iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
        ]) # type: iaa.Sequential
    def __len__(self) -> int:
        return len(self.infos)
    def get_example(self, i) -> Tuple[np.ndarray, np.ndarray]:
        info = self.infos[i]

        img = load_image(info, self.img_path)
        mask = create_mask(self.coco, info)

        if self.data_aug:
            # data augumentation
            seq_det = self.seq.to_deterministic()
            
            img  = np.expand_dims(img, axis=0)
            mask = np.expand_dims(mask, axis=0)
            img  = seq_det.augment_images(img)
            mask = seq_det.augment_images(mask)
            img  = self.seq_noise.augment_images(img)
            img  = np.squeeze(img)
            mask = np.squeeze(mask)

        # resize
        img  = cv2.resize(img, self.resize_shape)
        mask = cv2.resize(mask, self.resize_shape)

        mask = mask > 0

        return (img, mask)



if __name__ == '__main__':
    import argparse
    import cProfile
    import pstats
    import time
    from chainer.iterators import MultiprocessIterator, SerialIterator
    from train import convert_to_keras_batch
    import math

    parser = argparse.ArgumentParser(description='mscoco data generator self test')
    parser.add_argument("--dir", action='store', type=str, default="./", help='mscoco dir')
    args = parser.parse_args()

    resize_shape = (256, 256)

    train = CamVid(args.dir+"/annotations/person_keypoints_train2014.json", args.dir+"/train2014/", resize_shape, use_data_check=True, data_aug=True) # type: DatasetMixin
    valid = CamVid(args.dir+"/annotations/person_keypoints_val2014.json",   args.dir+"/val2014/",   resize_shape) # type: DatasetMixin

    print("train:"len(train),"valid:",len(valid))

    for mx in [train, valid]:
        print("start")
        it = convert_to_keras_batch(
            MultiprocessIterator(
                mx,
                batch_size=8,
                repeat=False,
                shuffle=False,
                n_processes=12,
                n_prefetch=120,
                shared_mem=1000*1000*5
            )
        ) # type: Iterator[Tuple[np.ndarray, np.ndarray]]

        for i,(_, (img,mask)) in enumerate(zip(range(math.floor(len(mx)/8)), it)):
            print(i, img.shape, mask.shape)
            assert img.shape == (8, 256, 256, 3)
            assert mask.shape == (8, 256, 256)
        print("stop")

    print("ok")
    exit()





