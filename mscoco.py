from typing import Tuple, List, Text, Dict, Any, Iterator, Union, Sized, Callable
import sys
import os
import numpy as np
from PIL import Image
sys.path.append("/usr/local/Cellar/opencv3/3.2.0/lib/python3.5/site-packages/") # mac opencv path
import cv2
import skimage.io as io
import scipy
from imgaug import augmenters as iaa
sys.path.append("./coco/PythonAPI/")

from pycocotools.coco import COCO
from pycocotools import mask as coco_mask

from chainer.iterators import MultiprocessIterator, SerialIterator
from chainer.dataset.dataset_mixin import DatasetMixin


def check(coco: COCO, info: dict, drop_crowd=False, drop_small=False, need_head=False, need_shoulder=False, need_elbow=False, need_llium=False)-> bool:
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
        if need_head and not(
            # 2 is visible
            keys[0*3+2] == 2 # nose
            or keys[1*3+2] == 2 # l-eye
            or keys[2*3+2] == 2 # r-eye
            or keys[3*3+2] == 2 # l-ear
            or keys[4*3+2] == 2 # r-ear
        ): return False

        # drop if no body
        if(need_shoulder and not(keys[5*3+2]  == 2 or keys[6*3+2]  == 2)): return False
        if(need_elbow    and not(keys[7*3+2]  == 2 or keys[8*3+2]  == 2)): return False
        if(need_llium    and not(keys[11*3+2] == 2 or keys[12*3+2] == 2)): return False

    return True

def load_image(info: dict, dir: str) -> np.ndarray :
    if(dir != None): img = io.imread(dir + info['file_name']) # type: np.ndarray
    else: img = io.imread(info['coco_url'])
    if    img.ndim == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif  img.ndim == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def create_mask(coco: COCO, info: dict, debug = False) -> np.ndarray:
    anns = coco.loadAnns(coco.getAnnIds(imgIds=[info['id']], iscrowd=False)) # type: List[dict]
    w, h = info["width"], info["height"] # type: Tuple[int, int]
    mask_all = np.zeros((h, w), np.uint8) # type: np.ndarray

    for ann in anns:
        rles = coco_mask.frPyObjects(ann["segmentation"], h, w) # type: List[dict]
        for rle in rles:
            mask = coco_mask.decode(rle) # type: np.ndarray
            mask[mask > 0] = 255
            mask_all += mask

        keys = ann["keypoints"] # type: List[int]
        if debug: 
            i, length = 0, len(keys)
            while i < length:
                x, y, v = keys[i], keys[i+1], keys[i+2]  # type: Tuple[int, int, int]
                text = str(int(i/3))
                i += 3
                if v != 2: continue
                mask_all[y, x] = 128
                cv2.putText(mask_all, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (128,128,128))

    return mask_all

def create_head_mask(coco: COCO, info: dict, debug = False) -> np.ndarray:
    # label = ["nose", "l-eye", "r-eye", "l-ear", "r-ear", "l-shoulder", "r-shoulder", "l-elbow", "r-elbow", "l-wrist", "r-wrist", "l-llium", "r-llium", "l-knee", "r-knee", "l-ankle", "r-ankle"]
    anns = coco.loadAnns(coco.getAnnIds(imgIds=[info['id']], iscrowd=False)) # type: List[dict]
    w, h = info["width"], info["height"] # type: Tuple[int, int]
    
    mask_all = np.zeros((h, w), np.uint8) # type: np.ndarray
    mask_head = np.zeros((h, w), np.uint8) # type: np.ndarray
    
    for ann in anns:
        mask_one = np.zeros((h, w), np.uint8) # type: np.ndarray
        rles = coco_mask.frPyObjects(ann["segmentation"], h, w) # type: List[dict]
        for rle in rles:
            # 飛び地なら複数回
            mask = coco_mask.decode(rle) # type: np.ndarray
            mask[mask > 0] = 255
            mask_one += mask

        keys = ann["keypoints"]
        parts = np.array(keys).reshape(17, 3) # type: np.ndarray

        # decide head center

        # nose, l-eye, r-eye
        face_parts = [(float(x),float(y)) for x,y,v in parts[0:3] if v == 2] # type: List[Tuple[float, float]]
        # l-ear, r-ear
        ear_parts = [(float(x),float(y)) for x,y,v in parts[3:5] if v == 2] # type: List[Tuple[float, float]]

        if len(face_parts) != 0:
            face_center = tuple(np.average(np.array(face_parts).T, axis=1).tolist()) # type: Tuple[float, float]
            head_parts = ear_parts + [face_center] # type: List[Tuple[float, float]]
        else:
            head_parts = ear_parts

        head_center = tuple(np.average(np.array(head_parts).T, axis=1).tolist()) # type: Tuple[float, float]

        # decide head region
        shoulder_parts = [(x,y) for x,y,v in parts[5:7] if v == 2] # type: List[Tuple[int, int]]
        distances = [np.sqrt(np.power(head_center[0] - sholder[0], 2) + np.power(head_center[1] - sholder[1], 2)) for sholder in shoulder_parts] # type: List[float]
        distance = np.average(distances) # type: float

        # annotate head position
        x, y = int(head_center[0]), int(head_center[1])

        kernel = gaussian_kernel(int(distance*2))
        kernel = (kernel/kernel.max()*255).astype("uint8")

        kernel_img = Image.fromarray(kernel)
        mask_head_img = Image.fromarray(mask_head)
        mask_head_img.paste(kernel_img, (x-int(kernel.shape[0]/2), y-int(kernel.shape[1]/2)))
        mask_head = np.asarray(mask_head_img)
        mask_head.flags.writeable = True
        
        if debug: 
            mask_head[y, x] = 255
            cv2.putText(mask_head, "+", (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255))
        # masking human region
        mask_all += (mask_one > 0).astype("uint8") * mask_head

    return mask_all



def gaussian(x, mu, sig): return (1/np.sqrt(2. * np.pi * sig)) * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
def gaussian_kernel(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(scipy.stats.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

class CamVid(DatasetMixin):
    def __init__(self, json_path: str, img_path: str, resize_shape: Tuple[int, int]=None,
        use_data_check: bool=False, data_aug: bool=False,
        drop_crowd=False, drop_small=False, need_head=False, need_shoulder=False, need_elbow=False, need_llium=False):
        self.data_aug = data_aug
        self.img_path = img_path
        self.resize_shape = resize_shape # type: Tuple[int, int]
        self.coco = COCO(json_path) # type: COCO
        coco = self.coco
        infos = coco.loadImgs(coco.getImgIds(catIds=coco.getCatIds(catNms=['person']))) # type: List[dict]
        self.infos = infos # type: List[dict]
        if use_data_check:
            self.infos = [info for info in infos if check(coco, info, drop_crowd, drop_small, need_head, need_shoulder, need_elbow, need_llium)]
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
    def get_example(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
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

        # binarize
        mask = mask > 0

        return (img, mask)

class CamVidHead(CamVid):
    def __init__(self, json_path: str, img_path: str, resize_shape: Tuple[int, int]=None, data_aug: bool=False,
        drop_crowd=False, drop_small=False, need_elbow=False, need_llium=False):
        super(CamVidHead, self).__init__(json_path=json_path, img_path=img_path, resize_shape=resize_shape, use_data_check=True, data_aug=data_aug,
            drop_crowd=drop_crowd, drop_small=drop_small, need_head=True, need_shoulder=True, need_elbow=need_elbow, need_llium=need_llium)
    def get_example(self, i) -> Tuple[np.ndarray, np.ndarray]:
        info = self.infos[i]

        img = load_image(info, self.img_path)
        alpha = create_mask(self.coco, info)
        mask = create_head_mask(self.coco, info)

        if self.data_aug:
            # data augumentation
            seq_det = self.seq.to_deterministic()
            
            img  = np.expand_dims(img, axis=0)
            alpha = np.expand_dims(alpha, axis=0)
            mask = np.expand_dims(mask, axis=0)
            img  = seq_det.augment_images(img)
            alpha = seq_det.augment_images(alpha)
            mask = seq_det.augment_images(mask)
            img  = self.seq_noise.augment_images(img)
            img  = np.squeeze(img)
            alpha  = np.squeeze(alpha)
            mask = np.squeeze(mask)

        # resize
        img  = cv2.resize(img, self.resize_shape)
        alpha = cv2.resize(alpha, self.resize_shape)
        mask = cv2.resize(mask, self.resize_shape)

        # binarize
        alpha = alpha > 0

        # concat
        img = np.dstack((img, alpha))

        return (img, mask)

class CamVidOneshot(CamVid):
    def __init__(self, json_path: str, img_path: str, resize_shape: Tuple[int, int]=None, data_aug: bool=False,
        drop_crowd=False, drop_small=False, need_elbow=False, need_llium=False):
        super(CamVidOneshot, self).__init__(json_path=json_path, img_path=img_path, resize_shape=resize_shape, use_data_check=True, data_aug=data_aug,
            drop_crowd=drop_crowd, drop_small=drop_small, need_head=True, need_shoulder=True, need_elbow=need_elbow, need_llium=need_llium)
    def get_example(self, i) -> Tuple[np.ndarray, np.ndarray]:
        info = self.infos[i]

        img = load_image(info, self.img_path)
        mask_all = create_mask(self.coco, info)
        mask_head = create_head_mask(self.coco, info)

        if self.data_aug:
            # data augumentation
            seq_det = self.seq.to_deterministic()
            
            img  = np.expand_dims(img, axis=0)
            mask_all = np.expand_dims(mask_all, axis=0)
            mask_head = np.expand_dims(mask_head, axis=0)
            img  = seq_det.augment_images(img)
            mask_all = seq_det.augment_images(mask_all)
            mask_head = seq_det.augment_images(mask_head)
            img  = self.seq_noise.augment_images(img)
            img  = np.squeeze(img)
            mask_all  = np.squeeze(mask_all)
            mask_head = np.squeeze(mask_head)

        # resize
        img  = cv2.resize(img, self.resize_shape)
        mask_all = cv2.resize(mask_all, self.resize_shape)
        mask_head = cv2.resize(mask_head, self.resize_shape)

        # concat
        mask = np.dstack((mask_all, mask_head))

        return (img, mask)


def convert_to_keras_batch(iter: Iterator[List[Tuple[np.ndarray, np.ndarray]]]) -> Iterator[Tuple[np.ndarray, np.ndarray]] :
    while True:
        batch = iter.__next__() # type: List[Tuple[np.ndarray, np.ndarray]]
        xs = [x for (x, _) in batch] # type: List[np.ndarray]
        ys = [y for (_, y) in batch] # type: List[np.ndarray]
        _xs = np.array(xs) # (n, 480, 360, 3)
        _ys = np.array(ys) # (n, 480, 360, n_classes)
        yield (_xs, _ys)


if __name__ == '__main__':
    import argparse
    import cProfile
    import pstats
    import time
    from chainer.iterators import MultiprocessIterator, SerialIterator
    import math

    parser = argparse.ArgumentParser(description='mscoco data generator self test')
    parser.add_argument("--dir", action='store', type=str, default="./", help='mscoco dir')
    args = parser.parse_args()

    resize_shape = (256, 256)


    train3 = CamVidHead(args.dir+"/annotations/person_keypoints_train2014.json", None, resize_shape, data_aug=True, drop_crowd=True) # type: DatasetMixin
    valid3 = CamVidHead(args.dir+"/annotations/person_keypoints_val2014.json",   None,   resize_shape) # type: DatasetMixin

    print("train3:",len(train3),"valid3:",len(valid3))

    for mx in [train3, valid3]:
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
        )

        for i,(_, (img,mask)) in enumerate(zip(range(math.floor(len(mx)/8)), it)):
            print(i, img.shape, mask.shape)
            assert img.shape == (8, 256, 256, 3)
            assert mask.shape == (8, 256, 256, 2)
        print("stop")

    print("ok")

    exit()



    train = CamVid(args.dir+"/annotations/person_keypoints_train2014.json", args.dir+"/train2014/", resize_shape, use_data_check=True, data_aug=True) # type: DatasetMixin
    valid = CamVid(args.dir+"/annotations/person_keypoints_val2014.json",   args.dir+"/val2014/",   resize_shape) # type: DatasetMixin

    print("train:",len(train),"valid:",len(valid))

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

    train2 = CamVidHead(args.dir+"/annotations/person_keypoints_train2014.json", args.dir+"/train2014/", resize_shape, data_aug=True, drop_crowd=True) # type: DatasetMixin
    valid2 = CamVidHead(args.dir+"/annotations/person_keypoints_val2014.json",   args.dir+"/val2014/",   resize_shape) # type: DatasetMixin

    print("train2:",len(train2),"valid2:",len(valid2))

    for mx in [train2, valid2]:
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
        )

        for i,(_, (img,mask)) in enumerate(zip(range(math.floor(len(mx)/8)), it)):
            print(i, img.shape, mask.shape)
            assert img.shape == (8, 256, 256, 4)
            assert mask.shape == (8, 256, 256)
        print("stop")

    print("ok")

    exit()





