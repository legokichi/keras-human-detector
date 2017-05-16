from typing import Tuple, List, Text, Dict, Any, Iterator, Union, Sized, Callable, cast
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
from keras.backend import cast_to_floatx

def check(coco: COCO, info: dict, drop_crowd=False, drop_small=False, drop_minkey=False)-> bool:
    '''
    mscoco の画像から使えそうなものを判定する
    '''
    anns = coco.loadAnns(coco.getAnnIds(imgIds=[info['id']], iscrowd=0)) # type: List[dict]
    
    # 矩形の中心
    centers = [(ann["bbox"][0]+ann["bbox"][2]/2, ann["bbox"][1]+ann["bbox"][3]/2) for ann in anns] # type: List[Tuple[float, float]]

    for i, ann in enumerate(anns):
        cat = coco.loadCats([ann["category_id"]])[0]
        if cat["name"] != "person": continue
        
        # drop if people are overlapping
        # 重なり除去
        if drop_crowd:
            x, y, w, h = ann["bbox"]
            for j, (cx, cy) in enumerate(centers):
                if i == j: continue
                if x < cx < x+w and y < cy < y+h: return False

        # drop this person if parts number is too low
        if drop_minkey and ann["num_keypoints"] < 5: return False

        # drop if segmentation area is too small
        if drop_small and ann["area"] < 64*64: return False
        
        keys = ann["keypoints"] # type: List[int]

        if not(
            keys[0*3+2] != 0 # nose
            or keys[1*3+2] != 0 # l-eye
            or keys[2*3+2] != 0 # r-eye
            or keys[3*3+2] != 0 # l-ear
            or keys[4*3+2] != 0 # r-ear
        ): return False

        # drop if no body
        if not(
            keys[5*3+2] != 0
            or keys[6*3+2] != 0
        ): return False

    return True

def load_image(info: dict, dir: Union[str, None]) -> np.ndarray :
    if(dir != None): img = io.imread(dir + info['file_name']) # type: np.ndarray
    else: img = io.imread(info['coco_url'])
    if    img.ndim == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif  img.ndim == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def create_mask(coco: COCO, info: dict, debug=False) -> np.ndarray:
    anns = coco.loadAnns(coco.getAnnIds(imgIds=[info['id']], iscrowd=False)) # type: List[dict]
    w, h = info["width"], info["height"] # type: Tuple[int, int]
    mask_all = np.zeros((h, w), np.uint8) # type: np.ndarray

    for ann in anns:
        cat = coco.loadCats([ann["category_id"]])[0]
        if cat["name"] != "person": continue

        rles = coco_mask.frPyObjects(ann["segmentation"], h, w) # type: List[dict]
        for rle in rles:
            mask = coco_mask.decode(rle) # type: np.ndarray
            mask[mask > 0] = 255
            mask_all += mask

    return mask_all

def create_head_mask(coco: COCO, info: dict, debug=False, DISTANCE_SCALE=2) -> np.ndarray:
    # label = ["nose", "l-eye", "r-eye", "l-ear", "r-ear", "l-shoulder", "r-shoulder", "l-elbow", "r-elbow", "l-wrist", "r-wrist", "l-llium", "r-llium", "l-knee", "r-knee", "l-ankle", "r-ankle"]
    anns = coco.loadAnns(coco.getAnnIds(imgIds=[info['id']], iscrowd=False)) # type: List[dict]
    w, h = info["width"], info["height"] # type: Tuple[int, int]
    
    mask_all = np.zeros((h, w), np.uint8) # type: np.ndarray
    mask_head = np.zeros((h, w), np.uint8) # type: np.ndarray
    
    for ann in anns:
        cat = coco.loadCats([ann["category_id"]])[0]
        if cat["name"] != "person": continue

        mask_one_person = np.zeros((h, w), np.uint8) # type: np.ndarray
        rles = coco_mask.frPyObjects(ann["segmentation"], h, w) # type: List[dict]
        for rle in rles:
            # 飛び地なら複数回
            mask = coco_mask.decode(rle) # type: np.ndarray
            mask[mask > 0] = 255
            mask_one_person += mask

        keys = ann["keypoints"]
        parts = np.array(keys).reshape(17, 3) # type: np.ndarray

        # decide head center

        # nose, l-eye, r-eye
        face_parts = [(float(x),float(y)) for x,y,v in parts[0:3] if v != 0] # type: List[Tuple[float, float]]
        # l-ear, r-ear
        ear_parts = [(float(x),float(y)) for x,y,v in parts[3:5] if v != 0] # type: List[Tuple[float, float]]

        if len(face_parts) != 0:
            face_center = cast(Tuple[float, float], tuple(np.average(np.array(face_parts).T, axis=1).tolist())) # type: Tuple[float, float]
            head_parts = ear_parts + [face_center] # type: List[Tuple[float, float]]
        else:
            head_parts = ear_parts

        head_center = cast(Tuple[float, float], tuple(np.average(np.array(head_parts).T, axis=1).tolist())) # type: Tuple[float, float]

        # decide head region
        shoulder_parts = [(x,y) for x,y,v in parts[5:7] if v != 0] # type: List[Tuple[int, int]]
        assert len(shoulder_parts) != 0
        distances = [np.sqrt(np.power(head_center[0] - sholder[0], 2) + np.power(head_center[1] - sholder[1], 2)) for sholder in shoulder_parts] # type: List[float]
        distance = np.average(distances) # type: float
        assert distance > 0

        kernel = gaussian_kernel(int(distance*DISTANCE_SCALE))
        # normalize
        kernel = (kernel/kernel.max()*255).astype("uint8")

        # annotate head position
        kernel_img = Image.fromarray(kernel)
        mask_head_img = Image.fromarray(mask_head)
        mask_head_img.paste(kernel_img, (int(head_center[0]-(kernel.shape[0]/2)), int(head_center[1]-(kernel.shape[1]/2))))
        mask_head = np.asarray(mask_head_img)
        mask_head.flags.writeable = True
        
        
        # masking human region
        mask_all += (mask_one_person > 0).astype("uint8") * mask_head

    return mask_all



#def gaussian(x, mu, sig): return (1/np.sqrt(2. * np.pi * sig)) * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
def gaussian_kernel(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(scipy.stats.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

class CamVid(DatasetMixin):
    def __init__(self, mode: str, keypoint_json_path: str, all_json_path: str, img_path: str, resize_shape: Tuple[int, int]=None,
        data_aug: bool=False, drop_crowd=False, drop_small=False, DISTANCE_SCALE=2.):
        self.mode = mode
        self.data_aug = data_aug
        self.img_path = img_path
        self.resize_shape = resize_shape # type: Tuple[int, int]
        self.coco = COCO(keypoint_json_path) # type: COCO
        coco = self.coco
        infos = coco.loadImgs(coco.getImgIds(catIds=coco.getCatIds(catNms=['person']))) # type: List[dict]
        print("original:", len(infos))
        self.infos = [info for info in infos if check(coco, info, drop_crowd, drop_small)] # type: List[dict]
        print("person:", len(self.infos))
        self.DISTANCE_SCALE = DISTANCE_SCALE
        if False and self.data_aug:
            coco = COCO(all_json_path)
            print(len(coco.getImgIds()))
            infos = coco.loadImgs(coco.getImgIds())
            _infos = []
            print("all:", len(infos))
            for info in infos:
                drop = False
                for ann in coco.loadAnns(coco.getAnnIds(imgIds=[info['id']], iscrowd=0)):
                    for cat in coco.loadCats([ann["category_id"]]):
                        if cat["name"] == "person":
                            drop = True
                            break
                if drop: continue
                _infos.append(info)
            # 1/4 くらいダミーデータを足す
            _infos = _infos[0:int(len(self.infos)/4)]
            print("_infos:", len(_infos))
            self.infos += _infos
        print("finaly:", len(self.infos))
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
            #iaa.GaussianBlur(sigma=(0, 0.5)),
            #iaa.AdditiveGaussianNoise(scale=(0., 0.1*255), per_channel=0.5),
            iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
        ]) # type: iaa.Sequential
    def __len__(self) -> int:
        return len(self.infos)
    def get_example(self, i) -> Tuple[np.ndarray, dict]:
        info = self.infos[i]

        img = load_image(info, self.img_path)
        mask_all = create_mask(self.coco, info)
        mask_head = create_head_mask(self.coco, info, DISTANCE_SCALE=self.DISTANCE_SCALE)

        assert img.dtype == "uint8"
        assert mask_all.dtype == "uint8"
        assert mask_head.dtype == "uint8"

        if self.data_aug:
            # data augumentation
            seq_det = self.seq.to_deterministic()
            
            img  = np.expand_dims(img, axis=0)
            mask_all = np.expand_dims(mask_all, axis=0)
            mask_head = np.expand_dims(mask_head, axis=0)
            img  = seq_det.augment_images(img)
            mask_all = seq_det.augment_images(mask_all)
            mask_head = seq_det.augment_images(mask_head)
            img  = self.seq_noise.augment_images(img) # add noise
            img  = np.squeeze(img)
            mask_all  = np.squeeze(mask_all)
            mask_head = np.squeeze(mask_head)

        # resize
        img  = cv2.resize(img, self.resize_shape)
        mask_all = cv2.resize(mask_all, self.resize_shape)
        mask_head = cv2.resize(mask_head, self.resize_shape)

        # binarize
        mask_all = mask_all > 0

        if self.mode == "multistage":
            return (img, {'output1': mask_all, 'output2': mask_head} )
        elif self.mode == "binarize":
            return (img, mask_all)
        elif self.mode == "heatmap":
            # concat
            img = np.dstack((img, mask_all))
            return (img, mask_head)
        elif self.mode == "integrated":
            return (img, mask_head)
        else:
            raise Exception("unknown mode: "+self.mode)


def convert_to_keras_batch(iter: Iterator[List[Tuple[np.ndarray, Union[np.ndarray, dict]]]]) -> Iterator[Tuple[np.ndarray, Union[np.ndarray, dict]]] :
    while True:
        batch = iter.__next__() # type: List[Tuple[np.ndarray, np.ndarray]]
        if isinstance(batch[0][1], dict):
            xs = [x for (x, _) in batch] # type: List[np.ndarray]
            ys = [y["output1"] for (_, y) in batch] # type: List[np.ndarray]
            zs = [y["output2"] for (_, y) in batch] # type: List[np.ndarray]
            xs = cast_to_floatx(np.array(xs)) # (n, 480, 360, 3)
            ys = cast_to_floatx(np.array(ys)) # (n, 480, 360, 1)
            zs = cast_to_floatx(np.array(ys)) # (n, 480, 360, 1)
            yield (xs, {"output1": ys, "output2": zs})
        else:
            xs = [x for (x, _) in batch]
            ys = [y for (_, y) in batch]
            xs = cast_to_floatx(np.array(xs)) # (n, 480, 360, 3)
            ys = cast_to_floatx(np.array(ys)) # (n, 480, 360, 1)
            yield (xs, ys)


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

    resize_shape = (512, 512)
    batch_size = 8

    train = CamVid("binarize", args.dir+"/annotations/person_keypoints_train2014.json", args.dir+"/annotations/instances_train2014.json", args.dir+"/train2014/", resize_shape, data_aug=True) # type: DatasetMixin
    valid = CamVid("binarize", args.dir+"/annotations/person_keypoints_val2014.json",   args.dir+"/annotations/instances_val2014.json",   args.dir+"/val2014/",   resize_shape) # type: DatasetMixin

    print("train:",len(train),"valid:",len(valid))

    for mx in [train, valid]:
        print("start")
        it = convert_to_keras_batch(
            MultiprocessIterator(
                mx,
                batch_size=batch_size,
                repeat=False,
                shuffle=False,
                n_processes=12,
                n_prefetch=120,
                shared_mem=1000*1000*5
            )
        ) # type: Iterator[Tuple[np.ndarray, np.ndarray]]

        for i,(_, (img,mask)) in enumerate(zip(range(math.floor(len(mx)/8)), it)):
            print(i, img.shape, mask.shape)
            assert img.shape == (batch_size, resize_shape[0], resize_shape[1], 3)
            assert mask.shape == (batch_size, resize_shape[0], resize_shape[1], 2)
        cast(MultiprocessIterator, it).finalize()
        print("stop")

    print("ok")
    exit()
