import sys
import os
from pycocotools import mask
import numpy as np
import cv2
import json
from collections import defaultdict
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from pymlutil.s3 import s3store, Connect
from pymlutil.jsonutil import ReadDictJson
from pymlutil.imutil import ImUtil, ImTransform

class CocoStore(ImUtil):

    def __init__(self, s3, bucket, dataset_desc, image_paths, class_dictionary, imflags=cv2.IMREAD_COLOR, name_decoration='' ):

        self.s3 = s3
        self.bucket = bucket
        self.dataset_desc = s3.GetDict(bucket,dataset_desc)
        self.class_dictionary = s3.GetDict(bucket,class_dictionary)
        self.image_paths = image_paths
        self.name_decoration = name_decoration
        self.imflags = imflags



        self.CreateIndex()
        super(CocoStore, self).__init__(dataset_desc=self.dataset_desc, class_dictionary=self.class_dictionary)
        self.i = 0

    def CreateIndex(self):
        # create index objDict rather than coco types
        anns, cats, imgs = {}, {}, {}
        imgToAnns = defaultdict(list)
        catToImgs = defaultdict(list)
        catToObj = {}
        objs = {}
        # make list based on 
        if 'annotations' in self.dataset_desc:
            for ann in self.dataset_desc['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset_desc:
            for img in self.dataset_desc['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset_desc:
            for cat in self.dataset_desc['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset_desc and 'categories' in self.dataset_desc:
            for ann in self.dataset_desc['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        if self.class_dictionary  is not None:
            for obj in self.class_dictionary ['objects']:
                catToObj[obj['id']] = obj
                objs[obj['trainId']] = {'category':obj['category'], 
                                        'color':obj['color'],
                                        'display':obj['display']
                                        }


        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats
        self.catToObj = catToObj
        self.objs = objs


    def __iter__(self):
        self.i = 0
        return self

    def classes(self, anns):
        class_vector = np.zeros(self.class_dictionary ['classes'], dtype=np.float32)

        for ann in anns:
            obj = self.catToObj[ann['category_id']]
            if obj['trainId'] < self.class_dictionary ["classes"]:
                class_vector[obj['trainId']] = 1.0

        return class_vector

    def DecodeImage(self, bucket, objectname):
        img = None
        numTries = 3
        for i in range(numTries):
            imgbuff = self.s3.GetObject(bucket, objectname)
            if imgbuff:
                imgbuff = np.frombuffer(imgbuff, dtype='uint8')
                img = cv2.imdecode(imgbuff, flags=self.imflags)
            if img is None:
                print('CocoStore::DecodeImage failed to load {}/{} try {}'.format(bucket, objectname, i))
            else:
                break
        return img

    def drawann(self, imgDef, anns):
        annimg = np.zeros(shape=[imgDef['height'], imgDef['width']], dtype=np.uint8)
        for ann in anns:
            obj = self.catToObj[ann['category_id']]
            if obj['trainId'] < self.class_dictionary ["classes"]:
                if type(ann['segmentation']) is list:
                    for i in range(len(ann['segmentation'])):
                        contour = np.rint(np.reshape(ann['segmentation'][i], (-1, 2))).astype(np.int32)
                        cv2.drawContours(image=annimg, contours=[contour], contourIdx=-1, color=obj['trainId'] , thickness=cv2.FILLED)
                elif type(ann['segmentation']) is dict:
                    rle = ann['segmentation']
                    compressed_rle = mask.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
                    annmask = mask.decode(compressed_rle)
                    annimg[annmask] = obj['trainId']
                else:
                    print('unexpected segmentation')
            else:
                print('trainId {} >= classes {}'.format(obj['trainId'], self.class_dictionary ["classes"]))    
        return annimg

    def len(self):
        return len(self.dataset_desc['images'])

    def __next__(self):
        if self.i < self.len():
            result = self.__getitem__(self.i)
            self.i += 1
            return result
        else:
            raise StopIteration

    def __getitem__(self, idx):
        if idx >= 0 and idx < self.len():
            img_entry = self.dataset_desc['images'][idx]
            imgFile = '{}/{}{}'.format(self.image_paths,self.name_decoration,img_entry['file_name'])
            img = self.DecodeImage(self.bucket, imgFile)
            ann_entry = self.imgToAnns[img_entry['id']]
            ann = self.drawann(img_entry, ann_entry)
            classes = self.classes(ann_entry)
            result = {'img':img, 'ann':ann, 'classes':classes}

            return result
        else:
            print('CocoStore.__getitem__ idx {} invalid.  Must be >=0 and < CocoStore.len={}'.format(idx, self.len()))
            return None

class CocoDataset(Dataset):
    def __init__(self, s3, bucket, dataset_desc, image_paths, class_dictionary, 
        height=640, 
        width=640, 
        imflags=cv2.IMREAD_COLOR, 
        image_transform=None,
        label_transform=None,
        name_decoration='',
        normalize=True, 
        enable_transform=True, 
        flipX=True, 
        flipY=False, 
        rotate=15, 
        scale_min=0.75, 
        scale_max=1.25, 
        offset=0.1,
        astype='float32'
    ):
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.height = height
        self.width = width
        self.imflags = imflags

        self.normalize = normalize
        self.enable_transform = enable_transform
        self.flipX = flipX
        self.flipY = flipY
        self.rotate = rotate
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.offset = offset
        self.astype = astype

        self.store = CocoStore(s3, bucket, dataset_desc, image_paths, class_dictionary, imflags=self.imflags, name_decoration=name_decoration)


        self.imTransform = ImTransform(height=height, width=width, 
                                     normalize=normalize, 
                                     enable_transform=enable_transform, 
                                     flipX=flipX, flipY=flipY, 
                                     rotate=rotate, 
                                     scale_min=scale_min, scale_max=scale_max, offset=offset, astype=astype)

    def __len__(self):
        return self.store.len()

    def __getitem__(self, idx):
        result = self.store.__getitem__(idx)
        if result is not None and result['img'] is not None and result['ann'] is not None:
            image = result['img']
            label = result['ann']

            if self.width is not None and self.height is not None:
                image, label, imgMean, imgStd = self.imTransform.random_resize_crop_or_pad(image, label)

            if image is not None and label is not None:
                if len(image.shape) < 3:
                    image = np.expand_dims(image, axis=-1)

                image = torch.from_numpy(image).permute(2, 0, 1)
                label = torch.from_numpy(label)

                if self.image_transform:
                    image = self.image_transform(image)
                if self.label_transform:
                    label = self.label_transform(label)
            
        else:
            image=None
            label=None
            imgMean = None
            imgStd = None
            print('CocoDataset.__getitem__ idx {} returned result=None.'.format(idx))
        return image, label, imgMean, imgStd

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

default_loaders = [{'set':'train', 'dataset': 'data/coco/annotations/instances_train2017.json', 'image_path':'data/coco/train2017' , 'enable_transform':True},
                   {'set':'test', 'dataset': 'data/coco/annotations/instances_val2017.json', 'image_path':'data/coco/val2017', 'enable_transform':False}]

def CreateCocoLoaders(s3, bucket, class_dict, 
                      batch_size = 2, shuffle=True, 
                      num_workers=0, cuda = True, timeout=0, loaders = default_loaders, 
                      height=640, width=640, 
                      image_transform=None, label_transform=None, 
                      normalize=True, flipX=True, flipY=False, 
                      rotate=3, scale_min=0.75, scale_max=1.25, offset=0.1, astype='float32',
                      random_seed = None):

    pin_memory = False
    if cuda:
        pin_memory = True

    startIndex = 0
    allocated = 0.0

    for i, loader in enumerate(loaders):
        dataset = CocoDataset(s3=s3, bucket=bucket, 
                    dataset_desc=loader['dataset'], 
                    class_dictionary=class_dict,
                    image_paths=loader['image_path'],
                    height=height, width=width, 
                    image_transform=image_transform, label_transform=label_transform, 
                    normalize=normalize,  enable_transform=loader['enable_transform'], 
                    flipX=flipX, flipY=flipY, 
                    rotate=rotate, scale_min=scale_min, scale_max=scale_max, offset=offset, astype=astype)

        # Creating PT data samplers and loaders:
        loader['batches'] =int(dataset.__len__()/batch_size)
        loader['length'] = loader['batches']*batch_size

        loader['dataloader'] = torch.utils.data.DataLoader(dataset=dataset, 
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            drop_last=True,
                                            num_workers=num_workers, 
                                            pin_memory=pin_memory,
                                            timeout=timeout,
                                            collate_fn=collate_fn)         

    return loaders