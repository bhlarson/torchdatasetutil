import sys
import os
from pycocotools import mask
import numpy as np
import cv2
import json
from collections import defaultdict
import functools
import random
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path, PurePath
from torch.utils.data.sampler import SubsetRandomSampler

from pymlutil.s3 import s3store, Connect
from pymlutil.jsonutil import ReadDict
from pymlutil.imutil import ImUtil, ImTransform

class ImagesStore(ImUtil):

    def __init__(self, s3, bucket, dataset_desc, class_dictionary, numTries=3):
        self.s3 = s3
        self.bucket = bucket
        self.dataset_desc = s3.GetDict(bucket,dataset_desc)
        self.class_dictionary = s3.GetDict(bucket,class_dictionary) 
        self.numTries=numTries

        super(ImagesStore, self).__init__(dataset_desc=self.dataset_desc, class_dictionary=self.class_dictionary)

        self.imflags = cv2.IMREAD_COLOR 
        if self.dataset_desc is not None and 'image_colorspace' in self.dataset_desc:
            if self.isGrayscale():
                self.imflags = cv2.IMREAD_GRAYSCALE
        self.anflags = cv2.IMREAD_GRAYSCALE 

        self.bare_image = self.dataset_desc['image_pattern'].replace('*','')
        self.bare_label = self.dataset_desc['label_pattern'].replace('*','')

        self.images = []
        self.labels = []

        self.CreateIndex()
        self.i = 0

    def ImagenameFromLabelname(self, lbl_filename):
        return lbl_filename.replace(self.bare_label, self.bare_image)


    def CreateIndex(self):
        file_list = self.s3.ListObjects( self.dataset_desc['bucket'], setname=self.dataset_desc['prefix'], pattern=None, recursive=self.dataset_desc['recursive'])
        imagedict = {}
        if self.dataset_desc['image_path']==self.dataset_desc['label_path']:
            for im_filename in file_list:
                if PurePath(im_filename).match(self.dataset_desc['image_pattern']):
                    imagedict[im_filename] = None
            for lbl_filename in file_list:
                if PurePath(lbl_filename).match(self.dataset_desc['label_pattern']):
                    im_filename = self.ImagenameFromLabelname(lbl_filename)
                    if im_filename in imagedict:
                        imagedict[im_filename] = lbl_filename

        for key in imagedict:
            if imagedict[key] is not None:
                self.images.append(key)
                self.labels.append(imagedict[key])



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

    def DecodeImage(self, bucket, objectname, flags):
        img = None
        for i in range(self.numTries):
            imgbuff = self.s3.GetObject(bucket, objectname)
            if imgbuff:
                imgbuff = np.frombuffer(imgbuff, dtype='uint8')

                try:
                    img = cv2.imdecode(imgbuff, flags=self.imflags)
                except:
                    print ("CocoStore::DecodeImage {}/{} cv2.imdecode exception i={}".format(bucket, objectname, i))
                    img = None

            if img is None or img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
                print('ImagesStore::DecodeImage failed to load {}/{} try {}'.format(bucket, objectname, i))
            else:
                break
        return img

    def ConvertLabels(self, ann):
        trainAnn = np.zeros_like(ann)
        for obj in self.class_dictionary ['objects']: # Load RGB colors as BGR
            if not (obj['id'] == obj['trainId'] and obj['id'] == 0):
                trainAnn[ann==obj['id']] = obj['trainId']
        return trainAnn

    def __len__(self):
        return len(self.images)

    def __next__(self):
        if self.i < self.__len__():
            result = self.__getitem__(self.i)
            self.i += 1
            return result
        else:
            raise StopIteration

    def __getitem__(self, idx):
        if idx >= 0 and idx < self.__len__():
            img = self.DecodeImage(self.bucket, self.images[idx], self.imflags)
            ann = self.DecodeImage(self.bucket, self.labels[idx], self.anflags)
            if img is not None and ann is not None:
                if ann is not None:
                    ann = self.ConvertLabels(ann)
                result = {'img':img, 'ann':ann}
            else:
                result = None

            return result
        else:
            print('ImagesStore.__getitem__ idx {} invalid.  Must be >=0 and < ImagesStore.__len__={}'.format(idx, self.__len__()))
            return None


class ImagesDataset(Dataset):
    def __init__(self, s3, bucket, dataset_desc, class_dictionary, 
        height=640, 
        width=640, 
        image_transform=None,
        label_transform=None,
        normalize=True, 
        enable_transform=True, 
        flipX=True, 
        flipY=False, 
        rotate=3, 
        scale_min=0.75, 
        scale_max=1.25, 
        offset=0.1,
        numTries=3
    ):
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.height = height
        self.width = width

        self.normalize = normalize
        self.enable_transform = enable_transform
        self.flipX = flipX
        self.flipY = flipY
        self.rotate = rotate
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.offset = offset

        self.store = ImagesStore(s3, bucket, dataset_desc, class_dictionary, numTries)

        self.imTransform = ImTransform(height=self.height, width=self.width, 
                                     normalize=normalize, 
                                     enable_transform=enable_transform, 
                                     flipX=flipX, flipY=flipY, 
                                     rotate=rotate, 
                                     scale_min=scale_min, scale_max=scale_max, offset=offset, astype=self.store.class_dictionary['input_type'])

    def __len__(self):
        return self.store.__len__()

    def __getitem__(self, idx):
        result = self.store.__getitem__(idx)
        if result is not None and result['img'] is not None and result['ann'] is not None:
            image = result['img']
            label = result['ann']

            if image is not None and label is not None:

                image, label, imgMean, imgStd = self.imTransform.random_resize_crop_or_pad(image, label)

                if len(image.shape) < 3:  
                    image = np.expand_dims(image, axis=-1) # Add a channel dimension

                image = torch.from_numpy(image).permute(2, 0, 1) # Convert from [channel, width, hight] to [channel, height, width]
                label = torch.from_numpy(label)

                if self.image_transform:
                    image = self.image_transform(image)
                if self.label_transform:
                    label = self.label_transform(label)

                assert(image.shape[-1] == self.width)
                assert(image.shape[-2] == self.height)
            
        else:           
            print('ImagesDataset.__getitem__ failed idx {} return result=None.'.format(idx))
            return None
        return image, label, imgMean, imgStd

# Handle corrupt images:
# https://github.com/pytorch/pytorch/issues/1137
# https://stackoverflow.com/questions/57815001/pytorch-collate-fn-reject-sample-and-yield-another/67583699#67583699
def collate_fn_replace_corrupted(batch, dataset, batch_size):
    original_batch_len = len(batch)
    batch = list(filter(lambda x: x is not None, batch)) # Filter out bad samples
    filtered_batch_len = len(batch)
    diff = original_batch_len - filtered_batch_len
    if filtered_batch_len < batch_size:                
        # Replace corrupted examples with another examples randomly
        print('imsgtore collate_fn_replace_corrupted diff = {}'.format(diff))
        batch.extend([dataset[random.randint(0, len(dataset))] for _ in range(diff)])
        # Recursive call to replace the replacements if they are corrupted
        return collate_fn_replace_corrupted(batch, dataset, batch_size)

    # Finally, when the whole batch is fine, return it
    return torch.utils.data.dataloader.default_collate(batch)

default_loaders = [{'set':'train', 'split':0.8, 'enable_transform':True},
                   {'set':'test', 'split':0.2, 'enable_transform':False}]

def CreateImageLoaders(s3, bucket, dataset_dfn, class_dict, 
                      batch_size = 2,  
                      num_workers=0, cuda = True, timeout=0, loaders = default_loaders, 
                      height=640, width=640, 
                      image_transform=None, label_transform=None, 
                      normalize=True, flipX=True, flipY=False, 
                      rotate=3, scale_min=0.75, scale_max=1.25, offset=0.1,
                      random_seed = None, numTries=3):

    dataset = ImagesDataset(s3, bucket, dataset_dfn, class_dict, 
                            height=height, width=width, 
                            image_transform=image_transform, label_transform=label_transform, 
                            normalize=normalize,  enable_transform=default_loaders[0]['enable_transform'], 
                            flipX=flipX, flipY=flipY, 
                            rotate=rotate, scale_min=scale_min, scale_max=scale_max, offset=offset,numTries=numTries)

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    pin_memory = False
    # if cuda:
    #     pin_memory = True

    startIndex = 0
    allocated = 0.0
    for i, loader in enumerate(loaders):
        allocated += loader['split']
        if allocated > 1.0:
            allocated = 1.0
        split = int(np.floor(allocated * dataset_size/batch_size))*batch_size
        if split > startIndex:

            if i > 0:
                dataset = ImagesDataset(s3, bucket, dataset_dfn, class_dict, 
                            height=height, width=width, 
                            image_transform=image_transform, label_transform=label_transform, 
                            normalize=normalize,  enable_transform=default_loaders[i]['enable_transform'], 
                            flipX=flipX, flipY=flipY, 
                            rotate=rotate, scale_min=scale_min, scale_max=scale_max, offset=offset)

            # Creating PT data samplers and loaders:
            loader['batches'] =int((split-startIndex)/batch_size)
            loader['length'] = loader['batches']*batch_size
            loader['dataset_dfn'] = dataset_dfn
            sampler = SubsetRandomSampler(indices[startIndex:split])
            startIndex = split
            collate_fn = functools.partial(collate_fn_replace_corrupted, dataset=dataset, batch_size=batch_size)
            loader['dataloader'] = torch.utils.data.DataLoader(dataset=dataset, 
                                                      batch_size=batch_size,
                                                      sampler=sampler,
                                                      num_workers=num_workers, 
                                                      pin_memory=pin_memory,
                                                      timeout=timeout,
                                                      collate_fn=collate_fn)         

    return loaders

