import sys
import os
from pycocotools import mask
import numpy as np
import cv2
import json
from collections import defaultdict
import unittest
import torch
import tqdm
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from pymlutil.s3 import s3store, Connect
from pymlutil.jsonutil import ReadDict
from pymlutil.imutil import ImUtil, ImTransform
from torchdatasetutil.cocostore import CocoStore, CreateCocoLoaders

test_config = 'test.yaml'
class Test(unittest.TestCase):      

    def test_iterator(self):
        parameters = ReadDict(test_config)

        if 'coco' not in parameters:
            raise ValueError('coco not in {}'.format(test_config))

        s3, creds, s3def = Connect(parameters['coco']['credentials'])

        dataset_desc = s3.GetDict(s3def['sets']['dataset']['bucket'],parameters['coco']['dataset_val'])
        class_dictionary = s3.GetDict(s3def['sets']['dataset']['bucket'],parameters['coco']['class_dict']) 
        imUtil = ImUtil(dataset_desc, class_dictionary)
                
        store = CocoStore(s3, bucket=s3def['sets']['dataset']['bucket'], 
                        dataset_desc=parameters['coco']['dataset_train'], 
                        image_paths=parameters['coco']['train_image_path'], 
                        class_dictionary=parameters['coco']['class_dict'])

        for i, iman in enumerate(store):
            img = store.MergeIman(iman['img'], iman['ann'])
            is_success, buffer = cv2.imencode(".png", img)
            if not is_success:
                raise ValueError('test_cocostore test_iterator cv2.imencode failure  image {}'.format(i))
            if img is None:
                raise ValueError('img is None')
            if 'test_images' in parameters['coco'] and i >= parameters['coco']['test_images']:
                break

    '''
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
    '''



    def test_CreateCocoLoaders(self):
        parameters = ReadDict(test_config)

        if 'coco' not in parameters:
            raise ValueError('coco not in {}'.format(test_config))

        s3, creds, s3def = Connect(parameters['coco']['credentials'])

        dataset_desc = s3.GetDict(s3def['sets']['dataset']['bucket'],parameters['coco']['dataset_val'])
        class_dictionary = s3.GetDict(s3def['sets']['dataset']['bucket'],parameters['coco']['class_dict']) 
        imUtil = ImUtil(dataset_desc, class_dictionary)

        loaders = CreateCocoLoaders(s3, bucket=s3def['sets']['dataset']['bucket'],
                                     class_dict=parameters['coco']['class_dict'])

        for i, data in enumerate(loaders[0]['dataloader']):
            inputs, labels, mean, stdev = data
            images = inputs.cpu().permute(0, 2, 3, 1).numpy()
            labels = np.around(labels.cpu().numpy()).astype('uint8')
            mean = mean.cpu().numpy()
            stdev = stdev.cpu().numpy()

            for j, image in enumerate(images):
                img = imUtil.MergeIman(images[j], labels[j], mean[j], stdev[j])
                is_success, buffer = cv2.imencode(".png", img)
                if not is_success:
                    raise ValueError('test_imstore test_CreateCocoLoaders cv2.imencode failure batch {} image {}'.format(i, j))
            if 'test_images' in parameters['coco'] and i >= parameters['coco']['test_images']:
                break

if __name__ == '__main__':
    unittest.main()