import sys
import os
from pycocotools import mask
import numpy as np
import cv2
import json
from collections import defaultdict
import unittest
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from pymlutil.s3 import s3store, Connect
from pymlutil.jsonutil import ReadDict
from pymlutil.imutil import ImUtil, ImTransform

# Force import from local files
sys.path.insert(0, os.path.abspath('')) # Test files from current path rather than installed module
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


        for i, iman in  tqdm(enumerate(store),
                             desc="CocoStore Reads",
                             total=len(store)):

            #img = store.MergeIman(iman['img'], iman['ann'])
            assert(iman['img'] is not None)
            assert(iman['ann'] is not None)
            if 'test_images' in parameters['coco'] and parameters['coco']['test_images'] is not None and i >= parameters['coco']['test_images']:
                break


    def test_CreateCocoLoaders(self):
        parameters = ReadDict(test_config)

        if 'coco' not in parameters:
            raise ValueError('coco not in {}'.format(test_config))

        s3, creds, s3def = Connect(parameters['coco']['credentials'])

        dataset_desc = s3.GetDict(s3def['sets']['dataset']['bucket'],parameters['coco']['dataset_val'])
        class_dictionary = s3.GetDict(s3def['sets']['dataset']['bucket'],parameters['coco']['class_dict']) 
        imUtil = ImUtil(dataset_desc, class_dictionary)

        loaders = CreateCocoLoaders(s3, bucket=s3def['sets']['dataset']['bucket'],
                                     class_dict=parameters['coco']['class_dict'], 
                                     batch_size=parameters['coco']['batch_size'], 
                                     height=parameters['coco']['height'], 
                                     width=parameters['coco']['width'],
                                     num_workers=parameters['coco']['num_workers'])

        for loader in tqdm(loaders, desc="CreateCocoLoaders"):
            for i, data in tqdm(enumerate(loader['dataloader']), 
                                desc="Batch Reads", 
                                total=loader['batches']):
                inputs, labels, mean, stdev = data
                assert(inputs.size(0)==parameters['coco']['batch_size'])
                assert(inputs.size(-1)==parameters['coco']['width'])
                assert(inputs.size(-2)==parameters['coco']['height'])
                assert(labels.size(0)==parameters['coco']['batch_size'])
                assert(labels.size(-1)==parameters['coco']['width'])
                assert(labels.size(-2)==parameters['coco']['height'])
                
                if 'test_images' in parameters['coco'] and parameters['coco']['test_images'] is not None and i >= parameters['coco']['test_images']:
                    break


if __name__ == '__main__':
    unittest.main()