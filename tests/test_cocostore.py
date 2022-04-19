import sys
import os
from pycocotools import mask
import numpy as np
import cv2
import json
from collections import defaultdict
import unittest
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from pymlutil.s3 import s3store, Connect
from pymlutil.jsonutil import ReadDictJson
from pymlutil.imutil import ImUtil, ImTransform
from torchdatasetutil.cocostore import CocoStore

class Test(unittest.TestCase):      

    '''def test_iterator(self):

        self.credentails = 'creds.yaml'
        self.dataset_train = 'data/coco/annotations/instances_train2017.json'
        self.dataset_val = 'data/coco/annotations/instances_val2017.json'
        self.train_image_path = 'data/coco/train2017'
        self.val_image_path = 'data/coco/val2017'
        self.class_dict = 'model/segmin/coco.json'
        self.imflags = cv2.IMREAD_COLOR
        s3, creds, s3def = Connect(self.credentails)

        dataset_desc = s3.GetDict(s3def['sets']['dataset']['bucket'],self.dataset_train)
        class_dictionary = s3.GetDict(s3def['sets']['dataset']['bucket'],self.class_dict) 
        imUtil = ImUtil(dataset_desc, class_dictionary)
                
        store = CocoStore(s3, bucket=s3def['sets']['dataset']['bucket'], 
                        dataset_desc=self.dataset_train, 
                        image_paths=self.train_image_path, 
                        class_dictionary=self.class_dict, 
                        imflags=self.imflags)

        for i, iman in enumerate(store):
            img = store.MergeIman(iman['img'], iman['ann'])
            # How to test if image is correct'''

    def test_dataset(self):
        print('test_dataset: create me!')

if __name__ == '__main__':
    unittest.main()