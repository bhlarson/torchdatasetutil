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
from torchdatasetutil.imstore import ImagesStore

test_config = 'test.yaml'
class Test(unittest.TestCase):      

    '''def test_iterator(self):
        parameters = ReadDict(test_config)

        if 'images' not in parameters:
            raise ValueError('images not in {}'.format(test_config))

        s3, creds, s3def = Connect(parameters['images']['credentials'])

        #dataset_desc = s3.GetDict(s3def['sets']['dataset']['bucket'],parameters['images']['dataset'])
        #class_dictionary = s3.GetDict(s3def['sets']['dataset']['bucket'],parameters['images']['class_dict']) 
        #imUtil = ImUtil(dataset_desc, class_dictionary)
                
        store = ImagesStore(s3, bucket=s3def['sets']['dataset']['bucket'], 
                        dataset_desc=parameters['images']['dataset'], 
                        class_dictionary=parameters['images']['class_dict'])

        for i, iman in enumerate(store):
            img = store.MergeIman(iman['img'], iman['ann'])
            if img is None:
                raise ValueError('img is None')
            if 'test_images' in parameters['coco'] and i >= parameters['coco']['test_images']:
                break'''
    



    def test_dataset(self):
        print('test_dataset: create me!')

if __name__ == '__main__':
    unittest.main()