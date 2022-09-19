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

sys.path.insert(0, os.path.abspath('')) # Test files from current path rather than installed module
from torchdatasetutil.cityscapesstore import CreateCityscapesLoaders

test_config = 'test.yaml'
class Test(unittest.TestCase):      

    
    def test_CreateImageLoaders(self):
        parameters = ReadDict(test_config)

        if 'images' not in parameters:
            raise ValueError('images not in {}'.format(test_config))

        s3, creds, s3def = Connect(parameters['images']['credentials'])

        dataset_desc = s3.GetDict(s3def['sets']['dataset']['bucket'],parameters['images']['dataset'])
        class_dictionary = s3.GetDict(s3def['sets']['dataset']['bucket'],parameters['images']['class_dict']) 
        imUtil = ImUtil(dataset_desc, class_dictionary)

        loaders = CreateCityscapesLoaders(s3, s3def, 
                        src = parameters['cityscapes']['obj_src'],
                        dest = parameters['cityscapes']['destination'],
                        bucket = s3def['sets']['dataset']['bucket'], 
                        width=parameters['cityscapes']['width'], 
                        height=parameters['cityscapes']['height'], 
                        batch_size = parameters['cityscapes']['batch_size'], 
                        num_workers=parameters['cityscapes']['num_workers'])

        for loader in tqdm(loaders, desc="CreateImageLoaders"):
            for i, data in tqdm(enumerate(loader['dataloader']), 
                                desc="Batch Reads", 
                                total=loader['batches']):
                inputs, labels, mean, std = data
                assert(inputs.size(0)==parameters['cityscapes']['batch_size'])
                assert(inputs.size(-2)==parameters['cityscapes']['width'])
                assert(inputs.size(-3)==parameters['cityscapes']['height'])
                assert(labels.size(0)==parameters['cityscapes']['batch_size'])
                assert(labels.size(-2)==parameters['cityscapes']['height'])
                assert(labels.size(-1)==parameters['cityscapes']['width'])

                if 'test_images' in parameters['images'] and parameters['images']['test_images'] is not None and i >= parameters['images']['test_images']:
                    break

if __name__ == '__main__':
    unittest.main()