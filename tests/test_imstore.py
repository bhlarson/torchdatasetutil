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

sys.path.insert(0, os.path.abspath('')) # Test files from current path rather than installed module
from torchdatasetutil.imstore import ImagesStore, CreateImageLoaders

test_config = 'test.yaml'
class Test(unittest.TestCase):      

    def test_iterator(self):
        parameters = ReadDict(test_config)

        if 'images' not in parameters:
            raise ValueError('images not in {}'.format(test_config))

        s3, creds, s3def = Connect(parameters['images']['credentials'])
                
        store = ImagesStore(s3, bucket=s3def['sets']['dataset']['bucket'], 
                        dataset_desc=parameters['images']['dataset'], 
                        class_dictionary=parameters['images']['class_dict'])

        for i, iman in enumerate(store):
            img = store.MergeIman(iman['img'], iman['ann'])
            is_success, buffer = cv2.imencode(".png", img)
            if not is_success:
                raise ValueError('test_imstore test_iterator cv2.imencode failure  image {}'.format(i))
            if img is None:
                raise ValueError('img is None')
            if 'test_images' in parameters['images'] and i >= parameters['images']['test_images']:
                break
    
    def test_CreateImageLoaders(self):
        parameters = ReadDict(test_config)

        if 'images' not in parameters:
            raise ValueError('images not in {}'.format(test_config))

        s3, creds, s3def = Connect(parameters['images']['credentials'])

        dataset_desc = s3.GetDict(s3def['sets']['dataset']['bucket'],parameters['images']['dataset'])
        class_dictionary = s3.GetDict(s3def['sets']['dataset']['bucket'],parameters['images']['class_dict']) 
        imUtil = ImUtil(dataset_desc, class_dictionary)

        loaders = CreateImageLoaders(s3, bucket=s3def['sets']['dataset']['bucket'],
                                     dataset_dfn=parameters['images']['dataset'],
                                     class_dict=parameters['images']['class_dict'])

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
                    raise ValueError('test_imstore test_CreateImageLoaders cv2.imencode failure batch {} image {}'.format(i, j))
            if 'test_images' in parameters['images'] and i >= parameters['images']['test_images']:
                break

if __name__ == '__main__':
    unittest.main()