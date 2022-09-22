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

        for i, iman in  tqdm(enumerate(store),
                             desc="ImagesStore iterator reads",
                             total=len(store)):
            #img = store.MergeIman(iman['img'], iman['ann'])
            assert(iman['img'] is not None)
            assert(iman['ann'] is not None)
            if 'test_images' in parameters['images'] and parameters['images']['test_images'] is not None  and i >= parameters['images']['test_images']:
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
                                     batch_size=parameters['images']['batch_size'], 
                                     height=parameters['images']['height'], 
                                     width=parameters['images']['width'],
                                     class_dict=parameters['images']['class_dict'])

        for loader in tqdm(loaders, desc="CreateImageLoaders"):
            for i, data in tqdm(enumerate(loader['dataloader']), 
                                desc="Batch Reads", 
                                total=loader['batches']):
                inputs, labels, mean, stdev = data
                assert(inputs.size(0)==parameters['images']['batch_size'])
                assert(inputs.size(-1)==parameters['images']['width'])
                assert(inputs.size(-2)==parameters['images']['height'])
                assert(labels.size(0)==parameters['images']['batch_size'])
                assert(labels.size(-1)==parameters['images']['width'])
                assert(labels.size(-2)==parameters['images']['height'])

                #images = inputs.cpu().permute(0, 3, 1, 2).numpy()
                #labels = np.around(labels.cpu().numpy()).astype('uint8')
                #mean = mean.cpu().numpy()
                #stdev = stdev.cpu().numpy()

                #for j, image in enumerate(images):
                #    img = imUtil.MergeIman(images[j], labels[j], mean[j], stdev[j])
                #    is_success, buffer = cv2.imencode(".png", img)
                #    if not is_success:
                #        raise ValueError('test_imstore test_CreateImageLoaders cv2.imencode failure batch {} image {}'.format(i, j))
                if 'test_images' in parameters['images'] and parameters['images']['test_images'] is not None and i >= parameters['images']['test_images']:
                    break

if __name__ == '__main__':
    unittest.main()