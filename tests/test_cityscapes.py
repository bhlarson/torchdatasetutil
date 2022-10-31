from dataclasses import replace
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
from torch.utils.data import DataLoader, WeightedRandomSampler

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

        class_dictionary = ReadDict(parameters['cityscapes']['class_dict'])
        imUtil = ImUtil({}, class_dictionary)

        sampler =parameters['cityscapes']['sampler']
        if sampler and 'sample_weights' in class_dictionary:
                train_sampler_weights = class_dictionary['sample_weights']['weights']
                upsampled_class = class_dictionary['sample_weights']['class']
        else:
            train_sampler_weights=None
            upsampled_class=None



        loaders = CreateCityscapesLoaders(s3, s3def, 
                        src = parameters['cityscapes']['obj_src'],
                        dest = parameters['cityscapes']['destination'],
                        class_dictionary =class_dictionary,
                        bucket = s3def['sets']['dataset']['bucket'], 
                        width=parameters['cityscapes']['width'], 
                        height=parameters['cityscapes']['height'], 
                        batch_size = parameters['cityscapes']['batch_size'], 
                        num_workers=parameters['cityscapes']['num_workers'],
                        train_sampler_weights=train_sampler_weights,)

        parameters['cityscapes']['test_path']=os.path.join(parameters['cityscapes']['test_path'], '') # Add trailing slash if not present
        os.makedirs(parameters['cityscapes']['test_path'], exist_ok=True)

        minority_class_list = []

        for loader in tqdm(loaders, desc="CreateImageLoaders"):
            for i, data in tqdm(enumerate(loader['dataloader']), 
                                desc="Batch Reads", 
                                total=loader['batches']):
                inputs, labels, mean, stdev = data
                assert(inputs.size(0)==parameters['cityscapes']['batch_size'])
                assert(inputs.size(-1)==parameters['cityscapes']['width'])
                assert(inputs.size(-2)==parameters['cityscapes']['height'])
                assert(labels.size(0)==parameters['cityscapes']['batch_size'])
                assert(labels.size(-1)==parameters['cityscapes']['width'])
                assert(labels.size(-2)==parameters['cityscapes']['height'])

                images = inputs.cpu().permute(0, 2, 3, 1).numpy()
                labels = np.around(labels.cpu().numpy()).astype('uint8')
                mean = mean.cpu().numpy()
                stdev = stdev.cpu().numpy()


                for j, image in enumerate(images):
                    if upsampled_class is not None:
                        minority_class_list.append(upsampled_class in labels[j])

                    img = imUtil.MergeIman(images[j], labels[j], mean[j], stdev[j])
                    write_path = '{}{}{:03d}{:03d}.png'.format(parameters['cityscapes']['test_path'], loader['set'], i,j)                   
                    cv2.imwrite(write_path,img)

                #    is_success, buffer = cv2.imencode(".png", img)
                #    if not is_success:
                #        raise ValueError('test_imstore test_CreateImageLoaders cv2.imencode failure batch {} image {}'.format(i, j))

                if sampler and i >= 200:
                    minority_class_ratio = sum(minority_class_list)/len(minority_class_list)
                    if minority_class_ratio < 0.2:
                        raise ValueError('Weighted Random Sampler maybe dysfunctional. {:4f} ratio is too low for specified minority class'.format(minority_class_ratio))
                    break

                if 'test_images' in parameters['cityscapes'] and parameters['cityscapes']['test_images'] is not None and i*parameters['cityscapes']['batch_size'] >= parameters['cityscapes']['test_images']:
                    break
            
            sampler=False

if __name__ == '__main__':
    unittest.main()