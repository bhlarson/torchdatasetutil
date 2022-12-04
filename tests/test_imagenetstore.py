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
from torchdatasetutil.imagenetstore import CreateImagenetLoaders

test_config = 'test.yaml'
class Test(unittest.TestCase):      
   
    def test_CreateImagenetLoaders(self):
        parameters = ReadDict(test_config)

        if 'imagenet' not in parameters:
            raise ValueError('images not in {}'.format(test_config))

        s3, creds, s3def = Connect(parameters['imagenet']['credentials'])

        loaders = CreateImagenetLoaders(s3, s3def, 
                                parameters['imagenet']['obj_src'], 
                                parameters['imagenet']['destination'], 
                                augment=parameters['imagenet']['augment'], 
                                normalize=parameters['imagenet']['normalize'], 
                                resize_width=parameters['imagenet']['resize_width'], 
                                resize_height=parameters['imagenet']['resize_height'],
                                crop_width=parameters['imagenet']['crop_width'], 
                                crop_height=parameters['imagenet']['crop_height'], 
                                batch_size=parameters['imagenet']['batch_size'], 
                                num_workers=parameters['imagenet']['num_workers'],
                                flipX=parameters['imagenet']['flipX'], 
                                flipY=parameters['imagenet']['flipY'], 
                                rotate=parameters['imagenet']['rotate'], 
                                scale_min=parameters['imagenet']['scale_min'], 
                                scale_max=parameters['imagenet']['scale_max'], 
                                offset=parameters['imagenet']['offset'], 
                                augment_noise=parameters['imagenet']['augment_noise'])

        parameters['imagenet']['test_path']=os.path.join(parameters['imagenet']['test_path'], '') # Add trailing slash if not present
        os.makedirs(parameters['imagenet']['test_path'], exist_ok=True)

        for loader in tqdm(loaders, desc="Loader"):
            for i, data in tqdm(enumerate(loader['dataloader']), 
                                desc="Batch Reads", 
                                total=loader['batches']):
                sample, target = data
                assert(sample is not None)
                assert(sample.shape[0] == parameters['imagenet']['batch_size'])
                if loader['height'] is not None:
                    assert(sample.shape[2] == loader['height'])
                if loader['width'] is not None:
                    assert(sample.shape[3] == loader['width'])
                    assert(target is not None)

                if parameters['imagenet']['save_image']:
                    sample =  sample.permute(0, 2, 3, 1) # Change to batch, height, width, channel for rendering
                    sample_max = sample.max()
                    sample_min = sample.min()
                    if sample_max > sample_min:
                        for j, image in enumerate(sample):
                            image = 255*(image - sample_min)/(sample_max-sample_min) # Convert to RGB color rane
                            image = image.cpu().numpy().astype(np.uint8)
                            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                            write_path = '{}{}{:03d}{:03d}.png'.format(parameters['imagenet']['test_path'], loader['set'], i,j)            
                            cv2.imwrite(write_path,image)

                if parameters['imagenet']['test_images'] is not None and parameters['imagenet']['test_images'] > 0 and (i+1)*parameters['imagenet']['batch_size'] >= parameters['imagenet']['test_images']:
                    print ('test_iterator complete')
                    break

if __name__ == '__main__':
    unittest.main()