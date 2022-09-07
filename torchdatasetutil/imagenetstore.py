import sys
import os
from pycocotools import mask
import numpy as np
import cv2
import json
import functools
import random
from collections import defaultdict
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from pymlutil.s3 import s3store, Connect
from pymlutil.jsonutil import ReadDict
from pymlutil.imutil import ImUtil, ImTransform

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def CreateImagesetLoaders(width=256, height=256, batch_size = 2, shuffle=True, 
                      num_workers=0, cuda = True, timeout=0, loaders = None, 
                      image_transform=None, label_transform=None, 
                      normalize=True, flipX=True, flipY=False, 
                      random_seed = None, numTries=3):

    pin_memory = False
    if cuda:
        pin_memory = True


    # Load dataset
    if loaders is None:
        train_transform = transforms.Compose([
            transforms.RandomAffine(degrees=10, 
               translate=(0.1, 0.1), 
               scale=(0.9, 1.1), 
               interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop( (width, height), padding=None, pad_if_needed=True, fill=0, padding_mode='constant'),
            transforms.ToTensor(), 
            transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), # Imagenet mean and standard deviation
            transforms.RandomHorizontalFlip(p=0.5),
            #AddGaussianNoise(0., 0.05)
        ])

        test_transform = transforms.Compose([
            transforms.RandomCrop( (width, height), padding=None, pad_if_needed=True, fill=0, padding_mode='constant'),
            transforms.ToTensor(), 
            transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)) # Imagenet mean and standard deviation
        ])

        default_loaders = [{'set':'train', 'dataset': '/nvmestore/mlstore/trainingset/imagenet/', 'enable_transform':True, 'transform':train_transform},
                        {'set':'val', 'dataset': '/nvmestore/mlstore/trainingset/imagenet/', 'enable_transform':False, 'transform':train_transform}]

        loaders = default_loaders

    startIndex = 0
    allocated = 0.0

    for i, loader in enumerate(loaders):


        imagenet_data = datasets.ImageNet(loader['dataset'], split=loader['set'], transform=loader['transform'])
        loader['dataloader'] = torch.utils.data.DataLoader(imagenet_data,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                num_workers=num_workers,
                                                pin_memory=pin_memory)

        # Creating PT data samplers and loaders:
        loader['batches'] =int(imagenet_data.__len__()/batch_size)
        loader['length'] = loader['batches']*batch_size      

    return loaders

def main(args):

    if args.test_dataset:
        loaders = CreateImagesetLoaders(args.width, args.height, batch_size=args.batch_size, 
                                    num_workers=args.num_workers,
                                    cuda = args.cuda)

        for loader in tqdm(loaders, desc="Loader"):
            for i, data in tqdm(enumerate(loader['dataloader']), 
                                desc="Batch Reads", 
                                total=loader['batches']):
                sample, target = data
                assert(sample is not None)
                assert(sample.shape[0] == args.batch_size)
                assert(sample.shape[2] == args.height)
                assert(sample.shape[3] == args.width)
                assert(target is not None)

            if args.num_images is not None and args.num_images > 0 and i >= args.num_images:
                print ('test_iterator complete')
                break

    print('Test complete')

#objdict = json.load(open('/data/git/mllib/datasets/coco.json'))
#Test(objdict, '/store/Datasets/coco/instances_val2017.json', '/store/Datasets/coco/val2014', 'COCO_val2014_')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process arguments')

    parser.add_argument('-d', '--debug', action='store_true',help='Wait for debuggee attach')   
    parser.add_argument('-debug_port', type=int, default=3000, help='Debug port')
    parser.add_argument('-debug_address', type=str, default='0.0.0.0', help='Debug port')
    parser.add_argument('-credentails', type=str, default='creds.yaml', help='Credentials file.')
    parser.add_argument('-num_images', type=int, default=0, help='Number of images to display')
    parser.add_argument('-num_workers', type=int, default=25, help='Data loader workers')
    parser.add_argument('-batch_size', type=int, default=4, help='Dataset batch size')
    parser.add_argument('-i', action='store_true', help='True to test iterator')
    parser.add_argument('-test_iterator', type=bool, default=False, help='True to test iterator')
    parser.add_argument('-ds', action='store_true', help='True to test dataset')
    parser.add_argument('-test_dataset', action='store_true', help='True to test dataset')
    parser.add_argument('-test_path', type=str, default='./datasets_test/', help='Test path ending in a forward slash')
    parser.add_argument('-test_config', type=str, default='test.yaml', help='Test configuration file')

    parser.add_argument('-height', type=int, default=256, help='Batch image height')
    parser.add_argument('-width', type=int, default=256, help='Batch image width')
    parser.add_argument('-imflags', type=int, default=cv2.IMREAD_COLOR, help='cv2.imdecode flags')
    parser.add_argument('-cuda', type=bool, default=True, help='pytorch CUDA flag') 
    parser.add_argument('-numTries', type=int, default=3, help="Read retries")

    args = parser.parse_args()

    if args.i:
        args.test_iterator = True

    if args.ds:
        args.test_dataset = True

    return args
    
if __name__ == '__main__':
    import argparse
    args = parse_arguments()

    if args.debug:
        print("Wait for debugger attach on {}:{}".format(args.debug_address, args.debug_port))
        import debugpy

        debugpy.listen(address=(args.debug_address, args.debug_port)) # Pause the program until a remote debugger is attached
        debugpy.wait_for_client()  # Pause the program until a remote debugger is attached
        print("Debugger attached")

    main(args)

