import sys
import os
from pycocotools import mask
import numpy as np
import cv2
import json
import functools
from collections import defaultdict
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from pymlutil.s3 import s3store, Connect
from pymlutil.jsonutil import ReadDict
from pymlutil.imutil import ImUtil, ImTransform, AddGaussianNoise, ResizePad

default_loaders = [{'set':'train', 'enable_transform':True, 'shuffle': True},
                   {'set':'test', 'enable_transform':False, 'shuffle': False}]

def CreateCifar10Loaders(dataset_path, batch_size = 2,  
                      num_workers=0, cuda = True, loaders = default_loaders, 
                      rotate=3, scale_min=0.75, scale_max=1.25, offset=0.1, augment_noise=0.0, width=32, height=32):

    Cifar10Classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    pin_memory = False
    # if cuda:
    #     pin_memory = True

    for i, loader in enumerate(loaders):
        if loader['enable_transform']:
            transform_list = []

            if width != 32 or height != 32:
                transform_list.append(ResizePad(width, height))

            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
            if rotate > 0 or offset > 0 or scale_min != 1.0 or scale_max != 1.0:
                transform_list.append(transforms.RandomAffine(degrees=rotate,
                        translate=(offset, offset), 
                        scale=(scale_min, scale_max), 
                        interpolation=transforms.InterpolationMode.BILINEAR))
            transform_list.append(transforms.ToTensor())
            transform_list.append(transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))) # Imagenet mean and standard deviation
            if augment_noise > 0.0:
                transform_list.append(AddGaussianNoise(0., augment_noise))

            transform = transforms.Compose(transform_list)
        else:
            transform_list = []
            if width != 32 or height != 32:
                transform_list.append(ResizePad(width, height))
            transform_list.append(transforms.ToTensor())
            transform_list.append(transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))) # Imagenet mean and standard deviation

            transform = transforms.Compose(transform_list)

        dataset = torchvision.datasets.CIFAR10(root=dataset_path, train=loader['set']=='train', download=True, transform=transform)
        # Creating PT data samplers and loaders:
        loader['width']=width
        loader['height']=height
        loader['in_channels']=3
        loader['num_classes']=len(Cifar10Classes)
        loader['classes']=list(Cifar10Classes)
        loader['batches'] =int(len(dataset)/batch_size)
        loader['length'] = loader['batches']*batch_size
        loader['dataloader'] = torch.utils.data.DataLoader(dataset=dataset, 
                                                    batch_size=batch_size,
                                                    shuffle=loader['shuffle'],
                                                    num_workers=num_workers, 
                                                    pin_memory=pin_memory)
    return loaders

def main(args):

    parameters = ReadDict(args.test_config)

    if args.test_dataset:
        loaders = CreateCifar10Loaders(args.dataset_path, 
                        batch_size = args.batch_size,  
                        num_workers=args.num_workers, 
                        cuda = args.cuda, 
                        rotate=args.augment_rotation, 
                        scale_min=args.augment_scale_min, 
                        scale_max=args.augment_scale_max, 
                        offset=args.augment_translate_x,
                        augment_noise = args.augment_noise,
                        width = parameters['cifar10']['width'],
                        height = parameters['cifar10']['height'],
                        )

        parameters['cifar10']['test_path']=os.path.join(parameters['cifar10']['test_path'], '') # Add trailing slash if not present
        os.makedirs(parameters['cifar10']['test_path'], exist_ok=True)

        for loader in tqdm(loaders, desc="Loader"):
            for i, data in tqdm(enumerate(loader['dataloader']), 
                                desc="Batch Reads", 
                                total=loader['batches']):
                sample, target = data
                assert(sample is not None)
                assert(sample.shape[0] == args.batch_size)
                assert(sample.shape[2] == loader['height'])
                assert(sample.shape[3] == loader['width'])
                assert(target is not None)

                sample =  sample.permute(0, 2, 3, 1) # Change to batch, height, width, channel for rendering
                sample_max = sample.max()
                sample_min = sample.min()
                if sample_max > sample_min:
                    for j, image in enumerate(sample):
                        image = 255*(image - sample_min)/(sample_max-sample_min) # Convert to RGB color rane
                        image = image.cpu().numpy().astype(np.uint8)
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        write_path = '{}{}{:03d}{:03d}.png'.format(parameters['cifar10']['test_path'], loader['set'], i,j)            
                        cv2.imwrite(write_path,image)

                if args.num_images is not None and args.num_images > 0 and i >= args.num_images:
                    print ('test_dataset complete')
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
    parser.add_argument('-dataset_path', type=str, default='./dataset', help='Local dataset path')
    parser.add_argument('-num_images', type=int, default=10, help='Number of images to display')
    parser.add_argument('-num_workers', type=int, default=0, help='Data loader workers')
    parser.add_argument('-batch_size', type=int, default=4, help='Dataset batch size')
    parser.add_argument('-ds', action='store_true', help='True to test dataset')
    parser.add_argument('-test_dataset', action='store_true', help='True to test dataset')
    parser.add_argument('-test_path', type=str, default='./datasets_test/', help='Test path ending in a forward slash')
    parser.add_argument('-test_config', type=str, default='test.yaml', help='Test configuration file')

    parser.add_argument('-cuda', type=bool, default=True, help='pytorch CUDA flag') 
    parser.add_argument('-numTries', type=int, default=3, help="Read retries")

    parser.add_argument('-augment_rotation', type=float, default=0.0, help='Input augmentation rotation degrees')
    parser.add_argument('-augment_scale_min', type=float, default=1.00, help='Input augmentation scale')
    parser.add_argument('-augment_scale_max', type=float, default=1.00, help='Input augmentation scale')
    parser.add_argument('-augment_translate_x', type=float, default=0.0, help='Input augmentation translation')
    parser.add_argument('-augment_translate_y', type=float, default=0.0, help='Input augmentation translation')
    parser.add_argument('-augment_noise', type=float, default=0.1, help='Augment image noise')

    args = parser.parse_args()

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

