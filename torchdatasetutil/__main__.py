import sys
import os
from pycocotools import mask
import numpy as np
import cv2
import json
from collections import defaultdict
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from pymlutil.s3 import s3store, Connect
from pymlutil.jsonutil import ReadDictJson
from pymlutil.imutil import ImUtil, ImTransform
from .cocostore import *
from .imstore import *

def Test(args):

    s3, creds, s3def = Connect(args.credentails)

    dataset_desc = s3.GetDict(s3def['sets']['dataset']['bucket'],args.dataset_train)
    class_dictionary = s3.GetDict(s3def['sets']['dataset']['bucket'],args.class_dict) 
    imUtil = ImUtil(dataset_desc, class_dictionary)
    
    if args.test_iterator:
        os.makedirs(args.test_path, exist_ok=True)
        
        store = CocoStore(s3, bucket=s3def['sets']['dataset']['bucket'], 
                          dataset_desc=args.dataset_train, 
                          image_paths=args.train_image_path, 
                          class_dictionary=args.class_dict, 
                          imflags=args.imflags)

        for i, iman in enumerate(store):
            img = store.MergeIman(iman['img'], iman['ann'])
            write_path = '{}cocostoreiterator{:03d}.png'.format(args.test_path, i)
            cv2.imwrite(write_path,img)
            if i >= args.num_images:
                print ('test_iterator complete')
                break

    if args.test_dataset:

        loaders_dfn = [{'set':'train', 'dataset': args.dataset_train, 'image_path': args.train_image_path, 'enable_transform':True},
                       {'set':'test', 'dataset':  args.dataset_val, 'image_path': args.val_image_path, 'enable_transform':False}]

        loaders = CreateCocoLoaders(s3=s3, 
                                    bucket=s3def['sets']['dataset']['bucket'],
                                    class_dict=args.class_dict, 
                                    batch_size=args.batch_size, 
                                    num_workers=args.num_workers, 
                                    cuda = args.cuda,
                                    loaders = loaders_dfn,
                                    height = args.height, width = args.width
                                )
        os.makedirs(args.test_path, exist_ok=True)

        for iDataset, loader in enumerate(loaders):
            for i, data  in enumerate(loader['dataloader']):
                images, labels, mean, stdev = data
                images = images.cpu().permute(0, 2, 3, 1).numpy()
                images = np.squeeze(images)
                labels = labels.cpu().numpy()

                for j in  range(args.batch_size):
                    img = imUtil.MergeIman(images[j], labels[j], mean[j].item(), stdev[j].item())
                    write_path = '{}cocostoredataset{}{:03d}{:03d}.png'.format(args.test_path, loader['set'], i,j)
                    cv2.imwrite(write_path,img)
                if i > min(np.ceil(args.num_images/args.batch_size), loader['batches']):
                    break
        print ('test_dataset complete')

    print('Test complete')

#objdict = json.load(open('/data/git/mllib/datasets/coco.json'))
#Test(objdict, '/store/Datasets/coco/instances_val2017.json', '/store/Datasets/coco/val2014', 'COCO_val2014_')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process arguments')

    parser.add_argument('-d', '--debug', action='store_true',help='Wait for debuggee attach')   
    parser.add_argument('-debug_port', type=int, default=3000, help='Debug port')
    parser.add_argument('-debug_address', type=str, default='0.0.0.0', help='Debug port')
    parser.add_argument('-dataset_path', type=str, default='./dataset', help='Local dataset path')
    parser.add_argument('-credentails', type=str, default='creds.json', help='Credentials file.')
    parser.add_argument('-dataset_train', type=str, default='data/coco/annotations/instances_train2017.json', help='Coco dataset train instance json file.')
    parser.add_argument('-dataset_val', type=str, default='data/coco/annotations/instances_val2017.json', help='Coco dataset validation instance json file.')
    parser.add_argument('-train_image_path', type=str, default='data/coco/train2017', help='Coco image path for dataset.')
    parser.add_argument('-val_image_path', type=str, default='data/coco/val2017', help='Coco image path for dataset.')
    parser.add_argument('-class_dict', type=str, default='model/segmin/coco.json', help='Model class definition file.')
    parser.add_argument('-num_images', type=int, default=10, help='Number of images to display')
    parser.add_argument('-num_workers', type=int, default=1, help='Data loader workers')
    parser.add_argument('-batch_size', type=int, default=4, help='Dataset batch size')
    parser.add_argument('-test_iterator', type=bool, default=True, help='True to test iterator')
    parser.add_argument('-test_dataset', type=bool, default=True, help='True to test dataset')
    parser.add_argument('-test_path', type=str, default='./datasets_test/', help='Test path ending in a forward slash')

    parser.add_argument('-height', type=int, default=640, help='Batch image height')
    parser.add_argument('-width', type=int, default=640, help='Batch image width')
    parser.add_argument('-imflags', type=int, default=cv2.IMREAD_COLOR, help='cv2.imdecode flags')
    parser.add_argument('-cuda', type=bool, default=True)

    args = parser.parse_args()
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

    Test(args)

