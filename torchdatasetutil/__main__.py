import sys
import os
from pycocotools import mask
import numpy as np
import cv2
import json
import unittest
from collections import defaultdict
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from pymlutil.s3 import s3store, Connect
from pymlutil.jsonutil import ReadDict
from pymlutil.imutil import ImUtil, ImTransform

from .cocostore import *
from .imstore import *
from .getcoco import *
from .getsceneflow import *

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process arguments')

    parser.add_argument('-d', '--debug', action='store_true',help='Wait for debuggee attach')   
    parser.add_argument('-debug_port', type=int, default=3000, help='Debug port')
    parser.add_argument('-debug_address', type=str, default='0.0.0.0', help='Debug port')
    parser.add_argument('-dataset_path', type=str, default='./dataset', help='Local dataset path')
    parser.add_argument('-credentials', type=str, default='creds.yaml', help='Credentials file.')
    parser.add_argument('-dataset_train', type=str, default='data/coco/annotations/instances_train2017.json', help='Coco dataset train instance json file.')
    parser.add_argument('-dataset_val', type=str, default='data/coco/annotations/instances_val2017.json', help='Coco dataset validation instance json file.')
    parser.add_argument('-train_image_path', type=str, default='data/coco/train2017', help='Coco image path for dataset.')
    parser.add_argument('-val_image_path', type=str, default='data/coco/val2017', help='Coco image path for dataset.')
    parser.add_argument('-class_dict', type=str, default='model/segmin/coco.json', help='Model class definition file.')
    parser.add_argument('-num_images', type=int, default=0, help='Number of images to display.  0 for full dataset')
    parser.add_argument('-num_workers', type=int, default=1, help='Data loader workers')
    parser.add_argument('-batch_size', type=int, default=4, help='Dataset batch size')
    parser.add_argument('-num_tries', type=int, default=3, help='Image read retries')
    

    parser.add_argument('-ci', action='store_true', help='True to test iterator')
    parser.add_argument('-coco_iterator', type=bool, default=False, help='True to test iterator')
    parser.add_argument('-cd', action='store_true', help='True to test dataset')
    parser.add_argument('-coco_dataset', action='store_true', help='True to test dataset')

    parser.add_argument('-coco_dataset_train', type=str, default='data/coco/annotations/instances_train2017.json', help='')
    parser.add_argument('-coco_dataset_val', type=str, default='data/coco/annotations/instances_val2017.json', help='')
    parser.add_argument('-coco_train_image_path', type=str, default='data/coco/train2017', help='')
    parser.add_argument('-coco_val_image_path', type=str, default='data/coco/val2017', help='')
    parser.add_argument('-coco_class_dict', type=str, default='model/segmin/coco.json', help='')

    parser.add_argument('-ii', action='store_true', help='True to test iterator')
    parser.add_argument('-image_iterator', type=bool, default=False, help='True to test iterator')
    parser.add_argument('-id', action='store_true', help='True to test dataset')
    parser.add_argument('-image_dataset', action='store_true', help='True to test dataset')

    parser.add_argument('-image_dataset_desc', type=str, default='data/lit/dataset.yaml', help='')
    parser.add_argument('-image_class_dict', type=str, default='model/crisplit/lit.json', help='')

    parser.add_argument('-test_path', type=str, default='./datasets_test/', help='Test path ending in a forward slash')

    parser.add_argument('-height', type=int, default=640, help='Batch image height')
    parser.add_argument('-width', type=int, default=640, help='Batch image width')
    parser.add_argument('-imflags', type=int, default=cv2.IMREAD_COLOR, help='cv2.imdecode flags')
    parser.add_argument('-cuda', type=bool, default=True)

    parser.add_argument('-getcoco', action='store_true',help='Get coco dataset') 
    parser.add_argument('-cocourl', type=json.loads, default=None, 
                        help='List of coco dataset URLs to load.  If none, coco 2017 datafiles are loaded from https://cocodataset.org/#download')
    parser.add_argument('-cocodatasetname', type=str, default='coco', help='coco dataset name in objet storage')

    parser.add_argument('-getsceneflow', action='store_true',help='Get sceneflow datasets')
    parser.add_argument('-sceneflowurl', type=json.loads, default=None, 
                        help='List of sceneflow dataset URLs to load.  If none, sceneflow_urls are loaded')
    parser.add_argument('-sceneflowdatasetname', type=str, default='sceneflow', help='Sintel dataset name in objet storage')

    args = parser.parse_args()

    if args.ci:
        args.coco_iterator = True

    if args.cd:
        args.coco_dataset = True

    if args.ii:
        args.image_iterator = True

    if args.id:
        args.image_dataset = True
    
    return args


def main(args):

    s3, creds, s3def = Connect(args.credentials)

    if args.getcoco:
        if args.cocourl is not None:
            getcoco(s3, s3def, cocourl=args.cocourl, dataset=args.cocodatasetname)
        else:
            getcoco(s3, s3def, dataset=args.cocodatasetname)

    if args.getsceneflow:
        if args.sceneflowurl is not None:
            getsceneflow(s3, s3def, cocourl=args.sceneflowurl, dataset=args.sceneflowdatasetname)
        else:
            getsceneflow(s3, s3def, dataset=args.sceneflowdatasetname)

        dataset_desc = s3.GetDict(s3def['sets']['dataset']['bucket'],args.dataset_train)
        class_dictionary = s3.GetDict(s3def['sets']['dataset']['bucket'],args.class_dict) 
        imUtil = ImUtil(dataset_desc, class_dictionary)

    if args.coco_iterator:

        store = CocoStore(s3, bucket=s3def['sets']['dataset']['bucket'], 
                        dataset_desc=args.coco_dataset_train, 
                        image_paths=args.coco_train_image_path, 
                        class_dictionary=args.coco_class_dict, numTries=args.num_tries,)

        for i, iman in tqdm(enumerate(store), desc='COCO read image', total=len(store),bar_format='{desc:<8.5}{percentage:3.0f}%|{bar:50}{r_bar}'):
            assert(iman['img'] is not None)
            assert(iman['ann'] is not None)

            if args.num_images > 0 and i >= args.num_images:
                print ('test_iterator complete')
                break

    if args.coco_dataset:
        loaders_dfn = [{'set':'train', 'dataset': args.dataset_train, 'image_path': args.train_image_path, 'enable_transform':True},
                       {'set':'test', 'dataset':  args.dataset_val, 'image_path': args.val_image_path, 'enable_transform':False}]

        loaders = CreateCocoLoaders(s3=s3, 
                                    bucket=s3def['sets']['dataset']['bucket'],
                                    class_dict=args.class_dict, 
                                    batch_size=args.batch_size, 
                                    num_workers=args.num_workers, 
                                    cuda = args.cuda,
                                    loaders = loaders_dfn,
                                    height = args.height, width = args.width,
                                    numTries=args.num_tries
                                )
        os.makedirs(args.test_path, exist_ok=True)

        for loader in tqdm(loaders, desc="Loader"):
            for i, data in tqdm(enumerate(loader['dataloader']), 
                                desc="Batch Reads", 
                                total=loader['batches']):
                inputs, labels, mean, stdev = data
                assert(inputs is not None)
                assert(labels is not None)

            if args.num_images > 0 and i >= args.num_images:
                print ('test_iterator complete')
                break


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
                if args.num_images > 0 and  i > min(np.ceil(args.num_images/args.batch_size), loader['batches']):
                    break
        print ('coco_dataset complete')

    if args.image_iterator:

        store = ImagesStore(s3, bucket=s3def['sets']['dataset']['bucket'], 
                        dataset_desc=args.image_dataset_desc, 
                        class_dictionary=args.image_class_dict, numTries=args.num_tries)
        failures = []
        for i, iman in  tqdm(enumerate(store),
                             desc="ImagesStore iterator reads",
                             total=len(store)):
            #img = store.MergeIman(iman['img'], iman['ann'])
            if iman is None:
                failures.append(i)
                print('torchdatasetutil ImagesStore failed to load image {}: {}'.format(i, store.images[i]))
            elif iman['ann'] is None:
                print('torchdatasetutil ImagesStore failed to load annotation {}: {}'.format(i, store.labels[i]))
                failures.append(i)
            if args.num_images > 0 and i >= args.num_images:
                print ('test_iterator complete')
                break

        print ('{} read failures'.format(len(failures)))
    if args.image_dataset:

        loaders = CreateImageLoaders(s3=s3, 
                                    bucket=s3def['sets']['dataset']['bucket'],
                                    dataset_dfn=args.image_dataset_desc,
                                    class_dict=args.image_class_dict, 
                                    batch_size=args.batch_size, 
                                    num_workers=args.num_workers, 
                                    cuda = args.cuda,
                                    height = args.height, width = args.width,
                                    numTries=args.num_tries,
                                )
        for loader in tqdm(loaders, desc="Loader"):
            for i, data in tqdm(enumerate(loader['dataloader']), 
                                desc="Batch Reads", 
                                total=loader['batches']):
                inputs, labels, mean, stdev = data
                assert(inputs is not None)
                assert(labels is not None)
                assert(inputs.shape[0] == args.batch_size)
                assert(inputs.shape[-1] == args.width)
                assert(inputs.shape[-2] == args.height)

            if args.num_images > 0 and i >= args.num_images:
                print ('test_iterator complete')
                break

    print('torchdatasetutil complete')

#objdict = json.load(open('/data/git/mllib/datasets/coco.json'))
#Test(objdict, '/store/Datasets/coco/instances_val2017.json', '/store/Datasets/coco/val2014', 'COCO_val2014_')


    
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


