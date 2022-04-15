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
from pathlib import Path, PurePath
from torch.utils.data.sampler import SubsetRandomSampler

from pymlutil.s3 import s3store, Connect
from pymlutil.jsonutil import ReadDictJson
from pymlutil.imutil import ImUtil, ImTransform

class ImagesStore(ImUtil):

    def __init__(self, s3, bucket, dataset_desc, class_dictionary):
        self.s3 = s3
        self.bucket = bucket
        self.dataset_desc = s3.GetDict(bucket,dataset_desc)
        self.class_dictionary = s3.GetDict(bucket,class_dictionary) 

        self.imflags = cv2.IMREAD_COLOR 
        if self.dataset_desc is not None and 'image_colorspace' in self.dataset_desc:
            if self.isGrayscale(self.dataset_desc['image_colorspace']):
                self.imflags = cv2.IMREAD_GRAYSCALE
        self.anflags = cv2.IMREAD_GRAYSCALE 

        self.bare_image = self.dataset_desc['image_pattern'].replace('*','')
        self.bare_label = self.dataset_desc['label_pattern'].replace('*','')

        self.images = []
        self.labels = []

        self.CreateIndex()
        super(ImagesStore, self).__init__(dataset_desc=self.dataset_desc, class_dictionary=self.class_dictionary)
        self.i = 0

    def isGrayscale(self, color_str):
        if color_str.lower() == 'grayscale':
            return True
        return False

    def ImagenameFromLabelname(self, lbl_filename):
        return lbl_filename.replace(self.bare_label, self.bare_image)


    def CreateIndex(self):
        file_list = self.s3.ListObjects( self.dataset_desc['bucket'], setname=self.dataset_desc['prefix'], pattern=None, recursive=self.dataset_desc['recursive'])
        imagedict = {}
        if self.dataset_desc['image_path']==self.dataset_desc['label_path']:
            for im_filename in file_list:
                if PurePath(im_filename).match(self.dataset_desc['image_pattern']):
                    imagedict[im_filename] = None
            for lbl_filename in file_list:
                if PurePath(lbl_filename).match(self.dataset_desc['label_pattern']):
                    im_filename = self.ImagenameFromLabelname(lbl_filename)
                    if im_filename in imagedict:
                        imagedict[im_filename] = lbl_filename

        for key in imagedict:
            if imagedict[key] is not None:
                self.images.append(key)
                self.labels.append(imagedict[key])



    def __iter__(self):
        self.i = 0
        return self

    def classes(self, anns):
        class_vector = np.zeros(self.class_dictionary ['classes'], dtype=np.float32)

        for ann in anns:
            obj = self.catToObj[ann['category_id']]
            if obj['trainId'] < self.class_dictionary ["classes"]:
                class_vector[obj['trainId']] = 1.0

        return class_vector

    def DecodeImage(self, bucket, objectname, flags):
        img = None
        numTries = 3
        for i in range(numTries):
            imgbuff = self.s3.GetObject(bucket, objectname)
            if imgbuff:
                imgbuff = np.frombuffer(imgbuff, dtype='uint8')
                img = cv2.imdecode(imgbuff, flags=self.imflags)
            if img is None or img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
                print('ImagesStore::DecodeImage failed to load {}/{} try {} img={}'.format(bucket, objectname, i, img))
            else:
                break
        return img

    def ConvertLabels(self, ann):
        trainAnn = np.zeros_like(ann)
        for obj in self.class_dictionary ['objects']: # Load RGB colors as BGR
            if not (obj['id'] == obj['trainId'] and obj['id'] == 0):
                trainAnn[ann==obj['id']] = obj['trainId']
        return trainAnn

    def len(self):
        return len(self.images)

    def __next__(self):
        if self.i < self.len():
            result = self.__getitem__(self.i)
            self.i += 1
            return result
        else:
            raise StopIteration

    def __getitem__(self, idx):
        if idx >= 0 and idx < self.len():
            img = self.DecodeImage(self.bucket, self.images[idx], self.imflags)
            ann = self.DecodeImage(self.bucket, self.labels[idx], self.anflags)
            if ann is not None:
                ann = self.ConvertLabels(ann)
            result = {'img':img, 'ann':ann}

            return result
        else:
            print('ImagesStore.__getitem__ idx {} invalid.  Must be >=0 and < ImagesStore.len={}'.format(idx, self.len()))
            return None


class ImagesDataset(Dataset):
    def __init__(self, s3, bucket, dataset_desc, class_dictionary, 
        height=640, 
        width=640, 
        image_transform=None,
        label_transform=None,
        normalize=True, 
        enable_transform=True, 
        flipX=True, 
        flipY=False, 
        rotate=3, 
        scale_min=0.75, 
        scale_max=1.25, 
        offset=0.1,
        astype='float32'
    ):
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.height = height
        self.width = width

        self.normalize = normalize
        self.enable_transform = enable_transform
        self.flipX = flipX
        self.flipY = flipY
        self.rotate = rotate
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.offset = offset
        self.astype = astype

        self.store = ImagesStore(s3, bucket, dataset_desc, class_dictionary)

        self.imTransform = ImTransform(height=height, width=width, 
                                     normalize=normalize, 
                                     enable_transform=enable_transform, 
                                     flipX=flipX, flipY=flipY, 
                                     rotate=rotate, 
                                     scale_min=scale_min, scale_max=scale_max, offset=offset, astype=astype)

    def __len__(self):
        return self.store.len()

    def __getitem__(self, idx):
        result = self.store.__getitem__(idx)
        if result is not None and result['img'] is not None and result['ann'] is not None:
            image = result['img']
            label = result['ann']

            if self.width is not None and self.height is not None:
                image, label, imgMean, imgStd = self.imTransform.random_resize_crop_or_pad(image, label)

            if image is not None and label is not None:
                if len(image.shape) < 3:
                    image = np.expand_dims(image, axis=-1)

                image = torch.from_numpy(image).permute(2, 0, 1)
                label = torch.from_numpy(label)

                if self.image_transform:
                    image = self.image_transform(image)
                if self.label_transform:
                    label = self.label_transform(label)
            
        else:
            image=None
            label=None
            imgMean = None
            imgStd = None
            print('ImagesDataset.__getitem__ idx {} returned result=None.'.format(idx))
        return image, label, imgMean, imgStd

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

default_loaders = [{'set':'train', 'split':0.8, 'enable_transform':True},
                   {'set':'test', 'split':0.2, 'enable_transform':False}]

def CreateDataLoaders(s3, bucket, dataset_dfn, class_dict, batch_size = 2, num_workers=0, cuda = True, loaders = default_loaders, 
                      height=640, width=640, 
                      image_transform=None, label_transform=None, 
                      normalize=True, flipX=True, flipY=False, 
                      rotate=3, scale_min=0.75, scale_max=1.25, offset=0.1, astype='float32',
                      random_seed = None):

    dataset = ImagesDataset(s3, bucket, dataset_dfn, class_dict, 
                            height=height, width=width, 
                            image_transform=image_transform, label_transform=label_transform, 
                            normalize=normalize,  enable_transform=default_loaders[0]['enable_transform'], 
                            flipX=flipX, flipY=flipY, 
                            rotate=rotate, scale_min=scale_min, scale_max=scale_max, offset=offset, astype=astype)

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    pin_memory = False
    if cuda:
        pin_memory = True

    startIndex = 0
    allocated = 0.0
    for i, loader in enumerate(loaders):
        allocated += loader['split']
        if allocated > 1.0:
            allocated = 1.0
        split = int(np.floor(allocated * dataset_size/batch_size))*batch_size
        if split > startIndex:

            if i > 0:
                dataset = ImagesDataset(s3, bucket, dataset_dfn, class_dict, 
                            height=height, width=width, 
                            image_transform=image_transform, label_transform=label_transform, 
                            normalize=normalize,  enable_transform=default_loaders[i]['enable_transform'], 
                            flipX=flipX, flipY=flipY, 
                            rotate=rotate, scale_min=scale_min, scale_max=scale_max, offset=offset, astype=astype)

            # Creating PT data samplers and loaders:
            loader['batches'] =int((split-startIndex)/batch_size)
            loader['length'] = loader['batches']*batch_size
            sampler = SubsetRandomSampler(indices[startIndex:split])
            startIndex = split

            loader['dataloader'] = torch.utils.data.DataLoader(dataset, 
                                                      batch_size=batch_size, 
                                                      sampler=sampler, 
                                                      num_workers=num_workers, 
                                                      pin_memory=pin_memory, 
                                                      collate_fn=collate_fn)         

    return loaders

def Test(args):

    s3, creds, s3def = Connect(args.credentails)

    if args.test_iterator:
        os.makedirs(args.test_path, exist_ok=True)
        store = ImagesStore(s3, s3def['sets']['dataset']['bucket'], args.dataset, args.class_dict)
        for i, iman in enumerate(store):
            img = store.MergeIman(iman['img'], iman['ann'])
            write_path = '{}imagesstoreiterator{:03d}.png'.format(args.test_path, i)
            cv2.imwrite(write_path,img)
            if i >= args.num_images:
                print ('test_iterator complete')
                break


    if args.test_dataset:
        loaders = CreateDataLoaders(s3=s3, 
                                    bucket=s3def['sets']['dataset']['bucket'], 
                                    dataset_dfn=args.dataset, 
                                    class_dict=args.class_dict, 
                                    batch_size=args.batch_size, 
                                    num_workers=args.num_workers, 
                                    cuda = args.cuda,
                                    height = args.height, width = args.width
                                )
        os.makedirs(args.test_path, exist_ok=True)

        for iDataset, loader in enumerate(loaders):
            i = 0
            while i < min(np.ceil(args.num_images/args.batch_size), loader['batches']):
                images, train_labels, train_mean, train_stdev = next(iter(loader['dataloader']))
                images = images.cpu().permute(0, 2, 3, 1).numpy()
                images = np.squeeze(images)
                j = 0
                while j < args.batch_size:
                    train_labels[j].cpu().numpy()
                    img = loader['dataloader'].dataset.store.MergeIman(images[j], train_labels[j].cpu().numpy(), train_mean[j].item(), train_stdev[j].item())
                    write_path = '{}imagestoredataset{}{:03d}{:03d}.png'.format(args.test_path, loader['set'], i,j)
                    cv2.imwrite(write_path,img)
                    j += 1
                i += 1
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
    parser.add_argument('-dataset', type=str, default='annotations/lit/dataset.yaml', help='Image dataset file')
    parser.add_argument('-class_dict', type=str, default='model/crisplit/lit.json', help='Model class definition file.')
    parser.add_argument('-num_images', type=int, default=10, help='Number of images to display')
    parser.add_argument('-num_workers', type=int, default=1, help='Data loader workers')
    parser.add_argument('-batch_size', type=int, default=4, help='Dataset batch size')
    parser.add_argument('-test_iterator', type=bool, default=True, help='True to test iterator')
    parser.add_argument('-test_dataset', type=bool, default=True, help='True to test dataset')
    parser.add_argument('-test_path', type=str, default='./datasets_test/', help='Test path ending in a forward slash')

    parser.add_argument('-height', type=int, default=640, help='Batch image height')
    parser.add_argument('-width', type=int, default=640, help='Batch image width')
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

