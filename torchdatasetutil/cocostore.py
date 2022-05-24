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
from pymlutil.jsonutil import ReadDict
from pymlutil.imutil import ImUtil, ImTransform

class CocoStore(ImUtil):

    def __init__(self, s3, # pymlutil.s3 s3 class object
                 bucket, # bucket name string
                 dataset_desc, # object in bucket containing coco format dataset definition 
                 image_paths, # path in bucket to dataset images
                 class_dictionary, # json or yaml class dictionary object described in  https://github.com/bhlarson/torchdatasetutil/blob/main/torchdatasetutil.ipynb#ClassDictionary
                 imflags=cv2.IMREAD_COLOR, # image colorspace to load in cv2 format
                 name_decoration='',  # Additional test to append to the filename to load
                 numTries=3 ): # Number of read retries

        self.s3 = s3 
        self.bucket = bucket 
        self.dataset_desc = s3.GetDict(bucket,dataset_desc) 
        self.class_dictionary = s3.GetDict(bucket,class_dictionary)
        self.image_paths = image_paths
        self.name_decoration = name_decoration
        self.imflags = imflags
        self.numTries = numTries



        self.CreateIndex()
        super(CocoStore, self).__init__(dataset_desc=self.dataset_desc, class_dictionary=self.class_dictionary)
        self.i = 0

    def CreateIndex(self):
        # create index objDict rather than coco types
        anns, cats, imgs = {}, {}, {}
        imgToAnns = defaultdict(list)
        catToImgs = defaultdict(list)
        catToObj = {}
        objs = {}
        # make list based on 
        if 'annotations' in self.dataset_desc:
            for ann in self.dataset_desc['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset_desc:
            for img in self.dataset_desc['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset_desc:
            for cat in self.dataset_desc['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset_desc and 'categories' in self.dataset_desc:
            for ann in self.dataset_desc['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        if self.class_dictionary  is not None:
            for obj in self.class_dictionary ['objects']:
                catToObj[obj['id']] = obj
                objs[obj['trainId']] = {'category':obj['category'], 
                                        'color':obj['color'],
                                        'display':obj['display']
                                        }


        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats
        self.catToObj = catToObj
        self.objs = objs


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

    def DecodeImage(self, bucket, objectname):
        img = None
        
        for i in range(self.numTries):
            imgbuff = self.s3.GetObject(bucket, objectname)
            if imgbuff:
                imgbuff = np.frombuffer(imgbuff, dtype='uint8')
                img = cv2.imdecode(imgbuff, flags=self.imflags)
            if img is None:
                print('CocoStore::DecodeImage failed to load {}/{} try {}'.format(bucket, objectname, i))
            else:
                break
        return img

    def drawann(self, imgDef, anns):
        annimg = np.zeros(shape=[imgDef['height'], imgDef['width']], dtype=np.uint8)
        for ann in anns:
            obj = self.catToObj[ann['category_id']]
            if obj['trainId'] < self.class_dictionary ["classes"]:
                if type(ann['segmentation']) is list:
                    for i in range(len(ann['segmentation'])):
                        contour = np.rint(np.reshape(ann['segmentation'][i], (-1, 2))).astype(np.int32)
                        cv2.drawContours(image=annimg, contours=[contour], contourIdx=-1, color=obj['trainId'] , thickness=cv2.FILLED)
                elif type(ann['segmentation']) is dict:
                    rle = ann['segmentation']
                    compressed_rle = mask.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
                    annmask = mask.decode(compressed_rle)
                    annimg[annmask] = obj['trainId']
                else:
                    print('unexpected segmentation')
            else:
                print('trainId {} >= classes {}'.format(obj['trainId'], self.class_dictionary ["classes"]))    
        return annimg

    def __len__(self):
        return len(self.dataset_desc['images'])

    def __next__(self):
        if self.i < self.__len__():
            result = self.__getitem__(self.i)
            self.i += 1
            return result
        else:
            raise StopIteration

    def __getitem__(self, idx):
        if idx >= 0 and idx < self.__len__():
            img_entry = self.dataset_desc['images'][idx]
            imgFile = '{}/{}{}'.format(self.image_paths,self.name_decoration,img_entry['file_name'])
            img = self.DecodeImage(self.bucket, imgFile)
            ann_entry = self.imgToAnns[img_entry['id']]
            ann = self.drawann(img_entry, ann_entry)
            classes = self.classes(ann_entry)
            result = {'img':img, 'ann':ann, 'classes':classes}

            return result
        else:
            print('CocoStore.__getitem__ idx {} invalid.  Must be >=0 and < CocoStore.len={}'.format(idx, self.__len__()))
            return None

class CocoDataset(Dataset):
    def __init__(self, s3, bucket, dataset_desc, image_paths, class_dictionary, 
        height=640, 
        width=640, 
        imflags=cv2.IMREAD_COLOR, 
        image_transform=None,
        label_transform=None,
        name_decoration='',
        normalize=True, 
        enable_transform=True, 
        flipX=True, 
        flipY=False, 
        rotate=15, 
        scale_min=0.75, 
        scale_max=1.25, 
        offset=0.1,
        numTries=3, # Number of read retries
    ):
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.height = height
        self.width = width
        self.imflags = imflags

        self.normalize = normalize
        self.enable_transform = enable_transform
        self.flipX = flipX
        self.flipY = flipY
        self.rotate = rotate
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.offset = offset
        self.numTries = numTries

        self.store = CocoStore(s3, bucket, dataset_desc, image_paths, class_dictionary, imflags=self.imflags, name_decoration=name_decoration, numTries=self.numTries)


        self.imTransform = ImTransform(height=height, width=width, 
                                     normalize=normalize, 
                                     enable_transform=enable_transform, 
                                     flipX=flipX, flipY=flipY, 
                                     rotate=rotate, 
                                     scale_min=scale_min, scale_max=scale_max, offset=offset, astype=self.store.class_dictionary['input_type'])

    def __len__(self):
        return self.store.__len__()

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
            print('CocoDataset.__getitem__ idx {} returned result=None.'.format(idx))
        return image, label, imgMean, imgStd

# Handle corrupt images:
# https://github.com/pytorch/pytorch/issues/1137
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

default_loaders = [{'set':'train', 'dataset': 'data/coco/annotations/instances_train2017.json', 'image_path':'data/coco/train2017' , 'enable_transform':True},
                   {'set':'test', 'dataset': 'data/coco/annotations/instances_val2017.json', 'image_path':'data/coco/val2017', 'enable_transform':False}]

def CreateCocoLoaders(s3, bucket, class_dict, 
                      batch_size = 2, shuffle=True, 
                      num_workers=0, cuda = True, timeout=0, loaders = default_loaders, 
                      height=640, width=640, 
                      image_transform=None, label_transform=None, 
                      normalize=True, flipX=True, flipY=False, 
                      rotate=3, scale_min=0.75, scale_max=1.25, offset=0.1,
                      random_seed = None, numTries=3):

    pin_memory = False
    if cuda:
        pin_memory = True

    startIndex = 0
    allocated = 0.0

    for i, loader in enumerate(loaders):
        dataset = CocoDataset(s3=s3, bucket=bucket, 
                    dataset_desc=loader['dataset'], 
                    class_dictionary=class_dict,
                    image_paths=loader['image_path'],
                    height=height, width=width, 
                    image_transform=image_transform, label_transform=label_transform, 
                    normalize=normalize,  enable_transform=loader['enable_transform'], 
                    flipX=flipX, flipY=flipY, 
                    rotate=rotate, scale_min=scale_min, scale_max=scale_max, offset=offset)

        # Creating PT data samplers and loaders:
        loader['batches'] =int(dataset.__len__()/batch_size)
        loader['length'] = loader['batches']*batch_size

        loader['dataloader'] = torch.utils.data.DataLoader(dataset=dataset, 
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            drop_last=True,
                                            num_workers=num_workers, 
                                            pin_memory=pin_memory,
                                            timeout=timeout,
                                            collate_fn=collate_fn)         

    return loaders

def main(args):

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
                          imflags=args.imflags, numTries=args.numTries)

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
                                    height = args.height, width = args.width,
                                    numTries=args.numTries
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
    parser.add_argument('-credentails', type=str, default='creds.yaml', help='Credentials file.')
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
    parser.add_argument('-cuda', type=bool, default=True, help='pytorch CUDA flag') 
    parser.add_argument('-numTries', type=int, default=3, help="Read retries")

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

    main(args)

