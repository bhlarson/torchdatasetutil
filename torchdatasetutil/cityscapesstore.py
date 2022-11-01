# From torchvision.datasets.cityscapes



import json
import os
import sys
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import cv2

import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.datasets.utils import extract_archive, iterable_to_str, verify_str_arg
from torchvision.datasets.vision import VisionDataset

from pymlutil.s3 import s3store, Connect
from pymlutil.jsonutil import ReadDict
from pymlutil.imutil import ImUtil, ImTransform

sys.path.insert(0, os.path.abspath('')) # Test files from current path rather than installed module
from torchdatasetutil.getcityscapes import getcityscapes

class CityscapesDataset(VisionDataset):
    """`Cityscapes <http://www.cityscapes-dataset.com/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory ``leftImg8bit``
            and ``gtFine`` or ``gtCoarse`` are located.
        split (string, optional): The image split to use, ``train``, ``test`` or ``val`` if mode="fine"
            otherwise ``train``, ``train_extra`` or ``val``
        mode (string, optional): The quality mode to use, ``fine`` or ``coarse``
        target_type (string or list, optional): Type of target to use, ``instance``, ``semantic``, ``polygon``
            or ``color``. Can also be a list to output a tuple with all specified target types.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.

    Examples:

        Get semantic segmentation target

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                                 target_type='semantic')

            img, smnt = dataset[0]

        Get multiple targets

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                                 target_type=['instance', 'color', 'polygon'])

            img, (inst, col, poly) = dataset[0]

        Validate on the "coarse" set

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='val', mode='coarse',
                                 target_type='semantic')

            img, smnt = dataset[0]
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple(
        "CityscapesClass",
        ["name", "id", "train_id", "category", "category_id", "has_instances", "ignore_in_eval", "color"],
    )

    classes = [
        CityscapesClass("unlabeled", 0, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("ego vehicle", 1, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("rectification border", 2, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("out of roi", 3, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("static", 4, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("dynamic", 5, 255, "void", 0, False, True, (111, 74, 0)),
        CityscapesClass("ground", 6, 255, "void", 0, False, True, (81, 0, 81)),
        CityscapesClass("road", 7, 0, "flat", 1, False, False, (128, 64, 128)),
        CityscapesClass("sidewalk", 8, 1, "flat", 1, False, False, (244, 35, 232)),
        CityscapesClass("parking", 9, 255, "flat", 1, False, True, (250, 170, 160)),
        CityscapesClass("rail track", 10, 255, "flat", 1, False, True, (230, 150, 140)),
        CityscapesClass("building", 11, 2, "construction", 2, False, False, (70, 70, 70)),
        CityscapesClass("wall", 12, 3, "construction", 2, False, False, (102, 102, 156)),
        CityscapesClass("fence", 13, 4, "construction", 2, False, False, (190, 153, 153)),
        CityscapesClass("guard rail", 14, 255, "construction", 2, False, True, (180, 165, 180)),
        CityscapesClass("bridge", 15, 255, "construction", 2, False, True, (150, 100, 100)),
        CityscapesClass("tunnel", 16, 255, "construction", 2, False, True, (150, 120, 90)),
        CityscapesClass("pole", 17, 5, "object", 3, False, False, (153, 153, 153)),
        CityscapesClass("polegroup", 18, 255, "object", 3, False, True, (153, 153, 153)),
        CityscapesClass("traffic light", 19, 6, "object", 3, False, False, (250, 170, 30)),
        CityscapesClass("traffic sign", 20, 7, "object", 3, False, False, (220, 220, 0)),
        CityscapesClass("vegetation", 21, 8, "nature", 4, False, False, (107, 142, 35)),
        CityscapesClass("terrain", 22, 9, "nature", 4, False, False, (152, 251, 152)),
        CityscapesClass("sky", 23, 10, "sky", 5, False, False, (70, 130, 180)),
        CityscapesClass("person", 24, 11, "human", 6, True, False, (220, 20, 60)),
        CityscapesClass("rider", 25, 12, "human", 6, True, False, (255, 0, 0)),
        CityscapesClass("car", 26, 13, "vehicle", 7, True, False, (0, 0, 142)),
        CityscapesClass("truck", 27, 14, "vehicle", 7, True, False, (0, 0, 70)),
        CityscapesClass("bus", 28, 15, "vehicle", 7, True, False, (0, 60, 100)),
        CityscapesClass("caravan", 29, 255, "vehicle", 7, True, True, (0, 0, 90)),
        CityscapesClass("trailer", 30, 255, "vehicle", 7, True, True, (0, 0, 110)),
        CityscapesClass("train", 31, 16, "vehicle", 7, True, False, (0, 80, 100)),
        CityscapesClass("motorcycle", 32, 17, "vehicle", 7, True, False, (0, 0, 230)),
        CityscapesClass("bicycle", 33, 18, "vehicle", 7, True, False, (119, 11, 32)),
        #CityscapesClass("license plate", -1, -1, "vehicle", 7, False, True, (0, 0, 142)),
    ]

    def __init__(
        self,
        root: str,
        class_dictionary: dict,
        split: str = "train",
        mode: str = "fine",
        target_type: Union[List[str], str] = "instance",
        transforms: Optional[Callable] = None,
        imflags = cv2.IMREAD_COLOR,
        anflags = cv2.IMREAD_GRAYSCALE,
        flipX=True, 
        flipY=False, 
        rotate=3, 
        scale_min=0.75, 
        scale_max=1.25, 
        offset=0.1,
    ) -> None:
        super().__init__(root)
        self.class_dictionary = class_dictionary
        self.mode = "gtFine" if mode == "fine" else "gtCoarse"
        self.images_dir = os.path.join(self.root, "leftImg8bit", split)
        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.target_type = target_type
        self.split = split
        self.images = []
        self.targets = []
        self.imflags = imflags
        self.anflags = anflags
        self.transforms = transforms

        verify_str_arg(mode, "mode", ("fine", "coarse"))
        if mode == "fine":
            valid_modes = ("train", "test", "val")
        else:
            valid_modes = ("train", "train_extra", "val")
        msg = "Unknown value '{}' for argument split if mode is '{}'. Valid values are {{{}}}."
        msg = msg.format(split, mode, iterable_to_str(valid_modes))
        verify_str_arg(split, "split", valid_modes, msg)

        if not isinstance(target_type, list):
            self.target_type = [target_type]
        [
            verify_str_arg(value, "target_type", ("instance", "semantic", "polygon", "color"))
            for value in self.target_type
        ]

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):

            if split == "train_extra":
                image_dir_zip = os.path.join(self.root, "leftImg8bit_trainextra.zip")
            else:
                image_dir_zip = os.path.join(self.root, "leftImg8bit_trainvaltest.zip")

            if self.mode == "gtFine":
                target_dir_zip = os.path.join(self.root, f"{self.mode}_trainvaltest.zip")
            elif self.mode == "gtCoarse":
                target_dir_zip = os.path.join(self.root, f"{self.mode}.zip")

            if os.path.isfile(image_dir_zip) and os.path.isfile(target_dir_zip):
                extract_archive(from_path=image_dir_zip, to_path=self.root)
                extract_archive(from_path=target_dir_zip, to_path=self.root)
            else:
                raise RuntimeError(
                    "Dataset not found or incomplete. Please make sure all required folders for the"
                    ' specified "split" and "mode" are inside the "root" directory'
                )

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)
            for file_name in os.listdir(img_dir):
                target_types = []
                for t in self.target_type:
                    target_name = "{}_{}".format(
                        file_name.split("_leftImg8bit")[0], self._get_target_suffix(self.mode, t)
                    )
                    target_types.append(os.path.join(target_dir, target_name))

                self.images.append(os.path.join(img_dir, file_name))
                self.targets.append(target_types)

    def classes(self, anns):
        class_vector = np.zeros(self.class_dictionary ['classes'], dtype=np.float32)

        for ann in anns:
            obj = self.catToObj[ann['category_id']]
            if obj['trainId'] < self.class_dictionary ["classes"]:
                class_vector[obj['trainId']] = 1.0

        return class_vector

    def ConvertLabels(self, ann):
        trainAnn = np.zeros_like(ann)
        for obj in self.class_dictionary ['objects']: # Load RGB colors as BGR
            if not (obj['id'] == obj['trainId'] and obj['id'] == 0):
                trainAnn[ann==obj['id']] = obj['trainId']
        return trainAnn

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        image = cv2.imread(self.images[index], self.imflags)

        targets: Any = []
        for i, t in enumerate(self.target_type):
            if t == "polygon":
                target = self._load_json(self.targets[index][i])
            else:
                target = cv2.imread(self.targets[index][i], self.anflags)

            if t == "semantic":
                target = self.ConvertLabels(target)

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            image, target, mean, std = self.transforms(image, target)
        else:
            mean = None
            std = None

        image = torch.from_numpy(image).permute(2, 0, 1)
        target = torch.from_numpy(target)

        assert(image.shape[-1] == self.transforms.width)
        assert(image.shape[-2] == self.transforms.height)

        return image, target, mean, std

    def __len__(self) -> int:
        return len(self.images)

    def extra_repr(self) -> str:
        lines = ["Split: {split}", "Mode: {mode}", "Type: {target_type}"]
        return "\n".join(lines).format(**self.__dict__)

    def _load_json(self, path: str) -> Dict[str, Any]:
        with open(path) as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode: str, target_type: str) -> str:
        if target_type == "instance":
            return f"{mode}_instanceIds.png"
        elif target_type == "semantic":
            return f"{mode}_labelIds.png"
        elif target_type == "color":
            return f"{mode}_color.png"
        else:
            return f"{mode}_polygons.json"

def CreateCityscapesLoaders(s3, s3def, src, dest, class_dictionary, bucket = None, width=256, height=256, batch_size = 2, shuffle=True, 
                      num_workers=0, cuda = True, timeout=0, loaders = None, 
                      transforms=True, 
                      normalize=True, 
                      random_seed = None, numTries=3,
                      flipX=True, 
                      flipY=False, 
                      rotate=3, 
                      scale_min=0.75, 
                      scale_max=1.25, 
                      offset=0.1,
                      astype='float32',
                      borderType=cv2.BORDER_CONSTANT,
                      borderValue=0,
                      train_sampler_weights=None):
    if not bucket:
        bucket = s3def['sets']['dataset']['bucket']

    if not os.path.exists(dest):
        s3.GetDir(bucket, src, dest)

    dest = os.path.join(dest, '') #Ensure there is a trailing slash

    pin_memory = False
    # if cuda:
    #     pin_memory = True

    # Load dataset
    if loaders is None:

        default_loaders = [{'set':'train', 'dataset': dest, 'enable_transform':True, 'mode':'fine', 'target_type':['semantic'], 'class_dictionary':class_dictionary } ,
                        {'set':'val', 'dataset': dest, 'enable_transform':False, 'mode':'fine', 'target_type':['semantic'], 'class_dictionary':class_dictionary}]

        loaders = default_loaders

    startIndex = 0
    allocated = 0.0

    for i, loader in enumerate(loaders):

        if train_sampler_weights is not None and loader['set'] == 'train':
            sampler=WeightedRandomSampler(weights=train_sampler_weights, num_samples=len(train_sampler_weights), replacement=True)
            shuffle=False
            print('Weighted Random Sampler is initiated!')
        else:
            sampler=None


        if transforms and loader['set'] == 'train':
            transform = ImTransform(height=height, width=width, 
                        normalize=normalize, 
                        enable_transform=loader['enable_transform'],
                        flipX=flipX, 
                        flipY=flipY, 
                        rotate=rotate, 
                        scale_min=scale_min, 
                        scale_max=scale_max, 
                        offset=offset, 
                        astype=astype,
                        borderType=borderType,
                        borderValue=borderValue,
                        )
        elif transforms:
            transform = ImTransform(height=height, width=width, 
                        normalize=normalize, 
                        astype=astype,
                        enable_transform=False, 
                        borderType=borderType,
                        borderValue=borderValue,
                        )
            


        dataset = CityscapesDataset(root=loader['dataset'],
                                    class_dictionary=loader['class_dictionary'],
                                    split=loader['set'], 
                                    mode=loader['mode'], 
                                    target_type=loader['target_type'], 
                                    transforms=transform
                                   )
        


        loader['dataloader'] = torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                num_workers=num_workers,
                                                pin_memory=pin_memory,
                                                sampler=sampler)

        # Creating PT data samplers and loaders:
        loader['batches'] =len(loader['dataloader'])
        loader['length'] = loader['batches']*batch_size
        loader['width']=width
        loader['height']=height
        loader['in_channels']=loader['class_dictionary']['input_channels']
        loader['num_classes']=loader['class_dictionary']['classes']
        loader['classes']=loader['class_dictionary']['objects']
        loader['dataset_dfn'] = {}

    return loaders