# Torch Dataset Utilities

The python library [torchdatasetutils](https://pypi.org/project/torchdatasetutil/) produces torch [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) classes and utility functions for several imaging datasets.  This currently includes sets of images and annotations from [CVAT](https://github.com/openvinotoolkit/cvat), [COCO dataset](https://cocodataset.org/).  "torchdatasetutil" uses an s3 object storage to hold dataset data.  This enables training and test to be performed on nodes different from where the dataset is stored with application defined credentials.  It uses torch PyTorch worker threads to prefetch data for efficient GPU or CPU training and inference.

"torchdatasetutils" takes as an input the [pymlutil](https://pypi.org/project/pymlutil/).s3 object to access the object storage.

Two json or yaml dictionaries are loaded from the object storage to identify and process the dataset: the dataset description and class dictionary.  The the dataset description is unique for each type of dataset.  The class dictionary is common to all datasets and describes data transformation and data augmentation.

## Library structure
- pymlutil.s3: access to object storage
- [torchdatasetutil](https://pypi.org/project/torchdatasetutil/)
    - [gitcoco.getcoco](https://github.com/bhlarson/torchdatasetutil/blob/main/torchdatasetutil/getcoco.py#L25): function to load the [COCO dataset](https://cocodataset.org/) from internet archives into object storage
    - [cocostore](https://github.com/bhlarson/torchdatasetutil/blob/main/torchdatasetutil/cocostore.py)
        - [CocoStore](https://github.com/bhlarson/torchdatasetutil/blob/main/torchdatasetutil/cocostore.py#L17): class providing a python iterator over the coco dataset in object storage
        - [CocoDataset](https://github.com/bhlarson/torchdatasetutil/blob/main/torchdatasetutil/cocostore.py)" class implementing the pytorch [Dataset class](https://pytorch.org/docs/stable/data.html#dataset-types) for the CocoStore iterator
    - [imstore](https://github.com/bhlarson/torchdatasetutil/blob/main/torchdatasetutil/imstore.py)

See [torchdatasetutil.ipynb](https://github.com/bhlarson/torchdatasetutil/blob/main/torchdatasetutil.ipynb) for library interface and usage
