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

## Class Dictionary

## COCO Dataset
To load coco dataset you must have a credentials yaml file identifying the final s3 location and credentials for the dataset with the following keys:

```yaml
s3:
- name: store
  type: trainer
  address: <address>:<port>
  access key: <access key>
  secret key: <secret key>
  tls: false
  cert verify: false
  cert path: null
  sets:
    dataset: {"bucket":"imgml","prefix":"data", "dataset_filter":"" }
    trainingset: {"bucket":"imgml","prefix":"training", "dataset_filter":"" }
    model: {"bucket":"imgml","prefix":"model", "dataset_filter":"" }
    test: {"bucket":"imgml","prefix":"test", "dataset_filter":"" }
```

Call torchdatasetutil.getcoco to retrieve the COCO dataset and stage it into object storage
```cmd
python3 -m torchdatasetutil.getcoco
```

To train with the coco dataset, first create dataset loaders
```python
from torchdatasetutil.cocostore import CreateCocoLoaders

# Create dataset loaders
dataset_bucket = s3def['sets']['dataset']['bucket']
if args.dataset=='coco':
    class_dictionary = s3.GetDict(s3def['sets']['dataset']['bucket'],args.coco_class_dict)
    loaders = CreateCocoLoaders(s3, dataset_bucket, 
        class_dict=args.coco_class_dict, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cuda = args.cuda,
        height = args.height,
        width = args.width,
    )

# Identify training and test loaders
trainloader = next(filter(lambda d: d.get('set') == 'train', loaders), None)
testloader = next(filter(lambda d: d.get('set') == 'test' or d.get('set') == 'val', loaders), None)

# Iterate through the dataset
for i, data in tqdm(enumerate(trainloader['dataloader']), 
                    bar_format='{desc:<8.5}{percentage:3.0f}%|{bar:50}{r_bar}', 
                    total=trainloader['batches'], desc="Train batches", disable=args.job):

    # Extract dataset data
    inputs, labels, mean, stdev = data

    # Remaining steps

```

# Cityscapes Dataset
To download cityscapes, your cityscapes credentials must be included in you credentials yaml file with the following structure

```yaml
cityscapes:
  username: <username>
  password: <password>
```
Call torchdatasetutil.getcityscapes to retrieve the cityscapes dataset and stage it into object storage
```cmd
python3 -m torchdatasetutil.getcityscapes
```
```python
if args.dataset=='cityscapes':
    class_dictionary = s3.GetDict(s3def['sets']['dataset']['bucket'],args.cityscapes_class_dict)
    loaders = CreateCityscapesLoaders(s3, s3def, 
        src = args.cityscapes_data,
        dest = args.dataset_path+'/cityscapes',
        class_dictionary = class_dictionary,
        batch_size = args.batch_size, 
        num_workers=args.num_workers,
        height=args.height,
        width=args.width, 
    )
```

# Imagenet:
1. Data from kaggle:
    # Data from https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data?select=LOC_sample_submission.csv
1. Extract and move validation folder data:
    https://discuss.pytorch.org/t/issues-with-dataloader-for-imagenet-should-i-use-datasets-imagefolder-or-datasets-imagenet/115742/7
1. Zip ILSVRC/Data/CLS-LOC/ to ILSVRC2012_devkit_t12.tar.gz
    ```cmd
    tar -czvf ILSVRC2012_devkit_t12.tar.gz ILSVRC/Data/CLS-LOC
    ```