# torchdatasetutil
Utilities to load and use pytorch datasets stored in Minio S3

## Credentials file
"torchdatasetutil" reads s3 and pypi credentails from a credential yaml or json file speciried with the "-credentails" parameter.  The structure of the credentials file is as follows: 

```yaml
pipy.org:
- package: torchdatasetutil
  username: __token__
  password: <token value>

s3:
- name: "store"
  type: "trainer"
  address: "<s3 url>"
  access key: "<s3 username>"
  secret key: "<s3 password>"
  tls: <true/false>
  cert verify: <true/false>
  cert path: <null/string>
  sets:
    dataset: {"bucket":"mllib","prefix":"data", "dataset_filter":"" }
    trainingset: {"bucket":"mllib","prefix":"training", "dataset_filter":"" }
    model: {"bucket":"mllib","prefix":"model", "dataset_filter":"" }
    test: {"bucket":"mllib","prefix":"test", "dataset_filter":"" }
```

## Class dictionary file
The dataset store expects a class dictionary that maps from dataset object types to training classes.  Below is an example class dictionary mapping the coco dataset classes to a set of new, simplified classes.  The class dictionary defines:
    - background: the background index
    - ignore: ignore index
    - classes: number of output classes
    - objects: array of classing mappings from the current dataset to the training dataset.  Objects include:
        - id: dataset index
        - name: dataset class name
        - category: output class
        - display: true/false if the output class is to be displayed
        - color: output class color 3 color RGB array
The dataset annotations are converted through this class dictionary for training, test, and display.
```json
{
    "background":0,
    "ignore":255,
    "classes":4,
    "objects":[
        {"id":0,    "name":"unlabeled",     "trainId":0 , "category":"void", "display":false, "color": [ 0,  0,  0]},
        {"id":1,    "name":"person",        "trainId":1 , "category":"person", "display":true, "color": [ 0,  255,  0]},
        {"id":2,    "name":"bicycle",       "trainId":2 , "category":"vehicle", "display": true, "color":[ 255,  0,  0]},
        {"id":3,    "name":"car",           "trainId":2 , "category":"vehicle", "display": true, "color":[ 255,  0,  0]},
        {"id":4,    "name":"motorcycle",    "trainId":2 , "category":"vehicle", "display": true, "color":[ 255,  0,  0]},
        {"id":5,    "name":"airplane",      "trainId":2 , "category":"vehicle", "display": true, "color":[ 255,  0,  0]},
        {"id":6,    "name":"bus",           "trainId":2 , "category":"vehicle", "display": true, "color":[ 255,  0,  0]},
        {"id":7,    "name":"train",         "trainId":2 , "category":"vehicle", "display": true, "color":[ 255,  0,  0]},
        {"id":8,    "name":"truck",         "trainId":2 , "category":"vehicle", "display": true, "color":[ 255,  0,  0]},
        {"id":9,    "name":"boat",          "trainId":2 , "category":"vehicle", "display": true, "color":[ 255,  0,  0]},
        ...
    ]
}
```

## Create Library
Create a PyPI account
- Run deploy -c to create and push this library to PyPI using your PyPI credentials
    ```cmd
    deploy -c
    ```
- Once this library is successfully created, open your [PyPI projects](https://pypi.org/manage/projects/), open your project, select "Setings" -> "Create a token".  
- Add the token to your project credentials

## Update library
```cmd
pip3 install --upgrade torchdatasetutil
```

## Load datasets
```cmd
py -m torchdatasetutil -d -getcoco
```

###
