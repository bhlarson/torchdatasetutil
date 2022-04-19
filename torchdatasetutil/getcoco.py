import sys
import argparse
import json
import os
import subprocess
import shutil
from os import fspath
import tempfile
from pymlutil.s3 import s3store
from zipfile import ZipFile
from tqdm import tqdm


cocourl=["http://images.cocodataset.org/zips/train2017.zip",
         "http://images.cocodataset.org/zips/val2017.zip",
         "http://images.cocodataset.org/zips/test2017.zip",
         "http://images.cocodataset.org/zips/unlabeled2017.zip",
         "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
         "http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip",
         "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip",
         "http://images.cocodataset.org/annotations/image_info_test2017.zip",
         "http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip",
        ]

def getcoco(s3, s3def, cocourl=cocourl, dataset='coco'):

    with tempfile.TemporaryDirectory() as tmpdir:

        for url in cocourl:
            outpath = '{}/{}'.format(tmpdir,os.path.basename(url))
            if os.path.isfile(outpath):
                print('{} exists.  Skipping'.format(outpath))
            else:
                sysmsg = 'wget -O {} {} '.format(outpath, url)
                print(sysmsg)
                os.system(sysmsg)

            dest = '{}/{}'.format(tmpdir,dataset)
            with ZipFile(outpath,"r") as zip_ref:
                for file in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist())):
                    zip_ref.extract(member=file, path=fspath(dest))

            os.remove(outpath) # Remove zip file once extracted

        saved_name = '{}/{}'.format(s3def['sets']['dataset']['prefix'] , dataset)
        print('Save {} to {}/{}'.format(tmpdir, s3def['sets']['dataset']['bucket'], saved_name))
        if s3.PutDir(s3def['sets']['dataset']['bucket'], tmpdir, saved_name):
            shutil.rmtree(tmpdir, ignore_errors=True)

        url = s3.GetUrl(s3def['sets']['dataset']['bucket'], saved_name)
        print("Complete. Results saved to {}".format(url))
