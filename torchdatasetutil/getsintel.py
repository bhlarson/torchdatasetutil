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


urls=["http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip",
      "https://lmb.informatik.uni-freiburg.de/data/FlyingChairs/FlyingChairs.zip",
      "https://lmb.informatik.uni-freiburg.de/data/FlyingChairs2.zip",
      "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip",
      "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_stereo_flow.zip"  ]

def getsintel(s3, s3def, urls=urls, dataset='sintel'):

    with tempfile.TemporaryDirectory() as tmpdir:

        for url in urls:
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
