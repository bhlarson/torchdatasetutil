import sys
import argparse
import json
import os
import subprocess
import shutil
from os import fspath
import tempfile
from pymlutil.s3 import s3store, Connect
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
    saved_name = '{}/{}'.format(s3def['sets']['dataset']['prefix'] , dataset)
    for url in tqdm(cocourl, bar_format='{desc:<8.5}{percentage:3.0f}%|{bar:50}{r_bar}'):
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = '{}/{}'.format(tmpdir,os.path.basename(url))
            if os.path.isfile(outpath):
                print('{} exists.  Skipping'.format(outpath))
            else:
                sysmsg = 'wget -O {} {} '.format(outpath, url)
                print(sysmsg)
                os.system(sysmsg)

            dest = '{}/{}'.format(tmpdir,dataset)
            with ZipFile(outpath,"r") as zip_ref:
                for file in tqdm(iterable=zip_ref.namelist(), 
                    bar_format='{desc:<8.5}{percentage:3.0f}%|{bar:50}{r_bar}', 
                    total=len(zip_ref.namelist())):
                    zip_ref.extract(member=file, path=fspath(dest))

            os.remove(outpath) # Remove zip

            print('Save {} to {}/{}'.format(dest, s3def['sets']['dataset']['bucket'], saved_name))
            s3.PutDir(s3def['sets']['dataset']['bucket'], dest, saved_name)

    url = s3.GetUrl(s3def['sets']['dataset']['bucket'], saved_name)
    print("Complete. Results saved to {}".format(url))

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process arguments')

    parser.add_argument('-d', '--debug', action='store_true',help='Wait for debuggee attach')   
    parser.add_argument('-debug_port', type=int, default=3000, help='Debug port')
    parser.add_argument('-debug_address', type=str, default='0.0.0.0', help='Debug port')
    parser.add_argument('-dataset_path', type=str, default='./dataset', help='Local dataset path')
    parser.add_argument('-credentails', type=str, default='creds.yaml', help='Credentials file.')
    parser.add_argument('-min', action='store_true',help='Minimum test')
    parser.add_argument('-cocourl', type=json.loads, default=cocourl, 
                        help='List of coco dataset URLs to load.  If none, coco 2017 datafiles are loaded from https://cocodataset.org/#download')


    args = parser.parse_args()
    return args

def main(args):

    s3, _, s3def = Connect(args.credentails)

    if args.min:
        args.cocourl = [args.cocourl[-2], args.cocourl[-1]]

    getcoco(s3, s3def, cocourl=args.cocourl, dataset='coco_test')

    print('{} {} complete'.format(__file__, __name__))

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