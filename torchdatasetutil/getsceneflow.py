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


urls=[{"name":"sintel", "url": "http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip"},
      {"name":"FlyingChairs", "url": "https://lmb.informatik.uni-freiburg.de/data/FlyingChairs/FlyingChairs.zip"},
      {"name":"FlyingChairs2", "url": "https://lmb.informatik.uni-freiburg.de/data/FlyingChairs2.zip"},
      {"name":"kitti", "url": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip"},
      {"name":"kitti", "url": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_stereo_flow.zip"}  ]

def getsceneflow(s3, s3def, urls=urls, dataset='sceneflow'):

    with tempfile.TemporaryDirectory() as tmpdir:

        for data in urls:
            outpath = '{}/{}'.format(tmpdir,os.path.basename(data['url']))
            if os.path.isfile(outpath):
                print('{} exists.  Skipping'.format(outpath))
            else:
                sysmsg = 'wget -O {} {} '.format(outpath, data['url'])
                print(sysmsg)
                os.system(sysmsg)

            dest = '{}/{}'.format(tmpdir,data['name'])
            with ZipFile(outpath,"r") as zip_ref:
                for file in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist()), desc="Extract zip"):
                    zip_ref.extract(member=file, path=fspath(dest))

            os.remove(outpath) # Remove zip file once extracted

            saved_name = '{}/{}'.format(s3def['sets']['dataset']['prefix'] , dataset)
            print('Save {} to {}/{}'.format(tmpdir, s3def['sets']['dataset']['bucket'], saved_name))
            if s3.PutDir(s3def['sets']['dataset']['bucket'], tmpdir, saved_name):
                shutil.rmtree(dest, ignore_errors=True)

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
    parser.add_argument('-urls', type=json.loads, default=urls, 
                        help='List of coco dataset URLs to load.  If none, the public urls list will be loaded')


    args = parser.parse_args()
    return args

def main(args):

    s3, _, s3def = Connect(args.credentails)

    if args.min:
        args.cocourl = [args.cocourl[-2], args.cocourl[-1]]

    getsceneflow(s3, s3def, urls=urls, dataset='sceneflow')

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