import sys
import argparse
import json
import os
import subprocess
import shutil
from os import fspath
import tempfile
from pymlutil.jsonutil import ReadDict
from pymlutil.s3 import s3store, Connect
from zipfile import ZipFile
from tqdm import tqdm


cityscapeurl='https://www.cityscapes-dataset.com'
citypackages=[
    {'id': 1, 'name': 'gtFine_trainvaltest.zip', 'size': '241MB'},
    {'id': 2, 'name': 'gtCoarse.zip', 'size': '1.3GB'},
    {'id': 3, 'name': 'leftImg8bit_trainvaltest.zip', 'size': '11GB'},
    {'id': 4, 'name': 'leftImg8bit_trainextra.zip', 'size': '44GB'},
    {'id': 8, 'name': 'camera_trainvaltest.zip', 'size': '2MB'},
    {'id': 9, 'name': 'camera_trainextra.zip', 'size': '8MB'},
    {'id': 10, 'name': 'vehicle_trainvaltest.zip', 'size': '2MB'},
    {'id': 11, 'name': 'vehicle_trainextra.zip', 'size': '7MB'},
    {'id': 12, 'name': 'leftImg8bit_demoVideo.zip', 'size': '6.6GB'},
    {'id': 28, 'name': 'gtBbox_cityPersons_trainval.zip', 'size': '2.2MB'},
]

def getcityscapes(s3, s3def, creds, dataset='cityscapes', cityscapeurl=cityscapeurl, citypackages=citypackages):

    tempcookie = tempfile.NamedTemporaryFile( prefix='cookie', suffix='txt')
    sysmsg = "wget --keep-session-cookies --save-cookies={} --post-data 'username={}&password={}&submit=Login' {}/login/".format(
        tempcookie.name,
        creds['cityscapes']['username'], 
        creds['cityscapes']['password'],
        cityscapeurl)
        
    #print(sysmsg)
    os.system(sysmsg)

    saved_name = '{}/{}'.format(s3def['sets']['dataset']['prefix'] , dataset)
    for citypackage in citypackages:
        with tempfile.TemporaryDirectory() as tmpdir:
            #https://www.cityscapes-dataset.com/file-handling/?packageID=1
            url = 'wget  --show-progress --load-cookies {} --content-disposition {}/file-handling/?packageID={}'.format(
                tempcookie.name,
                cityscapeurl,
                citypackage['id'])
            outpath = '{}/{}'.format(tmpdir, citypackage['name'])
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

            print('Save {} to {}/{}'.format(tmpdir, s3def['sets']['dataset']['bucket'], saved_name))
            s3.PutDir(s3def['sets']['dataset']['bucket'], tmpdir, saved_name)

    url = s3.GetUrl(s3def['sets']['dataset']['bucket'], saved_name)
    print("Complete. {} saved to {}".format(dataset, url))

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process arguments')

    parser.add_argument('-d', '--debug', action='store_true',help='Wait for debuggee attach')   
    parser.add_argument('-debug_port', type=int, default=3000, help='Debug port')
    parser.add_argument('-debug_address', type=str, default='0.0.0.0', help='Debug port')
    parser.add_argument('-dataset_path', type=str, default='./dataset', help='Local dataset path')
    parser.add_argument('-credentails', type=str, default='creds.yaml', help='Credentials file.')
    parser.add_argument('-min', action='store_true',help='Minimum test')
    parser.add_argument('-cityscapeurl', type=str, default=cityscapeurl, help='Cityscape URL')

    args = parser.parse_args()
    return args

def main(args):

    s3, _, s3def = Connect(args.credentails)
    creds = ReadDict(args.credentails)

    if args.min:
        packages = [citypackages[8], citypackages[9]]
    else:
        packages = citypackages

    getcityscapes(s3, s3def, creds=creds, dataset='cityscapes', cityscapeurl=cityscapeurl, citypackages=packages)

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