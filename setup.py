import shutil
import yaml
from setuptools import setup, find_packages

def ReadDictYaml(filepath):
    yamldict = {}
    try:
        with open(filepath) as yaml_file:
            yamldict = yaml.safe_load(yaml_file)
        if not yamldict:
            print('Failed to load {}'.format(filepath))
    except ValueError:
        print('Failed to load {} error {}'.format(filepath, ValueError))
    return yamldict

buildconfig = 'config/build.yaml'
config = ReadDictYaml(buildconfig)
print(config)

DESCRIPTION = 'Torch Dataset Utilities'

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Setting up
setup(
    # the name must match the folder name 'pymlutil'
    name="pymlutil", 
    version=config['version'],
    author="Brad Larson",
    author_email="<bhlarson@gmail.com>",
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=['pyyaml', 'prettytable', 'minio', 'numpy', 'opencv-python', 'torch', 'scikit-learn'], # add any additional packages that 
    url = 'https://github.com/bhlarson/pymlutil',
    keywords=['python', 'Machine Learning', 'Utilities'],
    classifiers= [
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
    ]
)