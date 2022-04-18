import sys, os
sys.path.insert(0, os.path.abspath('.'))
from pymlutil.jsonutil import ReadDict

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
config = ReadDict('{}/build.yaml'.format(__location__))

def VersionString(config):
    version_str = '{}.{}.{}'.format(config['version']['major'], config['version']['minor'], config['version']['patch'])
    if config['version']['label'] is not None and len(config['version']['label']) > 0:
        version_str += '-{}'.format(config['version']['label'])
    return version_str

def version():
    return VersionString(config)

__version__ = version()