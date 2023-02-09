import os

import pyregion
# drpy
from drpy import __version__ as version_drpy

__all__ = ['MODULE_PATH', 'loadList', 'loadLists']

LIBRARY_PATH = os.path.join(os.path.split(__file__)[0], 'lib')


def login(grism):
    """Print login message."""
    
    print('===========================================================================')
    print(f'                Data Reduction Pipeline for 2.16-m/BFOSC {grism}')
    print(f'                         (based on drpy v{version_drpy})')
    print('---------------------------------------------------------------------------')
    
    
def loadList(list_name, list_path=''):
    """Load a file list."""

    with open(os.path.join(list_path, f'{list_name}.list'), 'r') as f:
        file_list = [line.strip() for line in f.readlines()]
    
    return file_list


def loadLists(list_names, list_path=''):
    """Load file lists."""
    
    list_dict = dict()
    for list_name in list_names:
        list_dict[list_name] = loadList(list_name, list_path=list_path)
    
    return list_dict


def getMask(region_name, shape):
    """Generate custom mask from ds9 region."""
    
    r = pyregion.open(region_name)
    
    mask = r.get_mask(shape=shape)
    
    return mask