import os, re
from glob import glob

# NumPy
import numpy as np
# AstroPy
from astropy.io import fits
from astropy.time import Time
# ccdproc
from ccdproc import ImageFileCollection
# drpy
from drpy import __version__ as version_drpy
# pyregion
import pyregion

__all__ = ['makeDirectory', 'modifyHeader']


def login(instrument, width):
    """Print login message."""

    print('=' * width)
    print(f'Data Reduction Pipeline for {instrument}'.center(width))
    print(f'(based on drpy v{version_drpy})'.center(width))
    print('-' * width)


def makeDirectory(parent, child, verbose=False):
    """Make directory."""
    
    path = os.path.abspath(os.path.join(parent, child))
    
    if verbose:
        print(f'- Making {path}...')
        
    os.makedirs(path, exist_ok=True)
    
    return path


def fixHeader(file_name, verbose=False, **kwargs):
    """Fix non-standard keywords."""
    
    with fits.open(file_name, mode='update', **kwargs) as f:
        
        for hdu in f:
            
            # `FILE` -> `FILENAME`
            if ('FILE' in hdu.header) & ('FILENAME' not in hdu.header):
                
                if verbose:
                    print(f'{file_name}: rename `FILE` to `FILENAME`.')
                
                hdu.header.set(
                    'FILENAME', hdu.header['FILE'], hdu.header.comments['FILE'])
                del hdu.header['FILE']
            
            # `RADECSYS` -> `RADESYSa`
            if ('RADECSYS' in hdu.header) & ('RADESYSa' not in hdu.header):
                
                if verbose:
                    print(f'{file_name}: rename `RADECSYS` to `RADESYSa`.')
                
                hdu.header.set(
                    'RADESYSa', hdu.header['RADECSYS'], hdu.header.comments['RADECSYS'])
                del hdu.header['RADECSYS']
            
            # Set `MJD-OBS` from `DATE-OBS`
            if ('DATE-OBS' in hdu.header) & ('MJD-OBS' not in hdu.header):
                
                if verbose:
                    print(f'{file_name}: set `MJD-OBS` from `DATE-OBS`.')
                    
                hdu.header.set(
                    'MJD-OBS', Time(hdu.header['DATE-OBS'], format='fits').mjd, 
                    'Set from `DATE-OBS`.'
                )

            # Set `MJD-END` from `DATE-END`
            if ('DATE-END' in hdu.header) & ('MJD-END' not in hdu.header):
                
                if verbose:
                    print(f'{file_name}: set `MJD-END` from `DATE-END`.')
                    
                hdu.header.set(
                    'MJD-END', Time(hdu.header['DATE-END'], format='fits').mjd, 
                    'Set from `DATE-END`.'
                )

            for keyword in ['GAIN', 'RDNOISE', 'EPOCH']:
                
                # Ensure that ``keyword`` is of `float` type
                if keyword in hdu.header:

                    if isinstance(hdu.header[keyword], str):

                        try:
                            hdu.header.set(
                                keyword, float(hdu.header[keyword]), 
                                hdu.header.comments[keyword])

                            if verbose:
                                print(f'{file_name}: set `{keyword}` to `float`.')

                        except:
                            pass

            # Ensure that `EXTEND` is of `bool` type
            if 'EXTEND' in hdu.header:

                if not isinstance(hdu.header['EXTEND'], bool):
                
                    if verbose:
                        print(f'{file_name}: set `EXTEND` to `True`.')

                    hdu.header.set('EXTEND', True, hdu.header.comments['EXTEND'])

    return True


def fixKeyword(data_dir, ext, entry_file, verbose=False):
    """Fix typos in the header."""

    entries = np.loadtxt(entry_file, ndmin=2, dtype=str)

    for file_name, keyword, value in entries:
        
        if verbose:

            user_input = ''
            while user_input.lower() not in ['yes', 'y', 'no', 'n']:
                user_input = input(
                    f'{file_name}: set `{keyword}` to `{value}` (Y/N)? ')

        else:

            user_input = 'yes'

        if user_input.lower() in ['yes', 'y']:

            fits.setval(
                filename=os.path.join(data_dir, file_name), keyword=keyword, 
                value=value, ext=ext)

        else:

            continue

    return True


def getIndex(string, delimiter=','):
    """Generate a list of indices from input string."""

    idx_list = list()

    if string != '':

        for idx in string.split(delimiter):

            if idx.isdigit():
                idx_list.append(int(idx))

            else:

                assert re.match('^\d+-\d+$', idx), 'Use `-` for an index range, e.g., 1-8.'

                start, end = idx.split('-')
                idx_list.extend(list(range(int(start), (int(end) + 1))))
                
    return sorted(set(idx_list))


def getFileName(string, delimiter=',', prefix='image'):
    """Generate filenames from indices."""
    
    idx_list = getIndex(string, delimiter=delimiter)
    
    file_list = [f'{prefix}{idx:04}.fits' for idx in idx_list]
    
    return file_list


def getMask(path_to_region, shape):
    """Generate custom mask from ds9 region."""
    
    r = pyregion.open(path_to_region)
    
    mask = r.get_mask(shape=shape)
    
    return mask