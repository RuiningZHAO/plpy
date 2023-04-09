import os

# AstroPy
from astropy.io import fits
from astropy.time import Time


def makeDirectory(parent, child, verbose=False):
    """Make directory."""
    
    path = os.path.abspath(os.path.join(parent, child))
    
    if verbose:
        print(f'  - Making {path}...')
        
    os.makedirs(path, exist_ok=True)
    
    return path


def modifyHeader(file_name, verbose=False):
    """Modify fits header."""
    
    with fits.open(file_name, 'update') as f:
        
        for hdu in f:
            
            # `RADECSYS` -> `RADESYSa`
            if ('RADECSYS' in hdu.header) & ('RADESYSa' not in hdu.header):
                
                if verbose:
                    print(f'{file_name}: rename `RADECSYS` to `RADESYSa`.')
                
                hdu.header['RADESYSa'] = (
                    hdu.header['RADECSYS'], hdu.header.comments['RADECSYS'])
                del hdu.header['RADECSYS']
            
            # Set `MJD-OBS` from `DATE-OBS`
            if ('DATE-OBS' in hdu.header) & ('MJD-OBS' not in hdu.header):
                
                if verbose:
                    print(f'{file_name}: set `MJD-OBS` from `DATE-OBS`.')
                    
                hdu.header['MJD-OBS'] = (
                    Time(hdu.header['DATE-OBS'], format='fits').mjd, 
                    'Set from DATE-OBS'
                )
            
            # Set `GAIN` to `float`
            if 'GAIN' in hdu.header:
                
                if isinstance(hdu.header['GAIN'], str):
                    
                    try:
                        hdu.header['GAIN'] = (
                            float(hdu.header['GAIN']), hdu.header.comments['GAIN']
                        )
                        if verbose:
                            print(f'{file_name}: set `GAIN` to `float`.')
                        
                    except:
                        pass
            
            # Set `RDNOISE` to `float`
            if 'RDNOISE' in hdu.header:
                
                if isinstance(hdu.header['RDNOISE'], str):
                    
                    try:
                        hdu.header['RDNOISE'] = (
                            float(hdu.header['RDNOISE']), hdu.header.comments['RDNOISE']
                        )
                        if verbose:
                            print(f'{file_name}: set `RDNOISE` to `float`.')
                        
                    except:
                        pass