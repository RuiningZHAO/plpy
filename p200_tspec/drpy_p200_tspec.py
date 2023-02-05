import os
from glob import glob

# NumPy
import numpy as np
# AstroPy
from astropy.stats import mad_std
from astropy.table import Table
# ccdproc
from ccdproc import subtract_bias
# drpy
from drpy.batch import CCDDataList
from drpy.plotting import plot2d

    
def genFileTable(ccddatalist, keywords, verbose):
    """Generate file table."""
    
    if 'OBSTYPE' not in keywords:
        keywords.append('OBSTYPE')

    file_table = list()
    for ccd in ccddatalist:
        file_table.append([ccd.header[keyword] for keyword in keywords])

    file_table = Table(
        rows=file_table, names=[keyword for keyword in keywords])

    if verbose: file_table.pprint_all()
    
    return file_table


def loadData(data_dir, hdu, keywords, verbose):
    """Load and organize data"""

    file_list = sorted(glob(os.path.join(data_dir, '*.fits')))

    ccddatalist = CCDDataList.read(file_list=file_list, hdu=hdu)

    file_table = genFileTable(ccddatalist, keywords, verbose)

    mask_dark = file_table['OBSTYPE'] == 'dark'
    mask_flat = file_table['OBSTYPE'] == 'flat'
    mask_targ = file_table['OBSTYPE'] == 'object'

    return ccddatalist[mask_dark], ccddatalist[mask_flat], ccddatalist[mask_targ]


def getMasterFlat(flat_list, dark_list):
    """Master flat."""

    flat_with_lamp_on = flat_list.combine(
        method='average', sigma_clip=True, sigma_clip_low_thresh=3, 
        sigma_clip_high_thresh=3, sigma_clip_func=np.ma.median, 
        sigma_clip_dev_func=mad_std, mem_limit=500e6)

    flat_with_lamp_off = dark_list.combine(
        method='average', sigma_clip=True, sigma_clip_low_thresh=3, 
        sigma_clip_high_thresh=3, sigma_clip_func=np.ma.median, 
        sigma_clip_dev_func=mad_std, mem_limit=500e6)

    master_flat = subtract_bias(ccd=flat_with_lamp_on, master=flat_with_lamp_off)
    
    return master_flat


def main(data_dir, hdu, keywords, verbose):
    
    dark_list, flat_list, _ = loadData(
        data_dir=data_dir, hdu=hdu, keywords=keywords, verbose=verbose)

    # Assume that all the flats have the same configuration
    fsample, coadds, exptime = (
        flat_list[0].header['FSAMPLE'], flat_list[0].header['COADDS'], 
        flat_list[0].header['EXPTIME'])

    n_dark = len(dark_list)
    mask = np.zeros(n_dark, dtype=bool)
    for i in range(n_dark):
        if ((dark_list[i].header['FSAMPLE'] == fsample) & 
            (dark_list[i].header['COADDS'] == coadds) & 
            (dark_list[i].header['EXPTIME'] == exptime)):
            mask[i] = True

    # Mask flats with lamp off
    dark_list = dark_list[mask]

    for fccd, dccd in zip(flat_list, dark_list):
        print(fccd.header['FILENAME'], dccd.header['FILENAME'])

#     master_dark = dark_list.combine(
#         method='average', sigma_clip=True, sigma_clip_low_thresh=3, 
#         sigma_clip_high_thresh=3, sigma_clip_func=np.ma.median, 
#         sigma_clip_dev_func=mad_std, mem_limit=500e6)

#     master_dark.write('master_dark.fits', overwrite=True)

    master_flat = getMasterFlat(flat_list=flat_list, dark_list=dark_list)


    CCDData.writer.help('fits')

    master_flat.write('master_flat.fits', as_image_hdu=True, overwrite=True)

if __name__ == '__main__':

    data_dir = '/data3/zrn/data/Hale/TripleSpec/20221210'

    hdu = 0
    
    keywords = [
        'FILENAME', 'TIME', 'OBSTYPE', 'OBJECT', 'EXPTIME', 'COADDS', 'FSAMPLE', 
        'CATNAME', 'RA', 'DEC'
    ]
    
    verbose = True

    main(data_dir=data_dir, hdu=hdu, keywords=keywords, verbose=verbose)