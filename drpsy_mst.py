"""
mini-Sitian pipeline (version 2023)
"""

import os
from glob import glob

# NumPy
import numpy as np
# AstroPy
from astropy.stats import mad_std
# drpsy
from drpsy.batch import CCDDataList
from drpsy.plotting import plot2d
from drpsy.utils import imstatistics


def pipeline(data_dir, save_dir, telescope, target, row_range, col_range, verbose, 
             show, save):
    """
    """

    # Path
    plot_path = os.path.join(save_dir, 'figs')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    fits_path = os.path.join(save_dir, 'fits')
    if not os.path.exists(fits_path):
        os.makedirs(fits_path)
    
    # Load data
    biaslist = CCDDataList.read(
        file_list=sorted(glob(os.path.join(data_dir, f'{telescope}*bias*.fit'))), 
        hdu=0, unit='adu')
    flatlist = CCDDataList.read(
        file_list=sorted(glob(os.path.join(data_dir, f'{telescope}*flat*.fit'))), 
        hdu=0, unit='adu')
    targlist = CCDDataList.read(
        file_list=sorted(glob(os.path.join(data_dir, f'{telescope}*{target}*.fit'))), 
        hdu=0, unit='adu')
    
    # Trim
    biaslist_trimmed = biaslist.trim(row_range=row_range, col_range=col_range)
    flatlist_trimmed = flatlist.trim(row_range=row_range, col_range=col_range)
    targlist_trimmed = targlist.trim(row_range=row_range, col_range=col_range)
    
    # Combine bias
    master_bias = biaslist_trimmed.combine(
        method='average', sigma_clip=True, sigma_clip_low_thresh=3, 
        sigma_clip_high_thresh=3, sigma_clip_func=np.ma.median, 
        sigma_clip_dev_func=mad_std, mem_limit=500e6)
    
    # Plot master bias
    plot2d(
        master_bias.data, title=f'{telescope} master bias', show=show, save=save, 
        path=plot_path)
    
    # Write master bias to file
    master_bias.write(
        os.path.join(fits_path, f'{telescope}_master_bias.fits'), overwrite=True)

    # Check statistics 
    biaslist_trimmed.statistics(verbose=verbose)
    imstatistics(master_bias, verbose=verbose)
    
    # Subtract bias
    flatlist_bias_subtracted = flatlist_trimmed - master_bias
    
    # Combine flat
    master_flat = flatlist_bias_subtracted.combine(
        method='average', sigma_clip=True, sigma_clip_low_thresh=3, 
        sigma_clip_high_thresh=3, sigma_clip_func=np.ma.median, 
        sigma_clip_dev_func=mad_std, mem_limit=500e6)
    
    # Plot master flat
    plot2d(
        master_flat.data, title=f'{telescope} master flat', show=show, save=save, 
        path=plot_path)
    
    # Write master flat to file
    master_flat.write(
        os.path.join(fits_path, f'{telescope}_master_flat.fits'), overwrite=True)
    
    # Check statistics
    flatlist_trimmed.statistics(verbose=verbose)
    flatlist_bias_subtracted.statistics(verbose=verbose)
    imstatistics(master_flat, verbose=verbose)

    # Subtract bias
    targlist_bias_subtracted = targlist_trimmed - master_bias
    
    # Flat-fielding
    targ_corrected = targlist_bias_subtracted / master_flat
    
    # Write images to file
    for ccd in targ_corrected:
        fits_name = '{}_corrected.fits'.format(ccd.header['FILENAME'][:-4])
        ccd.write(
            os.path.join(fits_path, fits_name), overwrite=True)
    
if __name__ == '__main__':
    
    data_dir = '/data3/zrn/data/mst/20230109'
    save_dir = '/data3/zrn/workspace/data_reduction/20230109'
    telescopes = ['mst2', 'mst3']
    target = 'C2022E3'
    row_range = (2000, 4388)
    col_range=(3594, 5982)
    verbose = True
    show = False
    save = True
    
    for telescope in telescopes:
        pipeline(
            data_dir=os.path.join(data_dir, telescope), save_dir=save_dir, 
            telescope=telescope, target=target, row_range=row_range, 
            col_range=col_range, verbose=verbose, show=show, save=save)