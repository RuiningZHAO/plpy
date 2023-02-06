"""
mini-Sitian pipeline (version 2023)
"""

import os
from glob import glob
from copy import deepcopy

# NumPy
import numpy as np
# AstroPy
from astropy.io import fits
from astropy.time import Time
from astropy.stats import mad_std
from astropy.nddata import CCDData
# ccdproc
from ccdproc import ImageFileCollection
# drpy
from drpy.batch import CCDDataList
from drpy.plotting import plot2d
from drpy.utils import imstatistics

from .utils import makeDirectory, modifyHeader


def pipeline(save_dir, data_dir, keywords, telescope, target, gain, rdnoise, row_range, 
             col_range, steps, mem_limit, verbose, show, save):
    """
    """

    # Make subdirectories
    fig_path = makeDirectory(parent=save_dir, child=f'{target}/fig')
    pro_path = makeDirectory(parent=save_dir, child=f'{target}/pro')
    cal_path = makeDirectory(parent=save_dir, child=f'{target}/cal')
    
    if 'header' in steps:
        
        ifc = ImageFileCollection(
            location=data_dir, keywords=keywords, find_fits_by_reading=False, 
            filenames=None, glob_include=f'{telescope}*.fit', glob_exclude=None, 
            ext=0)
        for file_name in ifc.files_filtered(include_path=True):
            modifyHeader(file_name)
    
    # Construct image file collection
    ifc = ImageFileCollection(
        location=data_dir, keywords=keywords, find_fits_by_reading=False, 
        filenames=None, glob_include=f'{telescope}*.fit', glob_exclude=None, 
        ext=0)
    
    # Bias
    if 'bias' in steps:
        
        # Filter
        ifc_bias = ifc.filter(
            regex_match=True, file=f'{telescope}-bias-\d{{8}}-\d{{4}}.fit')
        if verbose:
            ifc_bias.summary.pprint_all()
        
        # Load data
        bias_list = CCDDataList.read(
            file_list=ifc_bias.files_filtered(include_path=True), hdu=0)
        
        # Trim
        bias_list_trimmed = bias_list.trim(row_range=row_range, col_range=col_range)
        
        # Combine bias
        master_bias = bias_list_trimmed.combine(
            method='average', sigma_clip=True, sigma_clip_low_thresh=3, 
            sigma_clip_high_thresh=3, sigma_clip_func=np.ma.median, 
            sigma_clip_dev_func=mad_std, mem_limit=mem_limit, 
            output_file=os.path.join(cal_path, f'{telescope}_master_bias.fits'), 
            overwrite_output=True)

        # Check statistics 
        bias_list_trimmed.statistics(verbose=verbose)
        imstatistics(master_bias, verbose=verbose)

        # Plot master bias
        plot2d(
            master_bias.data, title=f'{telescope} master bias', show=show, save=save, 
            path=fig_path)
        
        del bias_list, bias_list_trimmed, master_bias
    
    # Flat
    if 'flat' in steps:
        
        # Filter
        ifc_flat = ifc.filter(
            regex_match=True, file=f'{telescope}-flat-\d{{8}}-\d{{4}}.fit')
        if verbose:
            ifc_flat.summary.pprint_all()
        
        # Load data
        flat_list = CCDDataList.read(
            file_list=ifc_flat.files_filtered(include_path=True), hdu=0)
        
        # Trim
        flat_list_trimmed = flat_list.trim(row_range=row_range, col_range=col_range)
        
        # Subtract bias
        master_bias = CCDData.read(
            os.path.join(cal_path, f'{telescope}_master_bias.fits'))
        flat_list_bias_subtracted = flat_list_trimmed - master_bias
        
        # Combine flat
        scaling_func = lambda ccd: 1 / np.ma.average(ccd)
        master_flat = flat_list_bias_subtracted.combine(
            method='average', scale=scaling_func, sigma_clip=True, 
            sigma_clip_low_thresh=3, sigma_clip_high_thresh=3, 
            sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std, 
            mem_limit=mem_limit, 
            output_file=os.path.join(cal_path, f'{telescope}_master_flat.fits'), 
            overwrite_output=True)

        # Check statistics
        flat_list_trimmed.statistics(verbose=verbose)
        flat_list_bias_subtracted.statistics(verbose=verbose)
        imstatistics(master_flat, verbose=verbose)
        
        # Plot master flat
        plot2d(
            master_flat.data, title=f'{telescope} master flat', show=show, save=save, 
            path=fig_path)
        
        del (flat_list, flat_list_trimmed, master_bias, flat_list_bias_subtracted, 
             master_flat)
        
    # Target
    if 'target' in steps:

        # Filter
        ifc_targ = ifc.filter(
            regex_match=True, file=f'{telescope}-{target}-\d{{8}}-\d{{4}}.fit')
        if verbose:
            ifc_targ.summary.pprint_all()

        # Load data
        targ_list = CCDDataList.read(
            file_list=ifc_targ.files_filtered(include_path=True), hdu=0)

        # Trim
        targ_list_trimmed = targ_list.trim(row_range=row_range, col_range=col_range)

        # Subtract bias
        master_bias = CCDData.read(
            os.path.join(cal_path, f'{telescope}_master_bias.fits'))
        targ_list_bias_subtracted = targ_list_trimmed - master_bias

        # Generate real uncertainty
        targ_list_bias_subtracted_with_deviation = (
            targ_list_bias_subtracted.create_deviation(
                gain=gain, readnoise=rdnoise, disregard_nan=True)
        )
        
        # Flat-fielding
        master_flat = CCDData.read(
            os.path.join(cal_path, f'{telescope}_master_flat.fits'))
        targ_list_flat_fielded = targ_list_bias_subtracted_with_deviation / master_flat
        
        # Remove cosmic-ray
        targ_list_corrected = targ_list_flat_fielded.cosmicray(
            method='Laplacian', use_mask=False, gain=gain, readnoise=rdnoise, 
            sigclip=4.5, sigfrac=0.3, objlim=1, niter=5, verbose=verbose)

        # Write images to file
        for ccd in targ_list_corrected:    
            fits_name = '{}_corrected.fits'.format(ccd.header['FILENAME'][:-4])
            ccd.write(
                os.path.join(pro_path, fits_name), overwrite=True)
    
    
if __name__ == '__main__':
    
    save_dir = '/data3/zrn/workspace/data_reduction/20230124'
    data_dir = '/data3/zrn/data/mst/20230124'
    keywords = [
        'DATE-OBJ', 'OBJECT', 'OBJCTRA', 'OBJCTDEC', 'AIRMASS', 'EXPTIME', 'EGAIN']
    telescopes = ['mst2']
    target = 'C2022E3'
    gain = 0.24665763974189758
    rdnoise = 6.4
    row_range = (2250, 4250)
    col_range = (3750, 5750)
    steps = ['header', 'bias', 'flat', 'target']
    mem_limit = 16e9
    verbose = True
    show = False
    save = True
    
    for telescope in telescopes:
        pipeline(
            save_dir=save_dir, data_dir=os.path.join(data_dir, telescope), 
            keywords=keywords, telescope=telescope, target=target, gain=gain, 
            rdnoise=rdnoise, row_range=row_range, col_range=col_range, steps=steps, 
            mem_limit=mem_limit, verbose=verbose, show=show, save=save)
