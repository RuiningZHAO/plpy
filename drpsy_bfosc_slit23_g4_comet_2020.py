"""
2.16-m/BFOSC pipeline (version 2020)
"""

import sys
sys.path.append('/data3/zrn/workspace/github')

import os
from glob import glob

# NumPy
import numpy as np
from astropy.stats import mad_std
# AstroPy
from astropy.table import Table

from drspy import CCDDataList, concatenate, transform
from drspy.photometry.utils import getFWHM
from drspy.longslit import (response, illumination, align, fitcoords, dispcor, trace, 
                            background, extract)
from drspy.plotting import plotSpectrum1D, plot2d
from drspy.utils import imstatistics, invertCoordinateMap


def loadList(listname, listpath=''):
    """Load a file list.

    Parameters
    ----------
    listname : string
        Name of the list.

    listpath : string
        Path to the list.

    Returns
    -------
    filelist : list
        The loaded file list.
    """

    with open(os.path.join(listpath, f"{listname}.list"), 'r') as f:
        filelist = [line.strip() for line in f.readlines()]
    
    return filelist


def loadLists(listnames, listpath=''):
    """Load file lists.

    Parameters
    ----------
    listnames : list
        A list of list names.

    listpath : string
        Path to the lists.

    Returns
    -------
    filelists : dict
        Loaded lists stored in a dictionary.
    """
    
    lists = dict()
    for listname in listnames:
        lists[listname] = loadList(listname, listpath=listpath)
    
    return lists


def genFileTable(ccddatalist, listpath, verbose=False):
    """Generate file table."""
    
    lists = loadLists(listnames=['bias', 'flat', 'lamp', 'targ'], listpath=listpath)
    
    file_table = list()
    for ccd in ccddatalist:
        if ccd.header['FILENAME'] in lists['bias']:
            obstype = 'BIAS'
            obj = 'BIAS'
        elif ccd.header['FILENAME'] in lists['flat']:
            obstype = 'SPECLFLAT'
            obj = 'FLAT'
        elif ccd.header['FILENAME'] in lists['lamp']:
            obstype = 'SPECLLAMP'
            obj = 'FeAr'
        elif ccd.header['FILENAME'] in lists['targ']:
            obstype = 'SPECLTARGET'
            obj = 'TARGET'
        else:
            obstype = 'OTHER'
            obj = 'OTHER'
        file_table.append([
            ccd.header['FILENAME'], obstype, obj, ccd.header['EXPTIME']])

    file_table = Table(
        rows=file_table, names=['file_name', 'file_type', 'targ_name', 'exposure'])

    if verbose: file_table.pprint_all() # [2]
    
    return file_table


def main(data_dir, save_dir, hdu, row_range, col_range, slit_along, re_n_piece, 
         il_bins, il_n_piece, rdnoise, gain, index_concatenate, index_standard, 
         file_name, show, save, verbose):
    """2.16-m/BFOSC pipeline (version 2020)"""

    # ==================================================================================
    # Path
    # ==================================================================================
    plot_path = os.path.join(save_dir, 'figs')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    fits_path = os.path.join(save_dir, 'fits')
    if not os.path.exists(fits_path):
        os.makedirs(fits_path)
    
    # ==================================================================================
    # Load data
    # ==================================================================================
    file_list = sorted(glob(os.path.join(data_dir, '*.fit')))
    ccddatalist = CCDDataList.read(file_list=file_list, hdu=hdu)

    # ==================================================================================
    # Organize data
    # ==================================================================================
    file_table = genFileTable(ccddatalist, save_dir, verbose)

    mask_bias = file_table['file_type'] == 'BIAS'
    mask_flat = file_table['file_type'] == 'SPECLFLAT'
    mask_lamp = file_table['file_type'] == 'SPECLLAMP'
    mask_slit = file_table['file_type'] == 'OTHER'
    mask_spec = file_table['file_type'] == 'SPECLTARGET'
    
    bias_list = ccddatalist[mask_bias]
    flat_list = ccddatalist[mask_flat]
    lamp_list = ccddatalist[mask_lamp]
    slit_list = ccddatalist[mask_slit]
    spec_list = ccddatalist[mask_spec]

    # ==================================================================================
    # Trim
    # ==================================================================================
    if verbose: print('TRIM')

    bias_list_trimmed = bias_list.trim(row_range=row_range, col_range=col_range)
    flat_list_trimmed = flat_list.trim(row_range=row_range, col_range=col_range)
    lamp_list_trimmed = lamp_list.trim(row_range=row_range, col_range=col_range)
    spec_list_trimmed = spec_list.trim(row_range=row_range, col_range=col_range)

    # ==================================================================================
    # Bias combination
    # ==================================================================================
    if verbose: print('BIAS COMBINATION')

    master_bias = bias_list_trimmed.combine(
        method='average', sigma_clip=True, sigma_clip_low_thresh=3, 
        sigma_clip_high_thresh=3, sigma_clip_func=np.ma.median, 
        sigma_clip_dev_func=mad_std, mem_limit=500e6)

    # Check statistics 
    bias_list_trimmed.statistics(verbose=verbose)
    imstatistics(master_bias, verbose=verbose)

    # Plot master bias
    plot2d(master_bias.data, title='master bias', show=show, save=save, path=plot_path)

    # Write master bias to file
    master_bias.write(
        os.path.join(fits_path, 'master_bias.fits'), overwrite=True)

    # ==================================================================================
    # Bias subtraction
    # ==================================================================================
    if verbose: print('BIAS SUBTRACTION')

    # !!! Uncertainties assigned here (equal to that of ``master_bias``) are useless !!!
    flat_list_bias_subtracted = flat_list_trimmed - master_bias
    lamp_list_bias_subtracted = lamp_list_trimmed - master_bias
    spec_list_bias_subtracted = spec_list_trimmed - master_bias

    # Check statistics
    flat_list_trimmed.statistics(verbose=verbose) # [2]
    flat_list_bias_subtracted.statistics(verbose=verbose) # [2]

    # ==================================================================================
    # Flat Combination
    # ==================================================================================
    if verbose: print('FLAT COMBINATION')

    # !!! Old uncertainties are overwritten !!!
    combined_flat = flat_list_bias_subtracted.combine(
        method='average', sigma_clip=True, sigma_clip_low_thresh=3, 
        sigma_clip_high_thresh=3, sigma_clip_func=np.ma.median, 
        sigma_clip_dev_func=mad_std, mem_limit=500e6)

    # Check statistics
    imstatistics(combined_flat, verbose=verbose) # [2]

    # Plot combined flat
    plot2d(
        combined_flat.data, title='combined flat', show=show, save=save, path=plot_path)

    # Write combined flat to file
    combined_flat.write(
        os.path.join(fits_path, 'combined_flat.fits'), overwrite=True)
    
    # ==================================================================================
    # Flat normalization
    # ==================================================================================
    if verbose: print('FLAT NORMALIZATION')

    # Response calibration
    reflat = response(
        ccd=combined_flat, slit_along=slit_along, n_piece=re_n_piece, n_iter=5, 
        sigma_lower=3, sigma_upper=3, grow=10, use_mask=True, show=show, save=save, 
        path=plot_path)

    # Check statistics
    imstatistics(reflat, verbose=verbose) # [2]

    # Plot response calibrated flat
    plot2d(reflat.data, title='reflat', show=show, save=save, path=plot_path)
    
#     # Custom mask
#     reflat.mask[520:620, 1200:1750] = True
    
    # Plot response mask
    plot2d(
        reflat.mask.astype(int), vmin=0, vmax=1, title='response mask', show=show, 
        save=save, path=plot_path)

    # Write response calibrated flat to file
    reflat.write(os.path.join(fits_path, 'reflat.fits'), overwrite=True) # [3]
    
    # Illumination modeling
    ilflat = illumination(
        ccd=reflat, slit_along=slit_along, method='iraf', bins=il_bins, 
        n_piece=il_n_piece, n_iter=5, sigma_lower=1.5, sigma_upper=3, grow=3, 
        use_mask=True, show=show, save=save, path=plot_path)

    # Plot illumination
    plot2d(ilflat.data, title='illumination', show=show, save=save, path=plot_path)
    
    # Plot illumination mask
    plot2d(
        ilflat.mask.astype(int), vmin=0, vmax=1, title='illumination mask', show=show, 
        save=save, path=plot_path)
    
    # Write illumination to file
    ilflat.write(os.path.join(fits_path, 'illumination.fits'), overwrite=True) # [3]
    
    # Normalization
    normalized_flat = reflat.divide(ilflat, handle_meta='first_found')

    # Plot normalized flat
    plot2d(
        normalized_flat.data, title='normalized flat', show=show, save=save, 
        path=plot_path)

    # Write normalized flat to file
    normalized_flat.write(
        os.path.join(fits_path, 'normalized_flat.fits'), overwrite=True)  # [3]
    
    # [can be replaced by a pre-defined custom mask]
    normalized_flat.mask = None

    # ==================================================================================
    # Create deviation
    # ==================================================================================
    if verbose: print('CREATE DEVIATION')

    # !!! generate real uncertainty here!!!
    lamp_list_bias_subtracted_with_deviation = (
        lamp_list_bias_subtracted.create_deviation(
            gain=gain, readnoise=rdnoise, disregard_nan=True)
    )
    spec_list_bias_subtracted_with_deviation = (
        spec_list_bias_subtracted.create_deviation(
            gain=gain, readnoise=rdnoise, disregard_nan=True)
    )

    # ==================================================================================
    # Flat-fielding
    # ==================================================================================
    if verbose: print('FLAT-FIELDING')

    spec_list_flat_corrected = (
        spec_list_bias_subtracted_with_deviation / normalized_flat
    )

    # ==================================================================================
    # Cosmic ray removal
    # ==================================================================================
    if verbose: print('COSMIC RAY REMOVAL')

    spec_list_cosmicray_corrected = spec_list_flat_corrected.cosmicray(
        method='Laplacian', use_mask=False, gain=gain, readnoise=rdnoise, sigclip=4.5, 
        sigfrac=0.3, objlim=1, niter=5, verbose=True)

    # ==================================================================================
    # Concatenate lamp
    # ==================================================================================
    if verbose: print('CONCATENATE LAMP')

    if slit_along == 'col':
        row_range = (0, None); col_range = (0, index_concatenate)
    else:
        row_range = (0, index_concatenate); col_range = (0, None)

    scale = 1 / file_table[mask_lamp]['exposure'].data

    # Ensure that the first is the short exposure
    if scale[0] < scale[1]:
        lamp_list_bias_subtracted_with_deviation = (
            lamp_list_bias_subtracted_with_deviation[::-1]
        )
        scale = scale[::-1]

    concatenated_lamp = concatenate(
        lamp_list_bias_subtracted_with_deviation, row_range, col_range, scale=None)

    # Plot concatenated lamp
    plot2d(
        concatenated_lamp.data, title='concatenated lamp', show=show, save=save, 
        path=plot_path)

    # Write concatenated lamp to file
    concatenated_lamp.write(
        os.path.join(fits_path, 'concatenated_lamp.fits'), overwrite=True)  # [3]

    # ==================================================================================
    # Fit coordinates
    # ==================================================================================
    if verbose: print('FIT COORDINATES')

    U, _ = fitcoords(
        ccd=concatenated_lamp, slit_along=slit_along, order=1, n_med=15, n_piece=3, 
        prominence=1e-3, n_iter=3, sigma_lower=3, sigma_upper=3, grow=False, 
        use_mask=False, show=show, save=save, path=plot_path, height=0, threshold=0, 
        distance=5, width=5, wlen=15, rel_height=1, plateau_size=1)
    
    X, Y = invertCoordinateMap(U=U, V=None)

    # ==================================================================================
    # Distortion correction
    # ==================================================================================
    if verbose: print('DISTORTION CORRECTION')

    # Lamp
    transformed_lamp = transform(
        ccd=concatenated_lamp, X=X, Y=Y)

    # Plot transformed lamp
    plot2d(
        transformed_lamp.data, title='transformed lamp', show=show, save=save, 
        path=plot_path)
    
    # Write transformed lamp to file
    transformed_lamp.write(
        os.path.join(fits_path, 'transformed_lamp.fits'), overwrite=True) # [3]

    # Spectra
    target_list_corrected = list()
    for i, spec_cosmicray_corrected in enumerate(spec_list_cosmicray_corrected):

        old_name = spec_cosmicray_corrected.header['FILENAME'][:-4]

        if verbose: print('{} corrected.'.format(old_name))

        transformed_spec = transform(
            ccd=spec_cosmicray_corrected, X=X, Y=Y)
        
        transformed_spec.mask = None

        # Plot transformed spectrum
        plot2d(
            transformed_spec.data, title=old_name, show=show, save=save, 
            path=plot_path)

        target_list_corrected.append(transformed_spec)

    # ==================================================================================
    # Standard
    # ==================================================================================
    if verbose: print('OUTPUT STANDARD')

    standard = target_list_corrected[index_standard]

    # Plot standard
    plot2d(
        standard.data, title='corrected standard', show=show, save=save, path=plot_path)
    
    # Write standard to file
    standard.write(os.path.join(fits_path, 'standard.fits'), overwrite=True) # [3]

    # ==================================================================================
    # Target
    # ==================================================================================
    if verbose: print('OUTPUT TARGET')

    del target_list_corrected[index_standard]

    target_list_aligned = align(target_list_corrected, slit_along=slit_along)
    
    target = target_list_aligned.combine(
        method='average', sigma_clip=True, sigma_clip_low_thresh=3, 
        sigma_clip_high_thresh=3, sigma_clip_func=np.ma.median, 
        sigma_clip_dev_func=mad_std, mem_limit=500e6)

    # Plot target
    plot2d(
        target.data, title='corrected target', show=show, save=save, path=plot_path)
    
    # Write standard to file
    target.write(os.path.join(fits_path, 'target.fits'), overwrite=True) # [3]
    
    # ==================================================================================
    # Wavelength calibration
    # ==================================================================================
    if verbose: print('WAVELENGTH CALIBRATION')

    lamp1d = extract(
        ccd=transformed_lamp, slit_along=slit_along, trace=750, n_aper=1, 
        aper_width=10, show=show, save=save, path=plot_path)
    
    calibrated_lamp1d = dispcor(
        spectrum1d=lamp1d, reverse=True, file_name=file_name, n_piece=3, refit=True, 
        n_iter=5, sigma_lower=3, sigma_upper=3, grow=False, use_mask=True, show=show, 
        save=save, path=plot_path)
    
    # Plot calibrated arc spectrum
    plotSpectrum1D(
        spectrum1d=calibrated_lamp1d, xlabel='wavelength [A]', ylabel='count', 
        title='calibrated lamp', show=show, save=save, path=plot_path)
    
    # Write calibrated arc spectrum to file
    calibrated_lamp1d.write(
        os.path.join(fits_path, 'lamp1d.fits'), overwrite=True) # [3]
    

if __name__ == '__main__':

    # Path
    data_dir = '/data3/zrn/data/216/LRS/20200406'
    save_dir = '/data3/zrn/workspace/data_reduction/20200406'
#     data_dir = '/data3/zrn/data/216/LRS/20200413'
#     save_dir = '/data3/zrn/workspace/data_reduction/20200413'
#     data_dir = '/data3/zrn/data/216/LRS/20200420'
#     save_dir = '/data3/zrn/workspace/data_reduction/20200420'
#     data_dir = '/data3/zrn/data/216/LRS/20200423'
#     save_dir = '/data3/zrn/workspace/data_reduction/20200423'
    
    # Hyper parameters
    hdu = 0
    col_range = (0, 1950)
    row_range = (329, 1830)
    slit_along = 'col'
    response_dict = {
        'n_piece': 19
    }
    illumination_dict = {
        'bins': [0, 250, 500, 750, 1000, 1100, 1200, 1250, 1300, 1350, 1400, 1450, 
                 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950], 
        'n_piece': [3, 3, 3, 3, 12, 15, 15, 15, 15, 21, 15, 15, 12, 17, 21, 23, 25, 45, 
                    29, 30, 29]
    }
    rdnoise = 4.64
    gain = 1.41
    index_concatenate = 665
    index_standard = -1
    file_name = 'bfosc_slit23_g4_2020.fits'

    main(
        data_dir=data_dir, save_dir=save_dir, hdu=hdu, row_range=row_range, 
        col_range=col_range, slit_along=slit_along, 
        re_n_piece=response_dict['n_piece'], il_bins=illumination_dict['bins'], 
        il_n_piece=illumination_dict['n_piece'], rdnoise=rdnoise, 
        gain=gain, index_concatenate=index_concatenate, index_standard=index_standard, 
        file_name=file_name, show=False, save=True, verbose=True)
