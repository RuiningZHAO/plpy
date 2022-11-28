# [1] will be removed at last
# [2] will be replaced by logging
# [3] convert to np.float32
# ---------------------------------
import sys  # [1]
sys.path.append('/data3/zrn/workspace/github') # [1]

import os
from glob import glob

# NumPy
import numpy as np
from astropy.stats import mad_std, sigma_clip
# AstroPy
from astropy.table import Table

from drspy import CCDDataList
from drspy.longslit import (response, illumination, concatenate, fitcoords, resample, 
                            trace, background, extract, dispcor)
from drspy.plotting import plotSpectrum1D, plot2d
from drspy.utils import imstatistics



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

def genFileTable(ccddatalist):
    """
    """
    
    file_table = list()
    for ccd in ccddatalist:
        file_table.append([
            ccd.header['FILENAME'], ccd.header['OBSTYPE'], ccd.header['OBJECT'], 
            ccd.header['EXPTIME']])
    file_table = Table(
        rows=file_table, names=['file_name', 'file_type', 'targ_name', 'exposure'])
    file_table.pprint_all() # [2]
    
    return file_table

def main(data_dir, save_dir, hdu, row_range, col_range, slit_along, rdnoise, gain):
    """
    """
    
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
    file_table = genFileTable(ccddatalist)

    mask_bias = file_table['file_type'] == 'BIAS'
    mask_flat = file_table['file_type'] == 'SPECLFLAT'
    mask_lamp = file_table['file_type'] == 'SPECLLAMP'
    mask_spec = file_table['file_type'] == 'SPECLTARGET'
    
    bias_list = ccddatalist[mask_bias]
    flat_list = ccddatalist[mask_flat]
    lamp_list = ccddatalist[mask_lamp]
    spec_list = ccddatalist[mask_spec]

    # ==================================================================================
    # Trim
    # ==================================================================================
    bias_list_trimmed = bias_list.trim(row_range=row_range, col_range=col_range)
    flat_list_trimmed = flat_list.trim(row_range=row_range, col_range=col_range)
    lamp_list_trimmed = lamp_list.trim(row_range=row_range, col_range=col_range)
    spec_list_trimmed = spec_list.trim(row_range=row_range, col_range=col_range)

    # ==================================================================================
    # Bias combination
    # ==================================================================================
    master_bias = bias_list_trimmed.combine(
        method='average', sigma_clip=True, sigma_clip_low_thresh=3, 
        sigma_clip_high_thresh=3, sigma_clip_func=np.ma.median, 
        sigma_clip_dev_func=mad_std, mem_limit=500e6)

    # Check statistics 
    bias_list_trimmed.statistics(verbose=True) # [2]
    imstatistics(master_bias, verbose=True) # [2]

    # Plot master bias
    plot2d(master_bias.data, title='master bias', show=False, save=True, path=plot_path)

    # Write master bias to file
    master_bias.write(
        os.path.join(fits_path, 'master_bias.fits'), overwrite=True) # [3]
    
    # ==================================================================================
    # Bias subtraction
    # ==================================================================================
    # !!! Uncertainties assigned here (equal to that of ``master_bias``) is useless !!!
    flat_list_bias_subtracted = flat_list_trimmed - master_bias
    lamp_list_bias_subtracted = lamp_list_trimmed - master_bias
    spec_list_bias_subtracted = spec_list_trimmed - master_bias
    
    # Check statistics
    flat_list_trimmed.statistics(verbose=True) # [2]
    flat_list_bias_subtracted.statistics(verbose=True) # [2]
    
    # ==================================================================================
    # Flat Combination
    # ==================================================================================
    # !!! Old uncertainties are overwritten !!!
    combined_flat = flat_list_bias_subtracted.combine(
        method='average', sigma_clip=True, sigma_clip_low_thresh=3, 
        sigma_clip_high_thresh=3, sigma_clip_func=np.ma.median, 
        sigma_clip_dev_func=mad_std, mem_limit=500e6)

    # Check statistics
    imstatistics(combined_flat, verbose=True) # [2]

    # Plot combined flat
    plot2d(
        combined_flat.data, title='combined flat', show=False, save=True, path=plot_path)

    # Write combined flat to file
    combined_flat.write(
        os.path.join(fits_path, 'combined_flat.fits'), overwrite=True) # [3]
    
    # ==================================================================================
    # Flat normalization
    # ==================================================================================
    # Response calibration
    # [sometimes ``n_piece`` needs to be tuned.]
    reflat = response(
        ccd=combined_flat, slit_along=slit_along, n_piece=19, sigma_lower=3, 
        sigma_upper=3, grow=10, use_mask=True, show=False, save=True, path=plot_path)

    # Check statistics
    imstatistics(reflat, verbose=True) # [2]

    # Plot response calibrated flat
    plot2d(reflat.data, title='reflat', show=False, save=True, path=plot_path)

#     # Custom mask
#     reflat.mask[520:620, 1200:1750] = True
    
    # Plot response mask
    plot2d(
        reflat.mask.astype(int), vmin=0, vmax=1, title='response mask', show=False, 
        save=True, path=plot_path)

    # Write response calibrated flat to file
    reflat.write(os.path.join(fits_path, 'reflat.fits'), overwrite=True) # [3]
    
    # Illumination modeling
#     ilflat = illumination(
#         ccd=reflat, slit_along=slit_along, method='iraf',
#         bins=[0, 250, 500, 750, 1000, 1100, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 
#               1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950], n_piece=[3, 3, 3, 
#         3, 12, 15, 15, 15, 15, 21, 15, 15, 12, 17, 21, 23, 25, 45, 29, 30, 29], 
#         n_iter=5, sigma_lower=1.5, sigma_upper=3, grow=3, use_mask=True, show=False, 
#         save=True, path=plot_path)

#     ilflat = illumination(
#         ccd=reflat, slit_along=slit_along, method='iraf',
#         bins=[0, 250, 500, 750, 1000, 1100, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 
#               1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950], n_piece=[4, 4, 4, 
#         4, 15, 15, 15, 15, 15, 21, 15, 12, 10, 12, 21, 23, 23, 16, 20, 16, 25], 
#         n_iter=3, sigma_lower=1.5, sigma_upper=3, grow=3, use_mask=True, show=False, 
#         save=True, path=plot_path)

#     ilflat = illumination(
#         ccd=reflat, slit_along=slit_along, method='CubicSpline2D', n_piece=(21, 5), 
#         n_iter=3, sigma_lower=1.5, sigma_upper=3, grow=5, use_mask=True, show=False, 
#         save=True, path=plot_path)
    
    ilflat = illumination(
        ccd=reflat, slit_along=slit_along, method='Gaussian2D', sigma=(20, 20), bins=10, 
        n_iter=10, sigma_lower=1.5, sigma_upper=3, grow=5, use_mask=True, show=False, 
        save=True, path=plot_path)

    # Plot illumination
    plot2d(ilflat.data, title='illumination', show=False, save=True, path=plot_path)
    
    # Plot illumination mask
    plot2d(
        ilflat.mask.astype(int), vmin=0, vmax=1, title='illumination mask', show=False, 
        save=True, path=plot_path)
    
    # Write illumination to file
    ilflat.write(os.path.join(fits_path, 'illumination.fits'), overwrite=True) # [3]
    
    # Normalization
    normalized_flat = reflat.divide(ilflat, handle_meta='first_found')

    # Plot normalized flat
    plot2d(
        normalized_flat.data, title='normalized flat', show=False, save=True, 
        path=plot_path)

    # Write normalized flat to file
    normalized_flat.write(
        os.path.join(fits_path, 'normalized_flat.fits'), overwrite=True)  # [3]
    
    # [can be replaced by a pre-defined custom mask]
    normalized_flat.mask = None

    # ==================================================================================
    # Create deviation
    # ==================================================================================
    # !!! generate real uncertainty here!!!
    lamp_list_bias_subtracted_with_deviation = (
        lamp_list_bias_subtracted.create_deviation(gain=gain, readnoise=rdnoise)
    )
    spec_list_bias_subtracted_with_deviation = (
        spec_list_bias_subtracted.create_deviation(gain=gain, readnoise=rdnoise)
    )
    
    # ==================================================================================
    # Flat-fielding
    # ==================================================================================
    # Lamp
    lamp_list_flat_corrected = (
        lamp_list_bias_subtracted_with_deviation / normalized_flat
    )
    # Spec
    spec_list_flat_corrected = (
        spec_list_bias_subtracted_with_deviation / normalized_flat
    )

    # ==================================================================================
    # Cosmic ray removal
    # ==================================================================================
    spec_list_cosmicray_corrected = spec_list_flat_corrected.cosmicray(
        method='Laplacian', use_mask=False, gain=gain, readnoise=rdnoise, sigclip=4.5, 
        sigfrac=0.3, objlim=1, niter=5, verbose=True)

    # Plot cosmic-ray removed frames [how to output all the images???]
    plot2d(
        spec_list_cosmicray_corrected[0].data, title='cosmic-ray removed', show=False, 
        save=True, path=plot_path)

    # ==================================================================================
    # Concatenate lamp
    # ==================================================================================
    # Ensure that the first is the short exposure 
    if np.diff(file_table[mask_lamp]['exposure'].data)[0] < 0:
        lamp_list_flat_corrected = lamp_list_flat_corrected[::-1]

    concatenated_lamp = concatenate(
        lamp_list_flat_corrected, slit_along=slit_along, index=665)

    # Plot concatenated lamp
    plot2d(
        concatenated_lamp.data, title='concatenated lamp', show=False, save=True, 
        path=plot_path)

    # Write concatenated lamp to file
    concatenated_lamp.write(
        os.path.join(fits_path, 'concatenated_lamp.fits'), overwrite=True)  # [3]

    # ==================================================================================
    # Fit coordinates
    # ==================================================================================
    U, V = fitcoords(
        ccd=concatenated_lamp, slit_along=slit_along, method='2D', n_med=5, n_ext=10, 
        n_sub=5, n_piece=(3, 3), n_iter=3, sigma_lower=3, sigma_upper=3, grow=False, 
        height=0, threshold=0, distance=5, prominence=0.001, width=5, wlen=10, 
        rel_height=1, plateau_size=1, show=False, save=True, path=plot_path)
    
    # ==================================================================================
    # Distortion correction
    # ==================================================================================
    # Lamp
    resampled_lamp = resample(
        ccd=concatenated_lamp, slit_along=slit_along, U=U, V=V, use_uncertainty=True, 
        verbose=False)
    
    # Plot resampled lamp
    plot2d(
        resampled_lamp.data, title='resampled lamp', show=False, save=True, 
        path=plot_path)
    
    # Write resampled lamp to file
    resampled_lamp.write(
        os.path.join(fits_path, 'resampled_lamp.fits'), overwrite=True) # [3]

    # Spec
    for i, spec_cosmicray_corrected in enumerate(spec_list_cosmicray_corrected):
        
        old_name = spec_cosmicray_corrected.header['FILENAME']
        new_name = '{}_corrected.fits'.format(old_name[:-4])
        
        print('{} -> {}'.format(old_name, new_name))
        
        resampled_spec = resample(
            ccd=spec_cosmicray_corrected, slit_along=slit_along, U=U, V=V, 
            use_uncertainty=True, verbose=False)

        # Plot resampled spectrum
        plot2d(
            resampled_spec.data, title=old_name, show=False, save=True, 
            path=plot_path)

        # Write resampled standard spectrum to file
        resampled_spec.write(
            os.path.join(fits_path, new_name), overwrite=True) # [3]

    # ==================================================================================
    # Wavelength calibration
    # ==================================================================================
    lamp1d = extract(
        ccd=resampled_lamp, slit_along=slit_along, trace=750, n_aper=1, aper_width=10, 
        use_uncertainty=True, show=False, save=True, path=plot_path)
    
    calibrated_lamp1d = dispcor(
        spectrum1d=lamp1d, reverse=True, file_name='bfosc_slit23_g4.fits', n_ext=10, 
        n_sub=5, n_piece=3, refit=True, n_iter=5, sigma_lower=3, sigma_upper=3, 
        grow=False, show=False, save=True, path=plot_path)
    
    # Plot
    plotSpectrum1D(
        spectrum1d=calibrated_lamp1d, xlabel='wavelength [A]', ylabel='count', 
        title='extracted lamp', show=False, save=True, path=plot_path)
    
    calibrated_lamp1d.write(
        os.path.join(fits_path, 'lamp1d.fits'), overwrite=True)

if __name__ == '__main__':

    # Path
    data_dir = '/data3/zrn/workspace/data_reduction/demo_SN2021klg/data'
    save_dir = '/data3/zrn/workspace/data_reduction/demo_SN2021klg'
#     data_dir = '/data3/zrn/workspace/data_reduction/demo_standard/data'
#     save_dir = '/data3/zrn/workspace/data_reduction/demo_standard'
    
    # Hyper parameters
    hdu = 0
    col_range = (0, 1950)
    row_range = (329, 1830)
    slit_along = 'col'
    rdnoise = 4.64
    gain = 1.41

    main(
        data_dir=data_dir, save_dir=save_dir, hdu=hdu, row_range=row_range, 
        col_range=col_range, slit_along=slit_along, rdnoise=rdnoise, gain=gain)
