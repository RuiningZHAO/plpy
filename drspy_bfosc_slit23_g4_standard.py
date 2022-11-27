# [1] will be removed at last
# [2] will be replaced by logging
# [3] convert to np.float32
# ---------------------------------
import sys  # [1]
sys.path.append('/home/zrn/Workspace/') # [1]

import os
from glob import glob

# NumPy
import numpy as np
from astropy.stats import mad_std, sigma_clip
# AstroPy
from astropy.table import Table

from pyloss import CCDDataList
from pyloss.longslit import (response, illumination, concatenate, fitcoords, resample, 
                             trace, background, extract)
from pyloss.plotting import plot2d
from pyloss.utils import imstatistics



def pipeline(
        data_dir, save_dir, hdu, arc_name, std_name, obj_name, row_range, col_range, 
        slit_along, rdnoise, gain):
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
    file_table = list()
    for ccd in ccddatalist:
        file_table.append([
            ccd.header['FILENAME'], ccd.header['OBSTYPE'], ccd.header['OBJECT'], 
            ccd.header['EXPTIME']])
    file_table = Table(
        rows=file_table, names=['file_name', 'file_type', 'targ_name', 'exposure'])
    file_table.pprint_all() # [2]
    
    mask_slittarget = file_table['file_type'] == 'SLITTARGET'
    mask_bias = (~mask_slittarget) & (file_table['targ_name'] == 'BIAS')
    mask_flat = (~mask_slittarget) & (file_table['targ_name'] == 'FLAT')
    mask_arc = (~mask_slittarget) & (file_table['targ_name'] == arc_name)
    mask_std = (~mask_slittarget) & (file_table['targ_name'] == std_name)
    mask_obj = (~mask_slittarget) & (file_table['targ_name'] == obj_name)
    
    slittargetlist = ccddatalist[mask_slittarget]
    biaslist = ccddatalist[mask_bias]
    flatlist = ccddatalist[mask_flat]
    arclist = ccddatalist[mask_arc]
    stdlist = ccddatalist[mask_std]
    objlist = ccddatalist[mask_obj]

    # ==================================================================================
    # Trim
    # ==================================================================================
    biaslist_trimmed = biaslist.trim(row_range=row_range, 
                                     col_range=col_range)
    flatlist_trimmed = flatlist.trim(row_range=row_range, 
                                     col_range=col_range)
    arclist_trimmed = arclist.trim(row_range=row_range, 
                                   col_range=col_range)
    stdlist_trimmed = stdlist.trim(row_range=row_range, 
                                   col_range=col_range)
    objlist_trimmed = objlist.trim(row_range=row_range, 
                                   col_range=col_range)

    # ==================================================================================
    # Bias combination
    # ==================================================================================
    master_bias = biaslist_trimmed.combine(
        method='average', sigma_clip=True, sigma_clip_low_thresh=3, 
        sigma_clip_high_thresh=3, sigma_clip_func=np.ma.median, 
        sigma_clip_dev_func=mad_std, mem_limit=500e6)

    # Check statistics 
    biaslist_trimmed.statistics(verbose=True) # [2]
    imstatistics(master_bias, verbose=True) # [2]

    # Plot master bias
    plot2d(master_bias.data, title='master bias', save=True, path=plot_path)

    # Write master bias to file
    master_bias.write(
        os.path.join(fits_path, 'master_bias.fits'), overwrite=True) # [3]
    
    # ==================================================================================
    # Bias subtraction
    # ==================================================================================
    # !!! Uncertainties assigned here (equal to that of ``master_bias``) is useless !!!
    # Flat
    flatlist_bias_subtracted = flatlist_trimmed - master_bias
    
    # Arc
    arclist_bias_subtracted = arclist_trimmed - master_bias
    
    # Standand
    stdlist_bias_subtracted = stdlist_trimmed - master_bias
    
    # Object
    objlist_bias_subtracted = objlist_trimmed - master_bias
    
    # Check statistics
    flatlist_trimmed.statistics(verbose=True) # [2]
    flatlist_bias_subtracted.statistics(verbose=True) # [2]
    
    # ==================================================================================
    # Flat Combination
    # ==================================================================================
    # !!! Old uncertainties are overwritten !!!
    combined_flat = flatlist_bias_subtracted.combine(
        method='average', sigma_clip=True, sigma_clip_low_thresh=3, 
        sigma_clip_high_thresh=3, sigma_clip_func=np.ma.median, 
        sigma_clip_dev_func=mad_std, mem_limit=500e6)

    # Check statistics
    imstatistics(combined_flat, verbose=True) # [2]

    # Plot combined flat
    plot2d(combined_flat.data, title='combined flat', save=True, path=plot_path)

    # Write combined flat to file
    combined_flat.write(
        os.path.join(fits_path, 'combined_flat.fits'), overwrite=True) # [3]
    
    # ==================================================================================
    # Flat normalization
    # ==================================================================================
    # Response calibration
    reflat = response(
        ccd=combined_flat, slit_along=slit_along, n_piece=19, sigma_lower=3, 
        sigma_upper=3, grow=10, use_mask=True, save=True, path=plot_path)

    # Check statistics
    imstatistics(reflat, verbose=True) # [2]

    # Plot response calibrated flat
    plot2d(reflat.data, title='reflat', save=True, path=plot_path)

    # Plot response mask
    plot2d(
        reflat.mask.astype(int), vmin=0, vmax=1, title='response mask', save=True, 
        path=plot_path)

    # Write response calibrated flat to file
    reflat.write(os.path.join(fits_path, 'reflat.fits'), overwrite=True) # [3]

    # Illumination modeling
#     ilflat = illumination(
#         ccd=reflat, slit_along=slit_along, method='iraf',
#         bins=[0, 250, 500, 750, 1000, 1100, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 
#               1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950], n_piece=[4, 4, 4, 
#         4, 15, 15, 15, 15, 15, 21, 15, 12, 10, 12, 21, 23, 23, 16, 20, 16, 25], 
#         n_iter=3, sigma_lower=0.75, sigma_upper=3, grow=3, use_mask=True, show=False, 
#         save=True, path=plot_path)

#     ilflat = illumination(
#         ccd=reflat, slit_along=slit_along, method='CubicSpline2D', n_piece=(21, 5), 
#         n_iter=3, sigma_lower=1.5, sigma_upper=3, grow=5, use_mask=True, show=False, 
#         save=True, path=plot_path)
    
    ilflat = illumination(
        ccd=reflat, slit_along=slit_along, method='Gaussian2D', sigma=(50, 20), 
        n_iter=3, sigma_lower=1.5, sigma_upper=3, grow=5, use_mask=True, show=False, 
        save=True, path=plot_path)

    # Plot illumination
    plot2d(ilflat.data, title='illumination', save=True, path=plot_path)
    
    # Plot illumination mask
    plot2d(
        ilflat.mask.astype(int), vmin=0, vmax=1, title='illumination mask', save=True, 
        path=plot_path)
    
    # Output here???
    
    # Normalization
    normalized_flat = reflat.divide(ilflat, handle_meta='first_found')

    # Plot normalized flat
    plot2d(normalized_flat.data, title='normalized flat', save=True, path=plot_path)

    # Write normalized flat to file
    normalized_flat.write(
        os.path.join(fits_path, 'normalized_flat.fits'), overwrite=True)  # [3]
    
    # [can be replaced by a pre-defined custom mask]
    normalized_flat.mask = None

    # ==================================================================================
    # Create deviation
    # ==================================================================================
    # !!! generate real uncertainty here!!!
    # Arc
    arclist_bias_subtracted_with_deviation = arclist_bias_subtracted.create_deviation(
        gain=gain, readnoise=rdnoise)

    # Standand
    stdlist_bias_subtracted_with_deviation = stdlist_bias_subtracted.create_deviation(
        gain=gain, readnoise=rdnoise)

    # Object
    objlist_bias_subtracted_with_deviation = objlist_bias_subtracted.create_deviation(
        gain=gain, readnoise=rdnoise)
    
    # ==================================================================================
    # Flat-fielding
    # ==================================================================================
    # Arc
    arclist_flat_corrected = arclist_bias_subtracted_with_deviation / normalized_flat
    
    # Standard
    stdlist_flat_corrected = stdlist_bias_subtracted_with_deviation / normalized_flat
    
    # Object
    objlist_flat_corrected = objlist_bias_subtracted_with_deviation / normalized_flat

    # Output here???

    # ==================================================================================
    # Cosmic ray removal
    # ==================================================================================
    # Standard
    stdlist_cosmicray_corrected = stdlist_flat_corrected.cosmicray(
        method='Laplacian', use_mask=False, gain=gain, readnoise=rdnoise, sigclip=4.5, 
        sigfrac=0.3, objlim=1, niter=5, verbose=True)
    
    # Object
    objlist_cosmicray_corrected = objlist_flat_corrected.cosmicray(
        method='Laplacian', use_mask=False, gain=gain, readnoise=rdnoise, sigclip=4.5, 
        sigfrac=0.3, objlim=1, niter=5, verbose=True)

    # Plot cosmic-ray removed frames
    plot2d(
        stdlist_cosmicray_corrected[0].data, title='cosmic-ray removed standard', 
        save=True, path=plot_path)
    plot2d(
        objlist_cosmicray_corrected[0].data, title='cosmic-ray removed object', 
        save=True, path=plot_path)

    # Output here???

    # ==================================================================================
    # Concatenate arc spectrum
    # ==================================================================================
    concatenated_arc = concatenate(
        arclist_flat_corrected, slit_along=slit_along, index=665) # order???

    # Plot concatenated arc spectrum
    plot2d(concatenated_arc.data, title='concatenated arc', save=True, path=plot_path)

#     # ==================================================================================
#     # Fit coordinates
#     # ==================================================================================
#     X = fitcoords(
#         ccd=concatenated_arc, slit_along=slit_along, method='2D', n_med=5, n_ext=10, 
#         n_sub=5, n_piece=(3, 3), n_iter=3, sigma_lower=3, sigma_upper=3, grow=False, 
#         height=0, threshold=0, distance=5, prominence=0.001, width=5, wlen=10, 
#         rel_height=1, plateau_size=1, use_mask=True, show=False, save=True, 
#         path=plot_path)
    
#     # ==================================================================================
#     # Distortion correction
#     # ==================================================================================
#     # Arc
#     resampled_arc = resample(
#         ccd=concatenated_arc, slit_along=slit_along, X=X, use_uncertainty=True, 
#         verbose=False)
    
#     # Plot resampled arc spectrum
#     plot2d(resampled_arc.data, title='resampled arc', save=True, path=plot_path)
    
#     # Write resampled arc spectrum to file
#     resampled_arc.write(
#         os.path.join(fits_path, 'resampled_arc.fits'), overwrite=True) # [3]

#     # Standard
#     resampled_std = resample(
#         ccd=stdlist_cosmicray_corrected[0], slit_along=slit_along, X=X, 
#         use_uncertainty=True, verbose=False)
    
#     # Plot resampled standard spectrum
#     plot2d(
#         resampled_std.data, title='resampled standard', save=True, 
#         path=plot_path)
    
#     # Write resampled standard spectrum to file
#     resampled_std.write(
#         os.path.join(fits_path, 'resampled_standard.fits'), overwrite=True) # [3]
    
#     # Object???

#     # ==================================================================================
#     # Wavelength calibration
#     # ==================================================================================

    
#     # ==================================================================================
#     # Trace
#     # ==================================================================================
    trace_std = trace(
        ccd=stdlist_cosmicray_corrected[0], slit_along=slit_along, seeing=8, n_med=10, n_piece=3, 
        n_iter=5, sigma_lower=2, sigma_upper=2, grow=False, show=False, save=True, 
        path=plot_path)

    extract(
        ccd=concatenated_arc, slit_along=slit_along, trace=trace_std, n_aper=10, aper_width=400, 
        use_uncertainty=False, show=False, save=True, path=plot_path)
#     background_std = background(
#         ccd=resampled_std, slit_along=slit_along, trace=trace_std, distance=100, 
#         aper_width=100, degree=1, n_iter=5, sigma_lower=3, sigma_upper=3, grow=False)
#     # Plot background spectrum
#     plot2d(background_std.data, title='background', show=False, save=True, path=plot_path)
#     # Write background spectrum to file
#     background_std.write(
#         os.path.join(fits_path, 'background_std.fits'), overwrite=True) # [3]

#     # Background subtraction
#     std_background_subtracted = resampled_std.subtract(
#         background_std, handle_meta='first_found')
#     # Plot background subtracted standard spectrum
#     plot2d(
#         std_background_subtracted.data, title='background subtracted standard spectrum', 
#         show=False, save=True, path=plot_path)
#     # Write background subtracted standard spectrum to file
#     std_background_subtracted.write(
#         os.path.join(fits_path, 'background_subtracted_std.fits'), overwrite=True) # [3]

if __name__ == '__main__':

    # Parameters
    data_dir = '/home/zrn/Data_Reduction/20210504'
    std_name = 'Feige34'
    obj_name = '2021klg'
    
    # Hyper parameters
    hdu = 0
    arc_name = 'FeAr'
    col_range = (0, 1950)
    row_range = (329, 1830)
    slit_along = 'col'
    rdnoise = 4.64
    gain = 1.41

    pipeline(data_dir=data_dir, 
             save_dir=data_dir,
             hdu=hdu,
             arc_name=arc_name,
             std_name=std_name,
             obj_name=obj_name,
             row_range=row_range,
             col_range=col_range,
             slit_along=slit_along,
             rdnoise=rdnoise,
             gain=gain)
