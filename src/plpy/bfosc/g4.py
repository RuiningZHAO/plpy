"""
2.16-m/BFOSC pipeline
"""

import os, argparse

# NumPy
import numpy as np
# AstroPy
from astropy.stats import mad_std
from astropy.nddata import CCDData
# ccdproc
from ccdproc import ImageFileCollection
# drpy
from drpy.batch import CCDDataList
from drpy.utils import imstatistics
from drpy.plotting import plot2d#, plotSpectrum1D
from drpy.twodspec.longslit import (response, illumination)#, fitcoords, transform, 
#                                     trace, background, extract)

from .utils import loadLists
from ..utils import makeDirectory, modifyHeader

# from glob import glob

# from astropy.table import Table

# from drpy.image import concatenate
# from drpy.twodspec.utils import invertCoordinateMap
# from drpy.onedspec import dispcor


# def genFileTable(ccddatalist, listpath, verbose=False):
#     """Generate file table."""
    
#     lists = loadLists(list_names=['bias', 'flat', 'lamp', 'targ'], list_path=list_path)
    
#     file_table = list()
#     for ccd in ccddatalist:
#         if ccd.header['FILENAME'] in lists['bias']:
#             obstype = 'BIAS'
#             obj = 'BIAS'
#         elif ccd.header['FILENAME'] in lists['flat']:
#             obstype = 'SPECLFLAT'
#             obj = 'FLAT'
#         elif ccd.header['FILENAME'] in lists['lamp']:
#             obstype = 'SPECLLAMP'
#             obj = 'FeAr'
#         elif ccd.header['FILENAME'] in lists['targ']:
#             obstype = 'SPECLTARGET'
#             obj = 'TARGET'
#         else:
#             obstype = 'OTHER'
#             obj = 'OTHER'
#         file_table.append([
#             ccd.header['FILENAME'], obstype, obj, ccd.header['EXPTIME']])

#     file_table = Table(
#         rows=file_table, names=['file_name', 'file_type', 'targ_name', 'exposure'])

#     if verbose: file_table.pprint_all() # [2]
    
#     return file_table


def pipeline(save_dir, data_dir, hdu, slit_along, trim, row_range, col_range, , re_n_piece, 
             il_bins, il_n_piece, rdnoise, gain, index, file_name, steps, keywords, mem_limit, show, save, verbose):
    """2.16-m/BFOSC pipeline"""
    
    # Change working directory
    os.chdir(save_dir)
    
    # Make subdirectories
    fig_path = makeDirectory(parent='', child='fig')
    pro_path = makeDirectory(parent='', child='pro')
    cal_path = makeDirectory(parent='', child='cal')
    
    # Load lists
    list_dict = loadLists(list_names=['lamp', 'targ'])
    
    # Construct image file collection
    ifc = ImageFileCollection(
        location=data_dir, keywords=keywords, find_fits_by_reading=False, 
        filenames=None, glob_include='*.fit', glob_exclude=None, ext=hdu)

    # Modify fits header
    #   Note that image file collection is constructed before header modification.
    if 'header' in steps:
        if verbose:
            print('[HEADER MODIFICATION]')
        for file_name in ifc.files_filtered(include_path=True):
            modifyHeader(file_name, verbose=verbose)
    
    # Bias combination
    if ('bias.combine' in steps) or ('bias' in steps):
        if verbose:
            print('[BIAS COMBINATION]')
        # Filter
        ifc_bias = ifc.filter(regex_match=True, imagetyp='Bias Frame')
        if verbose:
            ifc_bias.summary.pprint_all()
        # Load bias
        bias_list = CCDDataList.read(
            file_list=ifc_bias.files_filtered(include_path=True), hdu=hdu)
        # Trim
        bias_list_trimmed = bias_list.trim(row_range=row_range, col_range=col_range)
        # Combine bias
        master_bias = bias_list_trimmed.combine(
            method='average', sigma_clip=True, sigma_clip_low_thresh=3, 
            sigma_clip_high_thresh=3, sigma_clip_func=np.ma.median, 
            sigma_clip_dev_func=mad_std, mem_limit=mem_limit, 
            output_file=os.path.join(cal_path, 'master_bias.fits'), 
            overwrite_output=True)
        # Check statistics 
        bias_list_trimmed.statistics(verbose=verbose)
        imstatistics(master_bias, verbose=verbose)
        # Plot master bias
        plot2d(
            master_bias.data, title='master bias', show=show, save=save, path=fig_path)
        # Release memory
        del bias_list, bias_list_trimmed
    
    # Flat combination
    if ('flat.combine' in steps) or ('flat' in steps):
        if verbose:
            print('[FLAT COMBINATION]')
        # Filter
        ifc_flat = ifc.filter(regex_match=True, imagetyp='Flat Field')
        if verbose:
            ifc_flat.summary.pprint_all()
        # Load flat
        flat_list = CCDDataList.read(
            file_list=ifc_flat.files_filtered(include_path=True), hdu=hdu)
        # Trim
        flat_list_trimmed = flat_list.trim(row_range=row_range, col_range=col_range)
        # Subtract bias
        #   Uncertainties created here (equal to that of ``master_bias``) are useless!!!
        if 'master_bias' not in locals():
            master_bias = CCDData.read(os.path.join(cal_path, 'master_bias.fits'))
        flat_list_bias_subtracted = flat_list_trimmed - master_bias
        # Combine flat
        #   Uncertainties created above are overwritten here!!!
        scaling_func = lambda ccd: 1 / np.ma.average(ccd)
        combined_flat = flat_list_bias_subtracted.combine(
            method='average', scale=scaling_func, sigma_clip=True, 
            sigma_clip_low_thresh=3, sigma_clip_high_thresh=3, 
            sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std, 
            mem_limit=mem_limit, 
            output_file=os.path.join(cal_path, 'combined_flat.fits'), 
            overwrite_output=True)
        # Check statistics
        flat_list_trimmed.statistics(verbose=verbose)
        flat_list_bias_subtracted.statistics(verbose=verbose)
        imstatistics(combined_flat, verbose=verbose)
        # Plot combined flat
        plot2d(
            combined_flat.data, title='combined_flat', show=show, save=save, 
            path=fig_path)
        # Release memory
        del flat_list, flat_list_trimmed, master_bias, flat_list_bias_subtracted

    # Response
    if ('flat.normalize.response' in steps) or \
       ('flat.normalize' in steps) or \
       ('flat' in steps):
        if verbose:
            print('[RESPONSE]')
        if 'combined_flat' not in locals():
            combined_flat = CCDData.read(os.path.join(cal_path, 'combined_flat.fits'))
        # Response calibration
        reflat = response(
            ccd=combined_flat, slit_along=slit_along, n_piece=re_n_piece, n_iter=5, 
            sigma_lower=3, sigma_upper=3, grow=10, use_mask=True, show=show, save=save, 
            path=fig_path)
        # Check statistics
        imstatistics(reflat, verbose=verbose)
        # Plot response calibrated flat
        plot2d(reflat.data, title='reflat', show=show, save=save, path=fig_path)
        # # Custom mask
        # reflat.mask[520:620, 1200:1750] = True
        # Plot response mask
        plot2d(
            reflat.mask.astype(int), vmin=0, vmax=1, title='response mask', show=show, 
            save=save, path=fig_path)
        # Write response calibrated flat to file
        reflat.write(os.path.join(cal_path, 'reflat.fits'), overwrite=True)
    
    # Illumination
    if ('flat.normalize.illumination' in steps) or \
       ('flat.normalize' in steps) or \
       ('flat' in steps):
        if verbose:
            print('[ILLUMINATION]')
        if 'reflat' not in locals():
            reflat = CCDData.read(os.path.join(cal_path, 'reflat.fits'))
        # Illumination modeling
        ilflat = illumination(
            ccd=reflat, slit_along=slit_along, method='iraf', bins=il_bins, 
            n_piece=il_n_piece, n_iter=5, sigma_lower=1.5, sigma_upper=3, grow=3, 
            use_mask=True, show=show, save=save, path=fig_path)
        # Plot illumination
        plot2d(ilflat.data, title='illumination', show=show, save=save, path=fig_path)
        # Plot illumination mask
        plot2d(
            ilflat.mask.astype(int), vmin=0, vmax=1, title='illumination mask', 
            show=show, save=save, path=fig_path)
        # Write illumination to file
        ilflat.write(os.path.join(cal_path, 'illumination.fits'), overwrite=True)

    # Flat normalization
    if ('flat.normalize' in steps) or ('flat' in steps):
        if verbose:
            print('[FLAT NORMALIZATION]')
        # Normalization
        normalized_flat = reflat.divide(ilflat, handle_meta='first_found')
        # Plot normalized flat
        plot2d(
            normalized_flat.data, title='normalized flat', show=show, save=save, 
            path=fig_path)
        # Write normalized flat to file
        normalized_flat.write(
            os.path.join(cal_path, 'normalized_flat.fits'), overwrite=True)
        # [can be replaced by a pre-defined custom mask]
        normalized_flat.mask = None

    # Lamp concatenation
    if ('lamp.concatenate' in steps) or ('lamp' in steps):
        if verbose:
            print('[CONCATENATE LAMP]')
        ifc_lamp = ImageFileCollection(
            location=data_dir, keywords=keywords, find_fits_by_reading=False, 
            filenames=list_dict['lamp'], glob_include=None, glob_exclude=None, ext=hdu)
        if verbose:
            ifc_lamp.summary.pprint_all()
        # Load lamp
        lamp_list = CCDDataList.read(
            file_list=ifc_lamp.files_filtered(include_path=True), hdu=hdu)
        # Trim
        lamp_list_trimmed = lamp_list.trim(row_range=row_range, col_range=col_range)
        # Subtract bias
        #   Uncertainties created here (equal to that of ``master_bias``) are useless!!!
        if 'master_bias' not in locals():
            master_bias = CCDData.read(os.path.join(cal_path, 'master_bias.fits'))
        lamp_list_bias_subtracted = lamp_list_trimmed - master_bias
        # if verbose:
        #     print('CREATE DEVIATION')
        #   Create real uncertainty here!!!
        lamp_list_bias_subtracted_with_deviation = (
            lamp_list_bias_subtracted.create_deviation(
                gain=gain, readnoise=rdnoise, disregard_nan=True)
        )
#         if slit_along == 'col':
#             row_range = (0, None); col_range = (0, index)
#         else:
#             row_range = (0, index); col_range = (0, None)

#         scale = 1 / file_table[mask_lamp]['exposure'].data

#         # Ensure that the first is the short exposure
#         if scale[0] < scale[1]:
#             lamp_list_bias_subtracted_with_deviation = (
#                 lamp_list_bias_subtracted_with_deviation[::-1]
#             )
#             scale = scale[::-1]

#         concatenated_lamp = concatenate(
#             lamp_list_bias_subtracted_with_deviation, row_range, col_range, scale=None)

#         # Plot concatenated lamp
#         plot2d(
#             concatenated_lamp.data, title='concatenated lamp', show=show, save=save, 
#             path=plot_path)

#         # Write concatenated lamp to file
#         concatenated_lamp.write(
#             os.path.join(fits_path, 'concatenated_lamp.fits'), overwrite=True)
    
#     if ('fitcoords' in steps) or ('lamp' in steps):
#         if verbose:
#             print('FIT COORDINATES')

#         U, _ = fitcoords(
#             ccd=concatenated_lamp, slit_along=slit_along, order=1, n_med=15, n_piece=3, 
#             prominence=1e-3, n_iter=3, sigma_lower=3, sigma_upper=3, grow=False, 
#             use_mask=False, show=show, save=save, path=plot_path, height=0, threshold=0, 
#             distance=5, width=5, wlen=15, rel_height=1, plateau_size=1)

#         X, Y = invertCoordinateMap(slit_along, U)
        
        
#         if verbose: print('DISTORTION CORRECTION')

#         # Lamp
#         transformed_lamp = transform(
#             ccd=concatenated_lamp, X=X, Y=Y)

#         # Plot transformed lamp
#         plot2d(
#             transformed_lamp.data, title='transformed lamp', show=show, save=save, 
#             path=plot_path)

#         # Write transformed lamp to file
#         transformed_lamp.write(
#             os.path.join(fits_path, 'transformed_lamp.fits'), overwrite=True) # [3]






    
#     # ==================================================================================
#     # Load data
#     # ==================================================================================
#     file_list = sorted(glob(os.path.join(data_dir, '*.fit')))
#     ccddatalist = CCDDataList.read(file_list=file_list, hdu=hdu)

#     # ==================================================================================
#     # Organize data
#     # ==================================================================================
#     file_table = genFileTable(ccddatalist, save_dir, verbose)

#     mask_lamp = file_table['file_type'] == 'SPECLLAMP'
#     mask_slit = file_table['file_type'] == 'OTHER'
#     mask_spec = file_table['file_type'] == 'SPECLTARGET'

#     lamp_list = ccddatalist[mask_lamp]
#     slit_list = ccddatalist[mask_slit]
#     spec_list = ccddatalist[mask_spec]

#     # ==================================================================================
#     # Trim
#     # ==================================================================================
#     spec_list_trimmed = spec_list.trim(row_range=row_range, col_range=col_range)

#     # ==================================================================================
#     # Bias subtraction
#     # ==================================================================================
#     # !!! Uncertainties assigned here (equal to that of ``master_bias``) are useless !!!
#     spec_list_bias_subtracted = spec_list_trimmed - master_bias

#     # ==================================================================================
#     # Create deviation
#     # ==================================================================================
#     if verbose: print('CREATE DEVIATION')

#     # !!! generate real uncertainty here!!!
#     lamp_list_bias_subtracted_with_deviation = (
#         lamp_list_bias_subtracted.create_deviation(
#             gain=gain, readnoise=rdnoise, disregard_nan=True)
#     )
#     spec_list_bias_subtracted_with_deviation = (
#         spec_list_bias_subtracted.create_deviation(
#             gain=gain, readnoise=rdnoise, disregard_nan=True)
#     )

#     # ==================================================================================
#     # Flat-fielding
#     # ==================================================================================
#     if verbose: print('FLAT-FIELDING')

#     spec_list_flat_corrected = (
#         spec_list_bias_subtracted_with_deviation / normalized_flat
#     )

#     # ==================================================================================
#     # Cosmic ray removal
#     # ==================================================================================
#     if verbose: print('COSMIC RAY REMOVAL')

#     spec_list_cosmicray_corrected = spec_list_flat_corrected.cosmicray(
#         method='Laplacian', use_mask=False, gain=gain, readnoise=rdnoise, sigclip=4.5, 
#         sigfrac=0.3, objlim=1, niter=5, verbose=True)

#     # ==================================================================================
#     # Concatenate lamp
#     # ==================================================================================
#     if verbose: print('CONCATENATE LAMP')

#     if slit_along == 'col':
#         row_range = (0, None); col_range = (0, index)
#     else:
#         row_range = (0, index); col_range = (0, None)

#     scale = 1 / file_table[mask_lamp]['exposure'].data

#     # Ensure that the first is the short exposure
#     if scale[0] < scale[1]:
#         lamp_list_bias_subtracted_with_deviation = (
#             lamp_list_bias_subtracted_with_deviation[::-1]
#         )
#         scale = scale[::-1]

#     concatenated_lamp = concatenate(
#         lamp_list_bias_subtracted_with_deviation, row_range, col_range, scale=None)

#     # Plot concatenated lamp
#     plot2d(
#         concatenated_lamp.data, title='concatenated lamp', show=show, save=save, 
#         path=plot_path)

#     # Write concatenated lamp to file
#     concatenated_lamp.write(
#         os.path.join(fits_path, 'concatenated_lamp.fits'), overwrite=True)  # [3]

#     # ==================================================================================
#     # Fit coordinates
#     # ==================================================================================
#     if verbose: print('FIT COORDINATES')

#     U, _ = fitcoords(
#         ccd=concatenated_lamp, slit_along=slit_along, order=1, n_med=15, n_piece=3, 
#         prominence=1e-3, n_iter=3, sigma_lower=3, sigma_upper=3, grow=False, 
#         use_mask=False, show=show, save=save, path=plot_path, height=0, threshold=0, 
#         distance=5, width=5, wlen=15, rel_height=1, plateau_size=1)

#     X, Y = invertCoordinateMap(slit_along, U)

#     # ==================================================================================
#     # Distortion correction
#     # ==================================================================================
#     if verbose: print('DISTORTION CORRECTION')

#     # Lamp
#     transformed_lamp = transform(
#         ccd=concatenated_lamp, X=X, Y=Y)
    
#     # Plot transformed lamp
#     plot2d(
#         transformed_lamp.data, title='transformed lamp', show=show, save=save, 
#         path=plot_path)
    
#     # Write transformed lamp to file
#     transformed_lamp.write(
#         os.path.join(fits_path, 'transformed_lamp.fits'), overwrite=True) # [3]

#     # Spectra
#     for i, spec_cosmicray_corrected in enumerate(spec_list_cosmicray_corrected):
        
#         old_name = spec_cosmicray_corrected.header['FILENAME']
#         new_name = '{}_corrected.fits'.format(old_name[:-4])
        
#         print('{} -> {}'.format(old_name, new_name))
        
#         transformed_spec = transform(
#             ccd=spec_cosmicray_corrected, X=X, Y=Y)

#         # Plot transformed spectrum
#         plot2d(
#             transformed_spec.data, title=old_name, show=show, save=save, 
#             path=plot_path)

#         # Write transformed spectrum to file
#         transformed_spec.write(
#             os.path.join(fits_path, new_name), overwrite=True) # [3]

#     # ==================================================================================
#     # Wavelength calibration
#     # ==================================================================================
#     if verbose: print('WAVELENGTH CALIBRATION')

#     lamp1d = extract(
#         ccd=transformed_lamp, slit_along=slit_along, trace1d=750, n_aper=1, 
#         aper_width=10, show=show, save=save, path=plot_path)
    
#     calibrated_lamp1d = dispcor(
#         spectrum1d=lamp1d, reverse=True, file_name=file_name, n_piece=3, refit=True, 
#         n_iter=5, sigma_lower=3, sigma_upper=3, grow=False, use_mask=True, show=show, 
#         save=save, path=plot_path)
    
#     # Plot calibrated arc spectrum
#     plotSpectrum1D(
#         spectrum1d=calibrated_lamp1d, title='calibrated lamp', show=show, save=save, 
#         path=plot_path)
    
#     # Write calibrated arc spectrum to file
#     calibrated_lamp1d.write(
#         os.path.join(fits_path, 'lamp1d.fits'), overwrite=True) # [3]


def main():
    """Command line tool."""
    

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--data_dir', default=None, 
        help='Name of the input file'
    )
    parser.add_argument(
        '-s', '--save_dir', default=None, 
        help='Format of the input file (`ascii`, `ecsv`, or `fits`).'
    )
    parser.add_argument(
        '-m', '--mode', default=False, 
        help='General or standard.'
    )

    # External parameters
    save_dir = '/data3/zrn/workspace/data_reduction/20200406'
    data_dir = '/data3/zrn/data/216/LRS/20200406'
    
    # Internal parameters
    hdu = 0
    slit_along = 'col'
    col_range = (0, 1950)
    row_range = (329, 1830)
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
    index = 665
    file_name = 'bfosc_slit23_g4.fits'
    steps = ['header', 'bias', 'lamp']
    keywords = ['imagetyp', 'object', 'exptime']
    mem_limit = 500e6

    pipeline(
        save_dir=save_dir, data_dir=data_dir, hdu=hdu, row_range=row_range, 
        col_range=col_range, slit_along=slit_along, 
        re_n_piece=response_dict['n_piece'], il_bins=illumination_dict['bins'], 
        il_n_piece=illumination_dict['n_piece'], rdnoise=rdnoise, 
        gain=gain, index=index, file_name=file_name, steps=steps, keywords=keywords,  mem_limit=mem_limit, show=False, save=True, 
        verbose=True)

if __name__ == '__main__':
    main()