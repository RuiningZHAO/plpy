"""
Pipeline for LJT/HRS
"""

import os, argparse, toml, warnings
from glob import glob
from copy import deepcopy

# NumPy
import numpy as np
# matplotlib
import matplotlib.pyplot as plt
# AstroPy
import astropy.units as u
# from astropy.io import fits
from astropy.stats import mad_std
from astropy.nddata import CCDData
from astropy.config import reload_config
from astropy.utils.exceptions import AstropyUserWarning
# ccdproc
from ccdproc import ImageFileCollection#, cosmicray_lacosmic
# # specutils
# from specutils import Spectrum1D
# drpy
from drpy.batch import CCDDataList
# from drpy.image import concatenate
from drpy.utils import imstatistics
from drpy.plotting import plot2d#, plotSpectrum1D
from drpy.twodspec import extract
# from drpy.twodspec.utils import invertCoordinateMap
from drpy.onedspec import loadSpectrum1D


from .. import conf
from ..utils import login, getIndex#, getMask

from .utils import getOrderInfo, traceEchelle, backgroundEchelle

# Load parameters from configuration file
reload_config(packageormod='plpy', rootname='plpy')


def pipeline(save_dir, data_dir, shouldCombine, keyword, verbose):
    """LJT/HRS pipeline."""

# #     # Custom mask
# #     path_to_semester = os.path.join(conf.path_to_library, semester)
# #     if not os.path.exists(path_to_semester):
# #         raise ValueError('Semester not found.')

# #     path_to_region = os.path.join(
# #         path_to_semester, f'bfosc_{grism}_slit{slit_width}_{semester}.reg')
# #     custom_mask = getMask(path_to_region=path_to_region, shape=conf.shape)

#     if not reference:
#         reference = sorted(glob(
#             os.path.join(
#                 conf.path_to_library, f'yfosc_{grism}_slit{slit_width}*.fits')))[-1]
#     else:
#         reference = os.path.abspath(reference)
#     if not os.path.exists(reference):
#         raise ValueError('Reference not found.')

    # Login message
    if verbose:
        login(f'LJT/HRS', 100)
    
    # Change working directory
    if verbose:
        print(f'  - Changing working directory to {save_dir}...')
    os.chdir(save_dir)
    
    # Check setup
    for directory in ['cal', 'fig', 'red']:
        if not os.path.isdir(directory):
            raise RuntimeError(f'Run command `plpy_setup` first to set up {save_dir}.')
    if not os.path.isfile('params.toml'):
        raise RuntimeError(f'Run command `plpy_setup` first to set up {save_dir}.')

    # Load inputs
    params = toml.load('params.toml')

    # Construct image file collection
    ifc = ImageFileCollection(
        location=data_dir, keywords=params['keywords'], find_fits_by_reading=False, 
        filenames=None, glob_include=params['include'], glob_exclude=params['exclude'], 
        ext=params['hdu'])

    if verbose:
        print('\n[OVERVIEW]')
        ifc.summary.pprint_all()

    # Load gain and readout noise
    gain = params['gain'] * u.photon / u.adu
    rdnoise = params['rdnoise'] * u.photon
#     first_file = ifc.files_filtered(include_path=True)[0]
#     gain = fits.getval(first_file, 'GAIN', ext=params['hdu']) * u.photon / u.adu
#     rdnoise = fits.getval(first_file, 'RDNOISE', ext=params['hdu']) * u.photon
    
    if 'trim' in params['steps']:
        # custom_mask = custom_mask[
        #     slice_from_string(conf.fits_section, fits_convention=True)
        # ]
        trim = True
    else:
        trim = False

    # Bias combination
    if 'bias' in params['steps']:
        
        if verbose:
            print('\n[BIAS COMBINATION]')
        
        # Load bias
        if verbose:
            print('  - Loading bias...')
        ifc_bias = ifc.filter(regex_match=True, obstype='DARK')
        bias_list = CCDDataList.read(
            file_list=ifc_bias.files_filtered(include_path=True), hdu=params['hdu'])

        # Trim
        if trim:
            if verbose:
                print('  - Trimming...')
            bias_list = bias_list.trim_image(fits_section=params['fits_section'])
        
        # Correct gain
        if verbose:
            print('  - Correcting gain...')
        bias_list_gain_corrected = bias_list.gain_correct(gain=gain)
        
        bias_list_gain_corrected.statistics(verbose=verbose)
        
        # Combine bias
        if verbose:
            print('  - Combining...')
        bias_combined = bias_list_gain_corrected.combine(
            method='average', mem_limit=conf.mem_limit, sigma_clip=True, 
            sigma_clip_low_thresh=3, sigma_clip_high_thresh=3, 
            sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std, 
            output_file='cal/bias_combined.fits', dtype=conf.dtype, 
            overwrite_output=True)
        
        imstatistics(bias_combined, verbose=verbose)
        
        # Plot combined bias
        plot2d(
            bias_combined.data, title='bias combined', show=conf.show, save=conf.save, 
            path='fig')
        
        # Release memory
        del bias_list

    # Flat combination
    if ('flat.combine' in params['steps']) or ('flat' in params['steps']):
        
        if verbose:
            print('\n[FLAT COMBINATION]')
        
        # Load flat
        if verbose:
            print('  - Loading flat...')
        ifc_flat = ifc.filter(regex_match=True, obstype='FLAT')
        flat_list = CCDDataList.read(
            file_list=ifc_flat.files_filtered(include_path=True), hdu=params['hdu'])

        # Trim
        if trim:
            if verbose:
                print('  - Trimming...')
            flat_list = flat_list.trim_image(fits_section=params['fits_section'])
        
        # Correct gain
        if verbose:
            print('  - Correcting gain...')
        flat_list_gain_corrected = flat_list.gain_correct(gain=gain)
        
        flat_list_gain_corrected.statistics(verbose=verbose)

        # Subtract bias
        #   Uncertainties created here (equal to that of ``bias_combined``) are useless!!!
        if verbose:
            print('  - Subtracting bias...')
        if 'bias_combined' not in locals():
            bias_combined = CCDData.read('cal/bias_combined.fits')
        flat_list_bias_subtracted = flat_list_gain_corrected.subtract_bias(bias_combined)
        
        flat_list_bias_subtracted.statistics(verbose=verbose)

        # Group
        ifc_flat_summary = ifc_flat.summary
        ifc_flat_summary_grouped = ifc_flat_summary.group_by(params['exposure'])
        keys = np.sort(ifc_flat_summary_grouped.groups.keys[params['exposure']].data)

        n_flat = keys.shape[0]
        if n_flat == 1:
            flat_names = ['flat']
        elif n_flat == 2:
            flat_names = ['rflat', 'bflat']
        elif n_flat == 3:
            flat_names = ['rflat', 'gflat', 'bflat']
        else:
            raise RuntimeError(
                f'Detected {n_flat} flat-field groups with different exposure times! '
                'Only up to 3 are allowed.'
            )

        if verbose:
            print('  - Grouping')
            print(f'    - {keys.shape[0]} groups: ' + ', '.join(keys.astype(str)))

        flat_combined_list = list()

        for i, (key, flat_name) in enumerate(zip(keys, flat_names)):

            # Combine flat
            #   Uncertainties created above are overwritten here!!!
            if verbose:
                print(
                    f'  - Combining group {key} ({(i + 1)}/{keys.shape[0]})...')
            mask = ifc_flat_summary[params['exposure']].data == key

            flat_combined = flat_list_bias_subtracted[mask].combine(
                method='average', mem_limit=conf.mem_limit, sigma_clip=True, 
                sigma_clip_low_thresh=3, sigma_clip_high_thresh=3, 
                sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std, 
                output_file=f'cal/{flat_name}_combined.fits', dtype=conf.dtype, 
                overwrite_output=True)

            # Plot combined flat
            plot2d(
                flat_combined.data, title=f'{flat_name} combined', show=conf.show, 
                save=conf.save, path='fig')

            flat_combined_list.append(flat_combined)

        flat_combined_list = CCDDataList(flat_combined_list)

        flat_combined_list.statistics(verbose=verbose)

        # Release memory
        del flat_list, flat_list_bias_subtracted

    # Lamp combination
    if 'lamp' in params['steps']:

        if verbose:
            print('\n[LAMP COMBINATION]')
        
        # Load flat
        if verbose:
            print('  - Loading lamp...')
        ifc_lamp = ifc.filter(regex_match=True, obstype='LAMP')
        lamp_list = CCDDataList.read(
            file_list=ifc_lamp.files_filtered(include_path=True), hdu=params['hdu'])

        # Trim
        if trim:
            if verbose:
                print('  - Trimming...')
            lamp_list = lamp_list.trim_image(fits_section=params['fits_section'])
        
        # Correct gain
        if verbose:
            print('  - Correcting gain...')
        lamp_list_gain_corrected = lamp_list.gain_correct(gain=gain)
        
        lamp_list_gain_corrected.statistics(verbose=verbose)

        # Subtract bias
        #   Uncertainties created here (equal to that of ``bias_combined``) are useless!!!
        if verbose:
            print('  - Subtracting bias...')
        if 'bias_combined' not in locals():
            bias_combined = CCDData.read('cal/bias_combined.fits')
        lamp_list_bias_subtracted = lamp_list_gain_corrected.subtract_bias(bias_combined)
        
        lamp_list_bias_subtracted.statistics(verbose=verbose)

        # Group
        ifc_lamp_summary = ifc_lamp.summary
        ifc_lamp_summary_grouped = ifc_lamp_summary.group_by(params['exposure'])
        keys = np.sort(ifc_lamp_summary_grouped.groups.keys[params['exposure']].data)

        n_lamp = keys.shape[0]
        if n_lamp == 1:
            lamp_names = ['lamp']
        else:
            n = int(np.log10(n_lamp)) + 1
            lamp_names = [f'lamp_{(i + 1):0{n}d}' for i in range(n_lamp)]

        if verbose:
            print('  - Grouping')
            print(f'    - {keys.shape[0]} groups: ' + ', '.join(keys.astype(str)))

        lamp_combined_list = list()

        for i, (key, lamp_name) in enumerate(zip(keys, lamp_names)):

            # Combine lamp
            #   Uncertainties created above are overwritten here!!!
            if verbose:
                print(
                    f'  - Combining group {key} ({(i + 1)}/{keys.shape[0]})...')
            mask = ifc_lamp_summary[params['exposure']].data == key

            lamp_combined = lamp_list_bias_subtracted[mask].combine(
                method='average', mem_limit=conf.mem_limit, sigma_clip=True, 
                sigma_clip_low_thresh=3, sigma_clip_high_thresh=3, 
                sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std, 
                output_file=f'cal/{lamp_name}_combined.fits', dtype=conf.dtype, 
                overwrite_output=True)

            # Plot combined flat
            plot2d(
                lamp_combined.data, title=f'{lamp_name} combined', show=conf.show, 
                save=conf.save, path='fig')

            lamp_combined_list.append(lamp_combined)

        lamp_combined_list = CCDDataList(lamp_combined_list)

        lamp_combined_list.statistics(verbose=verbose)

        # Release memory
        del lamp_list, lamp_list_bias_subtracted

    # Target combination
    if 'targ' in params['steps']:

        if verbose:
            print('\n[TARGET COMBINATION]')
        
        # Load target
        if verbose:
            print('  - Loading targ...')
        ifc_targ = ifc.filter(regex_match=True, obstype='OBJECT')
        targ_list = CCDDataList.read(
            file_list=ifc_targ.files_filtered(include_path=True), hdu=params['hdu'])

        # Trim
        if trim:
            if verbose:
                print('  - Trimming...')
            targ_list = targ_list.trim_image(fits_section=params['fits_section'])
        
        # Correct gain
        if verbose:
            print('  - Correcting gain...')
        targ_list_gain_corrected = targ_list.gain_correct(gain=gain)
        
        targ_list_gain_corrected.statistics(verbose=verbose)

        # Subtract bias
        #   Uncertainties created here (equal to that of ``bias_combined``) are useless!!!
        if verbose:
            print('  - Subtracting bias...')
        if 'bias_combined' not in locals():
            bias_combined = CCDData.read('cal/bias_combined.fits')
        targ_list_bias_subtracted = targ_list_gain_corrected.subtract_bias(bias_combined)
        
        targ_list_bias_subtracted.statistics(verbose=verbose)

        # Create real uncertainty!!!
        if verbose:
            print('  - Creating deviation...')
        targ_list_with_deviation = (
            targ_list_bias_subtracted.create_deviation(
                gain=None, readnoise=rdnoise, disregard_nan=True)
        )

        # Group
        ifc_targ_summary = ifc_targ.summary
        ifc_targ_summary_grouped = ifc_targ_summary.group_by(keyword)
        keys = np.sort(ifc_targ_summary_grouped.groups.keys[keyword].data)

        n_targ = keys.shape[0]

        if verbose:
            print('  - Grouping')
            print(f'    - {keys.shape[0]} groups: ' + ', '.join(keys))

        key_list = list()
        targ_combined_list = list()

        for i, key in enumerate(keys):

            # Combine target
            #   Uncertainties created above are overwritten here!!!
            if verbose:
                print(
                    f'  - Combining group {key} ({(i + 1)}/{n_targ})...')
            mask = ifc_targ_summary[keyword].data == key

            if shouldCombine:

#                 if mask.sum() >= 3:

#                     # Skip cosmic ray removal
#                     targ_list_cosmicray_corrected = targ_list_with_deviation[mask]

#                 else:

#                     # Remove cosmic ray
#                     if verbose:
#                         print('    - Removing cosmic ray...')
#                     targ_list_cosmicray_corrected = (
#                         targ_list_with_deviation[mask].cosmicray_lacosmic(
#                             use_mask=False, gain=(1 * u.dimensionless_unscaled), 
#                             readnoise=rdnoise, sigclip=4.5, sigfrac=0.3, objlim=1, 
#                             niter=5, verbose=True)
#                     )

                if mask.sum() > 1:

                    # Combine
                    if verbose:
                        print(f'    - Combining ({mask.sum()})...')
                    exptime = ifc_targ_summary[params['exposure']].data[mask]
                    scale = exptime.max() / exptime
                    targ_combined = targ_list_with_deviation[mask].combine(
                        method='average', scale=scale, mem_limit=conf.mem_limit, 
                        sigma_clip=True, sigma_clip_low_thresh=3, 
                        sigma_clip_high_thresh=3, sigma_clip_func=np.ma.median, 
                        sigma_clip_dev_func=mad_std, 
                        output_file=f'red/{key}_combined.fits', dtype=conf.dtype, 
                        overwrite_output=True)

                else:

                    targ_combined = targ_list_with_deviation[mask][0]
                    targ_combined.write(f'red/{key}_combined.fits', overwrite=True)

                if verbose:
                    print(f'    - Saving combined {key} to red/...')

                # Plot
                plot2d(
                    targ_combined.data, title=f'{key} combined', show=conf.show, 
                    save=conf.save, path='fig')
                
                key_list.append(key)
                targ_combined_list.append(targ_combined)

            else:

                # # Remove cosmic ray
                # if verbose:
                #     print('  - Removing cosmic ray...')
                # targ_list_cosmicray_corrected = (
                #     targ_list_with_deviation[mask].cosmicray_lacosmic(
                #         use_mask=False, gain=(1 * u.dimensionless_unscaled), 
                #         readnoise=rdnoise, sigclip=4.5, sigfrac=0.3, objlim=1, 
                #         niter=5, verbose=True)
                # )

                n = int(np.log10(mask.sum())) + 1

                for j, targ in enumerate(targ_list_with_deviation[mask]):

                    key_with_number = f'{key}_{(j + 1):0{n}d}'

                    # Write transformed spectrum to file
                    if verbose:
                        print(f'  - Saving corrected {key_with_number} to red/...')
                    targ.write(f'red/{key_with_number}_corrected.fits', overwrite=True)

                    # Plot
                    plot2d(
                        targ.data, title=f'{key_with_number} corrected', 
                        show=conf.show, save=conf.save, path='fig')

                    key_list.append(key_with_number)
                    targ_combined_list.append(targ)

        targ_combined_list = CCDDataList(targ_combined_list)

        # Release memory
        del (targ_list, targ_list_bias_subtracted, targ_list_with_deviation)

    if 'trace' in params['steps']:

        if verbose:
            print('\n[TRACE]')

        if 'flat_combined' in locals():
            flat_name = flat_names[params['trace']['flat_index']]
            flat_combined = deepcopy(flat_combined_list[params['trace']['flat_index']])

        else:
            n_flat = len(glob(f'cal/*flat_combined.fits'))
            if n_flat == 1:
                flat_name = 'flat'
            elif n_flat == 2:
                flat_name = 'rb'[params['trace']['flat_index']] + 'flat'
            elif n_flat == 3:
                flat_name = 'rgb'[params['trace']['flat_index']] + 'flat'
            else:
                raise RuntimeError(
                    f'Detected {n_flat} combined flat-fields. Only up to 3 are allowed.'
                )
            flat_combined = CCDData.read(f'cal/{flat_name}_combined.fits')

        orders, intervals = getOrderInfo(flat_combined)

        if verbose:
            print(f'  - Tracing flat-field ({flat_name})...')

        trace1d = traceEchelle(
            ccd=flat_combined, dispersion_axis=params['dispersion_axis'], 
            orders=orders, intervals=intervals, method='gaussian', fwhm=15, n_med=20, 
            order_ref_red=params['trace']['order_ref_red'], 
            order_ref_blue=params['trace']['order_ref_blue'], 
            range_red=params['trace']['range_red'], 
            range_blue=params['trace']['range_blue'], order=3, 
            n_piece=params['trace']['n_piece'], degree=params['trace']['degree'], 
            maxiters=5, sigma_lower=2, sigma_upper=2, grow=False, negative=False, 
            use_mask=False, title='order', show=False, save=True, path='fig')

        # Write
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=AstropyUserWarning)
            trace1d.write('cal/trace1d.fits', format='tabular-fits', overwrite=True)

    if 'background' in params['steps']:

        if verbose:
            print(f'\n[BACKGROUND]')

        # Load traces
        if 'trace1d' not in locals():

            if verbose:
                print('  - Loading traces...')

            trace1d = loadSpectrum1D('cal/trace1d.fits')

        # Load target
        if 'targ_combined_list' not in locals():

            if verbose:
                print('  - Loading targets...')

            if shouldCombine:
                file_list = sorted(glob('red/*_combined.fits'))
                key_list = [item[4:-14] for item in file_list]
            else:
                file_list = sorted(glob('red/*_corrected.fits'))
                key_list = [item[4:-15] for item in file_list]

            targ_combined_list = CCDDataList.read(file_list=file_list, hdu=0, unit=None)

        targ_background_subtracted_list = list()

        for key, targ in zip(key_list, targ_combined_list):

            if verbose:
                print(f'  - Modeling background of {key}...')

            background2d = backgroundEchelle(
                targ, dispersion_axis=params['dispersion_axis'], trace1d=trace1d, 
                mask_width=16, title=key, save=conf.save, path='fig')

            # Plot background
            plot2d(
                background2d.data, title=f'background2d {key}', show=conf.show, 
                save=conf.save, path='fig')

            # Write background to file
            background2d.write(f'red/background2d_{key}.fits', overwrite=True)

            if verbose:
                print(f'  - Subtracting background from {key}...')

            # Subtract background from target
            targ_background_subtracted = targ.subtract(
                background2d, handle_meta='first_found')
            
            # Plot background subtracted target
            plot2d(
                targ_background_subtracted.data, title=f'{key} background subtracted', 
                show=conf.show, save=conf.save, path='fig')

            # Write background subtracted target to file
            targ_background_subtracted.write(
                f'red/{key}_background_subtracted.fits', overwrite=True)

            targ_background_subtracted_list.append(targ_background_subtracted)

        targ_background_subtracted_list = CCDDataList(targ_background_subtracted_list)

    if 'extract' in params['steps']:

        if verbose:
            print(f'\n[EXTRACT]')

        indices = getIndex(params['extract']['orders'])

        if 'orders' not in locals():
            orders, _ = getOrderInfo()

        # Load traces
        if 'trace1d' not in locals():

            if verbose:
                print('  - Loading traces...')

            trace1d = loadSpectrum1D('cal/trace1d.fits')

        # Load target
        if 'targ_background_subtracted_list' not in locals():

            if verbose:
                print('  - Loading targets...')

            file_list = sorted(glob('red/*_background_subtracted.fits'))
            key_list = [item[4:-27] for item in file_list]

            targ_background_subtracted_list = CCDDataList.read(
                file_list=file_list, hdu=0, unit=None)

        # Load lamp
        if 'lamp_combined_list' not in locals():

            if verbose:
                print('  - Loading lamps...')

            file_list = sorted(glob('cal/lamp*.fits'))
            lamp_combined_list = CCDDataList.read(file_list=file_list, hdu=0, unit=None)

        for order_num in indices:

            i = np.where(orders == order_num)[0][0]

            if verbose:
                print(f'  - Extracting order #{order_num}...')
                print(f'    - Extracting lamp...')

            # todo: choose exposure according to order number
            lamp = lamp_combined_list[0]

            lamp1d = extract(
                ccd=lamp, dispersion_axis=params['dispersion_axis'], method='sum', 
                trace1d=trace1d[i], aper_width=params['extract']['aper_width'], 
                n_aper=1, show=False, save=False, path=None)

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=AstropyUserWarning)
                lamp1d.write(
                    f'red/lamp1d.o{order_num:03d}.fits', format='tabular-fits', 
                    overwrite=True)

            for key, targ in zip(key_list, targ_background_subtracted_list):

                if verbose:
                    print(f'    - Extracting {key}...')

                target1d = extract(
                    ccd=targ, dispersion_axis=params['dispersion_axis'], method='sum', 
                    trace1d=trace1d[i], aper_width=params['extract']['aper_width'], 
                    n_aper=1, show=False, save=False, path=None)

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=AstropyUserWarning)
                    target1d.write(
                        f'red/spec1d.{key.lower()}.o{order_num:03d}.fits', 
                        format='tabular-fits', overwrite=True)


def main():
    """Command line tool."""
    
    # External parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input_dir', required=True, type=str, 
        help='Input (data) directory.'
    )
    parser.add_argument(
        '-o', '--output_dir', default='', type=str, 
        help='Output (saving) directory.'
    )
    # parser.add_argument(
    #     '-r', '--reference', default=None, type=str, 
    #     help='Reference spectrum for wavelength calibration.'
    # )
    # parser.add_argument(
    #     '-s', '--standard', default=None, type=str, 
    #     help='Path to the standard spectrum in the library.'
    # )
    parser.add_argument(
        '-k', '--keyword', default='object', type=str, 
        help='Keyword for grouping.'
    )
    parser.add_argument(
        '-c', '--combine', action='store_true', 
        help='Combine or not.'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true', 
        help='Verbose or not.'
    )
        
    # Parse
    args = parser.parse_args()
    data_dir = os.path.abspath(args.input_dir)
    save_dir = os.path.abspath(args.output_dir)
    # reference = args.reference
    # standard = args.standard
    combine = args.combine
    keyword = args.keyword
    verbose = args.verbose

    pipeline(
        save_dir=save_dir, data_dir=data_dir, shouldCombine=combine, keyword=keyword, 
        verbose=verbose)