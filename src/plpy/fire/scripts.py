"""
Baade/FIRE pipeline
"""

import os, argparse, warnings, textwrap, toml
from glob import glob

# NumPy
import numpy as np
# SciPy
from scipy import ndimage
# matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# AstroPy
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
from astropy.stats import mad_std
from astropy.nddata import CCDData
from astropy.config import reload_config
from astropy.utils.exceptions import AstropyUserWarning
# ccdproc
from ccdproc import ImageFileCollection, trim_image#, cosmicray_lacosmic
from ccdproc.utils.slices import slice_from_string
# specutils
from specutils import Spectrum1D
# drpy
from drpy.batch import CCDDataList
from drpy.image import concatenate
from drpy.utils import imstatistics
from drpy.plotting import plot2d
from drpy.twodspec import (response, illumination, fitcoords, transform, trace, 
                           background, extract)
from drpy.twodspec.utils import invertCoordinateMap
from drpy.onedspec import dispcor

from .utils import getSkyIndex

from .. import conf
from ..utils import login, fixHeader, fixKeyword, getFileName, getMask

# Load parameters from configuration file
reload_config(packageormod='plpy', rootname='plpy')


def _arc(lamp, slit_along, fits_section, reference, save=False, fig_path=None, 
         verbose=False):
    """Generate rectification files and perform wavelength calibration."""

    title = lamp.header['object']

    # Trim
    if verbose:
        print('- Trimming...')
    lamp = trim_image(lamp, fits_section=fits_section)

    # Fit coordinates
    if verbose:
        print('- Fitting coordinates...')

    _, V = fitcoords(
        ccd=lamp, slit_along=slit_along, order=1, n_med=5, n_piece=3, 
        prominence=1e-3, maxiters=3, sigma_lower=3, sigma_upper=3, grow=False, 
        use_mask=False, plot=save, path=fig_path, height=0, threshold=0, 
        distance=5, width=5, wlen=15, rel_height=1, plateau_size=1)

    # Invert coordinate map
    if verbose:
        print('- Inverting coordinate map...')
    X, Y = invertCoordinateMap(slit_along, V)

    lamp_transformed = transform(ccd=lamp, X=X, Y=Y)

    # Extract
    if verbose:
        print('- Extracting 1-dimensional arc spectrum...')            
    arc1d = extract(
        ccd=lamp_transformed, slit_along=slit_along, method='sum', trace1d=100, 
        aper_width=30, n_aper=1, use_uncertainty=False, use_mask=False, title=title, 
        show=False, save=save, path=fig_path)

    # Correct dispersion of lamp spectrum (float64)
    if verbose:
        print('- Correcting dispersion of arc spectrum...')      
    arc1d_calibrated = dispcor(
        spectrum1d=arc1d, reverse=False, reference=reference, n_sub=20, 
        refit=True, degree=1, prominence=1e-3, height=0, threshold=0, 
        distance=5, width=5, wlen=15, rel_height=1, plateau_size=1, maxiters=5, 
        sigma_lower=None, sigma_upper=None, grow=False, use_mask=False, 
        title=title, show=False, save=save, path=fig_path)

    return V, X, Y, lamp_transformed, arc1d_calibrated


def arc():
    """Combine arc frames, rectify distortion and perform dispersion correction."""

    # External parameters
    parser = argparse.ArgumentParser(
        prog='plpy_fire_arc',
        description=(
            'Combine arc frames, rectify distortion and perform dispersion correction.'
        )
    )
    parser.add_argument(
        '-i', '--input_dir', required=True, type=str, 
        help='Input (data) directory.'
    )
    parser.add_argument(
        '-o', '--output_dir', default='', type=str, 
        help='Output (saving) directory.'
    )
    parser.add_argument(
        '-r', '--reference', default='', type=str, 
        help='Reference spectrum for wavelength calibration.'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true', 
        help='Verbose or not.'
    )
        
    # Parse
    args = parser.parse_args()
    data_dir = os.path.abspath(args.input_dir)
    save_dir = os.path.abspath(args.output_dir)
    reference = args.reference
    verbose = args.verbose

    if verbose:
        login(instrument='Baade/FIRE longslit mode', width=100)

    # Verify reference spectrum
    if not reference:
        reference = sorted(glob(
            os.path.join(os.path.split(__file__)[0], 'lib/fire_arc*.fits')))[-1]
    else:
        reference = os.path.abspath(reference)
    if not os.path.isfile(reference):
        raise ValueError('Reference not found.')

    # Change working directory
    if verbose:
        print(f'  - Changing working directory to {save_dir}...')
    os.chdir(save_dir)
    
    # Check setup
    for directory in ['cal', 'fig', 'red', 'sub']:
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

    # Load lamp
    if verbose:
        print('- Loading lamp...')
    ifc_lamp = ifc.filter(regex_match=True, obstype='Arc')
    lamp_list = CCDDataList.read(
        file_list=ifc_lamp.files_filtered(include_path=True), hdu=params['hdu'])

    # Group
    ifc_lamp_summary = ifc_lamp.summary
    ifc_lamp_summary_grouped = ifc_lamp_summary.group_by('object')
    keys = ifc_lamp_summary_grouped.groups.keys['object'].data
    if verbose:
        print('- Grouping...')
        print(textwrap.fill(f'  - {keys.shape[0]} groups: ' + ', '.join(keys), 88))

    for i, key in enumerate(keys):

        # Combine lamp
        if verbose:
            print(
                f'- Dealing with group {key} ({(i + 1)}/{keys.shape[0]})...')
        mask = ifc_lamp_summary['object'].data == key

        if mask.sum() > 1:

            if verbose:
                print(f'- Combining ({mask.sum()})...')

            lamp_combined = lamp_list[mask].combine(
                method='average', mem_limit=conf.mem_limit, sigma_clip=True, 
                sigma_clip_low_thresh=3, sigma_clip_high_thresh=3, 
                sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std)

        else:

            lamp_combined = lamp_list[mask][0]

        if 'flat' in key.lower():

            V, X, Y, lamp_transformed, arc1d_calibrated = _arc(
                lamp=lamp_combined, slit_along=params['slit_along'], 
                fits_section=params['fits_section'], reference=reference, 
                save=conf.save, fig_path='fig', verbose=verbose)

            # Write rectification files
            np.save(f'cal/V_{key}.npy', V)
            
            # Plot transformed lamp
            plot2d(
                lamp_transformed.data, aspect='auto', cbar=False, 
                title=f'{key}_transformed', show=conf.show, save=conf.save, 
                path='fig')

        else:

            _, X, Y, lamp_transformed, arc1d_calibrated = _arc(
                lamp=lamp_combined, slit_along=params['slit_along'], 
                fits_section=params['fits_section'], reference=reference, save=False, 
                fig_path=None, verbose=verbose)

        # Write rectification files
        np.save(f'cal/X_{key}.npy', X)
        np.save(f'cal/Y_{key}.npy', Y)

        # Write arc spectrum to file (float64)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=AstropyUserWarning)
            arc1d_calibrated.write(f'cal/{key}.fits', format='tabular-fits', 
                overwrite=True)


def flat():
    """Make flat-field."""
    
    # External parameters
    parser = argparse.ArgumentParser(
        prog='plpy_fire_flat',
        description='Make flat-field.'
    )
    parser.add_argument(
        '-i', '--input_dir', required=True, type=str, 
        help='Input (data) directory.'
    )
    parser.add_argument(
        '-o', '--output_dir', default='', type=str, 
        help='Output (saving) directory.'
    )
    parser.add_argument(
        '-b', '--blue_flat', required=True, type=str, 
        help='Value of the keyword `OBJECT` in the header of blue flat.'
    )
    parser.add_argument(
        '-r', '--red_flat', required=True, type=str, 
        help='Value of the keyword `OBJECT` in the header of red flat.'
    )
    parser.add_argument(
        '-a', '--arc_flat', default=None, type=str, 
        help='Value of the keyword `OBJECT` in the header of arc frame for flat.'
    )
    parser.add_argument(
        '-n', '--n_piece', default=19, type=int, 
        help='Number of equally spaced pieces for spline3 fitting of response.'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true', 
        help='Verbose or not.'
    )
        
    # Parse
    args = parser.parse_args()
    data_dir = os.path.abspath(args.input_dir)
    save_dir = os.path.abspath(args.output_dir)
    blue_flat = args.blue_flat
    red_flat = args.red_flat
    arc_flat = args.arc_flat
    n_piece = args.n_piece
    verbose = args.verbose

    if verbose:
        login(instrument='Baade/FIRE longslit mode', width=100)

    # Change working directory
    if verbose:
        print(f'  - Changing working directory to {save_dir}...')
    os.chdir(save_dir)

    # Check setup
    for directory in ['cal', 'fig', 'red', 'sub']:
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

    # Load gain
    first_file = ifc.files_filtered(include_path=True)[0]
    gain = fits.getval(first_file, 'EGAIN', ext=params['hdu']) * u.photon / u.adu

    # Find arc for flat
    if arc_flat is None:

        if verbose:
            print('- Searching for arc frame...')

        ifc_lamp = ifc.filter(regex_match=True, obstype='Arc')
        arc_arr = np.array(sorted(set(ifc_lamp.summary['object'].value)))
        idx_flat = np.where(['flat' in item.lower() for item in arc_arr])[0]
        idx_flat = idx_flat[0] if idx_flat.size > 0 else 0
        arc_flat = arc_arr[idx_flat]

    # Custom mask
    spectral_axis = Spectrum1D.read(f'cal/{arc_flat}.fits').spectral_axis.value
    custom_mask = np.zeros_like(spectral_axis, dtype=bool)
    idx_mask = np.arange(custom_mask.shape[0])
    bands = [(13300, 14250), (17800, 19900), (21200, 22200)]
    for band in bands:
        idxmin, idxmax = np.interp(band, spectral_axis, idx_mask).astype(int)
        custom_mask[idxmin:(idxmax + 1)] = True

    # Index for concatenate
    index = np.interp(11437, spectral_axis, idx_mask).astype(int)

    # Load flats
    if verbose:
        print('- Loading flat-fields...')
    ifc_bflat = ifc.filter(regex_match=True, object=blue_flat)
    bflat_list = CCDDataList.read(
        file_list=ifc_bflat.files_filtered(include_path=True), hdu=params['hdu'])
    ifc_rflat = ifc.filter(regex_match=True, object=red_flat)
    rflat_list = CCDDataList.read(
        file_list=ifc_rflat.files_filtered(include_path=True), hdu=params['hdu'])

    # Trim
    if verbose:
        print('- Trimming...')
    bflat_list = bflat_list.trim_image(fits_section=params['fits_section'])
    rflat_list = rflat_list.trim_image(fits_section=params['fits_section'])
        
    # Correct gain
    if verbose:
        print('- Correcting gain...')
    bflat_list_gain_corrected = bflat_list.gain_correct(gain=gain)
    rflat_list_gain_corrected = rflat_list.gain_correct(gain=gain)
        
    bflat_list_gain_corrected.statistics(verbose=verbose)
    rflat_list_gain_corrected.statistics(verbose=verbose)

    scaling_func = lambda ccd: 1 / np.ma.average(ccd)

    # Combine flats
    #   Uncertainties created above are overwritten here!!!
    if verbose:
        print('- Combining flats...')
    bflat_combined = bflat_list_gain_corrected.combine(
        method='average', scale=scaling_func, mem_limit=conf.mem_limit, 
        sigma_clip=True, sigma_clip_low_thresh=3, sigma_clip_high_thresh=3, 
        sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std, 
        output_file='cal/bflat_combined.fits', dtype=conf.dtype, 
        overwrite_output=True)
    rflat_combined = rflat_list_gain_corrected.combine(
        method='average', scale=scaling_func, mem_limit=conf.mem_limit, 
        sigma_clip=True, sigma_clip_low_thresh=3, sigma_clip_high_thresh=3, 
        sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std, 
        output_file='cal/rflat_combined.fits', dtype=conf.dtype, 
        overwrite_output=True)

    imstatistics(bflat_combined, verbose=verbose)
    imstatistics(rflat_combined, verbose=verbose)
        
    # Plot combined flats
    plot2d(
        bflat_combined.data, aspect='auto', cbar=False, title='bflat combined', 
        show=conf.show, save=conf.save, path='fig')
    plot2d(
        rflat_combined.data, aspect='auto', cbar=False, title='rflat combined', 
        show=conf.show, save=conf.save, path='fig')
        
    # Release memory
    del bflat_list, bflat_list_gain_corrected
    del rflat_list, rflat_list_gain_corrected

    # Concatenate flats
    if verbose:
        print('- Concatenating...')
        
    scaling_factor = np.median(
        rflat_combined[(index - 10):(index + 10), :].divide(
        bflat_combined[(index - 10):(index + 10), :]).data
    )
        
    flat_concatenated = concatenate(
        [bflat_combined, rflat_combined], fits_section=f'[:, :{index}]', 
        scale=[scaling_factor, 1])
        
    if (flat_concatenated.data <= 0).sum() > 0:
        warnings.warn(
            'Concatenated flat-field has negative values.', RuntimeWarning)
        
    # Plot concatenated flat
    plot2d(
        flat_concatenated.data, aspect='auto', cbar=False, title='flat concatenated', 
        show=conf.show, save=conf.save, path='fig')

    # Write concatenated flat to file
    flat_concatenated.write('cal/flat_concatenated.fits', overwrite=True)

    # Rectify curvature
    if verbose:
        print('- Rectifying curvature...')
    X = np.load(f'cal/X_{arc_flat}.npy')
    Y = np.load(f'cal/Y_{arc_flat}.npy')
    flat_transformed = transform(flat_concatenated, X=X, Y=Y)

    # Apply custom mask
    flat_transformed.mask |= custom_mask[:, np.newaxis]

    # Plot transformed flat
    plot2d(
        flat_transformed.data, aspect='auto', cbar=False, title='flat transformed', 
        show=conf.show, save=conf.save, path='fig')

    # Write transformed flat to file
    flat_transformed.write('cal/flat_transformed.fits', overwrite=True)
        
    # Model response
    if verbose:
        print('- Modeling response...')
        
    V = np.load(f'cal/V_{arc_flat}.npy')
    response2d = response(
        ccd=flat_transformed, slit_along=params['slit_along'], n_piece=n_piece, 
        coordinate=V, maxiters=0, sigma_lower=None, sigma_upper=None, grow=False, 
        use_mask=True, plot=conf.save, path='fig')
        
    # Plot modeled response
    plot2d(
        response2d.data, aspect='auto', cbar=False, title='response', show=conf.show, 
        save=conf.save, path='fig')
        
    # Write modeled response to file
    response2d.write('cal/response2d.fits', overwrite=True)
        
    # Normalize
    reflat = flat_concatenated.divide(
        response2d, handle_mask='first_found', handle_meta='first_found')
        
    imstatistics(reflat, verbose=verbose)
        
    # Plot response calibrated flat
    plot2d(
        reflat.data, aspect='auto', cbar=False, title='flat normalized', 
        show=conf.show, save=conf.save, path='fig')
        
    # Write response calibrated flat to file
    reflat.write('cal/flat_normalized.fits', overwrite=True)


#     # Model illumination
#     if verbose:
#         print('\n[ILLUMINATION]')

#     # Illumination modeling
#     illumination2d = illumination(
#         ccd=reflat, slit_along=slit_along, method='Gaussian2D', sigma=sigma, 
#         bins=20, maxiters=5, sigma_lower=3, sigma_upper=3, grow=1, 
#         use_mask=True, plot=save, path='fig')

#     # Plot modeled illumination
#     plot2d(
#         illumination2d.data, aspect='auto', cbar=False, title='illumination', 
#         show=show, save=save, path='fig')

#     # Plot illumination mask
#     plot2d(
#         illumination2d.mask.astype(int), vmin=0, vmax=1, aspect='auto', cbar=False, 
#         title='illumination mask', show=show, save=save, path='fig')

#     # Write illumination to file
#     illumination2d.write('cal/illumination.fits', overwrite=True)

#     # Normalize
#     normalized_flat = reflat.divide(
#         illumination2d, handle_mask='first_found', handle_meta='first_found')

#     imstatistics(normalized_flat, verbose=verbose)

#     # Plot normalized flat
#     plot2d(
#         normalized_flat.data, aspect='auto', cbar=False, title='normalized flat', 
#         show=show, save=save, path='fig')

#     # Plot normalized flat mask
#     plot2d(
#         normalized_flat.mask.astype(int), vmin=0, vmax=1, aspect='auto', 
#         cbar=False, title='normalized flat mask', show=show, save=save, 
#         path='fig')

#     # Write normalized flat to file
#     normalized_flat.write('cal/normalized_flat.fits', overwrite=True)


def _subtract(ccd_list, ifc_summary, objname, verbose):
    """
    ifc_summary : 
        Must contain `date-obs` and `ut-time`.
    """
    
    if verbose:
        print('- Pairing...')
    timetags = Time(
        [f'{date}T{time}' for date, time in ifc_summary[['date-obs', 'ut-time']]])
    idx_sky = getSkyIndex(
        timetags=timetags, objnames=ifc_summary['object'].value)

    # Subtract
    if verbose:
        print('- Subtracting...')

    sub_list = list()

    for ccd, sky in zip(ccd_list, ccd_list[idx_sky]):
        
        if verbose:
            print(f"  - {ccd.header['FILENAME']} - {sky.header['FILENAME']}")

        sub = ccd.subtract(sky, handle_mask='first_found', handle_meta='first_found')

        sub.header.set('OBJECT', objname)
        sub.header.set('BACKNAME', sky.header['FILENAME'])
        sub.header.set(
            'COMMENT', f"{ccd.header['FILENAME']} - {sky.header['FILENAME']}")

        sub_list.append(sub)
        
    return CCDDataList(sub_list)


def subtract():
    """Subtract sky background."""

    # External parameters
    parser = argparse.ArgumentParser(
        prog='plpy_fire_subtract',
        description='Subtract sky background.'
    )
    parser.add_argument(
        '-i', '--input_dir', required=True, type=str, 
        help='Input (data) directory.'
    )
    parser.add_argument(
        '-o', '--output_dir', default='', type=str, 
        help='Output (saving) directory.'
    )
    parser.add_argument(
        '-n', '--index', required=True, type=str, 
        help='Index of frames.'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true', 
        help='Verbose or not.'
    )
        
    # Parse
    args = parser.parse_args()
    data_dir = os.path.abspath(args.input_dir)
    save_dir = os.path.abspath(args.output_dir)
    filenames = getFileName(args.index, prefix='fire_')
    verbose = args.verbose

    if verbose:
        login(instrument='Baade/FIRE longslit mode', width=100)

    # Change working directory
    if verbose:
        print(f'  - Changing working directory to {save_dir}...')
    os.chdir(save_dir)
    
    # Check setup
    for directory in ['cal', 'fig', 'red', 'sub']:
        if not os.path.isdir(directory):
            raise RuntimeError(f'Run command `plpy_setup` first to set up {save_dir}.')
    if not os.path.isfile('params.toml'):
        raise RuntimeError(f'Run command `plpy_setup` first to set up {save_dir}.')

    # Load inputs
    params = toml.load('params.toml')

    ifc = ImageFileCollection(
        location=data_dir, keywords=params['keywords'], find_fits_by_reading=False, 
        filenames=filenames, glob_include=None, glob_exclude=None, ext=params['hdu'])

    ifc_lamp = ifc.filter(regex_match=True, obstype='Arc')
    arc_arr = np.array(sorted(set(ifc_lamp.summary['object'].value)))
    idx_flat = np.where(['flat' in item.lower() for item in arc_arr])[0]
    idx_flat = idx_flat[0] if idx_flat.size > 0 else 0

    # Group
    ifc_targ = ifc.filter(regex_match=True, obstype='Science|Telluric')
    ifc_targ_summary = ifc_targ.summary
    obj_pos = ifc_targ_summary['object'].value
    obj_arr = np.array(sorted(set([item.split('_')[0] for item in obj_pos])))
    pos_arr = np.array(sorted(set([item.split('_')[1] for item in obj_pos])))
    if verbose:
        print('- Grouping')
        print(
            textwrap.fill(f'  - {obj_arr.shape[0]} groups: ' + ', '.join(obj_arr), 100)
        )

    for i, obj in enumerate(obj_arr):

        # Filter
        regex = '|'.join([f'^{obj}_{pos}$' for pos in pos_arr]).replace('+', '\+')
        ifc_obj = ifc_targ.filter(regex_match=True, object=f'{regex}')

        # Load object
        if verbose:
            print(f'- Group {(i + 1)} (out of {obj_arr.shape[0]}): {obj}')
            ifc_obj.summary.pprint_all()
        obj_list = CCDDataList.read(
            file_list=ifc_obj.files_filtered(include_path=True), hdu=params['hdu'])
    
        # Trim
        if verbose:
            print('- Trimming...')
        obj_list = obj_list.trim_image(fits_section=params['fits_section'])
        
        # Correct gain
        if verbose:
            print('- Correcting gain...')
        first_file = ifc_obj.files_filtered(include_path=True)[0]
        gain = fits.getval(first_file, 'EGAIN', ext=params['hdu']) * u.photon / u.adu
        obj_list_gain_corrected = obj_list.gain_correct(gain=gain)
        
        # Create real uncertainty!!!
        if verbose:
            print('- Creating deviation...')
        rdnoise = fits.getval(first_file, 'ENOISE', ext=params['hdu']) * u.photon
        obj_list_with_deviation = (
            obj_list_gain_corrected.create_deviation(
                gain=None, readnoise=rdnoise, disregard_nan=True)
        )

        # Subtract
        obj_list_background_subtracted = _subtract(
            ccd_list=obj_list_with_deviation, ifc_summary=ifc_obj.summary, objname=obj, 
            verbose=verbose)
        
        # Find arc
        idx_obj = np.where([obj.lower() in item.lower() for item in arc_arr])[0]
        idx_obj = idx_obj[0] if idx_obj.size > 0 else idx_flat
        arc_obj = arc_arr[idx_obj]
        
        X = np.load(f'cal/X_{arc_obj}.npy')
        Y = np.load(f'cal/Y_{arc_obj}.npy')

        # Rectify curvature
        if verbose:
            print('- Rectifying curvature...')
        obj_list_transformed = obj_list_background_subtracted.apply_over_ccd(
            transform, X=X, Y=Y)

        # Extract
        for ccd in obj_list_transformed:

            # Write to file
            filename = (
                f"sub_{ccd.header['FILENAME'][5:9]}_{ccd.header['BACKNAME'][5:9]}.fits"
            )
            ccd.header.set('FILENAME', filename)
            ccd.write(os.path.join('sub', ccd.header['FILENAME']), overwrite=True)

    return True

#         # Flat-fielding
#         if verbose:
#             print('  - Flat-fielding...')
#         if 'normalized_flat' not in locals():
#             normalized_flat = CCDData.read('cal/normalized_flat.fits'))
#         targ_list_flat_fielded = (
#             targ_list_gain_corrected_with_deviation.flat_correct(normalized_flat)
#         )
        
#         # Bad pixel mask
#         threshold = 0.4
#         badpixel_mask = normalized_flat.data <= threshold
        
#         for entry in sub_entries:
            
#             # Fix bad pixels
#             if verbose:
#                 print(f'    - Fixing bad pixels...')
            
#             median_image = ndimage.median_filter(diff.data, size=7)
#             diff.data[badpixel_mask] = (median_image[badpixel_mask])
            
#             # Remove cosmic ray
#             if diff.header['OBSTYPE'] == 'Science':
                
#                 if verbose:
#                     print(f'    - Identifying cosmic ray pixels...')
                
#                 diff_cosmicray_corrected, cr_mask = cosmicray_lacosmic(
#                     ccd=np.abs(diff.data), sigclip=4.5, sigfrac=0.3, objlim=5.0, 
#                     invar=(diff.uncertainty.array**2), gain=1.0, readnoise=6.5, 
#                     satlevel=5e4, niter=5, sepmed=True, cleantype='meanmask', 
#                     fsmode='median', verbose=verbose)
                
#                 diff_cosmicray_corrected[diff.data < 0] *= -1
#                 diff.data[cr_mask] = diff_cosmicray_corrected[cr_mask]


def trace():
    """Trace."""
    
    # External parameters
    parser = argparse.ArgumentParser(
        prog='plpy_fire_trace',
        description='Trace.'
    )
    parser.add_argument(
        '-i', '--input_dir', required=True, type=str, 
        help='Input (data) directory.'
    )
    parser.add_argument(
        '-o', '--output_dir', default='', type=str, 
        help='Output (saving) directory.'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true', 
        help='Verbose or not.'
    )

    # Parse
    args = parser.parse_args()
    data_dir = os.path.abspath(args.input_dir)
    save_dir = os.path.abspath(args.output_dir)
    verbose = args.verbose

    if verbose:
        login(instrument='Baade/FIRE longslit mode', width=100)

    # Change working directory
    if verbose:
        print(f'  - Changing working directory to {save_dir}...')
    os.chdir(save_dir)
    
    # Check setup
    for directory in ['cal', 'fig', 'red', 'sub']:
        if not os.path.isdir(directory):
            raise RuntimeError(f'Run command `plpy_setup` first to set up {save_dir}.')
    if not os.path.isfile('params.toml'):
        raise RuntimeError(f'Run command `plpy_setup` first to set up {save_dir}.')

    # Load inputs
    params = toml.load('params.toml')

    ifc = ImageFileCollection(
        location='sub', keywords=params['keywords'], find_fits_by_reading=False, 
        filenames=None, glob_include='*.fits', glob_exclude=None, ext=params['hdu'])
    
    ifc_sci = ifc.filter(regex_match=True, obstype='Science|Telluric')
    # ifc_trc = ifc.filter(regex_match=True, obstype='Telluric')
    
    if verbose:
        ifc.summary.pprint_all()
    
    file_list = ifc_sci.files_filtered(include_path=True)

    trace1d_dict = dict()
    for i, file_name in enumerate(file_list):

        short_name = 'fire_' + os.path.split(file_name)[-1].split('_')[1] + '.fits'

        if verbose:
            print(f'- Tracing #{i + 1} (out of {len(file_list)}): {short_name}...')

        # Load data
        ccd = CCDData.read(file_name)

        try:
            # We expect that an exception is raised only when there is nothing in the 
            # image, otherwise a trace should be returned even the target is faint.
            trace1d_center = trace(
                ccd=ccd, slit_along=params['slit_along'], fwhm=10, method='center', 
                interval='[:]', n_med=400, reference_bin=0, show=False, save=False)
            trace1d_dict[short_name] = trace1d_center

        except Exception as e:
            warnings.warn('No trace found.', RuntimeWarning)

    with PdfPages('fig/background_aperture.pdf', keep_empty=False) as pdf:

        for i, file_name in enumerate(file_list):

            short_name = 'fire_' + os.path.split(file_name)[-1].split('_')[1] + '.fits'

            if verbose:
                print(f'- Modeling background #{i + 1} (out of {len(file_list)}): {file_name}...')

            # Load data
            ccd = CCDData.read(file_name)

            trace1d = 1 / 2 * (
                trace1d_dict[short_name].flux + trace1d_dict[ccd.header['BACKNAME']].flux
            ).value

            seperation = np.abs(
                trace1d_dict[ccd.header['BACKNAME']].flux - trace1d_dict[short_name].flux
            ).mean().value

            location = (
                -(seperation / 2 + params['background']['distance']), 
                params['background']['offset'], 
                (seperation / 2 + params['background']['distance'])
            )
            background2d = background(
                ccd=ccd, slit_along=params['slit_along'], trace1d=trace1d, 
                location=location, aper_width=params['background']['width'], 
                degree=params['background']['order'], maxiters=3, sigma_lower=4, 
                sigma_upper=4, use_uncertainty=False, use_mask=False, grow=False, 
                show=False, save=False)

            fig, ax = plt.subplots(1, 1, figsize=(6, 3))

            ax.step(
                np.arange(ccd.shape[1]), np.nanmedian(ccd.data[:400, :], axis=0), 
                where='mid', color='k')
            ax.plot(np.nanmedian(background2d.data[:400, :], axis=0), color='r')
            
            ax.axvline(x=trace1d_dict[short_name].flux.value[0], color='r', ls='--')
            ax.axvline(
                x=(trace1d[0] + params['background']['offset'] - params['background']['width'][1] / 2), 
                color='royalblue', ls='--')
            ax.axvline(
                x=(trace1d[0] + params['background']['offset'] + params['background']['width'][1] / 2), 
                color='royalblue', ls='--')
            ax.axvline(
                x=(trace1d[0] - seperation / 2 - params['background']['distance'] - params['background']['width'][0] / 2), 
                color='royalblue', ls='--')
            ax.axvline(
                x=(trace1d[0] - seperation / 2 - params['background']['distance'] + params['background']['width'][0] / 2), 
                color='royalblue', ls='--')
            ax.axvline(
                x=(trace1d[0] + seperation / 2 + params['background']['distance'] - params['background']['width'][0] / 2), 
                color='royalblue', ls='--')
            ax.axvline(
                x=(trace1d[0] + seperation / 2 + params['background']['distance'] + params['background']['width'][0] / 2), 
                color='royalblue', ls='--')

            ax.set_xlim(0, (ccd.shape[1] - 1))
            ax.set_xlabel('column', fontsize=16)
            ax.annotate(
                os.path.split(file_name)[-1], xy=(0.05, 0.9), xycoords='axes fraction', 
                fontsize=16)

            fig.tight_layout()

            pdf.savefig(fig, dpi=100)

            plt.close()

            ccd_background_subtracted = ccd.subtract(
                background2d, handle_meta='first_found')
            
            # Write
            ccd_background_subtracted.header.set('FILENAME', short_name)
            ccd_background_subtracted.write(
                os.path.join('red', short_name), overwrite=True)


def _extract(ccd, arc1d, slit_along, interval, location, aper_width, degree, 
             save=False, fig_path=None, verbose=None):
    """Extract."""

    # Trace
    if verbose:
        print(f"- Tracing {ccd.header['FILENAME']}")
    trace1d = trace(
        ccd=ccd, slit_along=slit_along, fwhm=10, method='center', interval=interval, 
        n_med=10, n_piece=5, maxiters=5, sigma_lower=2, sigma_upper=2, grow=False, 
        title=ccd.header['FILENAME'], show=False, save=save, path=fig_path)

#     # Model sky background
#     background2d = background(
#         ccd=ccd, slit_along=slit_along, trace1d=trace1d, location=location, 
#         aper_width=aper_width, degree=degree, maxiters=3, sigma_lower=4, sigma_upper=4, 
#         grow=False, use_uncertainty=False, use_mask=True, title=ccd.header['FILENAME'], 
#         show=False, save=save, path=fig_path)

#     spec1d = extract(
#         ccd=ccd, slit_along=slit_along, method='sum', trace1d=trace1d, 
#         aper_width=aper_width, n_aper=1, title=ccd.header['FILENAME'], show=False, 
#         save=save, path=fig_path)
    
#     return spec1d
    
    
def quick():
    """A quick look of FIRE data."""

    # External parameters
    parser = argparse.ArgumentParser(
        prog='plpy_fire_quick',
        description='A quick look of FIRE data.'
    )
    parser.add_argument(
        '-i', '--input_dir', required=True, type=str, 
        help='Input (data) directory.'
    )
    parser.add_argument(
        '-o', '--output_dir', default='', type=str, 
        help='Output (saving) directory.'
    )
    parser.add_argument(
        '-n', '--index', required=True, type=str, 
        help='Index of frames.'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true', 
        help='Verbose or not.'
    )
        
    # Parse
    args = parser.parse_args()
    data_dir = os.path.abspath(args.input_dir)
    save_dir = os.path.abspath(args.output_dir)
    filenames = getFileName(args.index, prefix='fire_')
    verbose = args.verbose

    if verbose:
        login(instrument='Baade/FIRE longslit mode', width=100)

    ifc = ImageFileCollection(
        location=data_dir, keywords=params['keywords'], find_fits_by_reading=False, 
        filenames=filenames, glob_include=None, glob_exclude=None, ext=params['hdu'])

    ifc_sci = ifc.filter(regex_match=True, obstype='Science')
    ifc_trc = ifc.filter(regex_match=True, obstype='Telluric')
    ifc_arc = ifc.filter(regex_match=True, obstype='Arc')
    
    # Verify
    assert ifc_sci.summary is not None, 'No `Science` frame found.'
    assert ifc_trc.summary is not None, 'No `Telluric` frame found.'
    assert ifc_arc.summary is not None, 'No `Arc` frame found.'

    # Arc
    if verbose:
        print('\n[ARC]')
    ifc_arc.summary.pprint_all()
    lamp = CCDDataList.read(
        file_list=ifc_arc.files_filtered(include_path=True), hdu=params['hdu'])[0]
    
    # reference spectrum
    reference = sorted(glob(
        os.path.join(os.path.split(__file__)[0], 'lib/fire_arc*.fits')))[-1]
    
    # Calibrate
    _, X, Y, _, arc1d_calibrated = _arc(
        lamp=lamp, slit_along=params['slit_along'], fits_section=params['fits_section'], 
        reference=reference, save=False, fig_path=None, verbose=verbose)

    # Science
    if verbose:
        print('\n[SCIENCE]')
    ifc_sci.summary.pprint_all()
    
    # Subtract sky background
    sci_list_background_subtracted = _subtract(ifc=ifc_sci, verbose=verbose)

    # Rectify curvature
    if verbose:
        print('- Rectifying curvature...')
    sci_list_transformed = sci_list_background_subtracted.apply_over_ccd(
        transform, X=X, Y=Y)

    # Extract
    for ccd in sci_list_transformed:
        
        # Write to file before extraction
        filename = (
            f"sub_{ccd.header['FILENAME'][5:9]}_{ccd.header['BACKNAME'][5:9]}.fits"
        )
        ccd.header.set('FILENAME', filename)
        ccd.write(os.path.join(save_dir, ccd.header['FILENAME']), overwrite=True)

        # _extract(ccd=ccd, arc1d=arc1d_calibrated, verbose=verbose)

#     # Telluric
#     if verbose:
#         print('\n[TELLURIC]')
#     ifc_trc.summary.pprint_all()

#     # Subtract sky background
#     trc_list_background_subtracted = _subtract(ifc=ifc_trc, verbose=verbose)

#     # Rectify curvature
#     if verbose:
#         print('- Rectifying curvature...')
#     trc_list_background_subtracted.apply_over_ccd(transform, X=X, Y=Y)