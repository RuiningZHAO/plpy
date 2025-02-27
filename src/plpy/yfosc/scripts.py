"""
Pipeline for YFOSC Grisms
"""

import os, argparse, warnings
from glob import glob

# NumPy
import numpy as np
# AstroPy
import astropy.units as u
from astropy.io import fits
from astropy.stats import mad_std
from astropy.nddata import CCDData
from astropy.config import reload_config
from astropy.utils.exceptions import AstropyUserWarning
# ccdproc
from ccdproc import ImageFileCollection, cosmicray_lacosmic
from ccdproc.utils.slices import slice_from_string
# specutils
from specutils import Spectrum1D
# drpy
from drpy.batch import CCDDataList
from drpy.image import concatenate
from drpy.utils import imstatistics
from drpy.plotting import plot2d, plotSpectrum1D
from drpy.twodspec import (response, illumination, align, fitcoords, transform, trace, 
                           background, profile, extract, calibrate2d)
from drpy.twodspec.utils import invertCoordinateMap
from drpy.onedspec import dispcor, sensfunc, calibrate1d

from . import conf
from ..utils import login, makeDirectory, getMask

# Load parameters from configuration file
reload_config(packageormod='plpy', rootname='plpy')


def pipeline(save_dir, data_dir, grism, slit_width, standard, reference, 
             shouldCombine, keyword, isPoint, shouldExtract, verbose):
    """YFOSC/G3 pipeline."""

#     # Custom mask
#     path_to_semester = os.path.join(conf.path_to_library, semester)
#     if not os.path.exists(path_to_semester):
#         raise ValueError('Semester not found.')

#     path_to_region = os.path.join(
#         path_to_semester, f'bfosc_{grism}_slit{slit_width}_{semester}.reg')
#     custom_mask = getMask(path_to_region=path_to_region, shape=conf.shape)

    if not reference:
        reference = sorted(glob(
            os.path.join(
                conf.path_to_library, f'yfosc_{grism}_slit{slit_width}*.fits')))[-1]
    else:
        reference = os.path.abspath(reference)
    if not os.path.exists(reference):
        raise ValueError('Reference not found.')

    # Login message
    if verbose:
        login(f'LJT/YFOSC {grism}', 100)
    
    # Make directories
    if verbose:
        print('\n[MAKE DIRECTORIES]')
        print(f'  - Changing working directory to {save_dir}...')
    os.chdir(save_dir)
    fig_path = makeDirectory(parent='', child='fig', verbose=verbose)
    pro_path = makeDirectory(parent='', child='pro', verbose=verbose)
    cal_path = makeDirectory(parent='', child='cal', verbose=verbose)

    # Construct image file collection
    ifc = ImageFileCollection(
        location=data_dir, keywords=conf.keywords, find_fits_by_reading=False, 
        filenames=None, glob_include=conf.include, glob_exclude=conf.exclude, 
        ext=conf.hdu)

    if verbose:
        print('\n[OVERVIEW]')
        ifc.summary.pprint_all()

    # Load gain and readout noise
    first_file = ifc.files_filtered(include_path=True)[0]
    gain = fits.getval(first_file, 'GAIN', ext=conf.hdu) * u.photon / u.adu
    rdnoise = fits.getval(first_file, 'RDNOISE', ext=conf.hdu) * u.photon
    
    if 'trim' in conf.steps:
        # custom_mask = custom_mask[
        #     slice_from_string(conf.fits_section, fits_convention=True)
        # ]
        trim = True
    else:
        trim = False
    
    # Bias combination
    if ('bias.combine' in conf.steps) or ('bias' in conf.steps):
        
        if verbose:
            print('\n[BIAS COMBINATION]')
        
        # Load bias
        if verbose:
            print('  - Loading bias...')
        ifc_bias = ifc.filter(regex_match=True, obstype='BIAS')
        bias_list = CCDDataList.read(
            file_list=ifc_bias.files_filtered(include_path=True), hdu=conf.hdu)

        # Trim
        if trim:
            if verbose:
                print('  - Trimming...')
            bias_list = bias_list.trim_image(fits_section=conf.fits_section)
        
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
            output_file=os.path.join(cal_path, 'bias_combined.fits'), dtype=conf.dtype, 
            overwrite_output=True)
        
        imstatistics(bias_combined, verbose=verbose)
        
        # Plot combined bias
        plot2d(
            bias_combined.data, title='bias combined', show=conf.show, save=conf.save, 
            path=fig_path)
        
        # Release memory
        del bias_list

    # Flat combination
    if ('flat.combine' in conf.steps) or ('flat' in conf.steps):
        
        if verbose:
            print('\n[FLAT COMBINATION]')
        
        # Load flat
        if verbose:
            print('  - Loading flat...')
        ifc_flat = ifc.filter(regex_match=True, obstype='LAMPFLAT')
        flat_list = CCDDataList.read(
            file_list=ifc_flat.files_filtered(include_path=True), hdu=conf.hdu)
        
        # Trim
        if trim:
            if verbose:
                print('  - Trimming...')
            flat_list = flat_list.trim_image(fits_section=conf.fits_section)
        
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
            bias_combined = CCDData.read(os.path.join(cal_path, 'bias_combined.fits'))
        flat_list_bias_subtracted = flat_list_gain_corrected.subtract_bias(bias_combined)
        
        flat_list_bias_subtracted.statistics(verbose=verbose)

        # Combine flat
        #   Uncertainties created above are overwritten here!!!
        if verbose:
            print('  - Combining...')
        scaling_func = lambda ccd: 1 / np.ma.average(ccd)
        flat_combined = flat_list_bias_subtracted.combine(
            method='average', scale=scaling_func, mem_limit=conf.mem_limit, sigma_clip=True, 
            sigma_clip_low_thresh=3, sigma_clip_high_thresh=3, 
            sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std, 
            output_file=os.path.join(cal_path, 'flat_combined.fits'), dtype=conf.dtype, 
            overwrite_output=True)

        imstatistics(flat_combined, verbose=verbose)

        # Plot combined flat
        plot2d(
            flat_combined.data, title='flat combined', show=conf.show, save=conf.save, 
            path=fig_path)

        # Release memory
        del flat_list, flat_list_bias_subtracted

    # Response
    if ('flat.normalize.response' in conf.steps) or \
       ('flat.normalize' in conf.steps) or \
       ('flat' in conf.steps):
        
        if verbose:
            print('\n[RESPONSE]')
        
        # Response calibration
        if 'flat_combined' not in locals():
            flat_combined = CCDData.read(os.path.join(cal_path, 'flat_combined.fits'))
        # flat_combined.mask |= custom_mask
        reflat = response(
            ccd=flat_combined, slit_along=conf.slit_along, n_piece=conf.n_piece, maxiters=0, 
            sigma_lower=None, sigma_upper=None, grow=False, use_mask=True, plot=conf.save, 
            path=fig_path)
        reflat = flat_combined.divide(reflat, handle_meta='first_found')
        
        imstatistics(reflat, verbose=verbose)
        
        # Plot response calibrated flat
        plot2d(
            reflat.data, title='flat response calibrated', show=conf.show, 
            save=conf.save, path=fig_path)
        
        # Plot response mask
        plot2d(
            reflat.mask.astype(int), vmin=0, vmax=1, title='mask response', 
            show=conf.show, save=conf.save, path=fig_path)
        
        # Write response calibrated flat to file
        reflat.write(
            os.path.join(cal_path, 'flat_response_calibrated.fits'), overwrite=True)
    
    # Illumination
    if ('flat.normalize.illumination' in conf.steps) or \
       ('flat.normalize' in conf.steps) or \
       ('flat' in conf.steps):

        if verbose:
            print('\n[ILLUMINATION]')
        
        # Illumination modeling
        if 'reflat' not in locals():
            reflat = CCDData.read(
                os.path.join(cal_path, 'flat_response_calibrated.fits.fits'))
        ilflat = illumination(
            ccd=reflat, slit_along=conf.slit_along, method='Gaussian2D', 
            sigma=conf.sigma, bins=10, maxiters=5, sigma_lower=3, sigma_upper=3, 
            grow=5, use_mask=True, plot=conf.save, path=fig_path)

        imstatistics(ilflat, verbose=verbose)

        # Plot illumination
        plot2d(
            ilflat.data, title='illumination', show=conf.show, save=conf.save, 
            path=fig_path)
        
        # Plot illumination mask
        plot2d(
            ilflat.mask.astype(int), vmin=0, vmax=1, title='mask illumination', 
            show=conf.show, save=conf.save, path=fig_path)
        
        # Write illumination to file
        ilflat.write(os.path.join(cal_path, 'illumination.fits'), overwrite=True)

    # Flat normalization
    if ('flat.normalize' in conf.steps) or ('flat' in conf.steps):
        
        if verbose:
            print('\n[FLAT NORMALIZATION]')
        
        # Normalization
        if 'reflat' not in locals():
            reflat = CCDData.read(
                os.path.join(cal_path, 'flat_response_calibrated.fits'))
        if 'ilflat' not in locals():
            ilflat = CCDData.read(os.path.join(cal_path, 'illumination.fits'))
        flat_normalized = reflat.divide(ilflat, handle_meta='first_found')

        imstatistics(flat_normalized, verbose=verbose)

        # Plot normalized flat
        plot2d(
            flat_normalized.data, title='flat normalized', show=conf.show, 
            save=conf.save, path=fig_path)
        
        # Plot normalized flat mask
        plot2d(
            flat_normalized.mask.astype(int), title='mask flat normalized', 
            show=conf.show, save=conf.save, path=fig_path)
        
        flat_normalized.mask = None
        
        # Write normalized flat to file
        flat_normalized.write(
            os.path.join(cal_path, 'flat_normalized.fits'), overwrite=True)

    # Curvature rectification
    if ('lamp.rectify' in conf.steps) or ('lamp' in conf.steps):

        if verbose:
            print('\n[CURVATURE RECTIFICATION]')
        
        # Load lamp
        if verbose:
            print('  - Loading lamp...')
        ifc_lamp = ifc.filter(regex_match=True, obstype='^LAMP$')
        lamp_list = CCDDataList.read(
            file_list=ifc_lamp.files_filtered(include_path=True), hdu=conf.hdu)

        # Trim
        if trim:
            if verbose:
                print('  - Trimming...')
            lamp_list = lamp_list.trim_image(fits_section=conf.fits_section)
        
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
            bias_combined = CCDData.read(os.path.join(cal_path, 'bias_combined.fits'))
        lamp_list_bias_subtracted = lamp_list_gain_corrected.subtract_bias(bias_combined)
        
        lamp_list_bias_subtracted.statistics(verbose=verbose)
        
        # Create real uncertainty!!!
        if verbose:
            print('  - Creating deviation...')
        lamp_list_bias_subtracted_with_deviation = (
            lamp_list_bias_subtracted.create_deviation(
                gain=None, readnoise=rdnoise, disregard_nan=True)
        )
        
        lamp = lamp_list_bias_subtracted_with_deviation[0]

        # Write transformed lamp to file
        lamp.write(
            os.path.join(cal_path, 'lamp.fits'), overwrite=True)
        
        # Fit coordinates
        if verbose:
            print('  - Fitting coordinates...')
        _, V = fitcoords(
            ccd=lamp, slit_along=conf.slit_along, order=1, n_med=15, n_piece=7, 
            prominence=1e-3, maxiters=3, sigma_lower=3, sigma_upper=3, grow=False, 
            use_mask=False, plot=conf.save, path=fig_path, height=0, threshold=0, 
            distance=5, width=5, wlen=15, rel_height=1, plateau_size=1)
        
        # Invert coordinate map
        if verbose:
            print('  - Inverting coordinate map...')
        X, Y = invertCoordinateMap(conf.slit_along, V)
        np.save(os.path.join(cal_path, 'X.npy'), X)
        np.save(os.path.join(cal_path, 'Y.npy'), Y)
        
        # Rectify curvature
        if verbose:
            print('  - Rectifying curvature...')
        lamp_transformed = transform(ccd=lamp, X=X, Y=Y)
        
        # Plot transformed lamp
        plot2d(
            lamp_transformed.data, title='lamp transformed', show=conf.show, save=conf.save, 
            path=fig_path)
        
        # Write transformed lamp to file
        lamp_transformed.write(
            os.path.join(cal_path, 'lamp_transformed.fits'), overwrite=True)
    
    # Correct targets
    if ('targ' in conf.steps):
        
        if verbose:
            print('\n[CORRECTION]')
        
        # Load targ
        if verbose:
            print('  - Loading targ...')
        ifc_targ = ifc.filter(regex_match=True, obstype='EXPOSE|FLUXREF')
        # The above line does not work as expected due to a bug in ccdproc. An issue is 
        # already reported to ccdproc through Github. The following lines should be 
        # removed after the bug is fixed.
        ifc_targ = ImageFileCollection(
            filenames=ifc_targ.files_filtered(include_path=True), 
            keywords=conf.keywords, ext=conf.hdu)
        # ------------------------------------------------------------------------------
        targ_list = CCDDataList.read(
            file_list=ifc_targ.files_filtered(include_path=True), hdu=conf.hdu)

        # Trim
        if trim:
            if verbose:
                print('  - Trimming...')
            targ_list = targ_list.trim_image(fits_section=conf.fits_section)
        
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
            bias_combined = CCDData.read(os.path.join(cal_path, 'bias_combined.fits'))
        targ_list_bias_subtracted = targ_list_gain_corrected.subtract_bias(bias_combined)
        
        targ_list_bias_subtracted.statistics(verbose=verbose)

        # Create real uncertainty!!!
        if verbose:
            print('  - Creating deviation...')
        targ_list_bias_subtracted_with_deviation = (
            targ_list_bias_subtracted.create_deviation(
                gain=None, readnoise=rdnoise, disregard_nan=True)
        )
        
        # Flat-fielding
        if verbose:
            print('  - Flat-fielding...')
        if 'flat_normalized' not in locals():
            flat_normalized = CCDData.read(
                os.path.join(cal_path, 'flat_normalized.fits'))
        targ_list_flat_fielded = (
            targ_list_bias_subtracted_with_deviation.flat_correct(flat_normalized)
        )

        # Identify flux standard
        isStandard = ifc_targ.summary['obstype'].data == 'FLUXREF'

        if isStandard.sum() > 0:
            
            # if isStandard.sum() > 1:
            #     raise RuntimeError('More than one standard spectrum found.')
                
            # Only the first standard is used
            index_standard = np.where(isStandard)[0][0]

            key_standard = ifc_targ.summary['object'].data[index_standard]
            standard_flat_fielded = targ_list_flat_fielded[index_standard]

            # Plot
            plot2d(
                standard_flat_fielded.data, title=f'{key_standard} flat-fielded', 
                show=conf.show, save=conf.save, path=fig_path)

            # Write standard spectrum to file
            if verbose:
                print('\n[STANDARD]')
                print(
                    f'  - Saving flat-fielded spectrum of {key_standard} to {pro_path}'
                    '...')
            standard_flat_fielded.write(
                os.path.join(pro_path, f'{key_standard}_flat_fielded.fits'), 
                overwrite=True)

            if verbose:
                print(f'  - Tracing {key_standard}...')
        
            # Trace (trace the brightest spectrum)
            trace1d_standard = trace(
                ccd=standard_flat_fielded, slit_along=conf.slit_along, fwhm=10, 
                method='trace', interval='[:]', n_med=10, n_piece=5, maxiters=5, 
                sigma_lower=2, sigma_upper=2, grow=False, title=key_standard, 
                show=conf.show, save=conf.save, path=fig_path)

            # Write standard trace to file (of type float64)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=AstropyUserWarning)
                trace1d_standard.write(
                    os.path.join(cal_path, f'trace1d_{key_standard}.fits'), 
                    format='tabular-fits', overwrite=True)

            if verbose:
                print('  - Extracting 1-dimensional lamp spectra...')

            # Extract lamp spectrum for standard (of type float64)
            if 'lamp' not in locals():
                lamp = CCDData.read(os.path.join(cal_path, 'lamp.fits'))
            lamp1d_standard = extract(
                ccd=lamp, slit_along=conf.slit_along, method='sum', 
                trace1d=trace1d_standard, aper_width=150, n_aper=1, 
                title=f'lamp1d {key_standard}', show=conf.show, save=conf.save, 
                path=fig_path)

            if verbose:
                print('  - Correcting dispersion axis of lamp spectra...')

            # Correct dispersion of lamp spectrum for standard (of type float64)
            lamp1d_standard_calibrated = dispcor(
                spectrum1d=lamp1d_standard, reverse=False, reference=reference, n_sub=20, 
                refit=True, degree=1, maxiters=5, sigma_lower=3, sigma_upper=3, grow=False, 
                use_mask=True, title=key_standard, show=conf.show, save=conf.save, 
                path=fig_path)

            if verbose:
                print(f'  - Saving calibrated lamp spectra to {cal_path}...')

            # Write calibrated lamp spectrum for standard to file (of type float64)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=AstropyUserWarning)
                lamp1d_standard_calibrated.write(
                    os.path.join(cal_path, f'lamp1d_{key_standard}.fits'), 
                    format='tabular-fits', overwrite=True)

            if verbose:
                print(f'  - Modeling sky background of {key_standard}...')

            # Model sky background of standard
            background2d_standard = background(
                ccd=standard_flat_fielded, slit_along=conf.slit_along, 
                trace1d=trace1d_standard, location=conf.bkg_location_stan, 
                aper_width=conf.bkg_width_stan, degree=conf.bkg_order_stan, maxiters=3, 
                sigma_lower=4, sigma_upper=4, grow=False, use_uncertainty=False, 
                use_mask=True, title=key_standard, show=conf.show, save=conf.save, 
                path=fig_path)

            # Plot sky background of standard
            plot2d(
                background2d_standard.data, title=f'background2d {key_standard}', 
                show=conf.show, save=conf.save, path=fig_path)

            # Write sky background of standard to file
            background2d_standard.write(
                os.path.join(pro_path, f'background2d_{key_standard}.fits'), 
                overwrite=True)

            if verbose:
                print(
                    f'  - Extracting sky background spectrum of {key_standard}...')

            # Extract background spectrum of standard
            background1d_standard = extract(
                ccd=background2d_standard, slit_along=conf.slit_along, method='sum', 
                trace1d=trace1d_standard, aper_width=150, n_aper=1, use_uncertainty=False, 
                use_mask=True, spectral_axis=lamp1d_standard_calibrated.spectral_axis, 
                show=False, save=False)

            # Plot background spectrum of standard
            plotSpectrum1D(
                background1d_standard, title=f'{key_standard} background1d', 
                show=conf.show, save=conf.save, path=fig_path)

            # Write sky background spectrum of standard to file (of type float64)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=AstropyUserWarning)
                background1d_standard.write(
                    os.path.join(cal_path, f'background1d_{key_standard}.fits'), 
                    format='tabular-fits', overwrite=True)

            if verbose:
                print(f'  - Subtracting sky background from {key_standard}...')

            # Subtract sky background from standard
            standard_background_subtracted = standard_flat_fielded.subtract(
                background2d_standard, handle_meta='first_found')

            # Plot background subtracted standard
            plot2d(
                standard_background_subtracted.data, 
                title=f'{key_standard} background subtracted', show=conf.show, 
                save=conf.save, path=fig_path)

            # Write background subtracted standard to file
            standard_background_subtracted.write(
                os.path.join(pro_path, f'{key_standard}_background_subtracted.fits'), 
                overwrite=True)

            # Extract standard spectrum
            if verbose:
                print(
                    f'  - Extracting spectrum of {key_standard} (standard) '
                    f'({conf.extract_stan})...'
                )

            if conf.extract_stan == 'sum':

                standard_cosmicray_corrected, crmask = cosmicray_lacosmic(
                    standard_background_subtracted.data, 
                    gain=(1 * u.dimensionless_unscaled), readnoise=rdnoise, 
                    sigclip=4.5, sigfrac=0.3, objlim=1, niter=5, verbose=True)

                standard_background_subtracted.data = standard_cosmicray_corrected
                standard_background_subtracted.mask = crmask

                # Extract (sum)
                standard1d = extract(
                    ccd=standard_background_subtracted, slit_along=conf.slit_along, 
                    method='sum', trace1d=trace1d_standard, 
                    aper_width=conf.aper_width_stan, n_aper=1, 
                    spectral_axis=lamp1d_standard_calibrated.spectral_axis, 
                    use_uncertainty=True, use_mask=True, title=key_standard, 
                    show=conf.show, save=conf.save, path=fig_path)

            else:

                # Model spatial profile of standard
                profile2d_standard, _ = profile(
                    ccd=standard_background_subtracted, slit_along=conf.slit_along, 
                    trace1d=trace1d_standard, profile_width=conf.aper_width_stan, 
                    window_length=conf.pfl_window_stan, polyorder=conf.pfl_order_stan, 
                    deriv=0, delta=1.0, title='profile', show=conf.fig_show, 
                    save=conf.fig_save, path=conf.fig_path)

                # Extract (optimal)
                standard1d = extract(
                    ccd=standard_background_subtracted, slit_along=conf.slit_along, 
                    method='optimal', profile2d=profile2d_standard, 
                    background2d=background2d_standard.data, rdnoise=rdnoise.value, maxiters=5, 
                    sigma_lower=5, sigma_upper=5, grow=False, 
                    spectral_axis=lamp1d_standard_calibrated.spectral_axis, 
                    use_uncertainty=True, use_mask=True, title=key_standard, show=conf.show, 
                    save=conf.save, path=fig_path)

            # Plot standard spectrum
            plotSpectrum1D(
                standard1d, title=f'{key_standard} extracted', show=conf.show, 
                save=conf.save, path=fig_path)

            # Write standard spectrum to file (of type float64)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=AstropyUserWarning)
                standard1d.write(
                    os.path.join(cal_path, f'{key_standard}_extracted.fits'), 
                    format='tabular-fits', overwrite=True)

            if standard is not None:

                if verbose:
                    print('\n[SENSITIVITY]')
                    print('  - Fitting sensitivity function...')

                # Fit sensitivity function
                sens1d, spl = sensfunc(
                    spectrum1d=standard1d, exptime=conf.exposure, airmass=conf.airmass, 
                    extinct=conf.extinct, standard=standard, bandwid=10, bandsep=10, 
                    n_piece=19, maxiters=5, sigma_lower=1, sigma_upper=3, grow=False, 
                    show=conf.show, save=conf.save, path=fig_path)
        
        if verbose:
            print('\n[TARGET]')

        # Remove flux standard
        targ_list_flat_fielded = targ_list_flat_fielded[~isStandard]

        # Group
        ifc_targ = ifc_targ.filter(regex_match=True, obstype='EXPOSE')
        # The above line does not work as expected due to a bug in ccdproc. An issue is 
        # already reported to ccdproc through Github. The following lines should be 
        # removed after the bug is fixed.
        ifc_targ = ImageFileCollection(
            filenames=ifc_targ.files_filtered(include_path=True), 
            keywords=conf.keywords, ext=conf.hdu)
        # ------------------------------------------------------------------------------
        ifc_targ_summary = ifc_targ.summary
        ifc_targ_summary_grouped = ifc_targ_summary.group_by(keyword)
        keys = ifc_targ_summary_grouped.groups.keys[keyword].data
        if verbose:
            print('  - Grouping')
            print(f'    - {keys.shape[0]} groups: ' + ', '.join(keys))

        key_list = list()
        targ_combined_list = list()

        for i, key in enumerate(keys):

            if verbose:
                print(
                    f'  - Dealing with group {key} ({(i + 1)}/{keys.shape[0]})...')
            mask = ifc_targ_summary[keyword].data == key

            if shouldCombine:

                if mask.sum() >= 3:

                    # Skip cosmic ray removal
                    targ_list_cosmicray_corrected = targ_list_flat_fielded[mask]

                else:

                    # Remove cosmic ray
                    if verbose:
                        print('    - Removing cosmic ray...')
                    targ_list_cosmicray_corrected = (
                        targ_list_flat_fielded[mask].cosmicray_lacosmic(
                            use_mask=False, gain=(1 * u.dimensionless_unscaled), 
                            readnoise=rdnoise, sigclip=4.5, sigfrac=0.3, objlim=1, 
                            niter=5, verbose=True)
                    )

                # Rectify curvature
                if verbose:
                    print('    - Rectifying curvature...')
                if 'X' not in locals():
                    X = np.load(os.path.join(cal_path, 'X.npy'))
                if 'Y' not in locals():
                    Y = np.load(os.path.join(cal_path, 'Y.npy'))
                targ_list_transformed = (
                    targ_list_cosmicray_corrected.apply_over_ccd(transform, X=X, Y=Y)
                )

                if mask.sum() > 1:

                    # Align
                    if verbose:
                        print('    - Aligning...')
                    targ_list_aligned = align(
                        targ_list_transformed, conf.slit_along, index=0, 
                        interval=conf.align_interval)

                    # Combine
                    if verbose:
                        print(f'    - Combining ({mask.sum()})...')
                    exptime = ifc_targ_summary['exptime'].data[mask]
                    scale = exptime.max() / exptime
                    targ_combined = targ_list_aligned.combine(
                        method='median', scale=scale, mem_limit=conf.mem_limit, 
                        sigma_clip=True, sigma_clip_low_thresh=3, 
                        sigma_clip_high_thresh=3, sigma_clip_func=np.ma.median, 
                        sigma_clip_dev_func=mad_std, 
                        output_file=os.path.join(pro_path, f'{key}_combined.fits'), 
                        dtype=conf.dtype, overwrite_output=True)

                else:

                    targ_combined = targ_list_transformed[0]
                    targ_combined.write(
                        os.path.join(pro_path, f'{key}_combined.fits'), overwrite=True)

                if verbose:
                    print(f'    - Saving combined {key} to {pro_path}...')

                # Plot
                plot2d(
                    targ_combined.data, title=f'{key} combined', show=conf.show, 
                    save=conf.save, path=fig_path)
                
                key_list.append(key)
                targ_combined_list.append(targ_combined)

            else:

                # Remove cosmic ray
                if verbose:
                    print('  - Removing cosmic ray...')
                targ_list_cosmicray_corrected = (
                    targ_list_flat_fielded[mask].cosmicray_lacosmic(
                        use_mask=False, gain=(1 * u.dimensionless_unscaled), 
                        readnoise=rdnoise, sigclip=4.5, sigfrac=0.3, objlim=1, 
                        niter=5, verbose=True)
                )

                # Rectify curvature
                if verbose:
                    print('  - Rectifying curvature...')
                if 'X' not in locals():
                    X = np.load(os.path.join(cal_path, 'X.npy'))
                if 'Y' not in locals():
                    Y = np.load(os.path.join(cal_path, 'Y.npy'))
                targ_list_transformed = targ_list_cosmicray_corrected.apply_over_ccd(
                        transform, X=X, Y=Y)

                n = int(np.log10(mask.sum())) + 1

                for j, targ_transformed in enumerate(targ_list_transformed):

                    if mask.sum() == 1:
                        new_name = f'{key}'
                    else:
                        new_name = f'{key}_{(j + 1):0{n}d}'

                    # Write transformed spectrum to file
                    if verbose:
                        print(f'  - Saving corrected {new_name} to {pro_path}...')
                    targ_transformed.write(
                        os.path.join(pro_path, f'{key}_corrected.fits'), overwrite=True)

                    # Plot
                    plot2d(
                        targ_transformed.data, title=f'{new_name} corrected', 
                        show=conf.show, save=conf.save, path=fig_path)

                    key_list.append(f'{key}_{(j + 1):0{n}d}')
                    targ_combined_list.append(targ_transformed)

        targ_combined_list = CCDDataList(targ_combined_list)

        if verbose:
            print('  - Extracting 1-dimensional lamp spectra...')

        # Extract lamp spectrum for target (of type float64)
        if 'lamp_transformed' not in locals():
            lamp_transformed = CCDData.read(
                os.path.join(cal_path, 'lamp_transformed.fits'))
        lamp1d_target = extract(
            ccd=lamp_transformed, slit_along=conf.slit_along, method='sum', 
            trace1d=750, aper_width=10, n_aper=1, title='lamp1d target', 
            show=conf.show, save=conf.save, path=fig_path)

        if verbose:
            print('  - Correcting dispersion axis of lamp spectra...')

        # Correct dispersion of lamp spectrum for target (of type float64)
        lamp1d_target_calibrated = dispcor(
            spectrum1d=lamp1d_target, reverse=False, reference=reference, n_sub=20, 
            refit=True, degree=1, maxiters=5, sigma_lower=3, sigma_upper=3, grow=False, 
            use_mask=True, title='target', show=conf.show, save=conf.save, 
            path=fig_path)

        if verbose:
            print(f'  - Saving calibrated lamp spectra to {cal_path}...')

        # Write calibrated lamp spectrum for target to file (of type float64)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=AstropyUserWarning)
            lamp1d_target_calibrated.write(
                os.path.join(cal_path, 'lamp1d_target.fits'), format='tabular-fits', 
                overwrite=True)

        if 'sens1d' in locals():

            sens1d = Spectrum1D(
                spectral_axis=lamp1d_target_calibrated.spectral_axis, 
                flux=(
                    spl(lamp1d_target_calibrated.spectral_axis.value) * sens1d.flux.unit
                ), 
                uncertainty=sens1d.uncertainty, 
                meta=sens1d.meta
            )

            if verbose:
                print(f'  - Saving sensitivity function to {cal_path}...')

            # Write sensitivity function to file (of type float64)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=AstropyUserWarning)
                sens1d.write(
                    os.path.join(cal_path, 'sens1d.fits'), format='tabular-fits', 
                    overwrite=True)

        for key, targ_combined in zip(key_list, targ_combined_list):

            # Trace (trace the brightest spectrum)
            if verbose:
                print(f'  - Tracing {key}...')

            if (conf.trace_targ is 'center') & (isStandard.sum() > 0):

                trace1d_target = trace(
                    ccd=targ_combined, slit_along=conf.slit_along, fwhm=10, 
                    method='center', interval='[:]', title=key, show=conf.show, 
                    save=conf.save, path=fig_path)
                shift = (
                    trace1d_standard.meta['header']['TRCENTER'] 
                    - trace1d_target.meta['header']['TRCENTER']) * u.pixel
                trace1d_target = Spectrum1D(
                    flux=(trace1d_standard.flux - shift), meta=trace1d_target.meta)

            else:

                trace1d_target = trace(
                    ccd=targ_combined, slit_along=conf.slit_along, fwhm=10, 
                    method='trace', interval='[:]', n_med=conf.trace_n_med, 
                    n_piece=conf.trace_n_piece, maxiters=5, sigma_lower=2, 
                    sigma_upper=2, grow=False, title=key, show=conf.show, 
                    save=conf.save, path=fig_path)

            if verbose:
                print(f'  - Modeling sky background of {key}...')

            # Model sky background of target
            background2d_target = background(
                ccd=targ_combined, slit_along=conf.slit_along, 
                trace1d=trace1d_target, location=conf.bkg_location_targ, 
                aper_width=conf.bkg_width_targ, degree=conf.bkg_order_targ, 
                maxiters=3, sigma_lower=4, sigma_upper=4, grow=False, 
                use_uncertainty=False, use_mask=False, title=key, 
                show=conf.show, save=conf.save, path=fig_path)

            # Write sky background of standard to file
            background2d_target.write(
                os.path.join(pro_path, f'background_{key}.fits'), 
                overwrite=True)

            if verbose:
                print(f'  - Subtracting sky background from {key} (target)...')

            # Subtract sky background from target
            target_background_subtracted = targ_combined.subtract(
                background2d_target, handle_meta='first_found')

            # Plot background subtracted target
            plot2d(
                target_background_subtracted.data, 
                title=f'{key} background subtracted', show=conf.show, save=conf.save, 
                path=fig_path)

            # Write background subtracted target to file
            target_background_subtracted.write(
                os.path.join(pro_path, f'{key}_background_subtracted.fits'), 
                overwrite=True)

            # Here ``isPoint`` is not needed.
            if shouldExtract:

                # Extract target spectrum
                if verbose:
                    print(
                        f'  - Extracting spectrum of {key} ({conf.extract_targ})...'
                    )

                if conf.extract_targ == 'sum':

                    # Extract (sum)
                    target1d = extract(
                        ccd=target_background_subtracted, slit_along=conf.slit_along, 
                        method='sum', trace1d=trace1d_target, 
                        aper_width=conf.aper_width_targ, n_aper=1, 
                        spectral_axis=lamp1d_target_calibrated.spectral_axis, 
                        use_uncertainty=True, use_mask=True, title=key, show=conf.show, 
                        save=conf.save, path=fig_path)

                else:

                    # Model spatial profile of target
                    profile2d_target, _ = profile(
                        ccd=target_background_subtracted, slit_along=conf.slit_along, 
                        trace1d=trace1d_target, profile_width=conf.aper_width_targ, 
                        window_length=conf.pfl_window_targ, 
                        polyorder=conf.pfl_order_targ, deriv=0, delta=1.0, 
                        title='profile', show=conf.fig_show, save=conf.fig_save, 
                        path=conf.fig_path)

                    # Extract (optimal)
                    target1d = extract(
                        ccd=target_background_subtracted, slit_along=conf.slit_along, 
                        method='optimal', profile2d=profile2d_target, 
                        background2d=background2d_target.data, rdnoise=rdnoise.value, 
                        maxiters=5, sigma_lower=5, sigma_upper=5, grow=False, 
                        spectral_axis=lamp1d_target_calibrated.spectral_axis, 
                        use_uncertainty=True, use_mask=True, title=key, show=conf.show, 
                        save=conf.save, path=fig_path)

                # Plot target spectrum
                plotSpectrum1D(
                    target1d, title=f'{key} extracted', show=conf.show, 
                    save=conf.save, path=fig_path)

                # Write calibrated spectrum to file
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=AstropyUserWarning)
                    target1d.write(
                        os.path.join(pro_path, f'{key}_extracted.fits'), 
                        format='tabular-fits', overwrite=True)

                # Calibrate target spectrum
                if 'sens1d' in locals():

                    if verbose:
                        print(f'    - Calibrating {key}...')
                    target1d_calibrated = calibrate1d(
                        spectrum1d=target1d, exptime=conf.exposure, 
                        airmass=conf.airmass, extinct=conf.extinct, sens1d=sens1d, 
                        use_uncertainty=False)

                    # Plot calibrated target spectrum
                    plotSpectrum1D(
                        target1d_calibrated, title=f'{key} calibrated', show=conf.show, 
                        save=conf.save, path=fig_path)

                    # Write calibrated spectrum to file
                    if verbose:
                        print(f'    - Saving calibrated {key} to {pro_path}...')
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', category=AstropyUserWarning)
                        target1d_calibrated.write(
                            os.path.join(pro_path, f'{key}_calibrated.fits'), 
                            format='tabular-fits', overwrite=True)

            elif 'sens1d' in locals():

                # Calibrate                    
                if verbose:
                    print(f'    - Calibrating {key}...')
                targ_calibrated = calibrate2d(
                    ccd=target_background_subtracted, slit_along=conf.slit_along, 
                    exptime=conf.exposure, airmass=conf.airmass, extinct=conf.extinct, 
                    sens1d=sens1d, use_uncertainty=False)

                # Write calibrated spectrum to file
                if verbose:
                    print(f'    - Saving calibrated {key} to {pro_path}...')
                targ_calibrated.write(
                    os.path.join(pro_path, f'{key}_calibrated.fits'), 
                    overwrite=True)


def spec():
    """Command line tool."""
    
    grism = 'g3'
    
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
    #     '-m', '--semester', required=True, type=str, 
    #     help='Observation semester.'
    # )
    parser.add_argument(
        '-w', '--slit_width', required=True, type=float, choices=[1.8, 2.5], 
        help='Slit width.'
    )
    parser.add_argument(
        '-r', '--reference', default=None, type=str, 
        help='Reference spectrum for wavelength calibration.'
    )
    parser.add_argument(
        '-s', '--standard', default=None, type=str, 
        help='Path to the standard spectrum in the library.'
    )
    parser.add_argument(
        '-k', '--keyword', default='object', type=str, 
        help='Keyword for grouping.'
    )
    parser.add_argument(
        '-c', '--combine', action='store_true', 
        help='Combine or not.'
    )
    parser.add_argument(
        '-x', '--extract', action='store_true', 
        help='Extract 1-dimensional spectra or not.'
    )
    parser.add_argument(
        '-p', '--point', action='store_true', 
        help='Point source or not. (not used)'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true', 
        help='Verbose or not.'
    )
        
    # Parse
    args = parser.parse_args()
    data_dir = os.path.abspath(args.input_dir)
    # semester = args.semester
    slit_width = str(args.slit_width).replace('.', '')
    save_dir = os.path.abspath(args.output_dir)
    reference = args.reference
    standard = args.standard
    combine = args.combine
    keyword = args.keyword
    extract = args.extract
    isPoint = args.point
    verbose = args.verbose

    pipeline(
        save_dir=save_dir, data_dir=data_dir, grism=grism, 
        slit_width=slit_width, standard=standard, reference=reference, 
        shouldCombine=combine, keyword=keyword, isPoint=isPoint, 
        shouldExtract=extract, verbose=verbose)