"""
2.16-m/BFOSC pipeline
"""

import os, argparse
from copy import deepcopy

# NumPy
import numpy as np
# AstroPy
from astropy.stats import mad_std
from astropy.nddata import CCDData
# ccdproc
from ccdproc import ImageFileCollection
# drpy
from drpy.batch import CCDDataList
from drpy.image import concatenate
from drpy.utils import imstatistics
from drpy.plotting import plot2d, plotSpectrum1D
from drpy.twodspec.longslit import (response, illumination, fitcoords, transform, 
                                    extract)
from drpy.twodspec.utils import invertCoordinateMap
from drpy.onedspec import dispcor

from ..utils import makeDirectory, modifyHeader
from .utils import LIBRARY_PATH, login, loadLists, getMask


def pipeline(save_dir, data_dir, hdu, keywords, steps, row_range, col_range, 
             slit_along, n_piece, sigma, index, rdnoise, gain, custom_mask, reference, 
             mem_limit, show, save, verbose, mode): 
    """2.16-m/BFOSC pipeline"""
    
    # Login message
    if verbose:
        login('G4')
    
    # Make directories
    if verbose:
        print('\n[MAKE DIRECTORIES]')
        print(f'  - Changing working directory to {save_dir}...')
    os.chdir(save_dir)
    fig_path = makeDirectory(parent='', child='fig', verbose=verbose)
    pro_path = makeDirectory(parent='', child='pro', verbose=verbose)
    cal_path = makeDirectory(parent='', child='cal', verbose=verbose)
    
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
            print('\n[HEADER MODIFICATION]')
        for file_name in ifc.files_filtered(include_path=True):
            modifyHeader(file_name, verbose=verbose)
    
    if 'trim' in steps:
        custom_mask = custom_mask[row_range[0]:row_range[1], col_range[0]:col_range[1]]
        trim = True
    else:
        trim = False
    
    # Bias combination
    if ('bias.combine' in steps) or ('bias' in steps):
        
        if verbose:
            print('\n[BIAS COMBINATION]')
        
        if verbose:
            print('  - Loading bias...')
        # Filter
        ifc_bias = ifc.filter(regex_match=True, imagetyp='Bias Frame')
        # Load bias
        bias_list = CCDDataList.read(
            file_list=ifc_bias.files_filtered(include_path=True), hdu=hdu)
        
        # Trim
        if trim:
            if verbose:
                print('  - Trimming...')
            bias_list = bias_list.trim(row_range=row_range, col_range=col_range)
        
        bias_list.statistics(verbose=verbose)

        # Combine bias
        if verbose:
            print('  - Combining...')
        master_bias = bias_list.combine(
            method='average', sigma_clip=True, sigma_clip_low_thresh=3, 
            sigma_clip_high_thresh=3, sigma_clip_func=np.ma.median, 
            sigma_clip_dev_func=mad_std, mem_limit=mem_limit, 
            output_file=os.path.join(cal_path, 'master_bias.fits'), 
            overwrite_output=True)
        
        imstatistics(master_bias, verbose=verbose)
        
        # Plot master bias
        plot2d(
            master_bias.data, title='master bias', show=show, save=save, path=fig_path)
        
        # Release memory
        del bias_list

    # Lamp concatenation
    if ('lamp.concatenate' in steps) or ('lamp' in steps):
        
        if verbose:
            print('\n[LAMP CONCATENATION]')
        
        # Load lamp
        if verbose:
            print('  - Loading lamp...')
        ifc_lamp = ImageFileCollection(
            location=data_dir, keywords=keywords, find_fits_by_reading=False, 
            filenames=list_dict['lamp'], glob_include=None, glob_exclude=None, ext=hdu)
        lamp_list = CCDDataList.read(
            file_list=ifc_lamp.files_filtered(include_path=True), hdu=hdu)
        
        # Trim
        if trim:
            if verbose:
                print('  - Trimming...')
            lamp_list = lamp_list.trim(row_range=row_range, col_range=col_range)
        
        lamp_list.statistics(verbose=verbose)
        
        # Subtract bias
        #   Uncertainties created here (equal to that of ``master_bias``) are useless!!!
        if verbose:
            print('  - Subtracting bias...')
        if 'master_bias' not in locals():
            master_bias = CCDData.read(os.path.join(cal_path, 'master_bias.fits'))
        lamp_list_bias_subtracted = lamp_list - master_bias
        
        lamp_list_bias_subtracted.statistics(verbose=verbose)
        
        # Create real uncertainty!!!
        if verbose:
            print('  - Creating deviation...')
        lamp_list_bias_subtracted_with_deviation = (
            lamp_list_bias_subtracted.create_deviation(
                gain=gain, readnoise=rdnoise, disregard_nan=True)
        )
        
        # Concatenate
        if verbose:
            print('  - Concatenating...')
        # Ensure that the first is the short exposure
        exptime = ifc_lamp.summary['exptime'].data
        if exptime[0] > exptime[1]:
            lamp_list_bias_subtracted_with_deviation = (
                lamp_list_bias_subtracted_with_deviation[::-1]
            )
        concatenated_lamp = concatenate(
            lamp_list_bias_subtracted_with_deviation, row_range=(0, None), 
            col_range=(0, index), scale=None)
        
        # Plot concatenated lamp
        plot2d(
            concatenated_lamp.data, title='concatenated lamp', show=show, save=save, 
            path=fig_path)
        
        # Write concatenated lamp to file
        concatenated_lamp.write(
            os.path.join(cal_path, 'concatenated_lamp.fits'), overwrite=True)
        
        # Release memory
        del (lamp_list, lamp_list_bias_subtracted, 
             lamp_list_bias_subtracted_with_deviation)
        
    # Curvature rectification
    if ('lamp.rectify' in steps) or ('lamp' in steps):
        
        if verbose:
            print('\n[CURVATURE RECTIFICATION]')
        
        # Fit coordinates
        if verbose:
            print('  - Fitting coordinates...')
        if 'concatenated_lamp' not in locals():
            concatenated_lamp = CCDData.read(
                os.path.join(cal_path, 'concatenated_lamp.fits'))
        U, _ = fitcoords(
            ccd=concatenated_lamp, slit_along=slit_along, order=1, n_med=15, n_piece=3, 
            prominence=1e-3, n_iter=3, sigma_lower=3, sigma_upper=3, grow=False, 
            use_mask=False, show=show, save=save, path=fig_path, height=0, threshold=0, 
            distance=5, width=5, wlen=15, rel_height=1, plateau_size=1)
        
        # Invert coordinate map
        if verbose:
            print('  - Inverting coordinate map...')
        X, Y = invertCoordinateMap(slit_along, U)
        np.save(os.path.join(cal_path, 'X.npy'), X)
        np.save(os.path.join(cal_path, 'Y.npy'), Y)
        
        # Rectify curvature
        if verbose:
            print('  - Rectifying curvature...')
        transformed_lamp = transform(ccd=concatenated_lamp, X=X, Y=Y)
        
        # Plot transformed lamp
        plot2d(
            transformed_lamp.data, title='transformed lamp', show=show, save=save, 
            path=fig_path)
        
        # Write transformed lamp to file
        transformed_lamp.write(
            os.path.join(cal_path, 'transformed_lamp.fits'), overwrite=True)
        
    # Wavelength calibration
    if ('lamp.calibrate' in steps) or ('lamp' in steps):
        
        if verbose:
            print('\n[WAVELENGTH CALIBRATION]')
        
        # Extract 1-dimensional spectrum
        if verbose:
            print('  - Extracting 1-dimensional lamp spectrum')
        if 'transformed_lamp' not in locals():
            transformed_lamp = CCDData.read(
                os.path.join(cal_path, 'transformed_lamp.fits'))
        lamp1d = extract(
            ccd=transformed_lamp, slit_along=slit_along, trace1d=750, n_aper=1, 
            aper_width=10, show=show, save=save, path=fig_path)

        # Correct dispersion
        if verbose:
            print('  - Correcting dispersion axis...')
        calibrated_lamp1d = dispcor(
            spectrum1d=lamp1d, reverse=True, reference=reference, n_piece=3, refit=True, 
            n_iter=5, sigma_lower=3, sigma_upper=3, grow=False, use_mask=True, show=show, 
            save=save, path=fig_path)

        # Plot calibrated lamp spectrum
        plotSpectrum1D(
            spectrum1d=calibrated_lamp1d, title='calibrated lamp', show=show, save=save, 
            path=fig_path)

        # Write calibrated lamp spectrum to file
        calibrated_lamp1d.write(
            os.path.join(cal_path, 'lamp1d.fits'), overwrite=True)
        
    # Flat combination
    if ('flat.combine' in steps) or ('flat' in steps):
        
        if verbose:
            print('\n[FLAT COMBINATION]')
            
        if verbose:
            print('  - Loading flat...')
        # Filter
        ifc_flat = ifc.filter(regex_match=True, imagetyp='Flat Field')
        # Load flat
        flat_list = CCDDataList.read(
            file_list=ifc_flat.files_filtered(include_path=True), hdu=hdu)
        
        # Trim
        if trim:
            if verbose:
                print('  - Trimming...')
            flat_list = flat_list.trim(row_range=row_range, col_range=col_range)
        
        flat_list.statistics(verbose=verbose)

        # Subtract bias
        #   Uncertainties created here (equal to that of ``master_bias``) are useless!!!
        if verbose:
            print('  - Subtracting bias...')
        if 'master_bias' not in locals():
            master_bias = CCDData.read(os.path.join(cal_path, 'master_bias.fits'))
        flat_list_bias_subtracted = flat_list - master_bias
        
        flat_list_bias_subtracted.statistics(verbose=verbose)

        # Combine flat
        #   Uncertainties created above are overwritten here!!!
        if verbose:
            print('  - Combining...')
        scaling_func = lambda ccd: 1 / np.ma.average(ccd)
        combined_flat = flat_list_bias_subtracted.combine(
            method='average', scale=scaling_func, sigma_clip=True, 
            sigma_clip_low_thresh=3, sigma_clip_high_thresh=3, 
            sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std, 
            mem_limit=mem_limit, 
            output_file=os.path.join(cal_path, 'combined_flat.fits'), 
            overwrite_output=True)

        imstatistics(combined_flat, verbose=verbose)
        
        # Plot combined flat
        plot2d(
            combined_flat.data, title='combined_flat', show=show, save=save, 
            path=fig_path)
        
        # Release memory
        del flat_list, flat_list_bias_subtracted

    # Response
    if ('flat.normalize.response' in steps) or \
       ('flat.normalize' in steps) or \
       ('flat' in steps):
        
        if verbose:
            print('\n[RESPONSE]')
        
        # Response calibration
        if 'combined_flat' not in locals():
            combined_flat = CCDData.read(os.path.join(cal_path, 'combined_flat.fits'))
        reflat = response(
            ccd=combined_flat, slit_along=slit_along, n_piece=n_piece, n_iter=5, 
            sigma_lower=3, sigma_upper=3, grow=10, use_mask=True, show=show, save=save, 
            path=fig_path)

        imstatistics(reflat, verbose=verbose)
        
        # Plot response calibrated flat
        plot2d(reflat.data, title='reflat', show=show, save=save, path=fig_path)
        
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
            print('\n[ILLUMINATION]')
        
        # Illumination modeling
        if 'reflat' not in locals():
            reflat = CCDData.read(os.path.join(cal_path, 'reflat.fits'))
        reflat.mask |= custom_mask
        ilflat = illumination(
            ccd=reflat, slit_along=slit_along, method='Gaussian2D', sigma=sigma, 
            bins=10, n_iter=5, sigma_lower=3, sigma_upper=3, grow=5, use_mask=True, 
            show=show, save=save, path=fig_path)

        imstatistics(ilflat, verbose=verbose)

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
            print('\n[FLAT NORMALIZATION]')
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

    # Correct targets
    if ('targ' in steps):
        
        if verbose:
            print('\n[TARGET CORRECTION]')
        
        # Load targ
        if verbose:
            print('  - Loading targ...')
        ifc_targ = ImageFileCollection(
            location=data_dir, keywords=keywords, find_fits_by_reading=False, 
            filenames=list_dict['targ'], glob_include=None, glob_exclude=None, ext=hdu)
        targ_list = CCDDataList.read(
            file_list=ifc_targ.files_filtered(include_path=True), hdu=hdu)
        
        # Trim
        if trim:
            if verbose:
                print('  - Trimming...')
            targ_list = targ_list.trim(row_range=row_range, col_range=col_range)

        targ_list.statistics(verbose=verbose)

        # Subtract bias
        #   Uncertainties created here (equal to that of ``master_bias``) are useless!!!
        if verbose:
            print('  - Subtracting bias...')
        if 'master_bias' not in locals():
            master_bias = CCDData.read(os.path.join(cal_path, 'master_bias.fits'))
        targ_list_bias_subtracted = targ_list - master_bias
        
        targ_list_bias_subtracted.statistics(verbose=verbose)

        # Create real uncertainty!!!
        if verbose:
            print('  - Creating deviation...')
        targ_list_bias_subtracted_with_deviation = (
            targ_list_bias_subtracted.create_deviation(
                gain=gain, readnoise=rdnoise, disregard_nan=True)
        )
        
        # Flat-fielding
        if verbose:
            print('  - Flat-fielding...')
        if 'normalized_flat' not in locals():
            normalized_flat = CCDData.read(os.path.join(cal_path, 'normalized_flat.fits'))
        targ_list_flat_fielded = (
            targ_list_bias_subtracted_with_deviation / normalized_flat
        )

        # Remove cosmic ray
        if verbose:
            print('  - Removing cosmic ray...')
        targ_list_cosmicray_corrected = targ_list_flat_fielded.cosmicray(
            method='Laplacian', use_mask=False, gain=gain, readnoise=rdnoise, sigclip=4.5, 
            sigfrac=0.3, objlim=1, niter=5, verbose=True)

        # Rectify curvature
        if verbose:
            print('  - Rectifying curvature...')
        if 'X' not in locals():
            X = np.load(os.path.join(cal_path, 'X.npy'))
        if 'Y' not in locals():
            Y = np.load(os.path.join(cal_path, 'Y.npy'))
        for i, targ_cosmicray_corrected in enumerate(targ_list_cosmicray_corrected):
            
            old_name = targ_cosmicray_corrected.header['FILENAME']
            new_name = f'{os.path.splitext(old_name)[0]}_corrected.fits'

            transformed_targ = transform(ccd=targ_cosmicray_corrected, X=X, Y=Y)

            # Plot transformed spectrum
            plot2d(
                transformed_targ.data, title=old_name, show=show, save=save, 
                path=fig_path)
            
            # Write transformed spectrum to file
            if verbose:
                print(f'  - Saving {new_name} to {pro_path}...')
            transformed_targ.write(
                os.path.join(pro_path, new_name), overwrite=True)


def main():
    """Command line tool."""
    
    # External parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--data_dir', required=True, type=str, 
        help='Data (input) directory.'
    )
    parser.add_argument(
        '-s', '--semester', required=True, type=str, 
        help='Observation semester.'
    )
    parser.add_argument(
        '-w', '--slit_width', required=True, type=float, choices=[1.8, 2.3], 
        help='Slit width.'
    )
    parser.add_argument(
        '-o', '--save_dir', default='', type=str, 
        help='Saving (output) directory.'
    )
    parser.add_argument(
        '-r', '--reference', default=None, type=str, 
        help='Reference spectrum for wavelength calibration.'
    )
    parser.add_argument(
        '-m', '--mode', default='general', type=str, choices=['general', 'standard'], 
        help='General or standard.'
    )
        
    # Parse
    args = parser.parse_args()
    data_dir = os.path.abspath(args.data_dir)
    semester = args.semester
    slit_width = str(args.slit_width).replace('.', '')
    save_dir = os.path.abspath(args.save_dir)
    reference = args.reference
    mode = args.mode
    
    # Internal parameters
    hdu = 0
    shape = (2048, 2048)
    keywords = ['imagetyp', 'object', 'exptime']
    steps = ['header', 'trim', 'bias', 'lamp', 'flat', 'targ']
    # steps = ['trim', 'flat', 'targ']
    row_range = (329, 1830)
    col_range = (0, 1900)
    slit_along = 'col'
    index = 665
    n_piece = 19
    sigma = (20, 30)
    rdnoise = 4.64
    gain = 1.41
    
    semester_path = os.path.join(LIBRARY_PATH, semester)
    if not os.path.exists(semester_path):
        raise ValueError('Semester not found.')
    
    # Note that slit18 can be used to calibrate slit23
    if not reference:
        reference = os.path.join(semester_path, f'bfosc_g4_slit18_{semester}.fits')
    else:
        reference = os.path.abspath(reference)
    if not os.path.exists(reference):
        raise ValueError('Reference not found.')
    
    region = os.path.join(semester_path, f'bfosc_g4_slit{slit_width}_{semester}.reg')
    custom_mask = getMask(region_name=region, shape=shape)
    
    pipeline(
        save_dir=save_dir, data_dir=data_dir, hdu=hdu, keywords=keywords, steps=steps, 
        row_range=row_range, col_range=col_range, slit_along=slit_along, 
        n_piece=n_piece, sigma=sigma, index=index, rdnoise=rdnoise, gain=gain, 
        custom_mask=custom_mask, reference=reference, mem_limit=500e6, show=False, 
        save=True, verbose=True, mode=mode)

if __name__ == '__main__':
    main()