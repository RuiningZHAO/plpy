"""
Magellan/FIRE pipeline
"""

import os, argparse, warnings
from glob import glob

# NumPy
import numpy as np
# SciPy
from scipy import ndimage
# AstroPy
import astropy.units as u
from astropy.io import fits
from astropy.stats import mad_std
from astropy.nddata import CCDData
# ccdproc
from ccdproc import ImageFileCollection, cosmicray_lacosmic
# drpy
from drpy.batch import CCDDataList
from drpy.image import concatenate
from drpy.utils import imstatistics
from drpy.plotting import plot2d
from drpy.twodspec.longslit import (response, illumination, fitcoords, transform, 
                                    background, extract)
from drpy.twodspec.utils import invertCoordinateMap
from drpy.onedspec import  dispcor, loadSpectrum1D, saveSpectrum1D

from .utils import LIBRARY_PATH, makeDirectory, modifyHeader


def pipeline(save_dir, data_dir, glob_include, procedures, hdu, keywords, fits_section, 
             slit_along, reference, high_voltage, low_voltage, index, n_piece, sigma, 
             sub_entries, hdr_entries, dtype, mem_limit, show, save, verbose):
    """Magella/FIRE pipeline."""
    
    # Login message
    if verbose:
        print('Magella/FIRE')
    
    # Make directories
    if verbose:
        print('\n[MAKE DIRECTORIES]')
        print(f'  - Changing working directory to {save_dir}...')
    os.chdir(save_dir)
    fig_path = makeDirectory(parent='', child='fig', verbose=verbose)
    pro_path = makeDirectory(parent='', child='pro', verbose=verbose)
    cal_path = makeDirectory(parent='', child='cal', verbose=verbose)
    
    # Filter files
    filenames = sorted(glob(os.path.join(data_dir, glob_include)))
    
    # Modify fits header
    if 'h' in procedures:
        if verbose:
            print('\n[HEADER MODIFICATION]')
        for file_name in filenames:
            modifyHeader(file_name, verbose=verbose, output_verify='warn')
        # Fix typos
        if hdr_entries is not None:
            for entry in hdr_entries:
                file_name = os.path.join(data_dir, entry[0])
                with fits.open(file_name, 'update', output_verify='warn') as hdu_list:
                    hdu_list[hdu].header.set(entry[1], entry[2])
    
    # Construct image file collection
    ifc = ImageFileCollection(
        location=data_dir, keywords=keywords, find_fits_by_reading=False, 
        filenames=filenames, glob_include=None, glob_exclude=None, ext=hdu)
    
    ifc.summary.pprint_all()
    ifc.summary.write(
        os.path.join(save_dir, 'summary.dat'), format='ascii.fixed_width_two_line', 
        overwrite=True)
    
    # Load gain and readout noise
    first_file = ifc.files_filtered(include_path=True)[0]
    gain = fits.getval(first_file, 'EGAIN', ext=hdu) * u.photon / u.adu
    rdnoise = fits.getval(first_file, 'ENOISE', ext=hdu) * u.photon
    
    if 't' in procedures:
        trim = True
    else:
        trim = False
    
    # Lamp
    if 'a' in procedures:
        
        if verbose:
            print('\n[LAMP COMBINATION]')
        
        # Load lamp
        if verbose:
            print('  - Loading lamp...')
        ifc_lamp = ifc.filter(regex_match=True, obstype='Arc')
        lamp_list = CCDDataList.read(
            file_list=ifc_lamp.files_filtered(include_path=True), hdu=hdu)
        
        # Trim
        if trim:
            if verbose:
                print('  - Trimming...')
            lamp_list = lamp_list.trim_image(fits_section=fits_section)
        
        # Correct gain
        if verbose:
            print('  - Correcting gain...')
        lamp_list_gain_corrected = lamp_list.gain_correct(gain=gain)
        
        lamp_list_gain_corrected.statistics(verbose=verbose)
        
        # Group
        ifc_lamp_summary = ifc_lamp.summary
        ifc_lamp_summary_grouped = ifc_lamp_summary.group_by('object')
        keys = ifc_lamp_summary_grouped.groups.keys['object'].data
        if verbose:
            print('  - Grouping')
            print(f'    - {keys.shape[0]} groups: ' + ', '.join(keys))
        
        for key in keys:
            
            # Combine lamp
            if verbose:
                print(f'  - Combining group {key}...')
            
            mask = ifc_lamp_summary['object'].data == key
            
            if mask.sum() > 1:
                combined_lamp = lamp_list_gain_corrected[mask].combine(
                    method='average', mem_limit=mem_limit, sigma_clip=True, 
                    sigma_clip_low_thresh=3, sigma_clip_high_thresh=3, 
                    sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std, 
                    output_file=os.path.join(cal_path, f'Arc_{key}.fits'), dtype=dtype, 
                    overwrite_output=True)
            else:
                combined_lamp = lamp_list_gain_corrected[mask][0]
                combined_lamp.write(
                    os.path.join(cal_path, f'Arc_{key}.fits'), overwrite=True)
        
            # Fit coordinates
            if verbose:
                print('    - Fitting coordinates...')
            
            _, V = fitcoords(
                ccd=combined_lamp, slit_along=slit_along, order=1, n_med=5, n_piece=3, 
                prominence=1e-3, maxiters=3, sigma_lower=3, sigma_upper=3, grow=False, 
                use_mask=False, plot=save, path=fig_path, height=0, threshold=0, 
                distance=5, width=5, wlen=15, rel_height=1, plateau_size=1)
            if key == 'flat':
                np.save(os.path.join(cal_path, 'V.npy'), V)

            # Invert coordinate map
            if verbose:
                print('    - Inverting coordinate map...')
            X, Y = invertCoordinateMap(slit_along, V)
            np.save(os.path.join(cal_path, f'X_{key}.npy'), X)
            np.save(os.path.join(cal_path, f'Y_{key}.npy'), Y)

            transformed_lamp = transform(ccd=combined_lamp, X=X, Y=Y)

            # Plot transformed lamp
            plot2d(
                transformed_lamp.data, aspect='auto', cbar=False, 
                title=f'transformed arc {key}', show=show, save=save, path=fig_path)
            
            # Extract
            if verbose:
                print('    - Extracting 1-dimensional lamp spectrum...')            
            arc1d = extract(
                ccd=transformed_lamp, slit_along=slit_along, method='sum', trace1d=100, 
                aper_width=30, n_aper=1, use_uncertainty=False, use_mask=False, 
                title=f'arc for {key}', show=show, save=save, path=fig_path)
            
            # Correct dispersion of lamp spectrum (of type float64)
            if verbose:
                print('    - Correcting dispersion axis of lamp spectrum...')      
            arc1d_calibrated = dispcor(
                spectrum1d=arc1d, reverse=False, reference=reference, n_sub=20, 
                refit=True, degree=1, prominence=1e-3, height=0, threshold=0, 
                distance=5, width=5, wlen=15, rel_height=1, plateau_size=1, maxiters=5, 
                sigma_lower=None, sigma_upper=None, grow=False, use_mask=False, 
                title='dispcor', show=show, save=save, path=fig_path)

            # Write arc spectrum to file
            saveSpectrum1D(
                os.path.join(cal_path, f'arc1d_{key}.fits'), arc1d_calibrated, 
                overwrite=True)
        
        # Release memory
        del lamp_list, lamp_list_gain_corrected
    
    # Flat combination
    if 'f' in procedures:
        
        if verbose:
            print('\n[FLAT COMBINATION]')
        
        # Load flats
        if verbose:
            print('  - Loading flats...')
        ifc_bflat = ifc.filter(regex_match=True, object=f'Domeflat{high_voltage}V')
        bflat_list = CCDDataList.read(
            file_list=ifc_bflat.files_filtered(include_path=True), hdu=hdu)
        ifc_rflat = ifc.filter(regex_match=True, object=f'Domeflat{low_voltage}V')
        rflat_list = CCDDataList.read(
            file_list=ifc_rflat.files_filtered(include_path=True), hdu=hdu)
        
        # Trim
        if trim:
            if verbose:
                print('  - Trimming...')
            bflat_list = bflat_list.trim_image(fits_section=fits_section)
            rflat_list = rflat_list.trim_image(fits_section=fits_section)
        
        # Correct gain
        if verbose:
            print('  - Correcting gain...')
        bflat_list_gain_corrected = bflat_list.gain_correct(gain=gain)
        rflat_list_gain_corrected = rflat_list.gain_correct(gain=gain)
        
        bflat_list_gain_corrected.statistics(verbose=verbose)
        rflat_list_gain_corrected.statistics(verbose=verbose)

        scaling_func = lambda ccd: 1 / np.ma.average(ccd)

        # Combine flats
        #   Uncertainties created above are overwritten here!!!
        if verbose:
            print('  - Combining flats...')
        combined_bflat = bflat_list_gain_corrected.combine(
            method='average', scale=scaling_func, mem_limit=mem_limit, sigma_clip=True, 
            sigma_clip_low_thresh=3, sigma_clip_high_thresh=3, 
            sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std, 
            output_file=os.path.join(cal_path, 'combined_bflat.fits'), dtype=dtype, 
            overwrite_output=True)
        combined_rflat = rflat_list_gain_corrected.combine(
            method='average', scale=scaling_func, mem_limit=mem_limit, sigma_clip=True, 
            sigma_clip_low_thresh=3, sigma_clip_high_thresh=3, 
            sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std, 
            output_file=os.path.join(cal_path, 'combined_rflat.fits'), dtype=dtype, 
            overwrite_output=True)

        imstatistics(combined_bflat, verbose=verbose)
        imstatistics(combined_rflat, verbose=verbose)
        
        # Plot combined flats
        plot2d(
            combined_bflat.data, aspect='auto', cbar=False, title='combined bflat', 
            show=show, save=save, path=fig_path)
        plot2d(
            combined_rflat.data, aspect='auto', cbar=False, title='combined rflat', 
            show=show, save=save, path=fig_path)
        
        # Release memory
        del bflat_list, bflat_list_gain_corrected, # flat_list_bias_subtracted
        del rflat_list, rflat_list_gain_corrected, # flat_list_bias_subtracted
        
        # Concatenate flats
        if verbose:
            print('\n[FLAT CONCATENATION]')
        
        scaling_factor = np.median(
            combined_rflat[(index - 10):(index + 10), :].divide(
            combined_bflat[(index - 10):(index + 10), :]).data
        )
        
        concatenated_flat = concatenate(
            [combined_bflat, combined_rflat], fits_section=f'[:, :{index}]', 
            scale=[scaling_factor, 1])
        
        if (concatenated_flat.data <= 0).sum() > 0:
            warnings.warn(
                'Concatenated flat-field has negative values.', RuntimeWarning)
        
        # Plot concatenated flat
        plot2d(
            concatenated_flat.data, aspect='auto', cbar=False, 
            title='concatenated flat', show=show, save=save, path=fig_path)
        
        # Plot concatenated flat mask
        plot2d(
            concatenated_flat.mask.astype(int), vmin=0, vmax=1, aspect='auto', 
            cbar=False, title='concatenated flat mask', show=show, save=save, path=fig_path)
        
        # Write concatenated flat to file
        concatenated_flat.write(
            os.path.join(cal_path, 'concatenated_flat.fits'), overwrite=True)

        # Rectify curvature
        if verbose:
            print('    - Rectifying curvature...')
        
        if 'X' not in locals():
            X = np.load(os.path.join(cal_path, 'X.npy'))
        if 'Y' not in locals():
            Y = np.load(os.path.join(cal_path, 'Y.npy'))
        transformed_flat = transform(concatenated_flat, X=X, Y=Y)
        
        # Plot transformed flat
        plot2d(
            transformed_flat.data, aspect='auto', cbar=False, title='transformed flat', 
            show=show, save=save, path=fig_path)
        
        # Write transformed flat to file
        transformed_flat.write(
            os.path.join(cal_path, 'transformed_flat.fits'), overwrite=True)
        
        # Model response
        if verbose:
            print('\n[RESPONSE]')
        
        if 'V' not in locals():
            V = np.load(os.path.join(cal_path, 'V.npy'))
        response2d = response(
            ccd=transformed_flat, slit_along=slit_along, n_piece=n_piece, coordinate=V, 
            maxiters=0, sigma_lower=None, sigma_upper=None, grow=False, use_mask=True, 
            plot=save, path=fig_path)
        
        # Plot modeled response
        plot2d(
            response2d.data, aspect='auto', cbar=False, title='response', show=show, 
            save=save, path=fig_path)
        
        # Plot response mask
        plot2d(
            response2d.mask.astype(int), vmin=0, vmax=1, aspect='auto', cbar=False, 
            title='response mask', show=show, save=save, path=fig_path)
        
        # Write modeled response to file
        response2d.write(os.path.join(cal_path, 'response2d.fits'), overwrite=True)
        
        # Normalize
        reflat = concatenated_flat.divide(
            response2d, handle_mask='first_found', handle_meta='first_found')
        
        imstatistics(reflat, verbose=verbose)
        
        # Plot response calibrated flat
        plot2d(
            reflat.data, aspect='auto', cbar=False, title='reflat', show=show, 
            save=save, path=fig_path)
        
        # Write response calibrated flat to file
        reflat.write(os.path.join(cal_path, 'reflat.fits'), overwrite=True)
    
        # Model illumination
        if verbose:
            print('\n[ILLUMINATION]')
        
        # Illumination modeling
        illumination2d = illumination(
            ccd=reflat, slit_along=slit_along, method='Gaussian2D', sigma=sigma, 
            bins=20, maxiters=5, sigma_lower=3, sigma_upper=3, grow=1, 
            use_mask=True, plot=save, path=fig_path)

        # Plot modeled illumination
        plot2d(
            illumination2d.data, aspect='auto', cbar=False, title='illumination', 
            show=show, save=save, path=fig_path)
        
        # Plot illumination mask
        plot2d(
            illumination2d.mask.astype(int), vmin=0, vmax=1, aspect='auto', cbar=False, 
            title='illumination mask', show=show, save=save, path=fig_path)
        
        # Write illumination to file
        illumination2d.write(
            os.path.join(cal_path, 'illumination.fits'), overwrite=True)
        
        # Normalize
        normalized_flat = reflat.divide(
            illumination2d, handle_mask='first_found', handle_meta='first_found')
        
        imstatistics(normalized_flat, verbose=verbose)
        
        # Plot normalized flat
        plot2d(
            normalized_flat.data, aspect='auto', cbar=False, title='normalized flat', 
            show=show, save=save, path=fig_path)
        
        # Plot normalized flat mask
        plot2d(
            normalized_flat.mask.astype(int), vmin=0, vmax=1, aspect='auto', 
            cbar=False, title='normalized flat mask', show=show, save=save, 
            path=fig_path)
        
        # Write normalized flat to file
        normalized_flat.write(
            os.path.join(cal_path, 'normalized_flat.fits'), overwrite=True)
    
    # Correct targets
    if 's' in procedures:
        
        if verbose:
            print('\n[TARGET CORRECTION]')
        
        # Load targ
        if verbose:
            print('  - Loading targ...')
        ifc_targ = ifc.filter(regex_match=True, obstype='Telluric|Science')
        file_list = ifc_targ.files_filtered(include_path=True)
        targ_list = CCDDataList.read(file_list=file_list, hdu=hdu)
        
        # Trim
        if trim:
            if verbose:
                print('  - Trimming...')
            targ_list = targ_list.trim_image(fits_section=fits_section)
        
        # Correct gain
        if verbose:
            print('  - Correcting gain...')
        targ_list_gain_corrected = targ_list.gain_correct(gain=gain)
        
        targ_list_gain_corrected.statistics(verbose=verbose)
        
        # Create real uncertainty!!!
        if verbose:
            print('  - Creating deviation...')
        targ_list_gain_corrected_with_deviation = (
            targ_list_gain_corrected.create_deviation(
                gain=None, readnoise=rdnoise, disregard_nan=True)
        )
        
        # Flat-fielding
        if verbose:
            print('  - Flat-fielding...')
        if 'normalized_flat' not in locals():
            normalized_flat = CCDData.read(
                os.path.join(cal_path, 'normalized_flat.fits'))
        targ_list_flat_fielded = (
            targ_list_gain_corrected_with_deviation.flat_correct(normalized_flat)
        )
        
        # Bad pixel mask
        threshold = 0.4
        badpixel_mask = normalized_flat.data <= threshold
        
        for entry in sub_entries:
            
            index_sub = file_list.index(os.path.join(data_dir, entry[0]))
            index_sky = file_list.index(os.path.join(data_dir, entry[1]))
            
            obstype = targ_list_flat_fielded[index_sub].header['OBSTYPE']
            
            # Subtract
            if verbose:
                print(f'  - Dealing with {obstype} pair: {entry[0]}, {entry[1]}')
                print(f'    - Subtracting...')
            
            diff = targ_list_flat_fielded[index_sub].subtract(
                targ_list_flat_fielded[index_sky], handle_meta='first_found')
            
            diff.header.set('OBJECT', entry[2], diff.header.comments['OBJECT'])
            diff.header.set('PAIRNAME', entry[1])
            diff.header.set('COMMENT', f'{entry[0]} - {entry[1]}')
            
            # Fix bad pixels
            if verbose:
                print(f'    - Fixing bad pixels...')
            
            median_image = ndimage.median_filter(diff.data, size=7)
            diff.data[badpixel_mask] = (median_image[badpixel_mask])
            
            # Remove cosmic ray
            if diff.header['OBSTYPE'] == 'Science':
                
                if verbose:
                    print(f'    - Identifying cosmic ray pixels...')
                
                diff_cosmicray_corrected, cr_mask = cosmicray_lacosmic(
                    ccd=np.abs(diff.data), sigclip=4.5, sigfrac=0.3, objlim=5.0, 
                    invar=(diff.uncertainty.array**2), gain=1.0, readnoise=6.5, 
                    satlevel=5e4, niter=5, sepmed=True, cleantype='meanmask', 
                    fsmode='median', verbose=verbose)
                
                diff_cosmicray_corrected[diff.data < 0] *= -1
                diff.data[cr_mask] = diff_cosmicray_corrected[cr_mask]
            
            # Transform
            X = np.load(os.path.join(cal_path, f'X_{entry[2]}.npy'))
            Y = np.load(os.path.join(cal_path, f'Y_{entry[2]}.npy'))
            diff_transformed = transform(ccd=diff, X=X, Y=Y)
            
            file_name = f"sub{entry[0].split('_')[-1][:-5]}-{entry[1].split('_')[-1]}"
            
            # Plot difference image
            plot2d(
                diff_transformed.data, aspect='auto', cbar=False, title=file_name[:-5], 
                show=show, save=save, path=fig_path)
            
            # Write difference image to file
            diff_transformed.write(os.path.join(pro_path, file_name), overwrite=True)


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
    parser.add_argument(
        '-p', '--procedures', default='htafs', type=str, 
        help='Procedures: h[eader], t[rim], a[rc], f[lat], s[ubtract].'
    )
    parser.add_argument(
        '-r', '--reference', default='', type=str, 
        help='Reference spectrum for wavelength calibration.'
    )
    parser.add_argument(
        '-hv', '--high_voltage', required=True, type=str, 
        help='Voltage of the high voltage flat.'
    )
    parser.add_argument(
        '-lv', '--low_voltage', required=True, type=str, 
        help='Voltage of the low voltage flat.'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true', 
        help='Verbose or not.'
    )
        
    # Parse
    args = parser.parse_args()
    data_dir = os.path.abspath(args.input_dir)
    save_dir = os.path.abspath(args.output_dir)
    procedures = args.procedures
    reference = os.path.abspath(args.reference)
    high_voltage = args.high_voltage
    low_voltage = args.low_voltage
    verbose = args.verbose

    # Internal parameters
    glob_include = '*.fits'
    hdu = 0
    shape = (2048, 2048)
    slit_along = 'row'
    keywords = [
        'date-obs', 'obstype', 'object', 'ra', 'dec', 'exptime', 'airmass', 'enoise', 
        'egain', 'readmode']
    
    fits_section = '[955:1186, 7:1990]'
    index = 1000
    n_piece = 101
    sigma = (10, 1)
    dtype = 'float32'
    show = False
    save = True
    mem_limit = 500e6
    
    if 'h' in procedures:
        hdr_entries = os.path.join(save_dir, 'hdr_entries.dat')
        if os.path.exists(hdr_entries):
            hdr_entries = np.loadtxt(hdr_entries, dtype=str)
        else:
            hdr_entries = None
    else:
        hdr_entries = None
    
    if 's' in procedures:
        sub_entries = np.loadtxt(os.path.join(save_dir, 'sub_entries.dat'), dtype=str)
    else:
        sub_entries = None
    
    if not os.path.isfile(reference):
        reference = os.path.join(LIBRARY_PATH, 'fire_arc1d.fits')
    reference = loadSpectrum1D(reference)
    
    # Run pipeline
    pipeline(
        save_dir=save_dir, data_dir=data_dir, glob_include=glob_include, 
        procedures=procedures, hdu=hdu, keywords=keywords, fits_section=fits_section, 
        slit_along=slit_along, reference=reference, high_voltage=high_voltage, 
        low_voltage=low_voltage, index=index, n_piece=n_piece, sigma=sigma, 
        sub_entries=sub_entries, hdr_entries=hdr_entries, dtype=dtype, 
        mem_limit=mem_limit, show=show, save=save, verbose=verbose)

if __name__ == '__main__':
        main()