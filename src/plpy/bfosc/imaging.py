"""
BFOSC photometry pipeline
"""

import os, argparse

# NumPy
import numpy as np
# AstroPy
from astropy.io import fits
from astropy.stats import mad_std
from astropy.nddata import CCDData
# ccdproc
from ccdproc import ImageFileCollection
# drpy
from drpy.batch import CCDDataList
from drpy.utils import imstatistics
from drpy.plotting import plot2d

from ..utils import makeDirectory, modifyHeader
from .utils import login


def pipeline(save_dir, data_dir, hdu, keywords, filters, steps, row_range, col_range, 
             custom_mask, mem_limit, show, save, verbose, mode):
    
    """BFOSC photometry pipeline."""
    
    # Login message
    if verbose:
        login('photometry')
    
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
        
        # Load bias
        if verbose:
            print('  - Loading bias...')
        ifc_bias = ifc.filter(regex_match=True, obstype='BIAS')
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
    
    # Flat combination
    if ('flat.combine' in steps) or ('flat' in steps):
        
        if verbose:
            print('\n[FLAT COMBINATION]')
        
        for f in filters:
            
            # Load flat
            if verbose:
                print(f'  - Loading {f} flat...')
            ifc_flat = ifc.filter(
                regex_match=True, obstype='PHOTFLAT', filter=f'Free_{f}_Free')
            flat_list = CCDDataList.read(
                file_list=ifc_flat.files_filtered(include_path=True), hdu=hdu)

            # Trim
            if trim:
                if verbose:
                    print('  - Trimming...')
                flat_list = flat_list.trim(row_range=row_range, col_range=col_range)

            flat_list.statistics(verbose=verbose)

            # Subtract bias
            #   Uncertainties created here (equal to that of ``master_bias``) are 
            #   useless!!!
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
                method='median', scale=scaling_func, sigma_clip=True, 
                sigma_clip_low_thresh=3, sigma_clip_high_thresh=3, 
                sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std, 
                mem_limit=mem_limit, 
                output_file=os.path.join(cal_path, f'combined_flat_{f}.fits'), 
                overwrite_output=True)

            imstatistics(combined_flat, verbose=verbose)

            # Plot combined flat
            plot2d(
                combined_flat.data, title=f'combined_flat_{f}', show=show, save=save, 
                path=fig_path)

            # Release memory
            del flat_list, flat_list_bias_subtracted, combined_flat
    
    
    # Correct targets
    if ('targ' in steps):
        
        if verbose:
            print('\n[TARGET CORRECTION]')
        
        # Target set
        targ_set = set(
            ifc.filter(regex_match=True, obstype='PHOTTARGET').summary['object'].data)
        
        for targ in targ_set:
            
            targ_path = makeDirectory(
                parent=pro_path, child=''.join(i for i in targ if i not in '\/:*?<>|'), 
                verbose=False)
                
            for f in filters:

                # Load targ
                if verbose:
                    print(f'  - Loading {f} of {targ}...')
                ifc_targ = ifc.filter(
                    regex_match=True, obstype='PHOTTARGET', object=targ, 
                    filter=f'Free_{f}_Free')
                file_list = ifc_targ.files_filtered(include_path=True)
                targ_list = CCDDataList.read(file_list=file_list, hdu=hdu)
                
                gain = float(fits.getval(file_list[0], 'GAIN', ext=hdu))
                rdnoise = float(fits.getval(file_list[0], 'RDNOISE', ext=hdu))

                # Trim
                if trim:
                    if verbose:
                        print('  - Trimming...')
                    targ_list = targ_list.trim(row_range=row_range, col_range=col_range)

                targ_list.statistics(verbose=verbose)

                # Subtract bias
                #   Uncertainties created here (equal to that of ``master_bias``) are 
                #   useless!!!
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
                combined_flat = CCDData.read(
                    os.path.join(cal_path, f'combined_flat_{f}.fits'))
                targ_list_flat_fielded = (
                    targ_list_bias_subtracted_with_deviation / combined_flat
                )

                # Remove cosmic ray
                if verbose:
                    print('  - Removing cosmic ray...')
                targ_list_cosmicray_corrected = targ_list_flat_fielded.cosmicray(
                    method='Laplacian', use_mask=False, gain=gain, readnoise=rdnoise, 
                    sigclip=4.5, sigfrac=0.3, objlim=1, niter=5, verbose=verbose)

                for i, targ_cosmicray_corrected in enumerate(targ_list_cosmicray_corrected):

                    file_name = targ_cosmicray_corrected.header['FILENAME']

                    # Plot corrected target
                    plot2d(
                        targ_cosmicray_corrected.data, title=file_name, show=show, 
                        save=save, path=fig_path)

                    # Write corrected target to file
                    if verbose:
                        print(f'  - Saving {file_name} to {targ_path}...')
                    targ_cosmicray_corrected.write(
                        os.path.join(targ_path, file_name), overwrite=True)
    
    
def main():
    """Command line tool."""
    
    # External parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--data_dir', required=True, type=str, 
        help='Data (input) directory.'
    )
    parser.add_argument(
        '-f', '--filters', required=True, type=str, 
        help='Filters.'
    )
    parser.add_argument(
        '-o', '--save_dir', default='', type=str, 
        help='Saving (output) directory.'
    )
    parser.add_argument(
        '-m', '--mode', default='general', type=str, choices=['general', 'standard'], 
        help='General or standard.'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true', 
        help='Verbose or not.'
    )
    
    # Parse
    args = parser.parse_args()
    data_dir = os.path.abspath(args.data_dir)
    filters = list(args.filters)
    save_dir = os.path.abspath(args.save_dir)
    mode = args.mode
    verbose = args.verbose
    
    # Internal parameters
    hdu = 0
    shape = (2048, 2048)
    keywords = [
        'date-obs', 'obstype', 'object', 'ra', 'dec', 'filter', 'exptime', 'rdnoise', 
        'gain']
    # steps = ['header', 'bias', 'flat', 'targ']
    steps = ['targ']
    row_range = None
    col_range = None
    
    for f in filters:
        if f not in list('UBVRI'):
            raise ValueError(f'Filter {f} not available.')
    
    custom_mask = np.zeros(shape, dtype=bool)
    
    pipeline(
        save_dir=save_dir, data_dir=data_dir, hdu=hdu, keywords=keywords, 
        filters=filters, steps=steps, row_range=row_range, col_range=col_range, 
        custom_mask=custom_mask, mem_limit=500e6, show=False, save=True, 
        verbose=verbose, mode=mode)

    
if __name__ == '__main__':
    main()