"""
mini-Sitian pipeline (version 2023)
"""

import os, argparse, toml
from glob import glob

# NumPy
import numpy as np
# AstroPy
import astropy.units as u
from astropy.io import fits
from astropy.stats import mad_std
from astropy.nddata import CCDData
from astropy.config import reload_config
# ccdproc
from ccdproc import ImageFileCollection
# drpy
from drpy.batch import CCDDataList
from drpy.plotting import plot2d
from drpy.utils import imstatistics

from .. import conf
from ..utils import login

# Load parameters from configuration file
reload_config(packageormod='plpy', rootname='plpy')


def pipeline(data_dir, telescope, glob_include, glob_exclude, keywords, hdu, rdnoise, 
             steps, verbose):
    """
    """

    data_dir = os.path.join(data_dir, telescope)
    
    # Construct image file collection
    ifc = ImageFileCollection(
        location=data_dir, keywords=keywords, find_fits_by_reading=False, 
        filenames=None, glob_include=glob_include, glob_exclude=glob_exclude, 
        ext=hdu)

    # Load gain and readout noise
    first_file = ifc.files_filtered(include_path=True)[0]
    gain = fits.getval(first_file, 'GAIN', ext=hdu) * u.photon / u.adu
    rdnoise = rdnoise * u.photon # !!! Calculate from bias or read from header !!!
    
    # Bias
    if 'bias' in steps:

        if verbose:
            print('\n[BIAS COMBINATION]')

        # Filter
        ifc_bias = ifc.filter(regex_match=True, imagetyp='BIAS')
        if verbose:
            ifc_bias.summary.pprint_all()
        
        # Load bias
        if verbose:
            print('  - Loading bias...')
        bias_list = CCDDataList.read(
            file_list=ifc_bias.files_filtered(include_path=True), hdu=hdu, unit=None)

        # Correct gain
        if verbose:
            print('  - Correcting gain...')
        bias_list_gain_corrected = bias_list.gain_correct(gain=gain)

        # Combine bias
        if verbose:
            print('  - Combining...')
        master_bias = bias_list_gain_corrected.combine(
            method='average', sigma_clip=True, sigma_clip_low_thresh=3, 
            sigma_clip_high_thresh=3, sigma_clip_func=np.ma.median, 
            sigma_clip_dev_func=mad_std, mem_limit=conf.mem_limit, 
            output_file=f'cal/master_bias_{telescope}.fits', 
            overwrite_output=True)

        # Check statistics 
        bias_list_gain_corrected.statistics(verbose=verbose)
        imstatistics(master_bias, verbose=verbose)

        # Plot master bias
        plot2d(
            master_bias.data, title=f'master bias {telescope}', show=conf.show, 
            save=conf.save, path='fig')
        
        del bias_list, bias_list_gain_corrected, master_bias
    
    # Flat
    if 'flat' in steps:

        if verbose:
            print('\n[FLAT COMBINATION]')

        # Filter
        ifc_flat = ifc.filter(regex_match=True, imagetyp='FLAT')
        if verbose:
            ifc_flat.summary.pprint_all()
        
        # Load flat
        if verbose:
            print('  - Loading flat...')
        flat_list = CCDDataList.read(
            file_list=ifc_flat.files_filtered(include_path=True), hdu=hdu, unit=None)

        # Correct gain
        if verbose:
            print('  - Correcting gain...')
        flat_list_gain_corrected = flat_list.gain_correct(gain=gain)

        # Subtract bias
        if verbose:
            print('  - Subtracting bias...')
        master_bias = CCDData.read(f'cal/master_bias_{telescope}.fits')
        flat_list_bias_subtracted = flat_list_gain_corrected - master_bias
        
        # Combine flat
        if verbose:
            print('  - Combining flat...')
        scaling_func = lambda ccd: 1 / np.ma.average(ccd)
        scale = 1 / ifc_flat.summary[params['exposure']].value
        master_flat = flat_list_bias_subtracted.combine(
            method='average', scale=scale, sigma_clip=True, 
            sigma_clip_low_thresh=3, sigma_clip_high_thresh=3, 
            sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std, 
            mem_limit=conf.mem_limit, 
            output_file=f'cal/master_flat_{telescope}.fits', 
            overwrite_output=True)

        # Check statistics
        flat_list_gain_corrected.statistics(verbose=verbose)
        flat_list_bias_subtracted.statistics(verbose=verbose)
        imstatistics(master_flat, verbose=verbose)
        
        # Plot master flat
        plot2d(
            master_flat.data, title=f'master flat, {telescope}', show=conf.show, 
            save=conf.save, path='fig')
        
        del (flat_list, flat_list_gain_corrected, master_bias, flat_list_bias_subtracted, 
             master_flat)
        
    # Target
    if 'target' in steps:

        if verbose:
            print('\n[TARGET CORRECTION]')

        # Filter
        ifc_targ = ifc.filter(regex_match=True, imagetyp='OBJECT')
        if verbose:
            ifc_targ.summary.pprint_all()

        # Load data
        if verbose:
            print('  - Loading target...')
        targ_list = CCDDataList.read(
            file_list=ifc_targ.files_filtered(include_path=True), hdu=hdu, unit=None)

        # Correct gain
        if verbose:
            print('  - Correcting gain...')
        targ_list_gain_corrected = targ_list.gain_correct(gain=gain)

        # Subtract bias
        if verbose:
            print('  - Subtracting bias...')
        master_bias = CCDData.read(f'cal/master_bias_{telescope}.fits')
        targ_list_bias_subtracted = targ_list_gain_corrected - master_bias

        # Create real uncertainty
        if verbose:
            print('  - Creating deviation...')
        targ_list_bias_subtracted_with_deviation = (
            targ_list_bias_subtracted.create_deviation(
                gain=None, readnoise=rdnoise, disregard_nan=True)
        )
        
        # Flat-fielding
        if verbose:
            print('  - Flat-fielding...')
        master_flat = CCDData.read(
            f'cal/master_flat_{telescope}.fits', unit=u.dimensionless_unscaled)
        targ_list_flat_fielded = targ_list_bias_subtracted_with_deviation / master_flat
        
        # Remove cosmic-ray
        if verbose:
            print('  - Removing cosmic rays...')
        targ_list_corrected = targ_list_flat_fielded.cosmicray_lacosmic(
            use_mask=False, gain=(1 * u.dimensionless_unscaled), readnoise=rdnoise, 
            sigclip=4.5, sigfrac=0.3, objlim=1, niter=5, verbose=verbose)

        # Write images to file
        for ccd in targ_list_corrected:
            fits_name = '{}_corrected.fits'.format(ccd.header['FILENAME'][:-5])
            if verbose:
                print(f'  - Writing to {fits_name}...')
            ccd.write(os.path.join('red', fits_name), overwrite=True)
    

def main():

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
        '-b', '--band', required=True, type=str, choices=['1', '2', '3'], 
        help='Band (123 for sloan gri).'
    )
    # parser.add_argument(
    #     '-c', '--combine', action='store_true', 
    #     help='Combine or not.'
    # )
    # parser.add_argument(
    #     '-k', '--keyword', default='object', type=str, 
    #     help='Keyword for grouping.'
    # )
    parser.add_argument(
        '-v', '--verbose', action='store_true', 
        help='Verbose or not.'
    )

    # Parse
    args = parser.parse_args()
    data_dir = os.path.abspath(args.input_dir)
    save_dir = os.path.abspath(args.output_dir)
    telescope = f'mst{args.band}'
    # combine = args.combine
    # keyword = args.keyword
    verbose = args.verbose

    if verbose:
        login(instrument='mini-Sitian', width=100)

    # Change working directory
    if verbose:
        print(f'- Changing working directory to {save_dir}...')
    os.chdir(save_dir)

    # Check setup
    for directory in ['cal', 'fig', 'red']:
        if not os.path.isdir(directory):
            raise RuntimeError(f'Run command `plpy_setup` first to set up {save_dir}.')
    if not os.path.isfile('params.toml'):
        raise RuntimeError(f'Run command `plpy_setup` first to set up {save_dir}.')

    # Load inputs
    params = toml.load('params.toml')

    pipeline(
        data_dir=data_dir, telescope=telescope, glob_include=params['include'], 
        glob_exclude=params['exclude'], keywords=params['keywords'], hdu=params['hdu'], 
        rdnoise=params['rdnoise'], steps=params['steps'], verbose=verbose)