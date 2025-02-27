"""
P200/TSPEC pipeline
"""

import os, argparse, warnings, toml
from glob import glob

# NumPy
import numpy as np
# SciPy
from scipy import signal
from scipy.ndimage import gaussian_filter
# matplotlib
import matplotlib.pyplot as plt
# AstroPy
import astropy.units as u
from astropy.io import fits
from astropy.stats import mad_std
from astropy.nddata import CCDData
from astropy.config import reload_config
from astropy.utils.exceptions import AstropyUserWarning
# ccdproc
from ccdproc import ImageFileCollection, cosmicray_lacosmic
# specutils
from specutils import Spectrum1D
# drpy
from drpy.batch import CCDDataList
from drpy.twodspec import fitcoords, transform, trace
from drpy.twodspec.utils import invertCoordinateMap
from drpy.onedspec import dispcor
from drpy.plotting import plot2d, _plot2d
from drpy.utils import imstatistics

from .. import conf
from ..utils import login, getFileName

# Load parameters from configuration file
reload_config(packageormod='plpy', rootname='plpy')


def _toRectangle(ccd, mask):
    """Transform the orders in the input frame to a rectangular array."""
    
    rect = ccd.copy()
    
    n_row = mask[:, 0].sum()
    
    rect.data = ccd.data.T[mask.T].reshape(-1, n_row).T

    if rect.uncertainty is not None:
        rect.uncertainty.array = ccd.uncertainty.array.T[mask.T].reshape(-1, n_row).T

    if rect.mask is not None:
        rect.mask = ccd.mask.T[mask.T].reshape(-1, n_row).T

    return rect


def flat():
    """Make flat-field."""
    
    # External parameters
    parser = argparse.ArgumentParser(
        prog='plpy_tspec_flat',
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
        '-v', '--verbose', action='store_true', 
        help='Verbose or not.'
    )
        
    # Parse
    args = parser.parse_args()
    data_dir = os.path.abspath(args.input_dir)
    save_dir = os.path.abspath(args.output_dir)
    verbose = args.verbose

    if verbose:
        login(instrument='P200/TSPEC', width=100)

    # Change working directory
    if verbose:
        print(f'- Changing working directory to {save_dir}...')
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
        filenames=None, glob_include=params['flat']['include'], 
        glob_exclude=params['flat']['exclude'], ext=params['hdu'])

    # Load gain
    first_file = ifc.files_filtered(include_path=True)[0]
    gain = fits.getval(first_file, 'GAIN', ext=params['hdu']) * u.photon / u.adu

    # Load data
    ifc_flat = ifc.filter(regex_match=True, obstype='^flat$')
    if verbose:
        print('- Loading flat-fields...')
        ifc_flat.summary.pprint_all()
    flat_list = CCDDataList.read(
        file_list=ifc_flat.files_filtered(include_path=True), hdu=params['hdu'])

    ifc_dark = ifc.filter(
        regex_match=True, obstype='^dark$', 
        exptime=ifc_flat.summary[params['exposure']].data[0])
    if verbose:
        print('- Loading dark frames...')
        ifc_dark.summary.pprint_all()
    dark_list = CCDDataList.read(
        file_list=ifc_dark.files_filtered(include_path=True), hdu=params['hdu'])


    # Correct gain
    if verbose:
        print('- Correcting gain...')
    dark_list_gain_corrected = dark_list.gain_correct(gain=gain)
    flat_list_gain_corrected = flat_list.gain_correct(gain=gain)
        
    dark_list_gain_corrected.statistics(verbose=verbose)
    flat_list_gain_corrected.statistics(verbose=verbose)

    # Combine dark frames
    if verbose:
        print('- Combining dark frames...')
    dark_combined = dark_list_gain_corrected.combine(
        method='average', mem_limit=conf.mem_limit, sigma_clip=True, 
        sigma_clip_low_thresh=3, sigma_clip_high_thresh=3, 
        sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std, 
        output_file='cal/dark_combined.fits', dtype=conf.dtype, 
        overwrite_output=True)

    imstatistics(dark_combined, verbose=verbose)

    # Plot combined dark
    plot2d(
        dark_combined.data, aspect='auto', cbar=False, title='dark combined', 
        show=conf.show, save=conf.save, path='fig')

    # Release memory
    del dark_list, dark_list_gain_corrected

    # Subtract dark
    #   Uncertainties created here (equal to that of ``dark_combined``) are useless!!!
    if verbose:
        print('- Subtracting dark frame...')
    flat_list_dark_subtracted = flat_list_gain_corrected.subtract_bias(dark_combined)

    flat_list_dark_subtracted.statistics(verbose=verbose)

    # Combine flats
    #   Uncertainties created above are overwritten here!!!
    if verbose:
        print('- Combining flat-fields...')
    flat_combined = flat_list_dark_subtracted.combine(
        method='average', mem_limit=conf.mem_limit, sigma_clip=True, 
        sigma_clip_low_thresh=3, sigma_clip_high_thresh=3, 
        sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std, 
        output_file='cal/flat_combined.fits', dtype=conf.dtype, 
        overwrite_output=True)

    imstatistics(flat_combined, verbose=verbose)

    # Plot combined flat
    plot2d(
        flat_combined.data, aspect='auto', cbar=False, title='flat combined', 
        show=conf.show, save=conf.save, path='fig')

    # Release memory
    del flat_list, flat_list_dark_subtracted

    # Locate orders
    if verbose:
        print('- Locating orders...')
    
    # Threshold for binarization (see below).
    threshold = np.percentile(flat_combined.data, params['flat']['threshold'] * 100)

    # What we need is something like this binarized map. The problem of this map is 
    # that the width may vary along the dispersion axis, which means the masked region 
    # may not be converted to a rectangular array. To fix this, we trace each order and 
    # set a common width.
    flat_binarized = np.where((flat_combined.data > threshold), 1.0, 0.0)

    flat_smoothed = gaussian_filter(flat_binarized, sigma=(25, 0))

    peaks, _ = signal.find_peaks(
        flat_smoothed[:, 0], height=0.9, distance=100, width=50, rel_height=1)

    if peaks.shape[0] != params['n_order']:
        raise RuntimeError(
            f"{params['n_order']} orders are expected, but {peaks.shape[0]} orders are found."
        )

    locations = np.zeros([(params['n_order'] + 1), params['shape'][1]])

    for i in range(params['n_order']):

        interval = (
            f"[{(peaks[i] - params['slit_length'] // 2)}:"
            f"{(peaks[i] + params['slit_length'] // 2)}]"
        )

        trace1d = trace(
            ccd=flat_smoothed, slit_along='col', fwhm=10, method='trace', 
            reference_bin=0, interval=interval, n_med=10, n_piece=2, maxiters=5, 
            sigma_lower=2, sigma_upper=2, grow=False, title=None, show=False, 
            save=False, path=None)

        if i == 0:
            locations[-1] = trace1d.spectral_axis.value

        locations[i] = np.round(trace1d.flux.value)

    # The following region is useless. Set to NaN for nicer plot.
    locations[0][800:] = np.nan

    if conf.show | conf.save:

        fig, ax = plt.subplots(1, 1)

        # Flat-field
        _plot2d(ax, flat_combined.data, aspect='auto', cbar=False)
        
        # Location of orders
        for i in range(params['n_order']):
            ax.plot(
                locations[-1], (locations[i] - params['slit_length'] // 2), 'r', lw=2, 
                zorder=0)
            ax.plot(
                locations[-1], (locations[i] + params['slit_length'] // 2), 'r', lw=2, 
                zorder=0)
            ax.annotate(
                f'order {7-i}', xy=(50, locations[i][50]), xycoords='data', 
                va='center', color='r', fontsize=16)

        ax.set_title('order location', fontsize=16)
        fig.set_figheight(params['shape'][0] / params['shape'][1] * fig.get_figwidth())
        fig.tight_layout()

        if conf.save:
            plt.savefig('fig/order_location.png', dpi=100)

        if conf.show:
            plt.show()

        plt.close()

    # Set to int type for slicing
    locations[0][800:] = 200
    locations = locations.astype(int)

    # Use a mask to store the location of the orders
    mask = np.zeros_like(flat_combined.data, dtype=bool)

    for i in range(params['n_order']):

        lower_edge = locations[i] - params['slit_length'] // 2
        upper_edge = locations[i] + params['slit_length'] // 2

        for j in locations[-1]:
            mask[lower_edge[j]:(upper_edge[j] + 1), j] = True

    np.save('cal/order.npy', mask)

    # Transform flat-field to rectangle
    if verbose:
        print('- Transforming flat-field to rectangle...')
    flat_rectangled = _toRectangle(flat_combined, mask)
    
    # Plot rectangled flat
    plot2d(
        flat_rectangled.data, aspect='auto', cbar=False, title='flat rectangled', 
        show=conf.show, save=conf.save, path='fig')

    # Write rectangled flat to file
    flat_rectangled.write('cal/flat_rectangled.fits', overwrite=True)

    # Normalize rectangular flat-field
    if verbose:
        print('- Normalizing rectangular flat-field...')

    flat_normalized = flat_rectangled.divide(
        (1 * flat_rectangled.unit), handle_mask='first_found', 
        handle_meta='first_found')
    flat_normalized.uncertainty = None

    width = 10

    for i in range(params['n_order']):

        lower_edge = i * params['slit_length']
        upper_edge = i * params['slit_length'] + params['slit_length']

        # Scale to the same level
        scaling_factor = (
            np.median(flat_normalized.data[lower_edge:upper_edge, 1024:(1024+width)]) 
            / np.median(flat_normalized.data[lower_edge:upper_edge, (1024-width):1024])
        )

        flat_normalized.data[lower_edge:upper_edge, :1024] = 1.0
        flat_normalized.data[lower_edge:upper_edge, 1024:] = scaling_factor

#         # A simple response
#         response = np.median(flat_normalized.data[lower_edge:upper_edge, :], axis=0)
#         response[1024:] /= scaling_factor

#         # Normalize
#         flat_normalized.data[lower_edge:upper_edge, :] /= response
#         flat_normalized.uncertainty.array[lower_edge:upper_edge, :] /= response

    # Plot normalized flat
    plot2d(
        flat_normalized.data, aspect='auto', cbar=False, title='flat normalized', 
        show=conf.show, save=conf.save, path='fig')

    # Write normalized flat to file
    flat_normalized.write('cal/flat_normalized.fits', overwrite=True)

    return


def _genSky(A, B):
    """Generate a sky frame.
    
    (A + B) - abs(A - B)
    """
    
    # A + B
    A_plus_B = A.add(B, handle_mask='first_found', handle_meta='first_found')
    
    # abs(A - B)
    A_minus_B = A.subtract(B, handle_mask='first_found', handle_meta='first_found')
    A_minus_B.data = np.abs(A_minus_B.data)

    # A + B - abs(A - B)
    sky2d = A_plus_B.subtract(
        A_minus_B, handle_mask='first_found', handle_meta='first_found')
    
    return sky2d


def _abba2abab(filenames):
    """Turn an ABBA sequence to an ABAB sequence."""

    n = len(filenames) // 4

    for i in range(n):
        filenames[(4 * i + 2)], filenames[(4 * i + 3)] = (
            filenames[(4 * i + 3)], filenames[(4 * i + 2)]
        )

    return filenames


def _subtract(ccd_list):
    """Subtract sky background.
    
    A - B. Take ABAB... sequence as input.
    """

    n = len(ccd_list) // 2

    sky_list = ccd_list.to_list()
    for i in range(n):
        sky_list[(2 * i)], sky_list[(2 * i + 1)] = (
            sky_list[(2 * i + 1)], sky_list[(2 * i)]
        )

    # Note that here ``ccd_list`` and ``sky_list`` are different data structures.
    ccd_list_background_subtracted = list()
    for ccd, sky in zip(ccd_list, sky_list):
        ccd_list_background_subtracted.append(
            ccd.subtract(sky, handle_mask='first_found', handle_meta='first_found')
        )

    return CCDDataList(ccd_list_background_subtracted)


def _fitcoords(ccd, shape, n_order, slit_length, save=False, fig_path=None):
    """Derive distortion."""

    X = np.zeros((n_order, slit_length, shape[1]))
    Y = np.zeros((n_order, slit_length, shape[1]))

    for i in range(n_order):

        lower_edge = i * slit_length
        upper_edge = i * slit_length + slit_length

        order = ccd[lower_edge:upper_edge]

        U, _ = fitcoords(
            ccd=order, slit_along='col', order=0, n_med=10, n_piece=1, prominence=0.01, 
            maxiters=3, sigma_lower=3, sigma_upper=3, grow=False, use_mask=True, 
            plot=save, path=fig_path, threshold=0, distance=3, width=3, wlen=10, 
            rel_height=1)
        
        X[i], Y[i] = invertCoordinateMap('col', U)

    return X, Y


def _rectify(ccd, n_order, slit_length, X, Y):
    """Rectify distortion."""

    ccd_transformed = ccd.copy()

    for i in range(n_order):
        
        lower_edge = i * slit_length
        upper_edge = i * slit_length + slit_length

        order = ccd[lower_edge:upper_edge]
        order_transformed = transform(ccd=order, X=X[i], Y=Y[i])

        # Data
        ccd_transformed.data[lower_edge:upper_edge] = order_transformed.data
        # Uncertainty
        if ccd_transformed.uncertainty is not None:
            ccd_transformed.uncertainty.array[lower_edge:upper_edge] = (
                order_transformed.uncertainty.array
            )
        # Mask
        if ccd_transformed.mask is not None:
            ccd_transformed.mask[lower_edge:upper_edge] = order_transformed.mask

    return ccd_transformed


def _dispcor(ccd, n_order, slit_length, save=False, fig_path=None, verbose=False):
    """Dispersion correction."""

    order1d_list_calibrated = list()

    for i in range(n_order):

        if verbose:
            print(f'  - Correcting order {(7 - i)}')

        lower_edge = i * slit_length
        upper_edge = i * slit_length + slit_length

        order = ccd[lower_edge:upper_edge]

        order1d = np.median(order.data, axis=0)
        order1d -= order1d.min()

        reference = os.path.join(
            os.path.split(__file__)[0], f'lib/tspec_order_{(7 - i)}.fits')

        order1d_calibrated = dispcor(
            spectrum1d=order1d, reverse=True, reference=reference, n_sub=20, 
            refit=True, prominence=1e-2, degree=1, maxiters=5, sigma_lower=3, 
            sigma_upper=3, grow=False, use_mask=False, title=f'order {(7 - i)}', 
            show=False, save=save, path=fig_path)
        order1d_calibrated.meta['header'] = ccd.meta
        
        order1d_list_calibrated.append(order1d_calibrated)
        
    return order1d_list_calibrated


def preprocess():
    """Subtract sky background and make a sky frame."""

    # External parameters
    parser = argparse.ArgumentParser(
        prog='plpy_tspec_prep',
        description='Subtract sky background and make a sky frame.'
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
        '-p', '--pattern', default='abba', type=str, choices=['abba', 'ab'], 
        help='Pattern of the input data.'
    )
    parser.add_argument(
        '-s', '--sky', required=True, type=str, 
        help='Index of sky frames.'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true', 
        help='Verbose or not.'
    )

    # Parse
    args = parser.parse_args()
    data_dir = os.path.abspath(args.input_dir)
    save_dir = os.path.abspath(args.output_dir)
    filenames = getFileName(args.index)
    pattern = args.pattern
    skynames = getFileName(args.sky)
    verbose = args.verbose

    # Verify ``filenames``
    if pattern == 'abba':
        assert len(filenames) % 4 == 0, (
            'For `abba` pattern, number of the data files should be a multiple of 4.')
        filenames = _abba2abab(filenames)

    elif pattern == 'ab':
        assert len(filenames) % 2 == 0, (
            'For `ab` pattern, number of the data files should be a multiple of 2.')

    # Verify ``skynames``
    assert len(skynames) == 2, (
        f'Two frames are expected to generate a sky frame. {len(skynames)} is given.')

    filenames.extend(skynames)

    if verbose:
        login(instrument='P200/TSPEC', width=100)

    # Change working directory
    if verbose:
        print(f'- Changing working directory to {save_dir}...')
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
        filenames=filenames, glob_include=None, glob_exclude=None, ext=params['hdu'])

    # Load gain & readout noise
    first_file = ifc.files_filtered(include_path=True)[0]
    gain = fits.getval(first_file, 'GAIN', ext=params['hdu']) * u.photon / u.adu
    rdnoise = (
        14.14 * u.photon / np.sqrt(fits.getval(first_file, 'FSAMPLE', ext=params['hdu']))
    )

    # Load data
    if verbose:
        print('- Load data...')
        ifc.summary.pprint_all()
    ccd_list = CCDDataList.read(
        file_list=ifc.files_filtered(include_path=True), hdu=params['hdu'])

    # Transform to rectangle
    if verbose:
        print('- Transforming to rectangle...')
    mask = np.load('cal/order.npy')
    ccd_list_rectangled = ccd_list.apply_over_ccd(_toRectangle, mask)

    # Correct gain
    if verbose:
        print('- Correcting gain...')
    ccd_list_gain_corrected = ccd_list_rectangled.gain_correct(gain=gain)

    # Create real uncertainty!!!
    if verbose:
        print('- Creating deviation...')
    ccd_list_with_deviation = (
        ccd_list_gain_corrected.create_deviation(
            gain=None, readnoise=rdnoise, disregard_nan=True)
    )

    # Flat-field
    if verbose:
        print('- Flat-fielding...')
    flat_normalized = CCDData.read('cal/flat_normalized.fits')
    ccd_list_flat_fielded = ccd_list_with_deviation.flat_correct(flat_normalized)

    # Generate sky frame
    if verbose:
        print(f'- Generating sky frame...')
    sky2d = _genSky(A=ccd_list_flat_fielded[-2], B=ccd_list_flat_fielded[-1])

    # Subtract sky background
    if verbose:
        print(f'- Subtracting sky background...')
    ccd_list_background_subtracted = _subtract(ccd_list_flat_fielded[:-2])

    # Rectify distortion
    if verbose:
        print(f'- Rectifying distortion...')
    X, Y = _fitcoords(
        sky2d, shape=params['shape'], n_order=params['n_order'], 
        slit_length=params['slit_length'], save=conf.save, fig_path='fig')
    sky2d_transformed = _rectify(sky2d, params['n_order'], params['slit_length'], X, Y)
    ccd_list_transformed = ccd_list_background_subtracted.apply_over_ccd(
        _rectify, params['n_order'], params['slit_length'], X, Y)

    # Plot sky
    skyname = 'sky2d_' + args.sky.replace(',', '_').replace('-', '_')
    plot2d(
        sky2d_transformed.data, aspect='auto', cbar=False, title=skyname, 
        show=conf.show, save=conf.save, path='fig')

    # Dispersion correction
    if verbose:
        print(f'- Correcting dispersion...')
    order1d_list_calibrated = _dispcor(
        sky2d_transformed, n_order=params['n_order'], slit_length=params['slit_length'], 
        save=conf.save, fig_path='fig', verbose=verbose)
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=AstropyUserWarning)
        for i, order1d_calibrated in enumerate(order1d_list_calibrated):
            ordername = skyname.replace('2d', '1d') + f'_order_{(7 - i)}.fits'
            order1d_calibrated.write(
                os.path.join('cal', ordername), format='tabular-fits', 
                overwrite=True)

    # Write transformed sky to file
    skyname += '.fits'
    if verbose:
        print(f"- Writing sky frame to {os.path.join('cal', skyname)}")
    sky2d_transformed.write(os.path.join('cal', skyname), overwrite=True)

    for i, ccd in enumerate(ccd_list_transformed):
        filename = f'sub{filenames[i][-9:-5]}.fits'
        if verbose:
            print(f"- Writing {filenames[i]} to {os.path.join('sub', filename)}")
        ccd.write(os.path.join('sub', filename), overwrite=True)

    return