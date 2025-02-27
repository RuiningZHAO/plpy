import os

# NumPy
import numpy as np
# SciPy
from scipy import ndimage, interpolate
# matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# AstroPy
import astropy.units as u
from astropy.time import Time
from astropy.nddata import CCDData
from astropy.config import reload_config
# ccdproc
from ccdproc.utils.slices import slice_from_string
# specutils
from specutils import Spectrum1D
# drpy
from drpy.twodspec.core import _trace
from drpy.validate import (_validateString, _validateBool, _validateCCD, _validateBins, 
                           _validateInteger, _validatePath)
from drpy.plotting import _plotFitting, _plot2d

from .. import conf

# Load parameters from configuration file
reload_config(packageormod='plpy', rootname='plpy')

# Set plot parameters
plt.rcParams['figure.figsize'] = [conf.fig_width, conf.fig_width]
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'


def getOrderInfo(ccd=None, sigma=3, extension=100):
    """
    """

    # Initial guess
    orders = np.load(os.path.join(os.path.split(__file__)[0], 'lib/order.npy'))
    intervals = orders[:, 1:]
    orders = orders[:, 0].astype(int)

    if ccd is not None:

        profile_lib = np.load(os.path.join(os.path.split(__file__)[0], 'lib/profile.npy'))
        profile_obs = np.median(ccd.data[:, 2000:2100], axis=1)

        profile_lib_smoothed = ndimage.gaussian_filter(profile_lib, sigma=sigma)
        profile_obs_smoothed = ndimage.gaussian_filter(profile_obs, sigma=sigma)

        profile_lib_extended = np.hstack(
            [profile_lib_smoothed[-extension:], profile_lib_smoothed, 
             profile_lib_smoothed[:extension]])

        profile_lib_normalized = profile_lib_extended / profile_lib_extended.max()
        profile_obs_normalized = profile_obs_smoothed / profile_obs_smoothed.max()

        shift = - extension + np.argmax(
            np.correlate(profile_lib_normalized, profile_obs_normalized, mode='valid'))

        intervals -= shift

    return orders, intervals


def traceEchelle(ccd, dispersion_axis, orders, intervals, method, fwhm=15, 
                 n_med=20, reference_bin=None, order_ref_red=None, 
                 order_ref_blue=None, range_red='[:]', range_blue='[:]', order=3, 
                 n_piece=3, degree=1, maxiters=5, sigma_lower=None, sigma_upper=None, 
                 grow=False, negative=False, use_mask=False, title='trace', 
                 show=conf.show, save=conf.save, path=None):
    """Trace on the 2-dimensional echelle spectrum.
    
    First the spatial profiles are binned by taking median along the dispersion axis. 
    Then the centroid of each order in the reference bin is determined. [...]
    
    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData` or `~numpy.ndarray`
        Input frame.
    
    dispersion_axis : str
        `col` or `row`. If `col`, the data array of ``ccd`` will be transposed during 
        calculations. Note that this will NOT affect the returned trace.

    orders : array_like
        Order numbers.

    intervals : `~numpy.ndarray`
        Intervals the orders lies in. Each element should be a 2-sequence specifying
        the boundary of the interval.
    
    method : str
        Centroiding method used in tracing. `gaussian` or `com`, and
        - if `gaussian`, peak center in each bin is determined using Gaussian fitting.
        This method is easy to fail when the data is noisy (for example, near the
        edges). Peak centers of these bins appear as NaNs in the trace fitting, so that
        they do not affect the result much as long as there are enough good peak
        centers.
        - if `com`, center of mass in each bin is determined. This method always
        provides peak center even when the data is quite noisy. The mask is not applied
        as well. A proper sigma-clipping is needed to avoid impact of bad peak centers
        on the trace fitting.

    fwhm : scalar
        Estimated full width at half maximum of the peak to be traced. (Rough 
        estimation is enough.)

    n_med : int, optional
        Number of spatial profiles to median. Must be >= `3`. Large number for faint 
        source.
        Default is `3`.
    
    reference_bin : int or `None`, optional
        Index of the reference bin.
        If `None`, the reference bin is the middle bin.

    
    order_ref_red : int or `None`, optional
        Reference order number for tracing orders at red end. This order and the redders
        are traced first. The trace of this order is then used to remove the curvature
        of the redder traces. Finally, the decurvated redder orders are fitted with
        low-order polynomials.

    order_ref_blue : int or `None`, optional
        Same as ``order_ref_red``, but for blue end.

    range_red : str, optional
        Traces at the red end can be very weak as they approach the edges of the frame.
        This argument specify a region that are bright enough to trace accurately.
        Regions beyond these specified areas are ignored in trace fitting to ensure
        that inaccurate centroiding in these weak regions does not affect the resulting
        traces.

    range_blue : str, optional
        Same as ``range_red``, but for blue end.
    
    order : int, optional
        Degree of the spline. Must be `5` >= ``order`` >= `1`.
        Default is `3`, a cubic spline.
    
    n_piece : int, optional
        Number of spline pieces. Lengths are all equal. Must be positive. 
        Default is `3`.

    degree : int, optional
        Degree of the polynomial used to fit decurvated traces. Could be `1` or `2`.
        Default is `1`, a linear fitting.

    maxiters : int, optional
        Maximum number of sigma-clipping iterations to perform. If convergence is 
        achieved prior to ``maxiters`` iterations, the clipping iterations will stop. 
        Must be >= `0`. 
        Default is `5`.

    sigma_lower : scalar or `None`, optional
        Number of standard deviations to use as the lower bound for the clipping limit. 
        If `None` (default), `3` is used.

    sigma_upper : scalar or `None`, optional
        Number of standard deviations to use as the upper bound for the clipping limit. 
        If `None` (default), `3` is used.
    
    grow : scalar or `False`, optional
        Radius within which to mask the neighbouring pixels of those that fall outwith 
        the clipping limits.

    negative : bool, optional
        The spectrum is negative or not. If `True`, the negative frame (the input frame 
        multiplied by `-1`) is used in traceing.
        Default is `False`.

    use_mask : bool, optional
        If `True` and a mask array is attributed to ``ccd``, the masked pixels are 
        ignored in the fitting. 
        Default is `False`.

    Returns
    -------
    trace1d : 
        Fitted trace.
    """

    _validateString(dispersion_axis, 'dispersion_axis', ['col', 'row'])

    fwhm = np.abs(fwhm)

    _validateBool(negative, 'negative')

    _validateBool(use_mask, 'use_mask')

    if dispersion_axis == 'row':
        nccd, data_arr, _, mask_arr = _validateCCD(ccd, 'ccd', False, use_mask, False)
    else:
        nccd, data_arr, _, mask_arr = _validateCCD(ccd, 'ccd', False, use_mask, True)

    if negative:
        data_arr *= -1

    n_row, n_col = data_arr.shape

    idx_row, idx_col = np.arange(n_row), np.arange(n_col)

    # Split into bins along dispersion axis
    _validateInteger(n_med, 'n_med', (3, None), (True, None))
    bin_edges, loc_bin, n_bin = _validateBins(n_med, n_col, isWidth=True)

    # If `None`, the reference bin will be the middle bin when the total number is odd 
    # while the first bin of the second half when the total number is even.
    if reference_bin is not None:
        _validateInteger(reference_bin, 'reference_bin', (0, (n_bin - 1)), (True, True))
    else:
        reference_bin = n_bin // 2

    count_bin = np.zeros([n_bin, n_row])
    mask_bin = np.zeros([n_bin, n_row], dtype=bool)

    for i in range(n_bin):

        bin_edge = bin_edges[i]

        # Bad pixels (NaNs or infs) in the original frame (if any) may lead to unmasked
        # elements in ``count_bin[i]`` and may cause an error in the Gaussian fitting
        # below.
        count_bin[i] = np.nanmedian(data_arr[:, bin_edge[0]:bin_edge[1]], axis=1)
        mask_bin[i] = np.all(mask_arr[:, bin_edge[0]:bin_edge[1]], axis=1)

    # todo: validate `orders`
    orders = np.array(orders)

    if order_ref_red is None:
        order_ref_red = orders[0]
    else:
        _validateInteger(
            order_ref_red, 'order_ref_red', (orders[0], orders[-1]), (True, False)
        )

    if order_ref_blue is None:
        order_ref_blue = orders[-1]
    else:
        _validateInteger(
            order_ref_blue, 'order_ref_blue', (order_ref_red, orders[-1]), (False, True)
        )
        

    idx_red = order_ref_red - orders[0]
    idx_blue = order_ref_blue - orders[0]

    n_order = orders.shape[0]
    fitted_trace_arr = np.zeros([n_order, n_col])
    refined_trace_arr = np.zeros([n_order, n_bin])
    residual_arr = np.zeros_like(refined_trace_arr)
    master_mask = np.zeros_like(refined_trace_arr, dtype=bool)
    threshold_lower = np.zeros(n_order)
    threshold_upper = np.zeros(n_order)

    for i in np.arange(n_order)[idx_red:(idx_blue + 1)]:

        interval = f'[{round(intervals[i][0])}:{round(intervals[i][1])}]'

        (_, fitted_trace_arr[i], refined_trace_arr[i], residual_arr[i], master_mask[i], 
         threshold_lower[i], threshold_upper[i]) = _trace(
            mode='trace', method=method, count_bin=count_bin, mask_bin=mask_bin, 
            loc_bin=loc_bin, n_bin=n_bin, k=reference_bin, interval=interval, 
            fwhm=fwhm, idx_row=idx_row, idx_col=idx_col, order=order, n_piece=n_piece, 
            maxiters=maxiters, sigma_lower=sigma_lower, sigma_upper=sigma_upper, 
            grow=grow)

    _validateInteger(degree, 'degree', (1, 2), (True, True))

    if order_ref_red > orders[0]:

        trace_red = np.interp(loc_bin, idx_col, fitted_trace_arr[idx_red])
        count_red = np.zeros_like(count_bin)
        mask_red = np.zeros_like(mask_bin, dtype=bool)
        for i in range(n_bin):
            count_red[i] = np.interp(
                idx_row + trace_red[i] - trace_red[reference_bin], idx_row, count_bin[i])
            mask_red[i] = 0.1 < np.interp(
                idx_row + trace_red[i] - trace_red[reference_bin], idx_row, mask_bin[i])

        idx_min, idx_max = idx_col[slice_from_string(range_red)][[0, -1]]
        mask_red[(loc_bin < idx_min) | (idx_max < loc_bin), :] = True

        for i in np.arange(n_order)[:idx_red]:

            interval = f'[{round(intervals[i][0])}:{round(intervals[i][1])}]'

            (_, fitted_trace_arr[i], refined_trace_arr[i], residual_arr[i], 
             master_mask[i], threshold_lower[i], threshold_upper[i]) = _trace(
                mode='trace', method=method, count_bin=count_red, mask_bin=mask_red, 
                loc_bin=loc_bin, n_bin=n_bin, k=reference_bin, interval=interval, 
                fwhm=fwhm, idx_row=idx_row, idx_col=idx_col, order=degree, n_piece=1, 
                maxiters=maxiters, sigma_lower=sigma_lower, sigma_upper=sigma_upper, 
                grow=grow)

        fitted_trace_arr[:idx_red] += (
            fitted_trace_arr[idx_red] - trace_red[reference_bin])
        refined_trace_arr[:idx_red] += trace_red - trace_red[reference_bin]

    if order_ref_blue < orders[-1]:

        trace_blue = np.interp(loc_bin, idx_col, fitted_trace_arr[idx_blue])
        count_blue = np.zeros_like(count_bin)
        mask_blue = np.zeros_like(mask_bin, dtype=bool)

        for i in range(n_bin):

            count_blue[i] = np.interp(
                idx_row + trace_blue[i] - trace_blue[reference_bin], idx_row, count_bin[i])
            mask_blue[i] = 0.1 < np.interp(
                idx_row + trace_blue[i] - trace_blue[reference_bin], idx_row, mask_bin[i])

        idx_min, idx_max = idx_col[slice_from_string(range_blue)][[0, -1]]
        mask_blue[(loc_bin < idx_min) | (idx_max < loc_bin), :] = True

        for i in np.arange(n_order)[(idx_blue + 1):]:

            interval = f'[{round(intervals[i][0])}:{round(intervals[i][1])}]'

            (_, fitted_trace_arr[i], refined_trace_arr[i], residual_arr[i], 
             master_mask[i], threshold_lower[i], threshold_upper[i]) = _trace(
                mode='trace', method=method, count_bin=count_blue, mask_bin=mask_blue, 
                loc_bin=loc_bin, n_bin=n_bin, k=reference_bin, interval=interval, 
                fwhm=fwhm, idx_row=idx_row, idx_col=idx_col, order=degree, n_piece=1, 
                maxiters=maxiters, sigma_lower=sigma_lower, sigma_upper=sigma_upper, 
                grow=grow)

        fitted_trace_arr[(idx_blue + 1):] += (
            fitted_trace_arr[idx_blue] - trace_blue[reference_bin])
        refined_trace_arr[(idx_blue + 1):] += trace_blue - trace_blue[reference_bin]

    # Plot
    _validateBool(show, 'show')
    _validateBool(save, 'save')

    if show | save:

        _validateString(title, 'title')
        if title != 'trace':
            title = f'{title} trace'

        height_ratios = (1 / 4.5, n_col / n_row)
        fig, ax = plt.subplots(2, 1, sharex=True, height_ratios=height_ratios, dpi=100)

        # Subplot 1
        ax[0].step(idx_row, count_bin[reference_bin], 'k-', lw=1.5, where='mid')
        # ax[0].axvline(x=center_ref, color='r', ls='--', lw=1.5)

        # Settings
        ax[0].grid(axis='both', color='0.95', zorder=-1)
        # ax[0].set_yscale('log')
        ax[0].tick_params(
            which='major', direction='in', top=True, right=True, length=5, width=1.5, 
            labelsize=12)
        ax[0].set_ylabel('pixel value', fontsize=16)
        ax[0].set_title(title, fontsize=16)

        # Subplot 2
        if dispersion_axis == 'row':
            xlabel, ylabel = 'row', 'column'
        else:
            xlabel, ylabel = 'column', 'row'
        _plot2d(
            ax=ax[1], ccd=data_arr.T, cmap='Greys_r', contrast=0.25, cbar=False, 
            xlabel=xlabel, ylabel=ylabel, aspect='auto')
        (xmin, xmax), (ymin, ymax) = ax[1].get_xlim(), ax[1].get_ylim()
        for fitted_trace in fitted_trace_arr:
            ax[1].plot(fitted_trace, idx_col, 'r--', lw=1.5)

        # Settings
        ax[1].set_xlim(xmin, xmax)
        ax[1].set_ylim(ymin, ymax)
        fig.set_figheight(fig.get_figwidth() * np.sum(height_ratios))
        fig.tight_layout()

        if save:
            fig_path = _validatePath(path, title)
            plt.savefig(fig_path, dpi=100)

        if show:
            plt.show()

        plt.close()

    # Fitting plot
    if save:

        fig_path = _validatePath(path, f'{title} fitting', '.pdf')
        with PdfPages(fig_path, keep_empty=False) as pdf:

            for i in np.arange(n_order):

                fig, ax = plt.subplots(2, 1, sharex=True, height_ratios=[3, 1], dpi=100)
                fig.subplots_adjust(hspace=0)

                _plotFitting(
                    ax=ax, x=loc_bin, y=refined_trace_arr[i], m=master_mask[i], 
                    x_fit=idx_col, y_fit=fitted_trace_arr[i], r=residual_arr[i], 
                    threshold_lower=threshold_lower[i], 
                    threshold_upper=threshold_upper[i], xlabel='dispersion axis [px]', 
                    ylabel='spatial axis [px]', use_relative=False)

                # Settings
                ax[0].set_title(f'order #{orders[i]}', fontsize=16)
                fig.align_ylabels()
                fig.tight_layout()

                pdf.savefig(fig, dpi=100)

                plt.close()

    header = nccd.header.copy()
    # header['TRINTERV'] = interval
    # header['TRCENTER'] = center_ref
    header['TRACE'] = '{} Trace ({}, n_med = {}, n_piece = {})'.format(
        Time.now().to_value('iso', subfmt='date_hm'), method, n_med, n_piece)

    meta = {'header': header}

    # No uncertainty or mask frame
    trace1d = Spectrum1D(flux=(fitted_trace_arr * u.pixel), meta=meta)

    return trace1d


def mask_from_trace(shape, trace_arr, width, head_and_foot=True):
    """
    """

    idx_col = np.arange(shape[1])

    trace_arr = np.round(trace_arr).astype(int)

    n_trace = trace_arr.shape[0]

    idx_trace = np.arange(n_trace)

    if (width % 2) == 0:
        width -= 1

    half_width = width // 2

    lower_arr = trace_arr - half_width
    upper_arr = trace_arr + half_width

    mask = np.zeros(shape, dtype=bool)

    for i in idx_trace:
        
        for j in idx_col:
            mask[lower_arr[i, j]:(upper_arr[i, j] + 1), j] = True

        if head_and_foot:

            if i == (n_trace - 1):

                gap_width = lower_arr[-1] - upper_arr[-2]
                uppermost = upper_arr[-1] + gap_width

                for j in idx_col:
                    mask[uppermost[j]:, j] = True

            if i == 0:

                gap_width = ((lower_arr[1] - upper_arr[0]) * 0.7).astype(int)
                lowermost = lower_arr[0] - gap_width

                for j in idx_col:
                    mask[:(lowermost[j] + 1), j] = True

    return mask


def backgroundEchelle(ccd, dispersion_axis, trace1d, mask_width, title='background', 
                      save=conf.save, path=None):
    """
    """

    _validateString(dispersion_axis, 'dispersion_axis', ['col', 'row'])

    if dispersion_axis == 'row':
        nccd, data_arr, _, mask_arr = _validateCCD(ccd, 'ccd', False, False, False)
    else:
        nccd, data_arr, _, mask_arr = _validateCCD(ccd, 'ccd', False, False, True)

    n_row, n_col = data_arr.shape

    idx_row, idx_col = np.arange(n_row), np.arange(n_col)

    # Assume that unit of the trace is [pixel]
    if isinstance(trace1d, Spectrum1D):
        trace_arr = trace1d.flux.value

    n_trace = trace_arr.shape[0]

    # Assume that dispersion axis is horizontal
    mask = ~mask_from_trace((n_row, n_col), trace_arr, width=mask_width)

    # Label each background sections
    label_arr, n_label = ndimage.label(mask)
    idx_label = np.arange(n_label) + 1

    if n_label > (n_trace + 1):
        raise ValueError(f'``mask_width`` = {mask_width} is too large.')

    bkgd_arr = data_arr.copy()

    for i in idx_col:

        med = ndimage.median(
            data_arr[:, i], labels=label_arr[:, i], index=idx_label)
        loc_med = ndimage.median(
            idx_row, labels=label_arr[:, i], index=idx_label)

        idx_start, idx_end = np.where(label_arr[:, i] != 0)[0][[0, -1]]

        tck = interpolate.splrep(loc_med, med, k=3, s=0)
        spl = interpolate.BSpline(*tck)
        bkgd_arr[idx_start:(idx_end + 1), i] = spl(idx_row[idx_start:(idx_end + 1)])
        
    rsdl_arr = data_arr - bkgd_arr
    threshold = np.array([rsdl_arr[mask[:, i], i].std(ddof=1) for i in idx_col]) * 3

    # Plot
    _validateBool(save, 'save')

    if save:

        _validateString(title, 'title')
        if title != 'background':
            title = f'{title} background'

        fig_path = _validatePath(path, f'{title} fitting', '.pdf')
        with PdfPages(fig_path, keep_empty=False) as pdf:

            for i in idx_col[::50]:

                fig, ax = plt.subplots(2, 1, sharex=True, height_ratios=[3, 1], dpi=100)
                fig.subplots_adjust(hspace=0)

                _plotFitting(
                    ax=ax, x=idx_row, y=data_arr[:, i], m=mask_arr[:, i], 
                    x_fit=idx_row, y_fit=bkgd_arr[:, i], r=rsdl_arr[:, i], 
                    threshold_lower=-threshold[i], threshold_upper=threshold[i], 
                    xlabel='spatial axis [px]', ylabel='pixel value', 
                    use_relative=False)

                # Settings
                ax[0].set_xlim(idx_row[0], idx_row[-1])
                ax[0].set_title(f'background at column {i}', fontsize=16)

                fig.align_ylabels()
                fig.tight_layout()

                pdf.savefig(fig, dpi=100)

                plt.close()

    # Background frame
    if dispersion_axis == 'row':

        nccd.data = bkgd_arr.copy()
        nccd.uncertainty = None
        if nccd.mask is not None:
            nccd.mask = mask_arr.copy()

    else:

        nccd.data = bkgd_arr.T
        nccd.uncertainty = None
        if nccd.mask is not None:
            nccd.mask = mask_arr.T

    # Output
    if isinstance(ccd, CCDData):
        nccd.header['MKBACKGR'] = '{} Background.'.format(
            Time.now().to_value('iso', subfmt='date_hm'))

    elif np.ma.isMaskedArray(ccd):
        nccd = np.ma.array(nccd.data, mask=nccd.mask)

    else:
        nccd = nccd.data.copy()

    return nccd


def to_rectangle(ccd, dispersion_axis, trace1d, width):
    """Transform the orders in the input frame to a rectangular array.
    """

    _validateString(dispersion_axis, 'dispersion_axis', ['col', 'row'])

    if dispersion_axis == 'row':
        nccd, data_arr, uncertainty_arr, mask_arr = _validateCCD(
            ccd, 'ccd', True, True, False)
    else:
        nccd, data_arr, uncertainty_arr, mask_arr = _validateCCD(
            ccd, 'ccd', True, True, True)

    n_row, n_col = data_arr.shape

    idx_row, idx_col = np.arange(n_row), np.arange(n_col)

    # Assume that unit of the trace is [pixel]
    if isinstance(trace1d, Spectrum1D):
        trace_arr = trace1d.flux.value

    trace_arr = np.round(trace_arr).astype(int)

    n_trace = trace_arr.shape[0]

    idx_trace = np.arange(n_trace)

    idx_gap = (idx_trace[:-1] + idx_trace[1:]) / 2
    gap_min = np.diff(trace_arr, axis=0).min(axis=1)

    # todo: validate maximum width?
    if (width % 2) == 0:
        width -= 1

    isWider = gap_min >= width

    mask_trace = np.interp(idx_trace, idx_gap, isWider) > 0.1

    data_rec = np.zeros([n_trace, width, n_col])
    uncertainty_rec = np.zeros([n_trace, width, n_col])
    mask_rec = np.zeros([n_trace, width, n_col], dtype=bool)

    if mask_trace.sum() > 0:

        mask = mask_from_trace(
            (n_row, n_col), trace_arr[mask_trace], width=width, head_and_foot=False)

        data_rec[mask_trace] = np.array(
            np.vsplit(
                data_arr.T[mask.T].reshape(-1, (mask_trace.sum() * width)).T, 
                mask_trace.sum()
            )
        )

        uncertainty_rec[mask_trace] = np.array(
            np.vsplit(
                uncertainty_arr.T[mask.T].reshape(-1, (mask_trace.sum() * width)).T, 
                mask_trace.sum()
            )
        )

        mask_rec[mask_trace] = np.array(
            np.vsplit(
                mask_arr.T[mask.T].reshape(-1, (mask_trace.sum() * width)).T, 
                mask_trace.sum()
            )
        )

    idx_trace = np.where(~mask_trace)[0]

    for i in idx_trace:

        mask = mask_from_trace(
            (n_row, n_col), trace_arr[i][np.newaxis, :], width=width, 
            head_and_foot=False)

        data_rec[i] = data_arr.T[mask.T].reshape(-1, width).T
        uncertainty_rec[i] = uncertainty_arr.T[mask.T].reshape(-1, width).T
        mask_rec[i] = mask_arr.T[mask.T].reshape(-1, width).T

    nccd.data = np.vstack(data_rec)

    if nccd.uncertainty is not None:
        nccd.uncertainty.array = np.vstack(uncertainty_rec)

    if nccd.mask is not None:
        nccd.mask = np.vstack(mask_rec)

    # Output
    if isinstance(ccd, CCDData):
        nccd.header['RECTANGL'] = '{} Orders decurvated.'.format(
            Time.now().to_value('iso', subfmt='date_hm'))

    elif np.ma.isMaskedArray(ccd):
        nccd = np.ma.array(nccd.data, mask=nccd.mask)

    else:
        nccd = nccd.data.copy()

    return nccd