"""
Data reduction pipelines for 2.16-m/BFOSC.
"""
import os

# AstroPy
import astropy.config as _config


class Conf(_config.ConfigNamespace):
    """Configuration parameters for `plpy.bfosc`."""

    path_to_library = _config.ConfigItem(
        os.path.join(os.path.split(__file__)[0], 'lib'), cfgtype='string', 
        module='plpy.bfosc', 
        description='Path to library.'
    )

    include = _config.ConfigItem(
        '*.fit', cfgtype='string', module='plpy.bfosc', 
        description=(
            'Unix-style filename pattern to select filenames to include in the file '
            'collection. Can be used in conjunction with `exclude` to easily select '
            'subsets of files in the input directory.'
        )
    )

    exclude = _config.ConfigItem(
        '', cfgtype='string', module='plpy.bfosc', 
        description=(
            'Unix-style filename pattern to select filenames to exclude from the file '
            'collection. Can be used in conjunction with `include` to easily select '
            'subsets of files in the input directory.'
        )
    )

    hdu = _config.ConfigItem(
        0, cfgtype='integer', module='plpy.bfosc', 
        description=(
            'HDU specification.'
        )
    )

    shape = _config.ConfigItem(
        [2048, 2048], cfgtype='int_list', module='plpy.bfosc', 
        description=(
            'Shape of the data array.'
        )
    )

    slit_along = _config.ConfigItem(
        'col', cfgtype='option(col, row)', module='plpy.bfosc', 
        description=(
            '`col` or `row`.'
        )
    )

    keywords = _config.ConfigItem(
        ['date-obs', 'obstype', 'object', 'ra', 'dec', 'filter', 'exptime', 'rdnoise', 'gain', 'airmass'], 
        cfgtype='list', module='plpy.bfosc', 
        description=(
            'Keywords to be printed out in verbose mode.'
        )
    )

    steps = _config.ConfigItem(
        ['header', 'trim', 'bias', 'flat', 'lamp', 'targ'], cfgtype='list', 
        module='plpy.bfosc', 
        description=(
            'Data reduction procedures.'
        )
    )
    
    fits_section = _config.ConfigItem(
        '[1:1900, 330:1830]', cfgtype='string', module='plpy.bfosc', 
        description=(
            'Section to be cutout (fits-style).'
        )
    )

    index = _config.ConfigItem(
        665, cfgtype='integer', module='plpy.bfosc', 
        description=(
            'The location to concatenate two arc frames (along the dispertion axis). An index '
            'between ArI 6871.2891 and ArI 6965.4307 is recommended.' 
        )
    )

    n_piece = _config.ConfigItem(
        23, cfgtype='integer', module='plpy.bfosc', 
        description=(
            'Number of equally spaced pieces for spline3 fitting (flat-field '
            'normalization).'
        )
    )

    sigma = _config.ConfigItem(
        [20, 30], cfgtype='int_list', module='plpy.bfosc', 
        description=(
            '2-dimensional Gaussian sigma for illumination modeling (flat-field '
            'normalization).'
        )
    )

    bkg_location_stan = _config.ConfigItem(
        [-200, 200], cfgtype='int_list', module='plpy.bfosc', 
        description=(
            'Location of the sky background apertures. (standard)'
        )
    )

    bkg_width_stan = _config.ConfigItem(
        [50, 50], cfgtype='int_list', module='plpy.bfosc', 
        description=(
            'Widths of the sky background apertures. (standard)'
        )
    )

    bkg_order_stan = _config.ConfigItem(
        0, cfgtype='integer', module='plpy.bfosc', 
        description=(
            'Order of polynomial for sky background fitting. (standard)'
        )
    )

    bkg_location_targ = _config.ConfigItem(
        [-200, 200], cfgtype='int_list', module='plpy.bfosc', 
        description=(
            'Location of the sky background apertures. (target)'
        )
    )

    bkg_width_targ = _config.ConfigItem(
        [50, 50], cfgtype='int_list', module='plpy.bfosc', 
        description=(
            'Widths of the sky background apertures. (target)'
        )
    )

    bkg_order_targ = _config.ConfigItem(
        0, cfgtype='integer', module='plpy.bfosc', 
        description=(
            'Order of polynomial for sky background fitting. (target)'
        )
    )

    extract_stan = _config.ConfigItem(
        'sum', cfgtype='option(sum, optimal)', module='plpy.bfosc', 
        description=(
            'Extraction method. (standard)'
        )
    )
        
    aper_width_stan = _config.ConfigItem(
        150, cfgtype='integer', module='plpy.bfosc', 
        description=(
            'Aperture width for extraction. For optimal extraction, this is the '
            'profile width, and a very wide aperture is preferred. Pixels outside the '
            'profile will be set to `0`. (standard)'
        )
    )

    pfl_window_stan = _config.ConfigItem(
        150, cfgtype='integer', module='plpy.bfosc', 
        description=(
            'Length of the Savgol filter window for profile smoothing along dispersion '
            'axis. Only used in optimal extraction. (standard)'
        )
    )

    pfl_order_stan  = _config.ConfigItem(
        3, cfgtype='integer', module='plpy.bfosc', 
        description=(
            'The order of the polynomial used to fit profile along dispersion axis. '
            'Only used in optimal extraction. (standard)'
        )
    )

    extract_targ = _config.ConfigItem(
        'sum', cfgtype='option(sum, optimal)', module='plpy.bfosc', 
        description=(
            'Extraction method. (target)'
        )
    )

    aper_width_targ = _config.ConfigItem(
        150, cfgtype='integer', module='plpy.bfosc', 
        description=(
            'Aperture width for extraction. For optimal extraction, this is the '
            'profile width, and a very wide aperture is preferred. Pixels outside the '
            'profile will be set to `0`. (target)'
        )
    )

    pfl_window_targ = _config.ConfigItem(
        150, cfgtype='integer', module='plpy.bfosc', 
        description=(
            'Length of the Savgol filter window for profile smoothing along dispersion '
            'axis. Only used in optimal extraction. (target)'
        )
    )

    pfl_order_targ = _config.ConfigItem(
        3, cfgtype='integer', module='plpy.bfosc', 
        description=(
            'The order of the polynomial used to fit profile along dispersion axis. '
            'Only used in optimal extraction. (target)'
        )
    )

    exposure = _config.ConfigItem(
        'EXPTIME', cfgtype='string', module='plpy.bfosc', 
        description=(
            'Keyword for exposure time.'
        )
    )

    airmass = _config.ConfigItem(
        'AIRMASS', cfgtype='string', module='plpy.bfosc', 
        description=(
            'Keyword for airmass.'
        )
    )

    extinct = _config.ConfigItem(
        'baoextinct.dat', cfgtype='string', module='plpy.bfosc', 
        description=(
            'Name of the extinction file.'
        )
    )

    dtype = _config.ConfigItem(
        'float32', cfgtype='string', module='plpy.bfosc', 
        description=(
            'Data type for output.'
        )
    )

    show = _config.ConfigItem(
        False, cfgtype='boolean', module='plpy.bfosc', 
        description=(
            'Whether to show plots.'
        )
    )

    save = _config.ConfigItem(
        True, cfgtype='boolean', module='plpy.bfosc', 
        description=(
            'Whether to save plots.'
        )
    )

    mem_limit = _config.ConfigItem(
        500e6, cfgtype='float', module='plpy.bfosc', 
        description=(
            'Maximum memory which should be used while combining (in bytes).'
        )
    )


conf = Conf()

del _config