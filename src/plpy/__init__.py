"""
Data reduction pipelines developed based on astro-drpy.
"""

__version__ = '1.0.3'

import os

# AstroPy
import astropy.config as _config


class Conf(_config.ConfigNamespace):
    """Configuration parameters for `plpy`."""

    dtype = _config.ConfigItem(
        'float32', cfgtype='string', module='plpy', 
        description=(
            'Data type for output.'
        )
    )

    fig_width = _config.ConfigItem(
        6, cfgtype='float', module='plpy', 
        description=(
            'Width of figures.'
        )
    )

    show = _config.ConfigItem(
        False, cfgtype='boolean', module='plpy', 
        description=(
            'Whether to show plots.'
        )
    )

    save = _config.ConfigItem(
        True, cfgtype='boolean', module='plpy', 
        description=(
            'Whether to save plots.'
        )
    )

    mem_limit = _config.ConfigItem(
        500e6, cfgtype='float', module='plpy', 
        description=(
            'Maximum memory which should be used while combining (in bytes).'
        )
    )

conf = Conf()

del _config