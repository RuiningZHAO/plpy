import os

# NumPy
import numpy as np
# AstroPy
from astropy.io import fits


def getSkyIndex(timetags, objnames, delimiter='_'):
    """Get indices of sky frames."""

    objname = [item.split(delimiter)[0] for item in objnames]

    assert len(set(objname)) == 1, 'The input frames contain more than one target.'

    positions = np.array([item.split(delimiter)[1].lower() for item in objnames])
    
    pair_list = list()
    for pos in sorted(set(positions)):
        idx_pos_arr = np.where(positions == pos)[0]
        idx_bkg_arr = np.where(positions != pos)[0]
        for idx_pos in idx_pos_arr:
            idx_bkg = idx_bkg_arr[
                np.abs(timetags[idx_bkg_arr] - timetags[idx_pos]).argmin()
            ]
            pair_list.append((idx_pos, idx_bkg))
    
    idx_list = [pair_list[i][1] for i in np.argsort([item[0] for item in pair_list])]

    return idx_list


def saveSpextool(spectrum1d, file_name, overwrite=False):
    """Save in spextool readable format."""
    
    header = spectrum1d.meta['header']
    header.set('NORDERS', 1, 'Number of orders')
    header.set('ORDERS', 0, "Placeholder, prisms don't have orders!")
    header.set('NAPS', 1, 'Number of apertures')
    header.set('XUNITS', 'um', 'Units of the X axis')
    header.set('YUNITS', 'DN', 'Units of the Y axis')
    header.set('XTITLE', '!7k!5 (!7l!5m)', 'IDL X title')
    header.set('YTITLE', 'f (!5DN)', 'IDL Y title')
    
    data = np.vstack([
        spectrum1d.spectral_axis.value / 1e4, 
        spectrum1d.flux.value, 
        spectrum1d.uncertainty.array])
    
    hdu = fits.PrimaryHDU(data=data, header=header)
    hdul = fits.HDUList([hdu])
    hdul.writeto(file_name, overwrite=overwrite)