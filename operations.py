import os, sys
from glob import glob

try:
    import numpy as np
except ImportError:
    print( 'Module `numpy` not found. Please install with: pip install numpy' )
    sys.exit()

try:
    from astropy import units as u
    from astropy.nddata import CCDData
except ImportError:
    print( 'Module `astropy` not found. Please install with: pip install astropy' )
    sys.exit()

try:
    import ccdproc
except ImportError:
    print( 'Module `ccdproc` not found. Please install with: pip install ccdproc' )
    sys.exit()

from utils import print_statistics, plot2d

def biascombine( path, ext, gain, rdnoise ):
    '''
    '''

    bias_frame_list = sorted( glob( os.path.join( path, '*.fit*' ) ) )

    biaslist = list()
    for i, bfile in enumerate( bias_frame_list ):
        # Load bias
        print( F'[Bias combination] Load {bfile} as bias frame [{i+1}/{len(bias_frame_list)}]' )
        bias = CCDData.read( bfile, hdu = ext, unit = u.adu )
        # Add uncertainty
        print( F'[Bias combination] Estimate uncertainty' )
        bias_with_deviation = ccdproc.create_deviation( bias, readnoise = rdnoise * u.adu )
        # Gain correction
        print( F'[Bias combination] Gain correction' )
        bias_gain_corrected = ccdproc.gain_correct( bias_with_deviation, gain * u.electron / u.adu )
        biaslist.append( bias_gain_corrected )

    print_statistics( biaslist )

    # Combine
    print( F'[Bias combination] Combine' )
    bias_combiner = ccdproc.Combiner( biaslist )
    master_bias = bias_combiner.median_combine()
    print_statistics( [master_bias] )

    # Plot
    print( F'[Bias combination] Plot' )
    plot2d( master_bias, 'Master Bias' )

    return master_bias

def flatcombine( filters, path, ext, expkw, gain, rdnoise, master_bias ):
    '''
    '''

    master_flat = dict()
    for filter in filters:

        flat_frame_list = sorted( glob( os.path.join( path, F'{filter}/flat/*.fit*' ) ) )

        flatlist = list(); expolist = list()
        for i, ffile in enumerate( flat_frame_list ):
            # Load flat-field
            print( F'[Flat-field combination] Load {ffile} as {filter} band flat-field [{i+1}/{len(flat_frame_list)}]' )
            flat = CCDData.read( ffile, hdu = ext, unit = u.adu )
            # Add uncertainty
            print( F'[Flat-field combination] Estimate uncertainty' )
            flat_with_deviation = ccdproc.create_deviation( flat, readnoise = rdnoise * u.adu )
            # Gain correction
            print( F'[Flat-field combination] Gain correction' )
            flat_gain_corrected = ccdproc.gain_correct( flat_with_deviation, gain * u.electron / u.adu )
            # Bias subtraction
            print( F'[Flat-field combination] Bias subtraction' )
            flat_bias_subtracted = ccdproc.subtract_bias( flat_gain_corrected, master_bias )
            flatlist.append( flat_bias_subtracted ); expolist.append( flat_bias_subtracted.meta[expkw] )

        print_statistics( flatlist )

        # Combine
        print( F'[Flat-field combination] Combine' )
        flat_combiner = ccdproc.Combiner( flatlist )
        flat_combiner.scaling = 1 / np.array( expolist )
        master_flat[filter] = flat_combiner.median_combine()
        print_statistics( [master_flat[filter]] )

        # Plot
        print( F'[Flat-field combination] Plot' )
        plot2d( master_flat[filter], F'{filter} Band Master Flat-field' )

    return master_flat

def imagecorrection( filters, path, ext, gain, rdnoise, master_bias, master_flat ):
    '''
    '''

    images = dict()
    for filter in filters:

        img_frame_list = sorted( glob( os.path.join( path, F'{filter}/*.fit*' ) ) )

        images[filter] = list()
        for i, imgfile in enumerate( img_frame_list ):
            # Load object
            print( F'[Image correction] Load {imgfile} as {filter} band object image [{i+1}/{len(img_frame_list)}]' )
            img = CCDData.read( imgfile, hdu = ext, unit = u.adu )
            # Add uncertainty
            print( F'[Image correction] Estimate uncertainty' )
            img_with_deviation = ccdproc.create_deviation( img, readnoise = rdnoise * u.adu )
            # Gain correction
            print( F'[Image correction] Gain correction' )
            img_gain_corrected = ccdproc.gain_correct( img_with_deviation, gain * u.electron / u.adu )
            # Bias subtraction
            print( F'[Image correction] Bias subtraction' )
            img_bias_subtracted = ccdproc.subtract_bias( img_gain_corrected, master_bias )
            # Flat-fielding
            print( F'[Image correction] Flat-fielding' )
            img_flat_fielded = ccdproc.flat_correct( img_bias_subtracted, master_flat[filter] )
            # Cosmic ray correction
            print( F'[Image correction] Mask cosmic ray pixels' )
            reduced_image = ccdproc.cosmicray_lacosmic( img_flat_fielded,
                                                        objlim    = 1,
                                                        sigclip   = 4.5, 
                                                        sigfrac   = 0.5,
                                                        readnoise = gain * rdnoise * u.electron,
                                                        verbose   = False )
            images[filter].append( reduced_image )

        print_statistics( images[filter] )
    
    return images