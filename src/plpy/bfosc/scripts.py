"""
Run pipeline
"""

import os, argparse


def spec():
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
        '-m', '--semester', required=True, type=str, 
        help='Observation semester.'
    )
    parser.add_argument(
        '-w', '--slit_width', required=True, type=float, choices=[1.8, 2.3], 
        help='Slit width.'
    )
    parser.add_argument(
        '-g', '--grism', required=True, type=int, choices=[3, 4], 
        help='Grism.'
    )
    parser.add_argument(
        '-r', '--reference', default=None, type=str, 
        help='Reference spectrum for wavelength calibration.'
    )
    parser.add_argument(
        '-s', '--standard', default=None, type=str, 
        help='Path to the standard spectrum in the library.'
    )
    parser.add_argument(
        '-k', '--keyword', default='object', type=str, 
        help='Keyword for grouping.'
    )
    parser.add_argument(
        '-c', '--combine', action='store_true', 
        help='Combine or not.'
    )
    parser.add_argument(
        '-x', '--extract', action='store_true', 
        help='Extract 1-dimensional spectra or not.'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true', 
        help='Verbose or not.'
    )
        
    # Parse
    args = parser.parse_args()
    data_dir = os.path.abspath(args.input_dir)
    save_dir = os.path.abspath(args.output_dir)
    semester = args.semester
    slit_width = str(args.slit_width).replace('.', '')
    grism = 'g' + str(args.grism)
    reference = args.reference
    standard = args.standard
    combine = args.combine
    keyword = args.keyword
    extract = args.extract
    verbose = args.verbose

    from .longslit import pipeline

    pipeline(
        save_dir=save_dir, data_dir=data_dir, semester=semester, grism=grism, 
        slit_width=slit_width, standard=standard, reference=reference, 
        shouldCombine=combine, keyword=keyword, shouldExtract=extract, 
        verbose=verbose)


def phot():
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

    from .imaging import pipeline

    pipeline(
        save_dir=save_dir, data_dir=data_dir, hdu=hdu, keywords=keywords, 
        filters=filters, steps=steps, row_range=row_range, col_range=col_range, 
        custom_mask=custom_mask, mem_limit=500e6, show=False, save=True, 
        verbose=verbose, mode=mode)