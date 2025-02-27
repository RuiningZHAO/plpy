import io, os, shutil, argparse
from glob import glob

# AstroPy
from astropy.logger import log
from astropy.config.configuration import (get_config_filename, generate_config, 
                                          is_unedited_config_file)
# ccdproc
from ccdproc import ImageFileCollection

from .utils import makeDirectory, fixHeader, fixKeyword


def config():
    """Create configuration file.
    
    Rewrite based on ~astropy.config.create_config_file.
    """

    # External parameters
    parser = argparse.ArgumentParser(
        prog='plpy_config',
        description='Create configuration file for `plpy`.'
    )
    parser.add_argument(
        '-w', '--overwrite', action='store_true', 
        help='Overwrite or not.'
    )

    # Parse
    args = parser.parse_args()
    overwrite = args.overwrite

    pkg = 'plpy'
    rootname = 'plpy'
    
    cfgfn = get_config_filename(pkg, rootname=rootname)

    # generate the default config template
    template_content = io.StringIO()
    generate_config(pkg, template_content)
    template_content.seek(0)
    template_content = template_content.read()

    doupdate = True

    # if the file already exists, check that it has not been modified
    if cfgfn is not None and os.path.isfile(cfgfn):

        with open(cfgfn, encoding='latin-1') as fd:
            content = fd.read()

        doupdate = is_unedited_config_file(content, template_content)

    if doupdate:

        with open(cfgfn, 'w', encoding='latin-1') as fw:
            fw.write(template_content)

        log.info(f'The configuration file has been successfully written to {cfgfn}')

        return True

    elif not doupdate:
        
        if not overwrite:

            log.warning(
                'The configuration file already exists and seems to have been '
                'customized, so it has not been updated. Use `-w` if you really want to '
                'update it.'
            )

            return False

        else:

            shutil.copyfile(cfgfn, f'{cfgfn}.bak')

            log.info(
                'The configuration file already exists and seems to have been '
                f'customized. It has been renamed to {cfgfn}.bak.'
            )

            with open(cfgfn, 'w', encoding='latin-1') as fw:
                fw.write(template_content)
            log.info(
                f'The new configuration file has been successfully written to {cfgfn}')

            return True


def setup():
    """Set up a directory for data reduction."""

    # External parameters
    parser = argparse.ArgumentParser(
        prog='plpy_setup',
        description='Set up a directory for data reduction.'
    )
    parser.add_argument(
        '-i', '--instrument', required=True, choices=['hrs', 'mst', 'tspec', 'bfosc'], 
        help='Instrument that the data is taken with.'
    )
    parser.add_argument(
        '-d', '--directory', default='', type=str, 
        help='Directory to be set up. Default is current directory.'
    )
    parser.add_argument(
        '-w', '--overwrite', action='store_true', 
        help='Overwrite or not if `params.toml` exists.'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true', 
        help='Verbose or not.'
    )

    # Parse
    args = parser.parse_args()
    instrument = args.instrument
    parent = os.path.abspath(args.directory)
    overwrite = args.overwrite
    verbose = args.verbose
    
    # Make directory
    makeDirectory(parent=parent, child='fig', verbose=verbose)
    makeDirectory(parent=parent, child='red', verbose=verbose)
    makeDirectory(parent=parent, child='cal', verbose=verbose)
    
    # Add more folders for some other instruments, for example, fire
    if instrument in ['fire', 'tspec']:
        makeDirectory(parent=parent, child='sub', verbose=verbose)

    # Generate input file
    if verbose:
        print(f'- Generating input file...')

    if (not overwrite) and os.path.isfile(os.path.join(parent, 'params.toml')):
            raise RuntimeError('`params.toml` already exists. Use `-w` to overwrite.')

    shutil.copyfile(
        os.path.join(os.path.split(__file__)[0], instrument, 'lib/params.toml'), 
        os.path.join(parent, 'params.toml'))
    
    if verbose:
        print(f'- {parent} is ready for {instrument} data reduction!')


def header():
    """Fix non-standard keywords and typos in header."""

    # External parameters
    parser = argparse.ArgumentParser(
        prog='plpy_header',
        description='Fix non-standard keywords and typos in the fits header.'
    )
    parser.add_argument(
        '-i', '--input_dir', required=True, type=str, 
        help='Input (data) directory.'
    )
    parser.add_argument(
        '-p', '--pattern', default='*.fits', type=str, 
        help='Unix-style filename pattern to select files from input directory.'
    )
    parser.add_argument(
        '-e', '--extension', default=0, type=int, 
        help='The extension from which the header will be read in all files.'
    )
    parser.add_argument(
        '-f', '--entry_file', default='', type=str, 
        help='A file containing entries.'
    )
    parser.add_argument(
        '-k', '--keywords', default='object', type=str, 
        help='Keywords to be printed out in verbose mode. (Use `,` as delimiter)'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true', 
        help='Verbose or not.'
    )

    # Parse
    args = parser.parse_args()
    data_dir = os.path.abspath(args.input_dir)
    include = args.pattern
    ext = args.extension
    entry_file = os.path.abspath(args.entry_file)
    keywords = args.keywords.split(',')
    verbose = args.verbose

    # Filter files
    filenames = sorted(glob(os.path.join(data_dir, include)))
    if not filenames:
        raise ValueError('File not found.')

    # Fix non-standard keywords
    if verbose:
        print('- Fixing non-standard keywords in header...')
    for file_name in filenames:
        fixHeader(file_name, verbose=verbose, output_verify='warn')
        
    # Fix typos
    if os.path.isfile(entry_file):
        if verbose:
            print('- Fixing typos in header...')
        fixKeyword(data_dir, ext, entry_file, verbose=verbose)

    if verbose:
        ifc = ImageFileCollection(
            location=data_dir, keywords=keywords, find_fits_by_reading=False, 
            filenames=filenames, glob_include=None, glob_exclude=None, 
            ext=ext)
        ifc.summary.pprint_all()

    return