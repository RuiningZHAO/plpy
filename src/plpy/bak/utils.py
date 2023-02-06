try:
    from prettytable import PrettyTable
    from prettytable import HEADER, NONE
except ImportError:
    print( 'Module `prettytable` not found. Please install with: pip install prettytable' )
    sys.exit()

try:
    import matplotlib.pyplot as plt
    # Set plot parameters
    plt.rcParams['axes.linewidth'] = 2.0
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
except ImportError:
    print( 'Module `matplotlib` not found. Please install with: pip install matplotlib' )
    sys.exit()

def print_statistics( imglist ):
    '''
    '''
    tab = PrettyTable( hrules = HEADER, vrules = NONE )
    tab.field_names = [ 'FRAME TYPE', 'SHAPE', 'MEAN', 'STDDEV', 'MIN', 'MAX' ]
    for img in imglist:
        tab.add_row([ 'Image',
                       img.data.shape,
                       round( img.data.mean(), 2 ),
                       round( img.data.std( ddof = 1 ), 2 ),
                       int( img.data.min() ),
                       int( img.data.max() ) ])
        tab.add_row([ 'Uncertainty',
                      '\"',
                      round( img.uncertainty.array.mean(), 2 ),
                      round( img.uncertainty.array.std( ddof = 1 ), 2 ),
                      round( img.uncertainty.array.min(), 2 ),
                      round( img.uncertainty.array.max(), 2 ) ])
    for col in ['MEAN', 'STDDEV', 'MIN', 'MAX']: tab.align[col] = 'r'
    print( '\n' + tab.get_string() + '\n' )

def plot2d( img, title, show = 1, save = 0 ):
    '''
    '''
    
    fig, ax = plt.subplots( 1, 1, figsize = ( 8, 8 ) )
    fig.subplots_adjust( right = 0.8 )
    
    # Image
    im = ax.imshow( img.data, cmap = 'Greys_r', origin = 'lower', extent = ( 0.5, img.data.shape[1]+0.5, 0.5, img.data.shape[0]+0.5) )

    # Colorbar
    cax = fig.add_axes([ ax.get_position().x1 + 0.02, ax.get_position().y0, 0.04, ax.get_position().height ])
    cb = fig.colorbar( im, cax = cax )

    # Settings
    ax.tick_params( which = 'major', direction = 'in', color = 'w', top = True, right = True, length = 7, width = 1.5, labelsize = 18 )
    ax.set_xlabel( 'X', fontsize = 22 ); ax.set_ylabel( 'Y', fontsize = 22 )
    cb.ax.tick_params( which = 'major', direction = 'in', color = 'w', right = True, length = 7, width = 1.5, labelsize = 18 )
    cb.ax.set_ylabel( 'ADU', fontsize = 22 )
    ax.set_title( title, fontsize = 22 )

    if save:
        fig_path = 'figs'
        if not os.path.exists( fig_path ): os.makedirs( fig_path )
        print( F'[Plotting] Save to { os.path.join( fig_path, F"{title}.png".replace( " ", "_" ) ) }' )
        plt.savefig( os.path.join( fig_path, F'{title}.png'.replace( ' ', '_' ) ), dpi = 144 )

    if show:
        print( '[Plotting] Show plot' )
        plt.show()
    plt.close()