import pandas as pd
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from sklearn import linear_model

def remove_duplicates( df_well ):
    df_well_no_duplicates = pd.DataFrame()

    num_double = 0
    num_single = 0
    for tile_10x in list(set(df_well['tile_10x'])):
        print( tile_10x, flush=True )
        for cell in df_well[df_well['tile_10x']==tile_10x]['cell']:
            if (df_well_no_duplicates.shape[0] > 0) and (cell in df_well_no_duplicates[df_well_no_duplicates['tile_10x']==tile_10x]['cell'].values):
                continue
            else:
                cell_df = df_well[(df_well['tile_10x']==tile_10x) &
                                  (df_well['cell']==cell)].head(1)
                df_well_no_duplicates = df_well_no_duplicates.append( cell_df )

            num_cells = cell_df.shape[0]
            if num_cells > 1: num_double += 1
            else: num_single += 1
    print( "Num double", num_double )
    print( "Num single", num_single )
    return df_well_no_duplicates

def remove_duplicates_tile( df_well_tile ):
    df_well_tile_no_duplicates = pd.DataFrame()

    print( "removing duplicates for tile", df_well_tile['tile_10x'].tolist()[0], flush=True )

    for cell in df_well_tile['cell']:
        if (df_well_tile_no_duplicates.shape[0] > 0) and (cell in df_well_tile_no_duplicates['cell'].values):
            continue
        else:
            cell_df = df_well_tile[(df_well_tile['cell']==cell)].head(1)
            df_well_tile_no_duplicates = df_well_tile_no_duplicates.append( cell_df )

    return df_well_tile_no_duplicates

def remove_duplicates_parallel( df_well, num_proc ):
    tile_list = list( set( df_well['tile_10x'] ) )

    with multiprocessing.Pool(processes=num_proc) as pool:
        filtered_results = pool.map( remove_duplicates_tile, [df_well[df_well['tile_10x']==tile_10x] for tile_10x in tile_list] )

        # combine all the results:
        df_well_no_duplicates = pd.DataFrame()
        for df_well_tile in filtered_results:
            df_well_no_duplicates = df_well_no_duplicates.append( df_well_tile )
    return df_well_no_duplicates

def read_write_remove_duplicates( df_file ):

    df = pd.read_csv( df_file )
    df_no_duplicates = remove_duplicates( df )
    df_no_duplicates.to_csv( df_file.replace('.csv', '_no_duplicates.csv' ) )

    return

def read_write_remove_duplicates_parallel( df_file, num_proc ):

    df = pd.read_csv( df_file )
    df_no_duplicates = remove_duplicates_parallel( df, num_proc)
    df_no_duplicates.to_csv( df_file.replace('.csv', '_no_duplicates.csv' ) )

    return

def filter_bad_barcodes( full_data, bad_barcodes ):

    bc1 = full_data['cell_barcode_0'].isin( bad_barcodes )
    bc2 = full_data['cell_barcode_1'].isin( bad_barcodes )
    bc1_or_bc2 = np.logical_or( bc1, bc2 )

    return full_data[ ~bc1_or_bc2 ]

def read_write_remove_duplicates_filter_bad_barcodes( df_file, bad_barcodes ):

    df = pd.read_csv( df_file )
    df_no_duplicates = remove_duplicates( df )
    df_no_duplicates_filtered = filter_bad_barcodes( df_no_duplicates, bad_barcodes )
    df_no_duplicates_filtered.to_csv( df_file.replace('.csv', '_no_duplicates.csv' ) )

    return

def read_write_remove_duplicates_filter_bad_barcodes_parallel( df_file, bad_barcodes, num_proc ):

    df = pd.read_csv( df_file )
    df_no_duplicates = remove_duplicates_parallel( df, num_proc )
    df_no_duplicates_filtered = filter_bad_barcodes( df_no_duplicates, bad_barcodes )
    df_no_duplicates_filtered.to_csv( df_file.replace('.csv', '_no_duplicates.csv' ) )

    return

def filter_matches_glcm(full_data, residual_scalar=0.08, glcm_cutoff_fitting_10x=15000,
                            show_plot=True, plot_save_name='', glcm_residual_offset=1):
    ransac = linear_model.RANSACRegressor(max_trials=500, min_samples=0.6)
    x_below_cutoff_intensity = full_data['match_glcm_dissim_10x'] < glcm_cutoff_fitting_10x
    X_all_intensities_10x = full_data['match_glcm_dissim_10x'][x_below_cutoff_intensity][:, np.newaxis]
    all_intensities_40x = full_data['match_glcm_dissim_40x'][x_below_cutoff_intensity]

    ransac.fit( X_all_intensities_10x, all_intensities_40x )
    line_X = np.arange( X_all_intensities_10x.min(), X_all_intensities_10x.max())[:, np.newaxis]
    line_y = ransac.predict( line_X )
    
    residuals = ransac.predict(full_data['match_glcm_dissim_10x'][:, np.newaxis]) - full_data['match_glcm_dissim_40x']


    residuals_binary = np.abs(residuals ) < (full_data['match_glcm_dissim_10x']*residual_scalar)+glcm_residual_offset
    line_y_residual_above = line_y + line_X[:,0]*residual_scalar+glcm_residual_offset
    line_y_residual_below = line_y - line_X[:,0]*residual_scalar-glcm_residual_offset

    plt.scatter( full_data['match_glcm_dissim_10x'], full_data['match_glcm_dissim_40x'],
               alpha=0.1, c=residuals_binary)
    plt.plot( line_X, line_y, color='red' )
    plt.plot( line_X, line_y_residual_above, color='cyan' )
    plt.plot( line_X, line_y_residual_below, color='cyan' )
    
    if plot_save_name != '':
        plt.savefig( plot_save_name )

    if show_plot:
        plt.show()
    plt.clf()
    
    fraction_filtered = np.sum(~residuals_binary)/len(residuals_binary)
    print( "Fraction filtered", fraction_filtered )
    
    return full_data[residuals_binary], fraction_filtered

def filter_matches_intensity(full_data, residual_scalar=0.08, intensity_cutoff_fitting_10x=15000,
                            show_plot=True, plot_save_name='', alpha=0.1, residual_offset=0):
    ransac = linear_model.RANSACRegressor(max_trials=500, min_samples=0.6)
    x_below_cutoff_intensity = full_data['match_intensity_10x'] < intensity_cutoff_fitting_10x
    X_all_intensities_10x = full_data['match_intensity_10x'][x_below_cutoff_intensity][:, np.newaxis]
    all_intensities_40x = full_data['match_intensity_40x'][x_below_cutoff_intensity]

    ransac.fit( X_all_intensities_10x, all_intensities_40x )
    line_X = np.arange( X_all_intensities_10x.min(), 2*X_all_intensities_10x.max())[:, np.newaxis]
    line_y = ransac.predict( line_X )
    
    residuals = ransac.predict(full_data['match_intensity_10x'][:, np.newaxis]) - full_data['match_intensity_40x']


    residuals_binary = np.abs(residuals ) < (full_data['match_intensity_10x']*residual_scalar)+residual_offset
    line_y_residual_above = line_y + line_X[:,0]*residual_scalar+residual_offset
    line_y_residual_below = line_y - line_X[:,0]*residual_scalar-residual_offset

    plt.scatter( full_data['match_intensity_10x'], full_data['match_intensity_40x'],
               alpha=alpha, c=residuals_binary)
    plt.plot( line_X, line_y, color='red' )
    plt.plot( line_X, line_y_residual_above, color='cyan' )
    plt.plot( line_X, line_y_residual_below, color='cyan' )

    if plot_save_name != '':
        plt.savefig( plot_save_name )
    
    if show_plot:
        plt.show()
    plt.clf()
    
    fraction_filtered = np.sum(~residuals_binary)/len(residuals_binary)
    print( "Fraction filtered", fraction_filtered, flush=True )
    
    return full_data[residuals_binary], fraction_filtered
    
def filter_matches_intensity_range(full_data, residual_scalar_range = [0.01, 0.05, 0.1, 0.15, 0.5, 1.0], 
                                   intensity_cutoff_fitting_10x=15000,
                            show_plot=True, plot_save_name='', alpha=0.1, residual_offset=0, fraction_filter_data_threshold=0.01):
    for residual_scalar in sorted(residual_scalar_range):
        filtered_data, fraction_filtered  = filter_matches_intensity( full_data, 
                                                                     residual_scalar, 
                                                                     intensity_cutoff_fitting_10x, show_plot=show_plot,
                                                                    plot_save_name=plot_save_name, alpha=alpha, 
                                                                    residual_offset=residual_offset)
        if fraction_filtered < fraction_filter_data_threshold:
            print( "FINAL INTENSITY selected residual_scalar", residual_scalar, flush=True )
            print( "FINAL INTENSITY fraction data filtered", fraction_filtered, flush=True )
            return filtered_data
        
def filter_matches_glcm_range(full_data, residual_scalar_range=[0.01, 0.05, 0.1, 0.15, 0.5, 1.0], glcm_cutoff_fitting_10x=15000,
                            show_plot=True, plot_save_name='', glcm_residual_offset=1, fraction_filter_data_threshold=0.005):
    for residual_scalar in sorted( residual_scalar_range ):
        filtered_data, fraction_filtered = filter_matches_glcm(full_data, residual_scalar=residual_scalar, 
                                                               glcm_cutoff_fitting_10x=glcm_cutoff_fitting_10x,
                                                            show_plot=show_plot, plot_save_name=plot_save_name, 
                                                               glcm_residual_offset=glcm_residual_offset)
        if fraction_filtered < fraction_filter_data_threshold:
            print( "FINAL GLCM selected residual_scalar", residual_scalar, flush=True )
            print( "FINAL GLCM fraction data filtered", fraction_filtered, flush=True ) 
            return filtered_data


if __name__ == '__main__':
    multiprocessing.freeze_support()
