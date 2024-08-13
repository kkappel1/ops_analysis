import image_analysis
from image_analysis.postprocess import *
import pandas as pd
import argparse
import datetime
import os

def filter_matches_well( data_file, no_overwrite ):

    begin_time = datetime.datetime.now()
    full_data = pd.read_csv( data_file )
    plot_save_name_intensity = os.path.dirname( os.path.abspath( data_file) ) + '/plot_filter_matches_intensity_' + data_file.split('/')[-1] + '.png'
    plot_save_name_glcm = os.path.dirname( os.path.abspath(data_file) ) + '/plot_filter_matches_glcm_' + data_file.split('/')[-1] + '.png'

    full_data_filt_int = filter_matches_intensity_range( full_data, residual_scalar_range=[0.01, 0.05, 0.1, 0.15, 0.5, 1.0], 
                                                         alpha=0.1, residual_offset=100, show_plot=False,
                                                         plot_save_name=plot_save_name_intensity)
    
    full_data_filt_int_glcm = filter_matches_glcm_range( full_data_filt_int, residual_scalar_range=[0.01, 0.05, 0.1, 0.15, 0.5, 1.0], 
                                                                      glcm_residual_offset=1, show_plot=False,
                                                                      plot_save_name=plot_save_name_glcm)

    save_data_file = data_file.replace( '.csv', '_filter_matches.csv' )
    full_data_filt_int_glcm.to_csv( save_data_file, index=False )



    end_time = datetime.datetime.now()
    print("Time filtering matches:", end_time - begin_time, flush=True )

    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser( description="remove duplicate cells and bad barcodes" )
    parser.add_argument( '-data_file', type=str, default="", help='csv file with the data for the well' )
    parser.add_argument( '-no_overwrite', default=False, action='store_true', 
            help='Dont overwrite existing nuclei segmentation files' )

    args = parser.parse_args()
    filter_matches_well( args.data_file, args.no_overwrite )
