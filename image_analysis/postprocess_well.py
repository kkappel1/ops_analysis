import image_analysis
from image_analysis.postprocess import *
import pandas as pd
import argparse
import datetime
import os

def postprocess_well( data_file, bad_barcode_file, num_proc, no_overwrite ):
    if no_overwrite:
        output_fname = data_file.replace('.csv', '_no_duplicates.csv' )
        if os.path.exists( output_fname ):
            return

    begin_time = datetime.datetime.now()
    if bad_barcode_file != '':
        bad_barcodes = pd.read_csv( bad_barcode_file, header=None)[0].to_list()
        if num_proc > 1:
            read_write_remove_duplicates_filter_bad_barcodes_parallel( data_file, bad_barcodes, num_proc )
        else:
            read_write_remove_duplicates_filter_bad_barcodes( data_file, bad_barcodes )
    else:
        if num_proc > 1:
            read_write_remove_duplicates_parallel( data_file, num_proc )
        else:
            read_write_remove_duplicates( data_file )
    end_time = datetime.datetime.now()
    print("Time postprocessing:", end_time - begin_time, flush=True )

    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser( description="remove duplicate cells and bad barcodes" )
    parser.add_argument( '-data_file', type=str, default="", help='csv file with the data for the well' )
    parser.add_argument( '-bad_barcode_file', type=str, default="", help='file that contains the bad barcodes' )
    parser.add_argument( '-num_proc', type=int, default=1, help='number of processors to use' )
    parser.add_argument( '-no_overwrite', default=False, action='store_true', 
            help='Dont overwrite existing nuclei segmentation files' )

    args = parser.parse_args()
    postprocess_well( args.data_file, args.bad_barcode_file, args.num_proc, args.no_overwrite )
