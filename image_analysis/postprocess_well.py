import image_analysis
from image_analysis.postprocess import *
import pandas as pd
import argparse

def postprocess_well( data_file, bad_barcode_file ):

    if bad_barcode_file != '':
        bad_barcodes = pd.read_csv( bad_barcode_file, header=None)[0].to_list()
        read_write_remove_duplicates_filter_bad_barcodes( data_file, bad_barcodes )
    else:
        read_write_remove_duplicates( data_file )

    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser( description="remove duplicate cells and bad barcodes" )
    parser.add_argument( '-data_file', type=str, default="", help='csv file with the data for the well' )
    parser.add_argument( '-bad_barcode_file', type=str, default="", help='file that contains the bad barcodes' )

    args = parser.parse_args()
    postprocess_well( args.data_file, args.bad_barcode_file )
