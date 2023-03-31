import pandas as pd
import numpy as np
import multiprocessing

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

if __name__ == '__main__':
    multiprocessing.freeze_support()
