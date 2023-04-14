import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
from scipy.stats import pearsonr
from image_analysis.params_library import *

def load_data_sublib( requested_sublib, requested_rep ):
    all_data_sublib = pd.DataFrame()
    for plate in plates:
        for well in wells:
            plate_well = f'{plate}_{well}'
            sublib_rep = map_plate_well_to_sublib[ plate_well ]
            sublib = sublib_rep.split('_')[0]
            rep = int( sublib_rep.split('_')[1] )
            if (sublib == requested_sublib) and (rep == requested_rep):
                data_file = f'reanalyze_plate{plate}/all_output_data_plate_{plate}_well_{well}_analyze20230323_no_duplicates_filter_matches.csv'
                data_plate_well = pd.read_csv( data_file )
                data_plate_well['plate'] = plate
                data_plate_well['replicate'] = rep
                # edit the file path for cell images
                data_plate_well['cell_img_gfp_file'] = f'reanalyze_plate{plate}/' + data_plate_well['cell_img_gfp_file']
                data_plate_well['cell_img_mask_file'] = f'reanalyze_plate{plate}/' + data_plate_well['cell_img_mask_file']
                data_plate_well['cell_img_dapi_file'] = f'reanalyze_plate{plate}/' + data_plate_well['cell_img_dapi_file']
                all_data_sublib = all_data_sublib.append( data_plate_well )
                print( f"Loading plate {plate} well {well} for sublib {requested_sublib} rep {requested_rep}" )
                
    print()
    return all_data_sublib

def apply_SNAP_intensity_correction( param_file, data_df_input):
    data_df = data_df_input.copy( deep=True )
    offset_params = pd.read_csv( param_file )
    offset = offset_params['offset'][0]
    slope = offset_params['slope'][0]
    
    features_to_correct = ['mean_GFP_intensity_GFP','std_GFP_intensity_GFP',
                          'mean_dilute_intensity_GFP','std_dilute_intensity_GFP',
                          'mean_condensate_intensity_GFP', 'mean_condensate_intensity_no_shrink_GFP',
                          'std_condensate_intensity_GFP', 'std_condensate_intensity_no_shrink_GFP', 
                          'total_cond_intensity_GFP', 'total_gfp_intensity_GFP'
                          ]

    for feature in features_to_correct:
        data_df[ feature ] = slope*(data_df[feature] - offset) +offset

    return data_df

def load_data_sublib_and_correct_SNAP( requested_sublib, requested_rep, SNAP_correction_param_file_dict ):
    all_data_sublib = pd.DataFrame()
    for plate in plates:
        for well in wells:
            plate_well = f'{plate}_{well}'
            sublib_rep = map_plate_well_to_sublib[ plate_well ]
            sublib = sublib_rep.split('_')[0]
            rep = int( sublib_rep.split('_')[1] )
            if (sublib == requested_sublib) and (rep == requested_rep):
                data_file = f'reanalyze_plate{plate}/all_output_data_plate_{plate}_well_{well}_analyze20230323_no_duplicates_filter_matches.csv'
                data_plate_well = pd.read_csv( data_file )
                data_plate_well['plate'] = plate
                data_plate_well['replicate'] = rep
                # edit the file path for cell images
                data_plate_well['cell_img_gfp_file'] = f'reanalyze_plate{plate}/' + data_plate_well['cell_img_gfp_file']
                data_plate_well['cell_img_mask_file'] = f'reanalyze_plate{plate}/' + data_plate_well['cell_img_mask_file']
                data_plate_well['cell_img_dapi_file'] = f'reanalyze_plate{plate}/' + data_plate_well['cell_img_dapi_file']
                if plate_well in SNAP_correction_param_file_dict.keys():
                    print( f"Correcting SNAP intensity in plate {plate} well {well}" )
                    SNAP_correction_param_file = SNAP_correction_param_file_dict[plate_well]
                    data_plate_well = apply_SNAP_intensity_correction( SNAP_correction_param_file, data_plate_well)
                all_data_sublib = all_data_sublib.append( data_plate_well )
                print( f"Loading plate {plate} well {well} for sublib {requested_sublib} rep {requested_rep}" )
                
    print()
    return all_data_sublib

def load_data_sublib_small_pools( requested_sublib, requested_rep ):
    all_data_sublib = pd.DataFrame()
    for plate in small_plates:
        for well in wells:
            plate_well = f'{plate}_{well}'
            sublib_rep = map_small_plate_well_to_sublib[ plate_well ]
            sublib = sublib_rep.split('_')[0]
            rep = int( sublib_rep.split('_')[1] )
            if (sublib == requested_sublib) and (rep == requested_rep):
                data_file = f'reanalyze_small_pools_plate{plate}/all_output_data_plate_{plate}_well_{well}_analyze20230323_no_duplicates_filter_matches.csv'
                data_plate_well = pd.read_csv( data_file )
                data_plate_well['plate'] = f'small_{plate}'
                data_plate_well['replicate'] = rep
                # edit the file path for cell images
                data_plate_well['cell_img_gfp_file'] = f'reanalyze_small_pools_plate{plate}/' + data_plate_well['cell_img_gfp_file']
                data_plate_well['cell_img_mask_file'] = f'reanalyze_small_pools_plate{plate}/' + data_plate_well['cell_img_mask_file']
                data_plate_well['cell_img_dapi_file'] = f'reanalyze_small_pools_plate{plate}/' + data_plate_well['cell_img_dapi_file']
                all_data_sublib = all_data_sublib.append( data_plate_well )
                print( f"Loading plate {plate} well {well} for sublib {requested_sublib} rep {requested_rep}" )
                
    print()
    return all_data_sublib

# plot mean intensity by barcode

def get_mean_intensity_by_barcode( data_1, data_2, barcodes, min_cells=30 ):

    data_1_int = []
    data_2_int = []
    data_1_single_bc = data_1[data_1['cell_barcode_1'].isnull()]
    data_2_single_bc = data_2[data_2['cell_barcode_1'].isnull()]
    for barcode in barcodes:
        data_1_bc = data_1_single_bc[ (data_1_single_bc['cell_barcode_0'] == barcode ) ]
        data_2_bc = data_2_single_bc[ (data_2_single_bc['cell_barcode_0'] == barcode ) ]

        if len( data_1_bc ) < min_cells: continue
        if len( data_2_bc ) < min_cells: continue
        data_1_int.append( np.mean( data_1_bc['mean_GFP_intensity_GFP'] ) )
        data_2_int.append( np.mean( data_2_bc['mean_GFP_intensity_GFP'] ) )
    data_1_int = np.array( data_1_int )
    data_2_int = np.array( data_2_int )
    return data_1_int, data_2_int

def plot_mean_intensity_by_barcode( data_1, data_2, barcodes, min_cells=30):
    data_1_int, data_2_int = get_mean_intensity_by_barcode( data_1, data_2, barcodes, min_cells )
    
    fig, ax = plt.subplots( 1, 1, figsize=(5, 5) )
    plt.scatter( data_1_int, data_2_int, alpha=0.5 )
    plt.plot( [min(data_1_int), max(data_1_int)], [min(data_1_int), max(data_1_int)], color='gray', ls='--' )
    plt.show()

def get_mean_num_cond_by_barcode( data_all, barcodes, min_cells=30, 
        min_intensity=100., max_intensity=10000., min_frac=0.3 ):

    data = data_all[ data_all['cell_barcode_1'].isnull() ]

    mean_num_cond_by_barcode = []
    for barcode in barcodes:
        data_bc = data[ (data['cell_barcode_0'] == barcode) ] 
        
        data_bc_range = data_bc[ (data_bc['mean_GFP_intensity_GFP'] > min_intensity ) &
                (data_bc['mean_GFP_intensity_GFP'] < max_intensity ) ]

        num_cells_bc = len( data_bc )
        num_cells_bc_range = len( data_bc_range )

        if num_cells_bc_range < (min_frac*num_cells_bc_range):
            mean_num_cond_by_barcode.append( np.nan )
            continue
        mean_num_cond = np.mean( data_bc_range['num_condensates_GFP'] )
        mean_num_cond_by_barcode.append( mean_num_cond )

    return mean_num_cond_by_barcode

def get_avg_property_by_barcode( data_all, barcodes, calc_property, 
        use_mean = True, min_cells=30,
        min_intensity=100., max_intensity=10000., min_frac=0.3 ):

    data = data_all[ data_all['cell_barcode_1'].isnull() ]

    avg_property_by_barcode = []
    for barcode in barcodes:
        data_bc = data[ (data['cell_barcode_0'] == barcode) ]

        data_bc_range = data_bc[ (data_bc['mean_GFP_intensity_GFP'] > min_intensity ) &
                (data_bc['mean_GFP_intensity_GFP'] < max_intensity ) ]

        num_cells_bc = len( data_bc )
        num_cells_bc_range = len( data_bc_range )

        if num_cells_bc_range < (min_frac*num_cells_bc_range):
            avg_property_by_barcode.append( np.nan )
            continue
        if use_mean:
            avg_property = np.mean( data_bc_range[calc_property] )
        else:
            avg_property = np.median( data_bc_range[calc_property] )
        avg_property_by_barcode.append( avg_property )

    return avg_property_by_barcode

