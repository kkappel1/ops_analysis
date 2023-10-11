import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
from scipy.stats import pearsonr, median_abs_deviation
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
                all_data_sublib = pd.concat( [all_data_sublib, data_plate_well] )
                #all_data_sublib = all_data_sublib.append( data_plate_well )
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

def load_data_sublib_and_correct_SNAP( requested_sublib, requested_rep, SNAP_correction_param_file_dict,
        base_dir=''):
    all_data_sublib = pd.DataFrame()
    for plate in plates:
        for well in wells:
            plate_well = f'{plate}_{well}'
            sublib_rep = map_plate_well_to_sublib[ plate_well ]
            sublib = sublib_rep.split('_')[0]
            rep = int( sublib_rep.split('_')[1] )
            if (sublib == requested_sublib) and (rep == requested_rep):
                if base_dir != '':
                    data_file = f'{base_dir}/reanalyze_plate{plate}/all_output_data_plate_{plate}_well_{well}_analyze20230323_no_duplicates_filter_matches.csv'
                else:
                    data_file = f'reanalyze_plate{plate}/all_output_data_plate_{plate}_well_{well}_analyze20230323_no_duplicates_filter_matches.csv'
                data_plate_well = pd.read_csv( data_file )
                data_plate_well['plate'] = plate
                data_plate_well['replicate'] = rep
                # edit the file path for cell images
                if base_dir != '':
                    data_plate_well['cell_img_gfp_file'] = f'{base_dir}/reanalyze_plate{plate}/' + data_plate_well['cell_img_gfp_file']
                    data_plate_well['cell_img_mask_file'] = f'{base_dir}/reanalyze_plate{plate}/' + data_plate_well['cell_img_mask_file']
                    data_plate_well['cell_img_dapi_file'] = f'{base_dir}/reanalyze_plate{plate}/' + data_plate_well['cell_img_dapi_file']
                else:
                    data_plate_well['cell_img_gfp_file'] = f'reanalyze_plate{plate}/' + data_plate_well['cell_img_gfp_file']
                    data_plate_well['cell_img_mask_file'] = f'reanalyze_plate{plate}/' + data_plate_well['cell_img_mask_file']
                    data_plate_well['cell_img_dapi_file'] = f'reanalyze_plate{plate}/' + data_plate_well['cell_img_dapi_file']
                if plate_well in SNAP_correction_param_file_dict.keys():
                    print( f"Correcting SNAP intensity in plate {plate} well {well}" )
                    SNAP_correction_param_file = SNAP_correction_param_file_dict[plate_well]
                    data_plate_well = apply_SNAP_intensity_correction( SNAP_correction_param_file, data_plate_well)
                all_data_sublib = pd.concat( [all_data_sublib, data_plate_well] )
                #all_data_sublib = all_data_sublib.append( data_plate_well )
                print( f"Loading plate {plate} well {well} for sublib {requested_sublib} rep {requested_rep}" )
                
    print()
    return all_data_sublib


def apply_SNAP_intensity_correction_extra_data( param_file, data_df_input):
    data_df = data_df_input.copy( deep=True )
    offset_params = pd.read_csv( param_file )
    offset = offset_params['offset'][0]
    slope = offset_params['slope'][0]
    
    features_to_correct = ['mean_GFP_intensity_no_holes']

    for feature in features_to_correct:
        data_df[ feature ] = slope*(data_df[feature] - offset) +offset

    return data_df

def load_extra_data_sublib_and_correct_SNAP( requested_sublib, requested_rep, SNAP_correction_param_file_dict,
        base_dir=''):
    all_data_sublib = pd.DataFrame()
    for plate in plates:
        for well in wells:
            plate_well = f'{plate}_{well}'
            sublib_rep = map_plate_well_to_sublib[ plate_well ]
            sublib = sublib_rep.split('_')[0]
            rep = int( sublib_rep.split('_')[1] )
            if (sublib == requested_sublib) and (rep == requested_rep):
                if base_dir != '':
                    data_file = f'{base_dir}/plate{plate}/phenotype_data_plate_{plate}_well_{well}_analyze20230728.csv'
                else:
                    data_file = f'plate{plate}/phenotype_data_plate_{plate}_well_{well}_analyze20230728.csv'
                data_plate_well = pd.read_csv( data_file )
                data_plate_well = data_plate_well[['cell_img_gfp_file','corr_GFP_dapi','mean_GFP_intensity_no_holes']]
                data_plate_well['plate'] = plate
                data_plate_well['replicate'] = rep
                # edit the file path for cell images
                data_plate_well['cell_img_gfp_file'] = data_plate_well['cell_img_gfp_file'].str.replace( f'process_phenotype/plate_{plate}//CELL_IMAGES_analyze20230728',f'process_phenotype//plate_{plate}//CELL_IMAGES_analyze20230323')
                if base_dir != '':
                    data_plate_well['cell_img_gfp_file'] = f'{base_dir}/reanalyze_plate{plate}/' + data_plate_well['cell_img_gfp_file']
                else:
                    data_plate_well['cell_img_gfp_file'] = f'reanalyze_plate{plate}/' + data_plate_well['cell_img_gfp_file']
                if plate_well in SNAP_correction_param_file_dict.keys():
                    print( f"Correcting SNAP intensity in plate {plate} well {well}" )
                    SNAP_correction_param_file = SNAP_correction_param_file_dict[plate_well]
                    data_plate_well = apply_SNAP_intensity_correction_extra_data( SNAP_correction_param_file, data_plate_well)
                all_data_sublib = pd.concat( [all_data_sublib, data_plate_well] )
                #all_data_sublib = all_data_sublib.append( data_plate_well )
                print( f"Loading plate {plate} well {well} for sublib {requested_sublib} rep {requested_rep}" )
                
    print()
    return all_data_sublib



def load_data_sublib_small_pools( requested_sublib, requested_rep, base_dir='' ):
    all_data_sublib = pd.DataFrame()
    for plate in small_plates:
        for well in wells:
            plate_well = f'{plate}_{well}'
            sublib_rep = map_small_plate_well_to_sublib[ plate_well ]
            sublib = sublib_rep.split('_')[0]
            rep = int( sublib_rep.split('_')[1] )
            if (sublib == requested_sublib) and (rep == requested_rep):
                if base_dir != '':
                    data_file = f'{base_dir}/reanalyze_small_pools_plate{plate}/all_output_data_plate_{plate}_well_{well}_analyze20230323_no_duplicates_filter_matches.csv'
                else:
                    data_file = f'reanalyze_small_pools_plate{plate}/all_output_data_plate_{plate}_well_{well}_analyze20230323_no_duplicates_filter_matches.csv'
                data_plate_well = pd.read_csv( data_file )
                data_plate_well['plate'] = f'small_{plate}'
                data_plate_well['replicate'] = rep
                # edit the file path for cell images
                if base_dir != '':
                    data_plate_well['cell_img_gfp_file'] = f'{base_dir}/reanalyze_small_pools_plate{plate}/' + data_plate_well['cell_img_gfp_file']
                    data_plate_well['cell_img_mask_file'] = f'{base_dir}/reanalyze_small_pools_plate{plate}/' + data_plate_well['cell_img_mask_file']
                    data_plate_well['cell_img_dapi_file'] = f'{base_dir}/reanalyze_small_pools_plate{plate}/' + data_plate_well['cell_img_dapi_file']
                else:
                    data_plate_well['cell_img_gfp_file'] = f'reanalyze_small_pools_plate{plate}/' + data_plate_well['cell_img_gfp_file']
                    data_plate_well['cell_img_mask_file'] = f'reanalyze_small_pools_plate{plate}/' + data_plate_well['cell_img_mask_file']
                    data_plate_well['cell_img_dapi_file'] = f'reanalyze_small_pools_plate{plate}/' + data_plate_well['cell_img_dapi_file']
                all_data_sublib = pd.concat( [all_data_sublib, data_plate_well] )
                #all_data_sublib = all_data_sublib.append( data_plate_well )
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
        min_intensity=100., max_intensity=10000., min_frac=0.3,
        remove_intensity_percentile=0 ):

    data = data_all[ data_all['cell_barcode_1'].isnull() ]

    avg_property_by_barcode = []
    for barcode in barcodes:
        if remove_intensity_percentile > 0:
            data_bc = data[ (data['cell_barcode_0'] == barcode) ]
            intensity_percentile = np.percentile( data_bc[ 'mean_GFP_intensity_GFP' ], 
                    remove_intensity_percentile )
            data_bc = data_bc[ data_bc['mean_GFP_intensity_GFP'] > intensity_percentile ]
        else:
            data_bc = data[ (data['cell_barcode_0'] == barcode) ]

        data_bc_range = data_bc[ (data_bc['mean_GFP_intensity_GFP'] > min_intensity ) &
                (data_bc['mean_GFP_intensity_GFP'] < max_intensity ) ]

        num_cells_bc = len( data_bc )
        num_cells_bc_range = len( data_bc_range )

        if num_cells_bc_range < min_cells:
            avg_property_by_barcode.append( np.nan )
            continue
        if num_cells_bc_range < (min_frac*num_cells_bc):
            avg_property_by_barcode.append( np.nan )
            continue
        if use_mean:
            avg_property = np.mean( data_bc_range[calc_property] )
        else:
            avg_property = np.median( data_bc_range[calc_property] )
        avg_property_by_barcode.append( avg_property )

    return avg_property_by_barcode

def get_avg_properties_for_barcode( data_all, barcode, list_calc_properties, 
        use_mean = True, min_cells=30,
        min_intensity=100., max_intensity=10000., min_frac=0.3,
        remove_intensity_percentile=0, 
        use_only_cells_with_condensates_properties=['mean_condensate_intensity_GFP',
            'mean_condensate_intensity_no_shrink_GFP','std_condensate_intensity_GFP',
            'std_condensate_intensity_no_shrink_GFP', 'mean_condensate_area',
            'std_condensate_area','mean_condensate_eccentricity', 'std_condensate_eccentricity']):

    data = data_all[ data_all['cell_barcode_1'].isnull() ]

    avg_properties_for_barcode = []
    std_properties_for_barcode = []
    if remove_intensity_percentile > 0:
        data_bc = data[ (data['cell_barcode_0'] == barcode) ]
        if len( data_bc ) < 1:
            avg_properties_for_barcode = [np.nan for calc_property in list_calc_properties ]
            std_properties_for_barcode = [np.nan for calc_property in list_calc_properties ]
            return avg_properties_for_barcode, std_properties_for_barcode

        intensity_percentile = np.percentile( data_bc[ 'mean_GFP_intensity_GFP' ], 
                remove_intensity_percentile )
        data_bc = data_bc[ data_bc['mean_GFP_intensity_GFP'] > intensity_percentile ]
    else:
        data_bc = data[ (data['cell_barcode_0'] == barcode) ]

    data_bc_range = data_bc[ (data_bc['mean_GFP_intensity_GFP'] > min_intensity ) &
            (data_bc['mean_GFP_intensity_GFP'] < max_intensity ) ]

    num_cells_bc = len( data_bc )
    num_cells_bc_range = len( data_bc_range )

    if num_cells_bc_range < min_cells:
        avg_properties_for_barcode = [np.nan for calc_property in list_calc_properties ]
        std_properties_for_barcode = [np.nan for calc_property in list_calc_properties ]
        return avg_properties_for_barcode, std_properties_for_barcode

    if num_cells_bc_range < (min_frac*num_cells_bc):
        avg_properties_for_barcode = [np.nan for calc_property in list_calc_properties ]
        std_properties_for_barcode = [np.nan for calc_property in list_calc_properties ]
        return avg_properties_for_barcode, std_properties_for_barcode

    for calc_property in list_calc_properties:
        # if this is a condensate property, get only the subset of cells with condensates
        if calc_property in use_only_cells_with_condensates_properties:
            data_for_calc = data_bc_range[ data_bc_range[ 'num_condensates_GFP' ] > 0 ]
        else:
            data_for_calc = data_bc_range
        if len( data_for_calc ) < min_cells:
            avg_properties_for_barcode.append( np.nan )
            std_properties_for_barcode.append( np.nan )
            continue

        if use_mean:
            avg_property = np.mean( data_for_calc[calc_property] )
            std_property = np.std( data_for_calc[calc_property] )
        else:
            avg_property = np.median( data_for_calc[calc_property] )
            std_property = median_abs_deviation( data_for_calc[calc_property] )
        avg_properties_for_barcode.append( avg_property )
        std_properties_for_barcode.append( std_property )

    return avg_properties_for_barcode, std_properties_for_barcode

def get_fraction_cells_with_condensates( data_all, barcode, min_intensity=100., 
        max_intensity=10000., remove_intensity_percentile=0 ):
    data = data_all[ data_all['cell_barcode_1'].isnull() ]
    if len( data ) < 1:
        return np.nan, 0
    if remove_intensity_percentile > 0:
        data_bc = data[ (data['cell_barcode_0'] == barcode) ]
        if len( data_bc ) < 1:
            return np.nan, 0
        intensity_percentile = np.percentile( data_bc[ 'mean_GFP_intensity_GFP' ], 
                remove_intensity_percentile )
        data_bc = data_bc[ data_bc['mean_GFP_intensity_GFP'] > intensity_percentile ]
    else:
        data_bc = data[ (data['cell_barcode_0'] == barcode) ]

    data_bc_range = data_bc[ (data_bc['mean_GFP_intensity_GFP'] > min_intensity ) &
            (data_bc['mean_GFP_intensity_GFP'] < max_intensity ) ]
    total_num_cells = len( data_bc_range )
    if total_num_cells < 1:
        return np.nan, 0

    data_bc_range_condensates = data_bc_range[ data_bc_range['num_condensates_GFP'] >0 ]
    fraction_cells_with_condensates = len( data_bc_range_condensates ) / total_num_cells

    return fraction_cells_with_condensates, total_num_cells

def get_percentile_property_for_barcode( data_all, barcode, calc_property,
        percentiles, min_cells=30,
        min_intensity=0., max_intensity=100000., min_frac=0.3 ):

    data = data_all[ data_all['cell_barcode_1'].isnull() ]

    data_bc = data[ (data['cell_barcode_0'] == barcode) ]

    data_bc_range = data_bc[ (data_bc['mean_GFP_intensity_GFP'] > min_intensity ) &
            (data_bc['mean_GFP_intensity_GFP'] < max_intensity ) ]


    num_cells_bc = len( data_bc )
    num_cells_bc_range = len( data_bc_range )

    if (num_cells_bc_range < (min_frac*num_cells_bc)) or (num_cells_bc_range < min_cells):
        percentiles_data = [ np.nan for x in percentiles ]
        return num_cells_bc_range, percentiles_data

    percentiles_data = []
    for percentile in percentiles:
        percentiles_data.append( np.percentile( data_bc_range[ calc_property ], percentile ) )
        
    return num_cells_bc_range, percentiles_data


