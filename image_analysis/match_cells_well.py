import image_analysis
from image_analysis.SBS_analysis_functions_v221121 import *
from image_analysis.matching_updated_centroids import * 
import skimage
import numpy as np
from ops.io import read_stack as read
from ops.io import save_stack as save
import ops.io
from skimage import (
    color, feature, filters, measure, morphology, segmentation, util
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.morphology import disk, watershed
from skimage.filters import threshold_otsu, rank
from scipy import ndimage as ndi
from ops.imports import *
import matplotlib.patches as mpatches
import random
import glob
from ops.process import Align
import ops.firesnake
from ops.firesnake import Snake
from collections import Counter
import operator
import math
import multiprocessing
import datetime
import argparse
from sklearn import linear_model


def combine_10x_40x_data( dict_40x_nuclei_to_10x_nuclei, file_40x, best_image_match_10x,
                            full_df_10x_cells, full_df_nuclei ):
    #print( full_df_10x_cells )
    df_merged_data = pd.DataFrame()
    for cell_40x in dict_40x_nuclei_to_10x_nuclei.keys():
        cell_10x = dict_40x_nuclei_to_10x_nuclei[ cell_40x ][ 'nucleus_10x' ]
        well_num_10x, tile_num_10x = get_well_and_tile_num_from_fname_10x( best_image_match_10x )


        data_10x = full_df_10x_cells[ (full_df_10x_cells['tile_10x'] == tile_num_10x)
                                     & (full_df_10x_cells['cell'] == cell_10x) &
                                    (full_df_10x_cells['well'] == well_num_10x)]

        # the cell may not be in the 10x df if it does not have a barcode
        data_40x = full_df_nuclei[ (full_df_nuclei['fname_dapi'] == file_40x) &
                                 (full_df_nuclei['nucleus_num'] == cell_40x)]

        # merge the data -- if there is a barcode for this cell
        if data_10x.shape[0] == 1 and data_40x.shape[0] == 1:
            merged_data = data_10x.merge( data_40x, on='well', how='left')
            df_merged_data = df_merged_data.append( merged_data )
    return df_merged_data


def combine_10x_40x_data_and_info( dict_40x_nuclei_to_10x_nuclei, file_40x, best_image_match_10x,
                            best_cc_mag, full_df_10x_cells, full_df_nuclei ):
    # include all the matching info in the dataframe 
    df_merged_data = pd.DataFrame()
    for cell_40x in dict_40x_nuclei_to_10x_nuclei.keys():
        cell_10x = dict_40x_nuclei_to_10x_nuclei[ cell_40x ][ 'nucleus_10x' ]
        well_num_10x, tile_num_10x = get_well_and_tile_num_from_fname_10x( best_image_match_10x )


        data_10x = full_df_10x_cells[ (full_df_10x_cells['tile_10x'] == tile_num_10x)
                                     & (full_df_10x_cells['cell'] == cell_10x) &
                                    (full_df_10x_cells['well'] == well_num_10x)]

        # the cell may not be in the 10x df if it does not have a barcode
        data_40x = full_df_nuclei[ (full_df_nuclei['fname_dapi'] == file_40x) &
                                 (full_df_nuclei['nucleus_num'] == cell_40x)]

        # merge the data -- if there is a barcode for this cell
        if data_10x.shape[0] == 1 and data_40x.shape[0] == 1:
            merged_data = data_10x.merge( data_40x, on='well', how='left')
            # add columns to merged_data
            merged_data['match_dist'] = [ dict_40x_nuclei_to_10x_nuclei[ cell_40x ][ 'min_dist' ]]
            merged_data['match_dist_second_min'] = [ dict_40x_nuclei_to_10x_nuclei[ cell_40x ][ 'second_min_dist' ]]
            merged_data['match_intensity_10x'] = [ dict_40x_nuclei_to_10x_nuclei[ cell_40x ][ 'intensity_10x' ]]
            merged_data['match_intensity_40x'] = [ dict_40x_nuclei_to_10x_nuclei[ cell_40x ][ 'intensity_40x' ]]
            merged_data['match_glcm_dissim_10x'] = [ dict_40x_nuclei_to_10x_nuclei[ cell_40x ][ 'glcm_dissim_10x' ]]
            merged_data['match_glcm_dissim_40x'] = [ dict_40x_nuclei_to_10x_nuclei[ cell_40x ][ 'glcm_dissim_40x' ]]
            merged_data['match_10x_centroid_x'] = [ dict_40x_nuclei_to_10x_nuclei[ cell_40x ][ 'match_10x_centroid' ][0]]
            merged_data['match_10x_centroid_y'] = [ dict_40x_nuclei_to_10x_nuclei[ cell_40x ][ 'match_10x_centroid' ][1]]
            merged_data['match_40x_centroid_x'] = [ dict_40x_nuclei_to_10x_nuclei[ cell_40x ][ 'match_40x_centroid' ][0]]
            merged_data['match_40x_centroid_y'] = [ dict_40x_nuclei_to_10x_nuclei[ cell_40x ][ 'match_40x_centroid' ][1]]
            merged_data['match_file_cc_mag'] = [ best_cc_mag ]
            df_merged_data = df_merged_data.append( merged_data )
    return df_merged_data

def get_well_and_tile_num_from_fname_10x( fname ):
    # Names look like:
    # WellA03_ChannelSBS_DAPI,FITC-Penta,SBS_Cy3,SBS_A594,SBS_Cy5,SBS_Cy7_Seq0000-tile0.tif
    base_fname = fname.split('/')[-1]
    well_num = base_fname.split('_')[0].replace('Well','')
    tile_num = int( base_fname.split('-')[-1].replace('.tif','').replace('tile','') )
    return well_num, tile_num



def match_well( plate_num, well_num, out_tag, NUM_PROCESSES, list_of_files_40x, 
                data_file_10x, data_file_40x, nd2_file_well, match_file_dir, 
                nuclei_mask_dir_40x, nuclei_mask_dir_10x,
                THRESHOLD_CC_MATCH=0.30 ):
    
    begin_time = datetime.datetime.now()
    # load the df for the 10x cells
    full_df_10x_cells = pd.read_csv( data_file_10x )
    
    # load the df for the 40x cells
    full_df_nuclei = pd.read_csv( data_file_40x )
    
    
    # get xy coords from nd2 file
    sites_to_xy_10x = map_10x_fields_to_xy_coords( nd2_file_well )
    
    # for 10x files from RNAscope and 40x files from phenix
    pixel_size_10x = 0.841792491782224
    pixel_size_40x = 0.14949402023919043

    dict_file40x_to_10x_shift = {}
    # rename so merging will be possible later
    if 'tile' in full_df_10x_cells:
        full_df_10x_cells = full_df_10x_cells.rename(columns={"tile": "tile_10x"})
    
    nuclei_masks_40x = []
    nuclei_masks_10x = []
    phenotype_10x_fnames = []
    phenotype_40x_fnames = []
    best_shifts = []
    best_cc_mags = []
    best_image_match_10xs = []
    plot=True
    files_40x_actually_used = []
    plot_tags = []
    tile_origins_10x = []
    # the settings I used before - they seem pretty good
    match_dist_thresh = 15.0
    match_ratio_min_to_next_thresh = 0.9 
    plot=True
    plot_phenotypes=False
    

    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        full_df_merged_data = pd.DataFrame()
    
        nuclei_mask_unique_indices = []
        for i, file_40x in enumerate(list_of_files_40x):
            base_fname = file_40x.split('/')[-1].split('.tif')[0]
            match_fname = f'{match_file_dir}/match_{base_fname}.txt'
            if not os.path.exists( match_fname ):
                continue
            # match file format:
            # {image} {cc_mag} {shift_x} {shift_y} {expected_x} {expected_y} {actual_x} {actual_y}
            #  0       1        2         3         4            5            6          7
            #  -8      -7       -6        -5        -4           -3           -2         -1
            #  -4      -3       -2        -1   (old format, without expected/actual xy)
            for line in open(match_fname):
                best_image_match_10x = line.split()[-8]
                if best_image_match_10x.startswith('SSD'):
                    best_image_match_10x = line.split()[-9] + ' ' + line.split()[-8]
                best_image_match_10x = best_image_match_10x.replace( '//','/')
                best_cc_mag = float(line.split()[-7])
                shift_x = float( line.split()[-6])
                shift_y = float( line.split()[-5])
                best_shift = np.array([shift_x,shift_y])
                expected_x = float( line.split()[-4] )
                expected_y = float( line.split()[-3] )
                actual_x = float( line.split()[-2] )
                actual_y = float( line.split()[-1] )
                dist_actual_expected = math.dist( [expected_x, expected_y], [actual_x, actual_y] )
            if best_cc_mag < THRESHOLD_CC_MATCH:
                continue
            # get the 40x nuclei mask
            nuclei_mask_40x_fname = f'{nuclei_mask_dir_40x}/nuclei_mask_{base_fname}.tif'
            if not os.path.exists( nuclei_mask_40x_fname ):
                print( "missing mask:", nuclei_mask_40x_fname )
                continue
            nuclei_mask_40x = read( nuclei_mask_40x_fname )
            # get the 10x nuclei mask 
            best_image_match_10x_tile = best_image_match_10x.split('/')[-1].split('-tile')[-1].replace('.tif','')
            nuclei_10x_mask_fname = f'{nuclei_mask_dir_10x}/10X_{well_num}_Tile-{best_image_match_10x_tile}.nuclei.tif'
            if not os.path.exists( nuclei_10x_mask_fname ):
                print( "missing 10x mask:", nuclei_10x_mask_fname )
                continue
            nuclei_10x_mask = read( nuclei_10x_mask_fname )

            # get the corresponding "phenotype" images for 10x and 40x
            phenotype_10x_fname = f'test_dapi_gfp_tifs/test_dapi_gfp-tile{best_image_match_10x_tile}.tif'
            phenotype_40x_fname = file_40x.replace( '-ch1', '-ch2' )
            if not os.path.exists( phenotype_10x_fname ):
                print( "missing phenotype 10x fname", phenotype_10x_fname )
                continue
            if not os.path.exists( phenotype_40x_fname ):
                print( "missing phenotype 40x fname", phenotype_40x_fname )
                continue

            dict_file40x_to_10x_shift[ file_40x ] = [best_image_match_10x, best_shift]
            best_image_match_10xs.append( best_image_match_10x )
            if phenotype_10x_fname not in phenotype_10x_fnames:
                nuclei_mask_unique_indices.append( i )
            nuclei_masks_40x.append( nuclei_mask_40x)
            nuclei_masks_10x.append( nuclei_10x_mask )
            phenotype_10x_fnames.append( phenotype_10x_fname )
            phenotype_40x_fnames.append( phenotype_40x_fname ) 
            best_shifts.append( best_shift )
            files_40x_actually_used.append( file_40x )
            best_cc_mags.append( best_cc_mag )
            tile_num_40x = get_field_num_from_phenix_name( file_40x )
            well_num_10x, tile_num_10x =  get_well_and_tile_num_from_fname_10x( best_image_match_10x )
            tile_origins_10x.append( sites_to_xy_10x[ tile_num_10x ] )
            plot_tags.append( 'well_' + str(well_num) + '_field_' + str(tile_num_40x) )
        print( "files_40x_actually_used", files_40x_actually_used )


        mapping_results = pool.starmap(map_40x_nuclei_to_10x_nuclei_centroid_dist_phenotype, [(nuclei_masks_40x[i],
                                            nuclei_masks_10x[i], phenotype_40x_fnames[i], phenotype_10x_fnames[i], 
                                            tile_origins_10x[i], best_shifts[i], pixel_size_10x,
                                            pixel_size_40x, match_dist_thresh, match_ratio_min_to_next_thresh,
                                            plot_tags[i], plot, plot_phenotypes) for i in range(len(best_shifts))] )
        
        # write the matching stats to a file
        with open( 'matching_stats_plate_{plate_num}_well_{well_num}_{out_tag}.txt'.format(plate_num=plate_num,well_num=well_num,out_tag=out_tag), 'w') as f:
            f.write( 'num_matched_nuclei frac_matched_nuclei duplicated_10x_nuclei\n' )
            for map_res in mapping_results:
                f.write( '%d %0.2f %d\n' %(map_res[1], map_res[2], map_res[3]))

        # fit a model for the 10x to 40x phenotype data which can be used for filtering matches
        all_intensities_10x = []
        all_intensities_40x = []
        all_dissim_10x = []
        all_dissim_40x = []

        for mapping_result in mapping_results:
            dict_all_40x_10x_match_data = mapping_result[0]
            for nucleus_40x in dict_all_40x_10x_match_data.keys():
                all_intensities_10x.append( dict_all_40x_10x_match_data[ nucleus_40x ][ 'intensity_10x' ] )
                all_intensities_40x.append( dict_all_40x_10x_match_data[ nucleus_40x ][ 'intensity_40x' ] )
                all_dissim_10x.append( dict_all_40x_10x_match_data[ nucleus_40x ][ 'glcm_dissim_10x' ] )
                all_dissim_40x.append( dict_all_40x_10x_match_data[ nucleus_40x ][ 'glcm_dissim_40x' ] )

        plt.subplots( 1,1,figsize=(5,5) )
        plt.scatter( all_dissim_10x, all_dissim_40x, alpha =0.1 )
        plt.savefig( f'all_dissim_compare_plate{plate_num}_well_{well_num}_{out_tag}.png' )
        plt.clf()

        plt.subplots( 1,1,figsize=(5,5) )
        plt.scatter( all_intensities_10x, all_intensities_40x, alpha =0.1 )
        plt.savefig( f'all_intensities_compare_plate{plate_num}_well_{well_num}_{out_tag}.png' )
        plt.clf()


        # combine data
        # parallelize this data combination
        mapping_results_dfs = pool.starmap( combine_10x_40x_data_and_info, [(mapping_results[i][0],
            files_40x_actually_used[i],
            best_image_match_10xs[i],
            best_cc_mags[i],
            full_df_10x_cells,
            full_df_nuclei) for i in range(len(mapping_results))] )
        for mapped_df in mapping_results_dfs:
            if len(mapped_df) > 0:
                full_df_merged_data = full_df_merged_data.append( mapped_df )
        
        full_df_merged_data.to_csv( 'all_output_data_plate_{plate_num}_well_{well_num}_{out_tag}.csv'.format(
            plate_num=plate_num, well_num=well_num, out_tag=out_tag), index=None )
        end_time = datetime.datetime.now()
        print("Time matching:", end_time - begin_time )

if __name__ == '__main__':
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser( description="10x to 40x cell matching for a single well" )
    parser.add_argument( '-plate_num', type=str, default="", help='plate number to analyze' )
    parser.add_argument( '-well_num', type=str, default="", help='well number to analyze' )
    parser.add_argument( '-out_tag', type=str, default="", help='output tag' )
    parser.add_argument( '-num_proc', type=int, default="", help='number of processors to run on' )
    parser.add_argument( '-list_dapi_files_40x', nargs='+', 
        help="List of all the 40x dapi files to match")
    parser.add_argument( '-data_file_10x', type=str, 
        help="csv file with all 10x sbs data, e.g. 10X_B2_Tile-9.filtered_cells.csv or filtered_cells_combined.csv")
    parser.add_argument( '-data_file_40x', type=str, 
        help="csv file with all 40x phenotype data, e.g. phenotype_data_plate_10_well_B2_out_tag.csv" )
    parser.add_argument( '-nd2_file_well_cycle_1', 
        help='nd2 file for the first cycle of SBS for this well' )
    parser.add_argument( '-match_file_dir', 
        help='directory with all the match files for this plate/well' )
    parser.add_argument( '-nuclei_mask_dir_40x', 
        help='directory with all the 40x nuclei masks, e.g. nuclei_masks_plate_10_well_B2_out_tag/' )
    parser.add_argument( '-nuclei_mask_dir_10x', 
        help='directory with all the 10x nuclei masks, e.g. process_cellpose_nophenotype_all/' )
    parser.add_argument( '-threshold_cc_match', type=float, default=0.30, 
        help='threshold for 40x to 10x raw image match' )

    args = parser.parse_args()
    match_well( args.plate_num, args.well_num, args.out_tag, args.num_proc,
                args.list_dapi_files_40x, args.data_file_10x, args.data_file_40x,
                args.nd2_file_well_cycle_1, args.match_file_dir, args.nuclei_mask_dir_40x,
                args.nuclei_mask_dir_10x, args.threshold_cc_match )

