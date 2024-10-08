import image_analysis
import image_analysis.preprocess_phenotype
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
from skimage.registration._masked_phase_cross_correlation import cross_correlate_masked
#from skimage.feature.masked_register_translation import cross_correlate_masked
import matplotlib
#matplotlib.use('Agg')
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

# this is not ideal
import warnings
warnings.filterwarnings("ignore")


def phenotype_well( plate_num, well_num, out_tag, output_tif_40x_base_dir, list_dapi_files, 
        check_match, match_dir, NUM_PROCESSES, ffc_file_dapi='', ffc_file_gfp='', do_ffc=True,
        nuclei_masks_dir=''):
    if check_match: 
        print( "Checking for match files", flush=True )
    else:
        print( "NOT checking for match files", flush=True )
    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        begin_time = datetime.datetime.now()
        min_size = 1000
        smooth_size = 5
        threshold_initial_guess = 150
        nuclei_smooth_value = 20
        area_min=2500
        plot_phenotype=False
        area_max = 20000
        condensate_cutoff_intensity = 300
        THRESH_STDS =3
        use_cellpose = True
        cellpose_diameter =87
    
        #dict_40x_nuclei_masks = {}
        full_df_nuclei = pd.DataFrame()
        files_and_tiles = []
        list_of_dapi_files = []
        list_of_gfp_files = []
        for f in list_dapi_files:
            base_fname = f.split('/')[-1].split('.tif')[0]
            if check_match and not os.path.exists( f'{match_dir}/match_{base_fname}.txt' ): 
                continue
            gfp_file = f.replace( '-ch1', '-ch2' )
            list_of_dapi_files.append( f )
            list_of_gfp_files.append( gfp_file )
            tile_num = get_field_num_from_phenix_name( f )
            file_save_dir = '{output_tif_40x_base_dir}/CELL_IMAGES_{out_tag}/well_{well_num}/field_{tile_num}/'.format(
                    output_tif_40x_base_dir=output_tif_40x_base_dir,out_tag=out_tag,well_num=well_num,tile_num=tile_num)
            if nuclei_masks_dir != '':
                nuclei_mask_file = f'{nuclei_masks_dir}/nuclei_mask_{base_fname}.tif'
                files_and_tiles.append( [f, gfp_file, tile_num, file_save_dir, nuclei_mask_file] )
            else:
                files_and_tiles.append( [f, gfp_file, tile_num, file_save_dir] )

        print( "len(files_and_tiles)", len(files_and_tiles), flush=True )

        if ffc_file_gfp != '' and ffc_file_dapi != '':
            print( "reading ffc files", flush=True )
            ffc_dapi = read( ffc_file_dapi )
            ffc_gfp = read( ffc_file_gfp )
        else:
            # get the flatfield correction for each channel
            ffc_dapi = image_analysis.preprocess_phenotype.get_flatfield_correction( list_of_dapi_files )
            ffc_gfp = image_analysis.preprocess_phenotype.get_flatfield_correction( list_of_gfp_files )

            save( f'{output_tif_40x_base_dir}/ffc_files/ffc_dapi_plate_{plate_num}_well_{well_num}_{out_tag}.tif', ffc_dapi )
            save( f'{output_tif_40x_base_dir}/ffc_files/ffc_gfp_plate_{plate_num}_well_{well_num}_{out_tag}.tif', ffc_gfp )

        if not do_ffc:
            print( "overwriting ffcs" )
            ffc_dapi = np.ones_like( ffc_dapi )
            ffc_gfp = np.ones_like( ffc_gfp )

        print( "min/max ffc", np.min( ffc_dapi ), np.max( ffc_dapi ), flush=True )
        print( "min/max ffc", np.min( ffc_gfp ), np.max( ffc_gfp ), flush=True )
        # write out dapi, gfp, and mask files for each cell
        # need file_save_dir and field_name
        # use output_tif_40x_base_dir

        start_time_ph = datetime.datetime.now()
        if nuclei_masks_dir != '':
            print( "running with nuclei masks" )
            phenotype_results = pool.starmap( prelim_phenotype_phenix_2channel_write_img_files, [(x[0], x[1], ffc_dapi, ffc_gfp, 
                                well_num, x[2], x[3], min_size, smooth_size, threshold_initial_guess, 
                                nuclei_smooth_value, area_min, plot_phenotype, area_max, 
                                condensate_cutoff_intensity, THRESH_STDS, use_cellpose, 
                                cellpose_diameter, x[4]) for x in files_and_tiles])
        else:
            phenotype_results = pool.starmap( prelim_phenotype_phenix_2channel_write_img_files, [(x[0], x[1], ffc_dapi, ffc_gfp, 
                                well_num, x[2], x[3], min_size, smooth_size, threshold_initial_guess, 
                                nuclei_smooth_value, area_min, plot_phenotype, area_max, 
                                condensate_cutoff_intensity, THRESH_STDS, use_cellpose, 
                                cellpose_diameter) for x in files_and_tiles])
        

        end_time_ph = datetime.datetime.now()
        print( "time phenotyping mapped:", end_time_ph - start_time_ph, flush=True )

        if nuclei_masks_dir == '':
            nuclei_mask_outdir = f'nuclei_masks_plate_{plate_num}_well_{well_num}_{out_tag}'
            if not os.path.exists( nuclei_mask_outdir ):
                os.makedirs( nuclei_mask_outdir )
            for i, result in enumerate(phenotype_results):
                fname = files_and_tiles[i][0]
                base_fname = fname.split('/')[-1].replace('.tiff','.tif')
                nuclei_mask = result[1]
                #dict_40x_nuclei_masks[fname] = nuclei_mask
                # write the nuclei mask out
                output_nuclei_mask_fname = f'{nuclei_mask_outdir}/nuclei_mask_{base_fname}'
                save( output_nuclei_mask_fname, nuclei_mask )
        for i, result in enumerate(phenotype_results):
            df_nuclei = result[0]
            full_df_nuclei = full_df_nuclei.append( df_nuclei )

        # write the full_df_nuclei out to a file
        full_df_nuclei.to_csv( "phenotype_data_plate_{plate_num}_well_{well_num}_{out_tag}.csv".format(
            plate_num=plate_num, well_num=well_num, out_tag=out_tag), index=None )

        end_time = datetime.datetime.now()
        print("Time phenotyping:", end_time - begin_time )


if __name__ == '__main__':
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser( description="basic phenotyping for images in a single well" ) 
    parser.add_argument( '-plate_num', type=str, default="", help='plate number to analyze' )
    parser.add_argument( '-well_num', type=str, default="", help='well number to analyze' )
    parser.add_argument( '-out_tag', type=str, default="", help='output tag' )
    parser.add_argument( '-output_tif_40x_base_dir', type=str, default="", 
                help='base dir for tiffs, e.g. phenotype_images/plate_10/' )
    parser.add_argument( '-check_for_match_file', default=False, action='store_true',
                help='check if there is a match file for each image before using for phenotyping' )
    parser.add_argument( '-no_ffc', default=False, action='store_true', 
            help='Do not do flatfield correction')
    parser.add_argument( '-match_dir', type=str, default='', help='directory that contains match files' )
    parser.add_argument( '-nuclei_masks_dir', type=str, default='', help='directory that contains nuclei masks' )
    parser.add_argument( '-ffc_file_dapi', type=str, default='', help='ffc file for dapi, optional' )
    parser.add_argument( '-ffc_file_gfp', type=str, default='', help='ffc file for gfp, optional' )
    parser.add_argument( '-list_dapi_files', nargs='+', help="List of all dapi files to analyze" )
    parser.add_argument( '-num_proc', type=int, default=1, help="number of processors to run on" )
    args = parser.parse_args()
    if args.no_ffc:
        do_ffc = False
    else:
        do_ffc = True
    phenotype_well( args.plate_num, args.well_num, args.out_tag,
                args.output_tif_40x_base_dir, args.list_dapi_files, 
                args.check_for_match_file, args.match_dir, args.num_proc, 
                args.ffc_file_dapi, args.ffc_file_gfp, do_ffc, nuclei_masks_dir )
