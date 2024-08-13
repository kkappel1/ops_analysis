from image_analysis.basic_phenotype_well import phenotype_well
from image_analysis.match_cells_well import *
import glob

####################
# inputs - required
####################
plate_num = 6
well = 'B5'
out_tag = 'example'
out_tag_nuclei_mask_dir = 'example'
out_tag_ffc = 'example'

base_dir_40x_files = 'raw_phenotype_images/'
num_proc = 6

# nd2 file from cycle 1
nd2_file_well_cycle_1 = 'raw_SBS_images/nd2_files/cycle_1/WellB5.nd2'
phenotype_10x_dir = 'raw_SBS_images/GFP_cycle_5/'

threshold_cc_match = 0.20
dist_threshold_match = 15.0
####################


####################
# inputs - optionally modify
####################
check_for_match_file = True
do_ffc = True
output_tif_40x_base_dir_top = 'process_phenotype/'
data_file_10x = 'process_cellpose_nophenotype_all/filtered_cells_combined.csv'
nuclei_mask_dir_10x = 'process_cellpose_nophenotype_all/'
####################


print( f"Plate {plate_num}, well {well}: running phenotyping" )
list_dapi_files = glob.glob( f'{base_dir_40x_files}/*-ch1*' )
match_dir = f'match_all_plate_{plate_num}_well_{well}/'
output_tif_40x_base_dir = output_tif_40x_base_dir_top + f'/plate_{plate_num}/'
nuclei_masks_dir = f'nuclei_masks_plate_{plate_num}_well_{well}_{out_tag_nuclei_mask_dir}'
ffc_file_dapi = f'{output_tif_40x_base_dir_top}/ffc_files/ffc_dapi_plate_{plate_num}_well_{well}_{out_tag_ffc}.tif'
ffc_file_gfp = f'{output_tif_40x_base_dir_top}/ffc_files/ffc_gfp_plate_{plate_num}_well_{well}_{out_tag_ffc}.tif'

# run the phenotype analysis
phenotype_well( plate_num=plate_num, well_num=well, out_tag=out_tag, 
        output_tif_40x_base_dir=output_tif_40x_base_dir, 
        list_dapi_files=list_dapi_files,
        check_match=check_for_match_file, match_dir=match_dir, 
        NUM_PROCESSES=num_proc, do_ffc=do_ffc, nuclei_masks_dir=nuclei_masks_dir,
        ffc_file_dapi=ffc_file_dapi, ffc_file_gfp=ffc_file_gfp)

data_file_40x = f'phenotype_data_plate_{plate_num}_well_{well}_{out_tag}.csv'
nuclei_mask_dir_40x = f'nuclei_masks_plate_{plate_num}_well_{well}_{out_tag}/'
print( f"Plate {plate_num}, well {well}: running matching" )
match_well( plate_num=plate_num, well_num=well, out_tag=out_tag, 
        NUM_PROCESSES=num_proc, list_of_files_40x=list_dapi_files,
        data_file_10x=data_file_10x, data_file_40x=data_file_40x, 
        nd2_file_well=nd2_file_well_cycle_1, match_file_dir=match_dir, 
        nuclei_mask_dir_40x=nuclei_mask_dir_40x, nuclei_mask_dir_10x=nuclei_mask_dir_10x, 
        phenotype_10x_dir=phenotype_10x_dir, 
        THRESHOLD_CC_MATCH=threshold_cc_match, DIST_THRESHOLD_MATCH=dist_threshold_match, 
        output_match_dir=output_tif_40x_base_dir)
