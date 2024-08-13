import image_analysis
from image_analysis.SBS_analysis_functions_v221121 import *
from image_analysis.matching_updated_centroids import *


##################
# Plate-specific settings
##################
wells = ['B5']
plate_num = 6
dapi_40x_base_dir = 'raw_phenotype_images/'
sbs_images_base_dir = 'raw_SBS_images/'
num_proc = 6
num_40x_files_match = 24

nd2_files_by_well = {
        'B5': "raw_SBS_images/nd2_files/cycle_1/WellB5.nd2",
}
##################

def well_to_well_phenix_24( well_name ):
    row = well_name[0]
    col = well_name[1]
    if row == 'A':
        phenix_row = 'r01'
    elif row == 'B':
        phenix_row = 'r02'
    elif row == 'C':
        phenix_row = 'r03'
    elif row == 'D':
        phenix_row = 'r04'
    phenix_col = 'c0' + col
    phenix_well = phenix_row + phenix_col
    return phenix_well

for well in wells:
    print( "Matching well" , well )

    well_phenix = well_to_well_phenix_24( well )
    list_of_40x_files_dapi = glob.glob( '{dapi_40x_base_dir}/{well_phenix}f*-ch1*.tiff'.format(
        dapi_40x_base_dir=dapi_40x_base_dir,
        well_phenix=well_phenix) )
    index_file_phenix = '{dapi_40x_base_dir}/Index.idx.xml'.format( dapi_40x_base_dir=dapi_40x_base_dir )
    list_of_files_10x = glob.glob( '{sbs_images_base_dir}/cycle_1/*tif'.format(
        sbs_images_base_dir=sbs_images_base_dir,
        well=well, 
        plate_num=plate_num))
    nd2_file_10x = nd2_files_by_well[ well ]
    pixel_size_10x = 0.841792491782224
    pixel_size_40x = 0.14949402023919043
    output_dir_brute_force_match = 'match_brute_force_plate_{plate_num}_well_{well}'.format( well=well, plate_num=plate_num ) 
    output_dir_match = 'match_all_plate_{plate_num}_well_{well}'.format( well=well, plate_num=plate_num )
    plot=True
    skip_brute_force = False
    
    map_40x_to_10x_files_with_10xtile_mapping( list_of_40x_files_dapi, well_phenix, index_file_phenix,
                        list_of_files_10x, nd2_file_10x, pixel_size_40x,
                        pixel_size_10x, output_dir_brute_force_match, output_dir_match,
                        num_proc, overlap_ratio_final_matches = 0.4,
                        num_40x_files_match=num_40x_files_match, plot=plot,
                        skip_brute_force=skip_brute_force)
