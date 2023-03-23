import image_analysis
from image_analysis.SBS_analysis_functions_v221121 import *
from image_analysis.matching_updated_centroids import * 


##################
# Plate-specific settings
##################
wells = ['C2']
#wells = ['B5','C2','C3','C4','C5']
#wells = ['B2','B3','B4','B5','C2','C3','C4','C5']
plate_num = 10
dapi_40x_base_dir = '/mnt/disks/data5/20230210_SBS_data/phenotype_images/'
sbs_images_base_dir = '/mnt/disks/data5/20230210_SBS_data/SBS_images/'
num_proc = 64
num_40x_files_match = 64

base_nd2_dir = sbs_images_base_dir + '/nd2_files/20230210_SBS_data/Kalli_SBS_24_20230203_multi_settings_plate10/Cycle1_20230203_141620_649/'

nd2_files_by_well = {
        'B2': base_nd2_dir + "/WellB2_ChannelKK_SBS_DAPI,KK_SBS_Cy3,KK_SBS_A594,KK_SBS_Cy5,KK_SBS_Cy7,SBS_Cy7_s4,SBS_Cy5_s1,SBS_Cy5_s4_Seq0000.nd2",
        'B3': base_nd2_dir + "/WellB3_ChannelKK_SBS_DAPI,KK_SBS_Cy3,KK_SBS_A594,KK_SBS_Cy5,KK_SBS_Cy7,SBS_Cy7_s4,SBS_Cy5_s1,SBS_Cy5_s4_Seq0001.nd2",
        'B4': base_nd2_dir + "/WellB4_ChannelKK_SBS_DAPI,KK_SBS_Cy3,KK_SBS_A594,KK_SBS_Cy5,KK_SBS_Cy7,SBS_Cy7_s4,SBS_Cy5_s1,SBS_Cy5_s4_Seq0002.nd2",
        'B5': base_nd2_dir + "/WellB5_ChannelKK_SBS_DAPI,KK_SBS_Cy3,KK_SBS_A594,KK_SBS_Cy5,KK_SBS_Cy7,SBS_Cy7_s4,SBS_Cy5_s1,SBS_Cy5_s4_Seq0003.nd2",
        'C2': base_nd2_dir + "/WellC2_ChannelKK_SBS_DAPI,KK_SBS_Cy3,KK_SBS_A594,KK_SBS_Cy5,KK_SBS_Cy7,SBS_Cy7_s4,SBS_Cy5_s1,SBS_Cy5_s4_Seq0007.nd2",
        'C3': base_nd2_dir + "/WellC3_ChannelKK_SBS_DAPI,KK_SBS_Cy3,KK_SBS_A594,KK_SBS_Cy5,KK_SBS_Cy7,SBS_Cy7_s4,SBS_Cy5_s1,SBS_Cy5_s4_Seq0006.nd2",
        'C4': base_nd2_dir + "/WellC4_ChannelKK_SBS_DAPI,KK_SBS_Cy3,KK_SBS_A594,KK_SBS_Cy5,KK_SBS_Cy7,SBS_Cy7_s4,SBS_Cy5_s1,SBS_Cy5_s4_Seq0005.nd2",
        'C5': base_nd2_dir + "/WellC5_ChannelKK_SBS_DAPI,KK_SBS_Cy3,KK_SBS_A594,KK_SBS_Cy5,KK_SBS_Cy7,SBS_Cy7_s4,SBS_Cy5_s1,SBS_Cy5_s4_Seq0004.nd2",
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
    list_of_40x_files_dapi = glob.glob( '{dapi_40x_base_dir}/plate_{plate_num}/well_{well}/{well_phenix}f*-ch1*.tiff'.format(
        dapi_40x_base_dir=dapi_40x_base_dir,
        well=well, 
        well_phenix=well_phenix,
        plate_num=plate_num) )
    index_file_phenix = '{dapi_40x_base_dir}/plate_{plate_num}/well_{well}/Index.idx.xml'.format( dapi_40x_base_dir=dapi_40x_base_dir, 
            well=well, plate_num=plate_num )
    list_of_files_10x = glob.glob( '{sbs_images_base_dir}/plate_{plate_num}_tiffs_rotated/well_{well}/cycle_1/*tif'.format(
        sbs_images_base_dir=sbs_images_base_dir,
        well=well, 
        plate_num=plate_num))
    nd2_file_10x = nd2_files_by_well[ well ]
    pixel_size_10x = 0.841792491782224
    pixel_size_40x = 0.14949402023919043
    output_dir_brute_force_match = 'match_brute_force_plate_{plate_num}_well_{well}'.format( well=well, plate_num=plate_num ) 
    output_dir_match = 'rematch_all_plate_{plate_num}_well_{well}'.format( well=well, plate_num=plate_num )
    plot=True
    skip_brute_force = True
    #skip_brute_force = False
    
    map_40x_to_10x_files_with_10xtile_mapping( list_of_40x_files_dapi, well_phenix, index_file_phenix,
                        list_of_files_10x, nd2_file_10x, pixel_size_40x,
                        pixel_size_10x, output_dir_brute_force_match, output_dir_match,
                        num_proc, overlap_ratio_final_matches = 0.4,
                        num_40x_files_match=num_40x_files_match, plot=plot,
                        skip_brute_force=skip_brute_force)
