import sys
sys.path.append( "/mnt/disks/data3/20221026_SBS_analysis" )
from matching_updated_centroids import * 

NUM_PROC = 16
wells = ['B2','B3','B4','B5','C2','C3','C4','C5']

base_out_dir = "/mnt/disks/data5/20230210_SBS_data/SBS_images/plate_10_tiffs_rotated/"

base_nd2_dir = "/mnt/disks/data5/20230210_SBS_data/SBS_images/nd2_files/20230210_SBS_data/Kalli_SBS_24_20230203_multi_settings_plate10/"
cycle_dirs = {
        "cycle_1": base_nd2_dir + 'Cycle1_20230203_141620_649/',
        "cycle_2": base_nd2_dir + 'Cycle2_20230204_165025_975/',
        "cycle_3": base_nd2_dir + 'Cycle3_20230206_104553_746/',
        "cycle_4": base_nd2_dir + 'Cycle4_20230207_095801_420/',
        "cycle_5": base_nd2_dir + 'Cycle5_20230207_191838_370/',
        "cycle_6": base_nd2_dir + 'Cycle6_20230208_164502_516/',
        "cycle_7": base_nd2_dir + 'Cycle7_20230209_152032_829/',
        "cycle_8": '/mnt/disks/data5/20230210_SBS_data/SBS_images/nd2_files/20230210_SBS_data/Kalli_SBS_24_20230203_multi_settings_plate10_cycle8/Cycle8_20230210_150423_538/',
}


if not os.path.exists( base_out_dir ):
    os.mkdir( base_out_dir )

def nd2_to_tif_rotate_180_parallel( args ):
    return nd2_to_tif_rotate_180( *args )

all_nd2_files = []
all_output_well_cycle_dirs = []
for well in wells:
    for cycle in cycle_dirs.keys():
        output_well_cycle_dir = base_out_dir +  'well_' + well + '/' + cycle 
        if not os.path.exists( output_well_cycle_dir ):
            os.makedirs( output_well_cycle_dir )
        nd2_file = glob.glob( cycle_dirs[cycle] + 'Well' + well + '*nd2' )[0]
        all_nd2_files.append( nd2_file )
        all_output_well_cycle_dirs.append( output_well_cycle_dir ) 

with multiprocessing.Pool( processes=NUM_PROC ) as pool:
    results = pool.map( nd2_to_tif_rotate_180_parallel, [(all_nd2_files[i],
        all_output_well_cycle_dirs[i], 
        0, 1, 2, 3, 4) for i in range(len(all_nd2_files) ) ] )
