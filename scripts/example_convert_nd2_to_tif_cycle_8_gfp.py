import multiprocessing
from image_analysis.preprocess_nd2 import *

NUM_PROC = 62
wells = ['B2','B3','B4','B5','C2','C3','C4','C5']

base_out_dir = "/mnt/disks/data5/20230210_SBS_data/SBS_images/plate_10_tiffs_rotated/"

base_nd2_dir_c8 =  '/mnt/disks/data5/20230210_SBS_data/SBS_images/nd2_files/20230210_SBS_data/Kalli_SBS_24_20230203_multi_settings_plate10_cycle8/Cycle8_20230210_150423_538/'

ch_dapi = 0
ch_gfp = 8


all_nd2_files = []
all_output_dirs = []
all_base_align_c1_dirs = []
for well in wells:
    base_align_c1_dir = f'{base_out_dir}/well_{well}/cycle_1/'
    output_dir = base_out_dir + 'well_' + well + '/' + 'GFP_cycle_8'
    if not os.path.exists( output_dir ):
        os.makedirs( output_dir )
    nd2_file = glob.glob( base_nd2_dir_c8 + 'Well' + well + '*nd2' )[0]
    all_nd2_files.append( nd2_file )
    all_output_dirs.append( output_dir )
    all_base_align_c1_dirs.append( base_align_c1_dir )


with multiprocessing.Pool( processes=NUM_PROC ) as pool:
    results = pool.starmap( GFP_nd2_to_tif_rotate_180_and_align, [(all_nd2_files[i],
        all_output_dirs[i], ch_dapi, ch_gfp, all_base_align_c1_dirs[i]
        ) for i in range(len(all_nd2_files) ) ] )

