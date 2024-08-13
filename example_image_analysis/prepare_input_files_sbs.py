import numpy as np
import glob
import os
import pandas as pd

# process input files into the format required by snakemake
# don't want to do this using the default OPS method because that will literally double the amount of tifs that I have
# (anyway I would need to automate the process of making the input_files spreadsheat)

######################
magnification = '10X'
cycles = [1,2,3,4,5]
wells = ['B5']
base_input_tiff_dir = '/om2/user/kkappel/image_analysis/example_data_analysis/raw_SBS_images/'
tiles = [10,11,12,13,14,15]
######################


well_tile_list_dict = {}
well_tile_list_dict['well'] = []
well_tile_list_dict['tile'] = []

for well in wells:
    for tile in tiles:
        for cycle in cycles:
            input_file_search = base_input_tiff_dir + f'/cycle_{cycle}/Well{well}*tile{tile}.tif'
            input_file_list = glob.glob( input_file_search )
            if len(input_file_list) != 1:
                print( "problem with input file names, not unique" )
                print( input_file_search )
                exit()
            input_file = input_file_list[0]

            output_dir = f'input/{magnification}_c{cycle}-SBS-{cycle}/'
            output_file = f'{magnification}_c{cycle}-SBS-{cycle}_{well}_Tile-{tile}.sbs.tif'

            if not os.path.exists( output_dir ):
                os.makedirs( output_dir ) 

            # link the input file to the output file
            # on mac, I have to use the absolute paths otherwise symbolic link creation does not work
            input_file_abspath = os.path.abspath( input_file )
            output_file_abspath = os.path.abspath( f'{output_dir}/{output_file}' )
            os.system( f'ln -s {input_file_abspath} {output_file_abspath}' )
        well_tile_list_dict['well'].append( well )
        well_tile_list_dict['tile'].append( tile )

df_well_tile_list = pd.DataFrame( well_tile_list_dict )
df_well_tile_list.to_csv( 'input/well_tile_list.csv', index=None )
