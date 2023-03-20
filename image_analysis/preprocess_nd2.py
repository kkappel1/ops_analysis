from pims_nd2 import ND2_Reader
import numpy as np
from ops.io import save_stack as save
from ops.io import read_stack as read
import ops.firesnake
from ops.firesnake import Snake
import os
import glob

# need to use pims_nd2 because my image has more than 7 channels (!!)
# for some reason nd2reader only reads in the first 7 channels
def GFP_nd2_to_tif_rotate_180_and_align(input_filename, output_dir, ch_dapi, ch_gfp, base_align_c1_dir ):
    with ND2_Reader(input_filename) as images:
        images.iter_axes = 'm'
        images.bundle_axes = 'cyx'

        # printing the metadata causes an error with pims_nd2
        #print( images.metadata )
        for site, image in enumerate(images):
            #output_filename = f'test_dapi_gfp_tifs/test_dapi_gfp-tile{site}.tif'
            output_filename = output_dir + '/' + input_filename.split('/')[-1].replace('.nd2','-tile' + str(site) + '.tif' )
            print( "Site {} for {} saved in {}.".format(site,input_filename.split('/')[-1],output_filename.split('/')[-1]))
            image = image.astype('uint16')
            image = image[[ch_dapi, ch_gfp],:]
            image = np.rot90( image, k=2, axes=(1,2))

            # align the image to cycle 1 before saving 
            #cycle1_f = f'plate_10_tiffs_rotated/well_B2/cycle_1/WellB2_ChannelKK_SBS_DAPI,KK_SBS_Cy3,KK_SBS_A594,KK_SBS_Cy5,KK_SBS_Cy7,SBS_Cy7_s4,SBS_Cy5_s1,SBS_Cy5_s4_Seq0000-tile{site}.tif'
            align_c1_files = glob.glob( f'{base_align_c1_dir}/*-tile{site}.tif')
            #print( f'{base_align_c1_dir}/*-tile{site}.tif' )
            if len( align_c1_files ) != 1:
                print( "probem with align c1 files", align_c1_files )
                return
            align_c1_file = align_c1_files[0]

            cycle1_data = read( align_c1_file )
            cycle1_2chan_data = cycle1_data[0:2]
            data_1andgfp = np.array( [cycle1_2chan_data, image] )
            aligned_1and_gfp = Snake._align_SBS( data_1andgfp, method='DAPI' )
            aligned_gfp_only = aligned_1and_gfp[1,:]
            save( output_filename, aligned_gfp_only )
