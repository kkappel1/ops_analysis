import datetime
import matplotlib.pyplot as plt
from ops.io import read_stack as read
import numpy as np


def plot_example_cells( df_data, barcodes, plot_save_name, NUM_IMGS=30, 
        mean_intensity_cutoff=150., vmax=3000., descriptions=[] ):
    ################## plot some example cells:
    begin_time = datetime.datetime.now()

    # can I plot a bunch of cells for a given barcode?
    fig, ax = plt.subplots(len(barcodes) , NUM_IMGS, figsize=(NUM_IMGS,len(barcodes)))
    for num, barcode in enumerate(barcodes):
        cells_with_barcode = df_data[(df_data['cell_barcode_0']==barcode) &
                                        (df_data['cell_barcode_1'].isnull())]
        img_num = 0
        print( "plotting", barcode )
        for index, cell in cells_with_barcode.iterrows():
            if img_num >(NUM_IMGS-1): break
            # get the gfp image
            # get the image mask
            cell_image_gfp_file = cell['cell_img_gfp_file']
            cell_image_mask_file = cell['cell_img_mask_file']
            cell_image_gfp = read( cell_image_gfp_file )
            cell_image_mask = np.load( cell_image_mask_file )
            masked_cell_image_gfp = cell_image_mask * cell_image_gfp
            masked_cell_pixels_gfp = cell_image_gfp[ cell_image_mask ]
            mean_intensity = np.mean( masked_cell_pixels_gfp )
            if mean_intensity < mean_intensity_cutoff:
                continue

            ax[num][img_num].set_axis_off()
            pad_x_size = max(0, 185 - np.shape(masked_cell_image_gfp)[0])
            pad_y_size = max(0, 185 - np.shape(masked_cell_image_gfp)[1])
            padded_image = np.pad( masked_cell_image_gfp, [(int(pad_x_size/2),int(pad_x_size/2)), 
                                            (int(pad_y_size/2),int(pad_y_size/2))],
                                            'constant', constant_values=0)

            ax[num][img_num].imshow( padded_image, cmap='gray', vmin=0, vmax=vmax )
            if len(descriptions) > 0:
                ax[num][img_num].set_title(descriptions[num],fontsize=2,color='blue')
            else:
                ax[num][img_num].set_title(barcode,fontsize=8,color='blue')
            ax[num][img_num].text(0.1,0.9,cell['num_condensates_GFP'],fontsize=6,color='blue', transform=ax[num][img_num].transAxes)
            ax[num][img_num].text(0.1,0.1,cell['frac_gfp_in_cond_GFP'],fontsize=6,color='blue' ,transform=ax[num][img_num].transAxes)
            img_num += 1

    # turn all axes off
    for i in range(len(barcodes)):
        for j in range(NUM_IMGS):
            ax[i][j].set_axis_off()

    plt.savefig( f'{plot_save_name}', dpi=300 )
    plt.clf()
    end_time = datetime.datetime.now()
    print("Time plotting:", end_time - begin_time )
