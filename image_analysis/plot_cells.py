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
                ax[num][img_num].set_title(barcode,fontsize=4,color='blue')
            ax[num][img_num].text(0.1,0.9,cell['num_condensates_GFP'],fontsize=6,color='blue', transform=ax[num][img_num].transAxes)
            ax[num][img_num].text(0.1,0.5,f"{cell['glcm_dissim']:.2f}",fontsize=4,color='blue' ,transform=ax[num][img_num].transAxes)
            ax[num][img_num].text(0.1,0.3,f"{cell['mean_GFP_intensity_GFP']:.2f}",fontsize=4,color='blue' ,transform=ax[num][img_num].transAxes)
            ax[num][img_num].text(0.1,0.1,f"{cell['frac_gfp_in_cond_GFP']:.2f}",fontsize=4,color='blue' ,transform=ax[num][img_num].transAxes)
            img_num += 1

    # turn all axes off
    for i in range(len(barcodes)):
        for j in range(NUM_IMGS):
            ax[i][j].set_axis_off()

    plt.savefig( f'{plot_save_name}', dpi=300 )
    plt.clf()
    end_time = datetime.datetime.now()
    print("Time plotting:", end_time - begin_time )

def plot_all_cells_barcode( df_data, barcode, plot_save_name, vmax=3000. ):

    begin_time = datetime.datetime.now()
    cells_with_barcode = df_data[ (df_data['cell_barcode_0']==barcode) &
            (df_data['cell_barcode_1'].isnull())]
    num_imgs = len( cells_with_barcode )
    print( "Number of cells:", num_imgs )

    num_cols=20
    num_rows= int( np.ceil(num_imgs/num_cols))
    fig, ax = plt.subplots(num_rows , num_cols, figsize=(num_cols,num_rows))

    # then sort the images by mean_GFP_intensity_GFP
    index = 0
    for df_index, cell in cells_with_barcode.sort_values('mean_GFP_intensity_GFP').iterrows():
        # get the row and column this cell should be plotted in
        row = int(index / num_cols)
        col = index - row*num_cols
        print( row, col, cell['mean_GFP_intensity_GFP'] )

        cell_image_gfp_file = cell['cell_img_gfp_file']
        cell_image_mask_file = cell['cell_img_mask_file']
        cell_image_gfp = read( cell_image_gfp_file )
        cell_image_mask = np.load( cell_image_mask_file )
        masked_cell_image_gfp = cell_image_mask * cell_image_gfp
        #masked_cell_pixels_gfp = cell_image_gfp[ cell_image_mask ]
        #mean_intensity = np.mean( masked_cell_pixels_gfp )

        ax[row][col].set_axis_off()
        pad_x_size = max(0, 185 - np.shape(masked_cell_image_gfp)[0])
        pad_y_size = max(0, 185 - np.shape(masked_cell_image_gfp)[1])
        padded_image = np.pad( masked_cell_image_gfp, [(int(pad_x_size/2),int(pad_x_size/2)), 
                                        (int(pad_y_size/2),int(pad_y_size/2))],
                                        'constant', constant_values=0)

        ax[row][col].imshow( padded_image, cmap='gray', vmin=0, vmax=vmax )

        ax[row][col].text(0.1,0.9,index+1,fontsize=6,color='blue', transform=ax[row][col].transAxes)
        #ax[row][col].text(0.1,0.9,cell['num_condensates_GFP'],fontsize=6,color='blue', transform=ax[row][col].transAxes)
        #ax[row][col].text(0.1,0.5,f"{cell['glcm_dissim']:.2f}",fontsize=4,color='blue' ,transform=ax[row][col].transAxes)
        #ax[row][col].text(0.1,0.3,f"{cell['mean_GFP_intensity_GFP']:.2f}",fontsize=4,color='blue' ,transform=ax[row][col].transAxes)
        #ax[row][col].text(0.1,0.1,f"{cell['frac_gfp_in_cond_GFP']:.2f}",fontsize=4,color='blue' ,transform=ax[row][col].transAxes)
        index += 1

    # turn all axes off
    for i in range(num_rows):
        for j in range(num_cols):
            ax[i][j].set_axis_off()

    plt.savefig( f'{plot_save_name}', dpi=300 )
    plt.clf()
    end_time = datetime.datetime.now()
    
    print("Time plotting:", end_time - begin_time )

def plot_example_cells_sublib_sorted( df_data_all, barcodes, sublibs_for_bcs, plot_save_name, NUM_IMGS=30, 
        mean_intensity_cutoff=150., vmax=3000., descriptions=[] ):
    ################## plot some example cells:
    begin_time = datetime.datetime.now()

    # can I plot a bunch of cells for a given barcode?
    print( "making axes" )
    print( len( barcodes ) )
    #fig, ax = plt.subplots(len(barcodes) , NUM_IMGS)
    fig, ax = plt.subplots(len(barcodes) , NUM_IMGS, figsize=(NUM_IMGS,len(barcodes)))
    print( "done making axes" )
    for num, barcode in enumerate(barcodes):
        print( "getting sublib" )
        df_data = df_data_all[df_data_all['sublibrary']==sublibs_for_bcs[num]]
        print( "getting bc etc" )
        cells_with_barcode = df_data[(df_data['cell_barcode_0']==barcode) &
                                        (df_data['cell_barcode_1'].isnull()) & 
                                        (df_data['mean_GFP_intensity_GFP'] > mean_intensity_cutoff)]
        print( "getting num sample" )
        num_sample = min( NUM_IMGS, len( cells_with_barcode ) )
        print( "sorting" )
        cells_with_barcode_subset_sorted = cells_with_barcode.sample( n=num_sample ).sort_values('mean_GFP_intensity_GFP')
        img_num = 0
        print( "plotting", barcode )
        for index, cell in cells_with_barcode_subset_sorted.iterrows():
            if img_num >(NUM_IMGS-1): break
            # get the gfp image
            # get the image mask
            cell_image_gfp_file = cell['cell_img_gfp_file']
            cell_image_mask_file = cell['cell_img_mask_file']
            cell_image_gfp = read( cell_image_gfp_file )
            cell_image_mask = np.load( cell_image_mask_file )
            masked_cell_image_gfp = cell_image_mask * cell_image_gfp
            masked_cell_pixels_gfp = cell_image_gfp[ cell_image_mask ]

            ax[num][img_num].set_axis_off()
            pad_x_size = max(0, 185 - np.shape(masked_cell_image_gfp)[0])
            pad_y_size = max(0, 185 - np.shape(masked_cell_image_gfp)[1])
            padded_image = np.pad( masked_cell_image_gfp, [(int(pad_x_size/2),int(pad_x_size/2)), 
                                            (int(pad_y_size/2),int(pad_y_size/2))],
                                            'constant', constant_values=0)

            ax[num][img_num].imshow( padded_image, cmap='gray', vmin=0, vmax=vmax )
            if len(descriptions) > 0:
                ax[num][img_num].set_title(descriptions[num],fontsize=2,color='blue')
                if img_num == 0:
                    ax[num][img_num].set_ylabel(descriptions[num],fontsize=4,color='blue', rotation=0)
            else:
                ax[num][img_num].set_title(barcode,fontsize=4,color='blue')
            ax[num][img_num].text(0.1,0.9,cell['num_condensates_GFP'],fontsize=4,color='blue', transform=ax[num][img_num].transAxes)
            ax[num][img_num].text(0.1,0.5,f"{cell['glcm_dissim']:.2f}",fontsize=4,color='blue' ,transform=ax[num][img_num].transAxes)
            ax[num][img_num].text(0.1,0.3,f"{cell['mean_GFP_intensity_GFP']:.2f}",fontsize=4,color='blue' ,transform=ax[num][img_num].transAxes)
            ax[num][img_num].text(0.1,0.1,f"{cell['frac_gfp_in_cond_GFP']:.2f}",fontsize=4,color='blue' ,transform=ax[num][img_num].transAxes)
            img_num += 1

    # turn all axes off
    for i in range(len(barcodes)):
        for j in range(NUM_IMGS):
            ax[i][j].set_axis_off()

    plt.savefig( f'{plot_save_name}', dpi=300 )
    plt.clf()
    end_time = datetime.datetime.now()
    print("Time plotting:", end_time - begin_time )

def plot_labeled_cells_sorted( df_data, barcodes, plot_save_name, NUM_IMGS=30, 
        mean_intensity_cutoff=150., vmax=3000., descriptions=[] ):
    ################## plot some example cells:
    begin_time = datetime.datetime.now()

    # can I plot a bunch of cells for a given barcode?
    print( "making axes" )
    print( len( barcodes ) )
    #fig, ax = plt.subplots(len(barcodes) , NUM_IMGS)
    fig, ax = plt.subplots(len(barcodes) , NUM_IMGS, figsize=(NUM_IMGS,len(barcodes)))
    print( "done making axes" )
    for num, barcode in enumerate(barcodes):
        print( "getting bc etc" )
        cells_with_barcode = df_data[(df_data['cell_barcode_0']==barcode) &
                                        (df_data['cell_barcode_1'].isnull()) & 
                                        (df_data['mean_GFP_intensity_GFP'] > mean_intensity_cutoff)]
        print( "getting num sample" )
        num_sample = min( NUM_IMGS, len( cells_with_barcode ) )
        print( "sorting" )
        cells_with_barcode_subset_sorted = cells_with_barcode.sample( n=num_sample ).sort_values('mean_GFP_intensity_GFP')
        img_num = 0
        print( "plotting", barcode )
        for index, cell in cells_with_barcode_subset_sorted.iterrows():
            if img_num >(NUM_IMGS-1): break
            # get the gfp image
            # get the image mask
            cell_image_gfp_file = cell['cell_img_gfp_file']
            cell_image_mask_file = cell['cell_img_mask_file']
            cell_image_gfp = read( cell_image_gfp_file )
            cell_image_mask = np.load( cell_image_mask_file )
            masked_cell_image_gfp = cell_image_mask * cell_image_gfp
            masked_cell_pixels_gfp = cell_image_gfp[ cell_image_mask ]

            ax[num][img_num].set_axis_off()
            pad_x_size = max(0, 185 - np.shape(masked_cell_image_gfp)[0])
            pad_y_size = max(0, 185 - np.shape(masked_cell_image_gfp)[1])
            padded_image = np.pad( masked_cell_image_gfp, [(int(pad_x_size/2),int(pad_x_size/2)), 
                                            (int(pad_y_size/2),int(pad_y_size/2))],
                                            'constant', constant_values=0)

            ax[num][img_num].imshow( padded_image, cmap='gray', vmin=0, vmax=vmax )
            if len(descriptions) > 0:
                ax[num][img_num].set_title(descriptions[num],fontsize=2,color='blue')
                if img_num == 0:
                    ax[num][img_num].set_ylabel(descriptions[num],fontsize=4,color='blue', rotation=0)
            else:
                ax[num][img_num].set_title(barcode,fontsize=4,color='blue')
            #ax[num][img_num].text(0.1,0.9,cell['num_condensates_GFP'],fontsize=4,color='blue', transform=ax[num][img_num].transAxes)
            #ax[num][img_num].text(0.1,0.5,f"{cell['glcm_dissim']:.2f}",fontsize=4,color='blue' ,transform=ax[num][img_num].transAxes)
            #ax[num][img_num].text(0.1,0.3,f"{cell['mean_GFP_intensity_GFP']:.2f}",fontsize=4,color='blue' ,transform=ax[num][img_num].transAxes)
            #ax[num][img_num].text(0.1,0.1,f"{cell['frac_gfp_in_cond_GFP']:.2f}",fontsize=4,color='blue' ,transform=ax[num][img_num].transAxes)
            img_num += 1

    # turn all axes off
    for i in range(len(barcodes)):
        for j in range(NUM_IMGS):
            ax[i][j].set_axis_off()

    plt.savefig( f'{plot_save_name}', dpi=300 )
    plt.clf()
    end_time = datetime.datetime.now()
    print("Time plotting:", end_time - begin_time )
