import datetime
import matplotlib.pyplot as plt
from ops.io import read_stack as read
import numpy as np
import matplotlib.colors as mcolors


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
    plt.close()
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
                #print( num, img_num, len( descriptions ) )
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

def plot_cell_colocalization(df_data, wells, img_1_name='cell_img_gfp_file', img_2_name='', vmax=2000.,
        vmax_other=2000., NUM_IMGS=3, mean_intensity_cutoff=300., compare_stain='pml', plot_save_name=''):

    fig, ax = plt.subplots(len(wells), NUM_IMGS*3, figsize=(NUM_IMGS*3,len(wells)))
    for num, well in enumerate( wells ):
        print( well )
        well_data = df_data[ #(df_data['plate_tag'] == plate_tag) &
                       (df_data['well']==well) &
                       (df_data['mean_GFP_intensity_GFP']>mean_intensity_cutoff)]

        # get representative cells
        mean_num_cond = np.mean( well_data['num_condensates_GFP'] )
        std_num_cond = np.std( well_data['num_condensates_GFP'] )
        mean_intensity = np.mean( well_data['mean_GFP_intensity_GFP'] )
        std_intensity = np.std( well_data['mean_GFP_intensity_GFP'] )

        mean_num_cond_stain = np.mean( well_data[f'num_condensates_{compare_stain}'] )
        std_num_cond_stain = np.std( well_data[f'num_condensates_{compare_stain}'] )

        well_data_avg = well_data[ (well_data['num_condensates_GFP'].between(mean_num_cond-std_num_cond, mean_num_cond+std_num_cond )) &
            (well_data['mean_GFP_intensity_GFP'].between(mean_intensity-std_intensity, mean_intensity+std_intensity)) &
            (well_data[f'num_condensates_{compare_stain}'].between(mean_num_cond_stain-std_num_cond_stain, mean_num_cond_stain+std_num_cond_stain)) ]

        print( mean_num_cond, mean_intensity, mean_num_cond_stain )
        print( len(well_data), len(well_data_avg) )

        img_num = 0
        for index, cell in well_data_avg.iterrows():
            if img_num > (NUM_IMGS-1): break
            cell_image_gfp_file = cell['cell_img_gfp_file']
            cell_image_mask_file = cell['cell_img_mask_file']
            other_stain_file = cell[img_2_name]

            other_stain_image = read( other_stain_file )
            cell_image_gfp = read( cell_image_gfp_file )
            cell_image_mask = np.load( cell_image_mask_file )

            masked_cell_image_gfp = cell_image_mask * cell_image_gfp
            masked_cell_image_other_stain = cell_image_mask * other_stain_image

            ax[num][img_num].set_axis_off()

            pad_x_size = max(0, 185 - np.shape(masked_cell_image_gfp)[0])
            pad_y_size = max(0, 185 - np.shape(masked_cell_image_gfp)[1])
            padded_image = np.pad( masked_cell_image_gfp, [(int(pad_x_size/2),int(pad_x_size/2)), 
                                            (int(pad_y_size/2),int(pad_y_size/2))],
                                            'constant', constant_values=0)

            padded_image_other = np.pad( masked_cell_image_other_stain, [(int(pad_x_size/2),int(pad_x_size/2)), 
                                            (int(pad_y_size/2),int(pad_y_size/2))],
                                            'constant', constant_values=0)


            colors = [(0, 'black'), (1, 'green')]
            cmap_black_green = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors)
            colorsbr = [(0, 'black'), (1, 'red')]
            cmap_black_red = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colorsbr)

            ax[num][img_num].imshow( padded_image, cmap=cmap_black_green, vmin=0, vmax=vmax )
            #ax[num][img_num].imshow( padded_image, cmap='gray', vmin=0, vmax=vmax )
            max_for_other = np.max([np.max(padded_image_other), vmax_other])
            ax[num][img_num+NUM_IMGS].imshow( padded_image_other, cmap=cmap_black_red, vmin=0, vmax=max_for_other )
            #ax[num][img_num+NUM_IMGS].imshow( padded_image_other, cmap=cmap_black_red, vmin=0, vmax=vmax_other )
            #ax[num][img_num+NUM_IMGS].imshow( padded_image_other, cmap='gray', vmin=0, vmax=vmax_other )
            #ax[num][img_num+2*NUM_IMGS].imshow( padded_image, cmap=cmap_black_green, vmin=0, vmax=vmax, alpha=0.5 )
            #ax[num][img_num+2*NUM_IMGS].imshow( padded_image, cmap='Greens', vmin=0, vmax=vmax, alpha=0.5 )
            #ax[num][img_num+2*NUM_IMGS].imshow( padded_image_other, cmap=cmap_black_red, vmin=0, vmax=vmax_other, alpha=0.5 )
            #ax[num][img_num+2*NUM_IMGS].imshow( padded_image_other, cmap='Purples', vmin=0, vmax=vmax_other, alpha=0.5 )

            green_channel = padded_image / np.max(padded_image)
            #green_channel = padded_image / vmax
            above_one = green_channel > 1.
            green_channel[ above_one ] = 1.

            #red_channel = padded_image_other / np.max(padded_image_other)
            red_channel = padded_image_other / np.max([np.max(padded_image_other), vmax_other])
            #red_channel = padded_image_other / vmax_other
            above_one = red_channel > 1.
            red_channel[ above_one ] = 1.

            rgb_image = np.zeros((padded_image.shape[0], padded_image.shape[1], 3))
            rgb_image[:,:,0] = red_channel
            rgb_image[:,:,1] = green_channel
            ax[num][img_num+2*NUM_IMGS].imshow( rgb_image )

            if img_num==0:
                ax[num][img_num].set_ylabel(well, fontsize=8, color='blue')
                print( compare_stain, "max red:", np.max([np.max(padded_image_other), vmax_other]) )
            ax[num][img_num].set_title(well,fontsize=8,color='blue')
            ax[num][img_num+2*NUM_IMGS].set_title(well,fontsize=8,color='blue')
            ax[num][img_num+NUM_IMGS].set_title(well,fontsize=8,color='blue')
        
            img_num += 1

    for i in range( len(wells)):
        for j in range( NUM_IMGS*3 ):
            ax[i][j].set_axis_off()

    plt.savefig( f'{plot_save_name}', dpi=300 )
    plt.clf()
    plt.close()

def plot_cells_pooled_vs_arrayed( df_data_pool, df_data_array, barcodes, sublibs_for_bcs, wells, plot_save_name,
        NUM_IMGS=5, vmax=3000. ):

    ################## plot some example cells:
    begin_time = datetime.datetime.now()

    fig, ax = plt.subplots(len(barcodes) , NUM_IMGS*2+1, figsize=(NUM_IMGS*2+1,len(barcodes)))
    print( "done making axes" )
    for num, barcode in enumerate(barcodes):
        cells_with_barcode_pool = df_data_pool[(df_data_pool['cell_barcode_0']==barcode) &
                                        (df_data_pool['cell_barcode_1'].isnull()) &
                                        (df_data_pool['sublibrary']==sublibs_for_bcs[num])]
        cells_array = df_data_array[(df_data_array['well']==wells[num])]
        num_sample = min( NUM_IMGS, len( cells_with_barcode_pool  ) )
        cells_with_barcode_pool_subset_sorted = cells_with_barcode_pool.sample( n=num_sample ).sort_values('mean_GFP_intensity_GFP').reset_index()
        cells_array_subset_sorted = cells_array.sample( n=num_sample ).sort_values('mean_GFP_intensity_GFP').reset_index()
        img_num = 0
        print( "plotting", barcode )
        for row_index in range( len(cells_array_subset_sorted) ):
            cell_pool = cells_with_barcode_pool_subset_sorted.iloc[row_index]
            cell_array = cells_array_subset_sorted.iloc[row_index]

            if img_num >(NUM_IMGS-1): break
            # get the gfp image
            # get the image mask
            cell_image_gfp_file_pool = cell_pool['cell_img_gfp_file']
            cell_image_mask_file_pool = cell_pool['cell_img_mask_file']
            cell_image_gfp_pool = read( cell_image_gfp_file_pool )
            cell_image_mask_pool = np.load( cell_image_mask_file_pool )
            masked_cell_image_gfp_pool = cell_image_mask_pool * cell_image_gfp_pool

            ax[num][img_num].set_axis_off()
            pad_x_size = max(0, 185 - np.shape(masked_cell_image_gfp_pool)[0])
            pad_y_size = max(0, 185 - np.shape(masked_cell_image_gfp_pool)[1])
            padded_image_pool = np.pad( masked_cell_image_gfp_pool, [(int(pad_x_size/2),int(pad_x_size/2)), 
                                            (int(pad_y_size/2),int(pad_y_size/2))],
                                            'constant', constant_values=0)

            ax[num][img_num].imshow( padded_image_pool, cmap='gray', vmin=0, vmax=vmax )

            cell_image_gfp_file_array = cell_array['cell_img_gfp_file']
            cell_image_mask_file_array = cell_array['cell_img_mask_file']
            cell_image_gfp_array = read( cell_image_gfp_file_array )
            cell_image_mask_array = np.load( cell_image_mask_file_array )
            masked_cell_image_gfp_array = cell_image_mask_array * cell_image_gfp_array

            ax[num][img_num+NUM_IMGS+1].set_axis_off()
            pad_x_size = max(0, 185 - np.shape(masked_cell_image_gfp_array)[0])
            pad_y_size = max(0, 185 - np.shape(masked_cell_image_gfp_array)[1])
            padded_image_array = np.pad( masked_cell_image_gfp_array, [(int(pad_x_size/2),int(pad_x_size/2)), 
                                            (int(pad_y_size/2),int(pad_y_size/2))],
                                            'constant', constant_values=0)

            ax[num][img_num+NUM_IMGS+1].imshow( padded_image_array, cmap='gray', vmin=0, vmax=vmax )
            ax[num][img_num+NUM_IMGS+1].set_title( f'{wells[num]}', color='blue', fontsize=6 )
            ax[num][img_num].set_title( f'{barcode}', color='blue', fontsize=6 )

            img_num += 1

    # turn all axes off
    for i in range(len(barcodes)):
        for j in range(NUM_IMGS*2+1):
            ax[i][j].set_axis_off()

    plt.savefig( f'{plot_save_name}', dpi=300 )
    plt.clf()
    end_time = datetime.datetime.now()
    print("Time plotting:", end_time - begin_time )

    return

def plot_livecell_barcode( data_allt, barcode, timepoints, num_cells_to_plot, plot_save_name, vmax=3000. ):

    begin_time = datetime.datetime.now()

    # number of columns = number of timepoints
    # number of rows = num_cells_to_plot
    fig, ax = plt.subplots(num_cells_to_plot, len(timepoints), figsize=(len(timepoints),num_cells_to_plot))

    # add unique tracked nucleus num
    data_allt['unique_tracked_nucleus_num'] = data_allt['tile'].astype(str) + '_' +data_allt['tracked_nucleus_num'].astype(str)

    # get all the cells with the specified barcode
    unique_tracked_nucleus_nums = data_allt[ (data_allt['cell_barcode_0']==barcode) &
            (data_allt['cell_barcode_1'].isnull())]['unique_tracked_nucleus_num'].tolist()
    #data_allt_bc = data_allt[data_allt['unique_tracked_nucleus_num'].isin(unique_tracked_nucleus_nums )]

    for cell_num, cell_id in enumerate( unique_tracked_nucleus_nums ):
        print( cell_num )
        if cell_num > (num_cells_to_plot -1): break
        data_allt_cell = data_allt[ data_allt['unique_tracked_nucleus_num'] == cell_id ]
        for img_num, time in enumerate(timepoints):
            data_timep = data_allt_cell[ data_allt_cell['frame']==time ]
            cell_image_gfp_file = data_timep['cell_img_gfp_file'].iloc[0]
            cell_image_mask_file = data_timep['cell_img_mask_file'].iloc[0]

            cell_image_gfp = read( cell_image_gfp_file )
            cell_image_mask = np.load( cell_image_mask_file )

            masked_cell_image_gfp = cell_image_mask * cell_image_gfp

            ax[cell_num][img_num].set_axis_off()
            pad_x_size = max(0, 185 - np.shape(masked_cell_image_gfp)[0])
            pad_y_size = max(0, 185 - np.shape(masked_cell_image_gfp)[1])
            padded_image = np.pad( masked_cell_image_gfp, [(int(pad_x_size/2),int(pad_x_size/2)), 
                                            (int(pad_y_size/2),int(pad_y_size/2))],
                                            'constant', constant_values=0)

            ax[cell_num][img_num].imshow( padded_image, cmap='gray', vmin=0, vmax=vmax )


    # turn all axes off
    for i in range(num_cells_to_plot):
        for j in range(len(timepoints)):
            ax[i][j].set_axis_off()

    plt.savefig( f'{plot_save_name}', dpi=300 )
    plt.clf()
    plt.close()
    end_time = datetime.datetime.now()
    print("Time plotting:", end_time - begin_time )
    
    return
