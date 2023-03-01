from nd2reader import ND2Reader
from ops.io import save_stack as save
from scipy.spatial.distance import cdist
import random
import multiprocessing
from SBS_analysis_functions_v221121 import *
from scipy.stats import linregress
matplotlib.use('Agg')

def nd2_to_tif_rotate_180( input_filename, output_dir, ch_dapi, ch_G, ch_T, ch_A, ch_C ):
    with ND2Reader(input_filename) as images:
        images.iter_axes = 'v'
        #images.bundle_axes = 'cxy' # this flips the image, below fixes it.
        images.bundle_axes = 'cyx'

        for site, image in zip(images.metadata['fields_of_view'],images):
            output_filename = output_dir + '/' + input_filename.split('/')[-1].replace('.nd2','-tile' + str(site) + '.tif' )
            print( "Site {} for {} saved in {}.".format(site,input_filename.split('/')[-1],output_filename.split('/')[-1]))
            image = image.astype('uint16')
            image = image[[ch_dapi, ch_G, ch_T, ch_A, ch_C],:]
            image = np.rot90( image, k=2, axes=(1,2))
            save( output_filename, image )

def map_10x_fields_to_xy_coords( nd2_file ):
    sites_to_xy_10x = {}
    with ND2Reader(nd2_file) as img:
        xs = img.parser._raw_metadata.x_data
        ys = img.parser._raw_metadata.y_data
        sites = []
        for i, site in enumerate(img.metadata['fields_of_view']):
            sites.append( site )
            sites_to_xy_10x[site] = [-1*xs[i], ys[i]]
    return sites_to_xy_10x

def map_40x_fields_to_xy_coords_phenix( well_phenix, index_file_phenix):
    # well_phenix should be something like 'r02c03'
    # index_file_phenix should be something like 'Index.idx.xml'
    image_name = ''
    images_40x_to_xy = {}
    x = []
    y = []
    for line in open( index_file_phenix ):
        if "<URL>" in line:
            image_name = line.split('>')[1].split('<')[0]
            if ('-ch1' not in image_name) or not image_name.startswith( well_phenix):
                image_name = ''
                continue
        if image_name != '':
            if "PositionX" in line:
                posx = float(line.split('>')[1].split('<')[0])
                x.append( posx )
            elif "PositionY" in line:
                posy = float(line.split('>')[1].split('<')[0])
                y.append( posy )
        if len(x) == 1 and len(y) == 1:
            images_40x_to_xy[ image_name ] = [x[0], y[0]]
            x = []
            y = []

    return images_40x_to_xy

def fit_linear_40x_to_10x_from_brute_force_matches( brute_force_match_dir,
                                                    sites_to_xy_10x,
                                                    images_40x_to_xy,
                                                    pixel_size_10x,
                                                    image_size_10x,
                                                    CORR_THRESH=0.35,
                                                    plot=False ):

    x10x_for_fitting = []
    x40x_for_fitting = []
    y10x_for_fitting = []
    y40x_for_fitting = []
    
    # 10x image width = pixel size * number of pixels

    for match_file_base in os.listdir( brute_force_match_dir ):
        #print( match_file_base )
        if os.path.isdir( brute_force_match_dir + '/' + match_file_base ): continue
        fbase = match_file_base.split('/')[-1].replace('.txt', '.tiff').replace('match_','')
        match_file = brute_force_match_dir + '/' + match_file_base
    
        for line in open( match_file):
            file_10x_match = line.split()[-4]
            corr = float(line.split()[-3])
            shift_x = float(line.split()[-2])
            shift_y = float( line.split()[-1])
        # skip if this file was not actually matched properly
        if corr < CORR_THRESH: continue
        #print( '\n', fbase, file_10x_match, corr, shift_x, shift_y)  
        # get the corresponding 10x and 40x coordinates
        site_10x = int(file_10x_match.split('/')[-1].split('tile')[-1].split('.tif')[0])
        site_10x_x = sites_to_xy_10x[site_10x][0]
        site_10x_y = sites_to_xy_10x[site_10x][1]
        
        # this is the transformation for the right camera
        # and should also be the transformation for left camera if I did fitting with properly rotated images
        x_10x = site_10x_x + shift_y * pixel_size_10x
        y_10x = site_10x_y + (image_size_10x-shift_x)* pixel_size_10x
        #y_10x = site_10x_y + (1480-shift_x)* pixel_size_10x
        

        x_40x = images_40x_to_xy[fbase][0]
        y_40x = images_40x_to_xy[fbase][1]
        
        x10x_for_fitting.append( x_10x )
        x40x_for_fitting.append( x_40x )
        y10x_for_fitting.append( y_10x )
        y40x_for_fitting.append( y_40x )
        #print( site_10x, y_10x, y_40x)
        #print( site_10x, x_10x, x_40x)
    # make a unique file name
    unique_file_name = 'fitting_'
    for match_file_base in os.listdir( brute_force_match_dir ):
        if os.path.isdir( brute_force_match_dir + '/' + match_file_base ): continue
        # match_r02c02f1026p01-ch1sk1fk1fl1.txt
        fbase = match_file_base.split('/')[-1].replace('.txt', '.tiff').replace('match_','')
        unique_file_name += fbase[0:6] + '_'
        break
    if plot:
        fig, ax = plt.subplots( 1,1,figsize=(5,5))
        ax.scatter( y10x_for_fitting, y40x_for_fitting, alpha=0.3, c=np.arange(len(y10x_for_fitting)), cmap='plasma')
        #plt.savefig( 'fitting_y_10x_vs_40x.png' )
        # save with unique file name 
        plt.savefig( brute_force_match_dir + '/images/' + unique_file_name + 'y_10x_vs_40x.png' )
        plt.clf()
        
        fig, ax = plt.subplots( 1,1,figsize=(5,5))
        ax.scatter( x10x_for_fitting, x40x_for_fitting, alpha=0.3, c=np.arange(len(y10x_for_fitting)), cmap='plasma')
        #plt.savefig( 'fitting_x_10x_vs_40x.png' )
        plt.savefig( brute_force_match_dir + '/images/' + unique_file_name + 'x_10x_vs_40x.png' )
        plt.clf()
    
    # fit a model
    linregress_x = linregress( x40x_for_fitting, x10x_for_fitting )
    linregress_y = linregress( y40x_for_fitting, y10x_for_fitting )

    return linregress_x, linregress_y

def get_10x_coords_from_40x_coords( x40x, y40x, linregress_x, linregress_y ):
    x10x = linregress_x.slope*x40x + linregress_x.intercept
    y10x = linregress_y.slope*y40x + linregress_y.intercept
    return x10x, y10x

def get_field_num_from_phenix_name( name ):
    field_num = name.split('/')[-1].split('-')[0].split('f')[1].split('p')[0]
    return int(field_num)

def get_10x_tiles_from_coords(x10x, y10x, sites_to_xy_10x, tile_size_10x_x, tile_size_10x_y ):
    likely_tiles = []
    ########## debug
    #all_xs = []
    #all_ys = []
    #for site in sites_to_xy_10x.keys():
    #    all_xs.append( sites_to_xy_10x[site][0] )
    #    all_ys.append( sites_to_xy_10x[site][1] )
    #plt.scatter( all_xs, all_ys )
    #plt.scatter( x10x, y10x )
    #plt.savefig( 'debug_coords.png' )
    #plt.clf()
    ###############
    for tile in sites_to_xy_10x.keys():
        tile_x = sites_to_xy_10x[tile][0]
        tile_y = sites_to_xy_10x[tile][1]
        if ((x10x > (tile_x - tile_size_10x_x*0.15)) and (x10x < (tile_x + 1.15*tile_size_10x_x))
            and ((y10x > (tile_y - tile_size_10x_y*0.15)) and (y10x < (tile_y + 1.15*tile_size_10x_y)))):
            likely_tiles.append( tile )
            
    return likely_tiles

# files_10x_base_name should be something like "/Volumes/Extreme SSD/20221026_SBS_plate_1_tiffs/well_B3/cycle_1/WellB3_ChannelSBS_DAPI,SBS_Cy3,SBS_A594,SBS_Cy5,SBS_Cy7_Seq0000-"
def get_best_10x_image_match_from_model( file_40x, files_10x_base_name, images_40x_to_xy, sites_to_xy_10x, 
                                            linregress_x, linregress_y, tile_size_10x_x, tile_size_10x_y,
                                            output_match_dir, scale_factor, overlap_ratio=0.5, plot=False ):
    # get the likley 10x tiles that this matches to from the xy coords
    base_file_40x = file_40x.split('/')[-1]
    if base_file_40x not in images_40x_to_xy.keys():
        return "", 0.0, [-1,-1], 0
    x40x, y40x = images_40x_to_xy[ base_file_40x ]
    x10x, y10x = get_10x_coords_from_40x_coords( x40x, y40x, linregress_x, linregress_y )
    likely_tiles_10x = get_10x_tiles_from_coords( x10x, y10x, sites_to_xy_10x, tile_size_10x_x, tile_size_10x_y )
    files_likely_tiles_10x = []
    for likely_tile in likely_tiles_10x:
        full_file_name = files_10x_base_name + 'tile' + str(likely_tile) + '.tif'
        files_likely_tiles_10x.append( full_file_name ) 
    #print( "likely_tiles_10x", likely_tiles_10x )
    #print( "files_likely_tiles_10x", files_likely_tiles_10x )
    if len( files_likely_tiles_10x ) < 1: 
        return "", 0.0, [-1,-1], 0
    # find the best tile match from these likely tiles 
    (best_image_match, best_max_cc_mag, best_shift, final_shifted_40x_image) = get_best_10x_image_match_phenix_left_nomove( 
                file_40x, files_likely_tiles_10x, scale_factor, output_dir=output_match_dir, overlap_ratio=overlap_ratio, plot=plot)

    return best_image_match, best_max_cc_mag, best_shift, final_shifted_40x_image


def map_40x_nuclei_to_10x_nuclei_centroid_dist( nuclei_mask_40x, nuclei_mask_10x,
                                              tile_origin_10x, best_shift, pixel_size_10x,
                                               DIST_THRESH=15.0,
                                               RATIO_MIN_TO_NEXT_THRESH=0.9,
                                               plot_tag='test_2',
                                              plot=False):
    
    # get the 10x centroids
    image_size_10x = np.shape( nuclei_mask_10x )[0]
    image_size_40x = np.shape( nuclei_mask_40x )[0]
    
    cells_to_centroids_10x = {}
    for cell in skimage.measure.regionprops(nuclei_mask_10x):
        centroid = cell.centroid
        # get the centroid in actual x y positions
        x = tile_origin_10x[0] + centroid[1] * pixel_size_10x
        y = tile_origin_10x[1] + (image_size_10x-centroid[0])* pixel_size_10x
        cells_to_centroids_10x[ cell.label ] = [x, y]
        
        
    # get the 40x centroids
    shift_x = best_shift[0]
    shift_y = best_shift[1]
    
    nucleii_to_centroids_40x = {}
    for nucleus in skimage.measure.regionprops(nuclei_mask_40x):
        centroid = nucleus.centroid
        tile_origin_40x_x = tile_origin_10x[0] + shift_y*pixel_size_10x
        tile_origin_40x_y = tile_origin_10x[1] + (image_size_10x-shift_x)*pixel_size_10x - (image_size_40x*pixel_size_40x)
        
        centroid_x = tile_origin_40x_x + centroid[1]*pixel_size_40x
        centroid_y = tile_origin_40x_y + (image_size_40x - centroid[0])*pixel_size_40x

        nucleii_to_centroids_40x[nucleus.label] = [centroid_x, centroid_y]
        
        
    dict_40x_nuclei_to_10x_nuclei = {}
    if (len(nucleii_to_centroids_40x) < 1) or (len(cells_to_centroids_10x) < 1):
        return dict_40x_nuclei_to_10x_nuclei, 0, 0.0, 0
        
    df_10x_centroids = pd.DataFrame.from_dict( cells_to_centroids_10x, orient='index')
    df_40x_centroids = pd.DataFrame.from_dict( nucleii_to_centroids_40x, orient='index')

    distances = cdist( df_40x_centroids[[0,1]].values, 
                     df_10x_centroids[[0,1]].values)
    index_min = distances.argmin( axis=1 )

    #DIST_THRESH = 15.0
    # may want to make this ratio distance dependent?
    #RATIO_MIN_TO_NEXT_THRESH = 0.9


    # loop through the nuclei and see whether they can be matched
    for nuclei_index_40x in range( len( df_40x_centroids)):
        nucleus_label_40x = df_40x_centroids.iloc[ [nuclei_index_40x] ].index[0]
        # distances from this 40x nucleus to all 10x nuclei
        distances_nucleus = distances[nuclei_index_40x]
        # the minimum distance to a 10x nucleus and its index
        index_min = distances_nucleus.argmin()
        # which 40x nucleus is min dist from this 10x nucleus?
        index_min_10x_to_40x = distances.T[index_min].argmin()
        if index_min_10x_to_40x != nuclei_index_40x:
            # then this nucleus cannot be matched
            dict_40x_nuclei_to_10x_nuclei[ nucleus_label_40x ] = 0
            continue
        match_nucleus_label_10x = df_10x_centroids.iloc[[index_min]].index[0]    
        min_dist = distances_nucleus[ index_min ]
        # then find the next smallest distance to a 10x nucleus
        sorted_distances = sorted( distances_nucleus )
        second_min_dist = sorted_distances[1]
        ratio_min_to_next = min_dist / second_min_dist
        if (min_dist < DIST_THRESH) and (ratio_min_to_next < RATIO_MIN_TO_NEXT_THRESH):
            # then this is a match!
            # not checking here if this 10x nucleus had already been matched
            # will need to resolve any double matches after
            dict_40x_nuclei_to_10x_nuclei[ nucleus_label_40x ] = match_nucleus_label_10x

            # check if this 10x nucleus had already been matched
            # if so, check whether the distance for that match is less than the distance for this match
        else:
            dict_40x_nuclei_to_10x_nuclei[ nucleus_label_40x ] = 0
        # resolve any 40x nuclei that have been matched to the same 10x nuclei
        # (shouldn't actually need to do that if I'm checking that it's the min dist from 10x to 40x and 40x to 10x)
    num_matched_nuclei = sum([x>0 for x in dict_40x_nuclei_to_10x_nuclei.values()])
    matched_nuclei = [x for x in dict_40x_nuclei_to_10x_nuclei.values() if x>0]
    frac_matched_nuclei = num_matched_nuclei/len(df_40x_centroids)
    duplicated_10x_nuclei = len(matched_nuclei) - len(set(matched_nuclei))
    #print( "Num matched nuclei:", num_matched_nuclei)
    #print( "Fraction matched nuclei:", num_matched_nuclei/len(df_40x_centroids))
    #print( "duplicated 10x nuclei:", len(matched_nuclei) - len(set(matched_nuclei)))
    if plot:
        # plot the matched nuclei
        plt.subplots(1,1,figsize=(20, 20))
        plt.scatter( df_10x_centroids[[0]].values, df_10x_centroids[[1]].values)
        plt.scatter( df_40x_centroids[[0]].values, df_40x_centroids[[1]].values)
        for mindex, matched_nuc_40x in enumerate(dict_40x_nuclei_to_10x_nuclei.keys()):
            if dict_40x_nuclei_to_10x_nuclei[matched_nuc_40x] == 0: 
                # this cell was not matched
                continue
            centroid_40x = nucleii_to_centroids_40x[matched_nuc_40x]
            centroid_10x = cells_to_centroids_10x[ dict_40x_nuclei_to_10x_nuclei[matched_nuc_40x]]

            plt.scatter( [centroid_10x[0], centroid_40x[0]], [centroid_10x[1],centroid_40x[1]] )
            plt.text( centroid_10x[0], centroid_10x[1], mindex )
            plt.text( centroid_40x[0], centroid_40x[1], mindex )
        plt.savefig( "overlay_matches_centroid_{plot_tag}.pdf".format(plot_tag=plot_tag) )
        plt.clf()
        #plt.show()
        
    return dict_40x_nuclei_to_10x_nuclei, num_matched_nuclei, frac_matched_nuclei, duplicated_10x_nuclei


def get_best_10x_image_match_mask_phenix_left_nomove( dapi_file_40x, list_of_files_10x, scale_factor,
                                         output_dir, overlap_ratio=1., plot=False ):

    if len( list_of_files_10x ) < 1: 
        return

    # load the image
    image_40x = read( dapi_file_40x )

    # mask the image
    threshold_initial_guess = 150
    smooth_size = 5
    nuclei_smooth_value = 20
    min_size = 1000
    area_min = 2500
    area_max = 50000
#    image_40x = segment_nuclei_phenotype( image_40x, threshold_initial_guess, smooth_size, nuclei_smooth_value, min_size, area_min, area_max )



    # these are the 2 lines that always give a warning "bad rank filter performance"
    smoothed = rank.median( image_40x, disk(smooth_size))
    smoothed = rank.enhance_contrast( smoothed, disk(smooth_size))
    init_guess = min( np.max( smoothed)-1, threshold_initial_guess)
    smoothed_thresh_global = filters.threshold_li( smoothed, initial_guess=init_guess )
    mask = smoothed > smoothed_thresh_global
    mask = skimage.morphology.remove_small_objects(mask, min_size=min_size)

    labeled = skimage.measure.label(mask)
    labeled = ops.process.filter_by_region(labeled,
                                           score = lambda r: r.mean_intensity,
                                           threshold = lambda x: 100, intensity_image=image_40x) > 0

    # fill holes below minimum area
    filled = ndi.binary_fill_holes(labeled)
    difference = skimage.measure.label(filled!=labeled)
    change = ops.process.filter_by_region(difference, lambda r: r.area < area_min, 0) > 0
    labeled[change] = filled[change]

    ### the smooth value here is really important!!
    ### using a value that is too small results in a bunch of nuclei get segmented into pieces
    nuclei = ops.process.apply_watershed(labeled, smooth=nuclei_smooth_value)

    #cleared = skimage.segmentation.clear_border(nuclei)
    final_nuclei = ops.process.filter_by_region(nuclei,
                                                score=lambda r: area_min < r.area < area_max,
                                                threshold = lambda x: 100)
    mean_intensity_nuclei = np.mean(image_40x[final_nuclei > 0])
    print( "mean_intensity_nuclei" , mean_intensity_nuclei )
    image_40x = final_nuclei > 0
    #image_40x = mean_intensity_nuclei * image_40x










    image_10x_0 = read( list_of_files_10x[0] )
    
    image_size_10x = np.shape( image_10x_0[0] )[0]
    padded_image_40x_rescaled, _ = rescale_40x_to_10x_and_pad_phenix_left_nomove( image_40x,
                                                                     scale_factor, image_size_10x )
    
    # create 40x image mask (only want to align the actual image -- not the pad)
    target_mask = padded_image_40x_rescaled > 0
    
    #fig, ax = plt.subplots( 1,1,figsize=(5,5))
    #ax.imshow(padded_image_40x_rescaled, interpolation='none')
    #plt.savefig( "test_padded_rescaled_image.pdf" )
    #plt.show()
    #fig, ax = plt.subplots( 1,1,figsize=(5,5))
    #ax.imshow(image_10x_0[0], interpolation='none')
    #plt.savefig( "test_10x_image.pdf")
    #plt.show()
    
    # loop through the 10x images to find the correct alignment
    best_shift = []
    best_max_cc_mag = -1.
    best_image_match = ''
    for file_10x in list_of_files_10x:
        print( "testing", file_10x)
        image_10x = read( file_10x )
        dapi_image_10x = image_10x[0]

        # mask the 10x dapi image by nuclei 

        THRESHOLD_DAPI = 2000
        NUCLEUS_AREA_MIN = 150
        NUCLEUS_AREA_MAX = 5000
        nuclei = Snake._segment_nuclei(dapi_image_10x, THRESHOLD_DAPI,
                                   area_min=NUCLEUS_AREA_MIN, area_max=NUCLEUS_AREA_MAX)
        mean_intensity_10x_nuclei = np.mean( dapi_image_10x[nuclei > 0] )
        dapi_image_10x = nuclei > 0
        #dapi_image_10x = mean_intensity_10x_nuclei * dapi_image_10x
        fig, ax = plt.subplots(1,1,figsize=(5,5))
        ax.imshow( dapi_image_10x, interpolation='none', cmap='Reds' )
        plt.savefig( 'test_10x_segment.pdf' )
        plt.clf()








        
        # create 10x image mask (does not mask anything out, full image is valid)
        src_mask = np.ones( np.shape(dapi_image_10x), dtype=bool)
        
        # get the masked correlation - use DAPI only
        # this code is essentially equivalent to masked_register_translation
        # but I want it to return the cross correlation and don't want to compute twice
        # so need to write out here        
        #image_10x_rescaled_0 = skimage.transform.rescale( image_10x[0], 
        #                                     0.5, anti_aliasing=False, 
        #                                     multichannel=False, preserve_range=True)
        shift, cc = masked_register_translation_return_CC( dapi_image_10x,
        #shift, cc = masked_register_translation_return_CC( image_10x_rescaled_0, 
                                                          padded_image_40x_rescaled, src_mask, target_mask,
                                                         overlap_ratio=overlap_ratio)
        max_cc_mag = np.abs( cc.max() )
        if max_cc_mag > best_max_cc_mag:
            best_shift = shift
            best_max_cc_mag = max_cc_mag
            best_image_match = file_10x
    
    st = skimage.transform.SimilarityTransform( translation=-1*best_shift[::-1] )
    shifted_40x_image = skimage.transform.warp( padded_image_40x_rescaled, st, preserve_range=True )
    final_shifted_40x_image = shifted_40x_image.astype(image_40x.dtype)
    
    base_fname = dapi_file_40x.split('/')[-1].split('.tif')[0]
    
    # plot the best image
    if plot:
        fig, ax = plt.subplots( 1,1,figsize=(5,5))
        best_image_match_image = read(best_image_match)
        best_image_match_image_dapi = best_image_match_image[0]
        #best_image_match_image_dapi = skimage.transform.rotate( best_image_match_image_dapi, 180, preserve_range=True )
        ax.imshow( best_image_match_image_dapi, interpolation='none',alpha=0.5, cmap='Reds')
        ax.imshow(final_shifted_40x_image, interpolation='none',alpha=0.5, cmap='Blues')
        if not os.path.exists( "{output_dir}".format(output_dir=output_dir) ):
            os.makedirs(  "{output_dir}".format(output_dir=output_dir), exist_ok=True )
        if not os.path.exists( "{output_dir}/images/".format(output_dir=output_dir) ):
            os.makedirs( "{output_dir}/images/".format(output_dir=output_dir), exist_ok=True )
        plt.savefig( "{output_dir}/images/test_overlay_best_match_{name}.pdf".format(output_dir=output_dir,name=base_fname) )
        #plt.savefig( "test_overlay_best_match_{name}.pdf".format(name=base_fname) )
        plt.clf()
        #plt.show()

    # write to a file
    if not os.path.exists( output_dir ):
        os.makedirs( output_dir, exist_ok=True )
    with open( '{output_dir}/match_{name}.txt'.format(name=base_fname, output_dir=output_dir), 'w') as f:
        f.write( '{image} {cc_mag} {shift_x} {shift_y}\n'.format(
            image=best_image_match, cc_mag=best_max_cc_mag, shift_x=best_shift[0], shift_y=best_shift[1] ))
    
    return best_image_match, best_max_cc_mag, best_shift, final_shifted_40x_image



def get_best_10x_image_match_phenix_left_nomove( dapi_file_40x, list_of_files_10x, scale_factor, 
                                         output_dir, overlap_ratio=1., plot=False ):

    if len( list_of_files_10x ) < 1: 
        return

    # load the image
    image_40x = read( dapi_file_40x )

    image_10x_0 = read( list_of_files_10x[0] )
    
    image_size_10x = np.shape( image_10x_0[0] )[0]
    padded_image_40x_rescaled, _ = rescale_40x_to_10x_and_pad_phenix_left_nomove( image_40x,
                                                                     scale_factor, image_size_10x )
    
    # create 40x image mask (only want to align the actual image -- not the pad)
    target_mask = padded_image_40x_rescaled > 0
    
    #fig, ax = plt.subplots( 1,1,figsize=(5,5))
    #ax.imshow(padded_image_40x_rescaled, interpolation='none')
    #plt.savefig( "test_padded_rescaled_image.pdf" )
    #plt.show()
    #fig, ax = plt.subplots( 1,1,figsize=(5,5))
    #ax.imshow(image_10x_0[0], interpolation='none')
    #plt.savefig( "test_10x_image.pdf")
    #plt.show()
    
    # loop through the 10x images to find the correct alignment
    best_shift = []
    best_max_cc_mag = -1.
    best_image_match = ''
    for file_10x in list_of_files_10x:
        print( "testing", file_10x)
        image_10x = read( file_10x )
        dapi_image_10x = image_10x[0]
        
        # create 10x image mask (does not mask anything out, full image is valid)
        src_mask = np.ones( np.shape(dapi_image_10x), dtype=bool)
        
        # get the masked correlation - use DAPI only
        # this code is essentially equivalent to masked_register_translation
        # but I want it to return the cross correlation and don't want to compute twice
        # so need to write out here        
        #image_10x_rescaled_0 = skimage.transform.rescale( image_10x[0], 
        #                                     0.5, anti_aliasing=False, 
        #                                     multichannel=False, preserve_range=True)
        shift, cc = masked_register_translation_return_CC( dapi_image_10x,
        #shift, cc = masked_register_translation_return_CC( image_10x_rescaled_0, 
                                                          padded_image_40x_rescaled, src_mask, target_mask,
                                                         overlap_ratio=overlap_ratio)
        max_cc_mag = np.abs( cc.max() )
        if max_cc_mag > best_max_cc_mag:
            best_shift = shift
            best_max_cc_mag = max_cc_mag
            best_image_match = file_10x
    
    st = skimage.transform.SimilarityTransform( translation=-1*best_shift[::-1] )
    shifted_40x_image = skimage.transform.warp( padded_image_40x_rescaled, st, preserve_range=True )
    final_shifted_40x_image = shifted_40x_image.astype(image_40x.dtype)
    
    base_fname = dapi_file_40x.split('/')[-1].split('.tif')[0]
    
    # plot the best image
    if plot:
        fig, ax = plt.subplots( 1,1,figsize=(5,5))
        best_image_match_image = read(best_image_match)
        best_image_match_image_dapi = best_image_match_image[0]
        #best_image_match_image_dapi = skimage.transform.rotate( best_image_match_image_dapi, 180, preserve_range=True )
        ax.imshow( best_image_match_image_dapi, interpolation='none',alpha=0.5, cmap='Reds')
        ax.imshow(final_shifted_40x_image, interpolation='none',alpha=0.5, cmap='Blues')
        if not os.path.exists( "{output_dir}".format(output_dir=output_dir) ):
            os.makedirs(  "{output_dir}".format(output_dir=output_dir), exist_ok=True )
        if not os.path.exists( "{output_dir}/images/".format(output_dir=output_dir) ):
            os.makedirs( "{output_dir}/images/".format(output_dir=output_dir), exist_ok=True )
        plt.savefig( "{output_dir}/images/test_overlay_best_match_{name}.pdf".format(output_dir=output_dir,name=base_fname) )
        #plt.savefig( "test_overlay_best_match_{name}.pdf".format(name=base_fname) )
        plt.clf()
        #plt.show()

    # write to a file
    if not os.path.exists( output_dir ):
        os.makedirs( output_dir, exist_ok=True )
    with open( '{output_dir}/match_{name}.txt'.format(name=base_fname, output_dir=output_dir), 'w') as f:
        f.write( '{image} {cc_mag} {shift_x} {shift_y}\n'.format(
            image=best_image_match, cc_mag=best_max_cc_mag, shift_x=best_shift[0], shift_y=best_shift[1] ))
    
    return best_image_match, best_max_cc_mag, best_shift, final_shifted_40x_image


# this should be used for 10x images taken from the left RNAscope camera which have already been rotated 180 degrees
def rescale_40x_to_10x_and_pad_phenix_left_nomove( image_40x_to_rescale, scale_factor, image_size_10x ):
    image_40x_rescaled = skimage.transform.rescale( image_40x_to_rescale,
                                                         scale_factor, anti_aliasing=False, 
                                                         multichannel=False, preserve_range=True)
    rescaled_size = np.shape( image_40x_rescaled )[0]
    pad_size = image_size_10x - rescaled_size
    padded_image_40x_rescaled = skimage.util.pad( image_40x_rescaled, 
                                                       [(0,pad_size), (0,pad_size)], 
                                                       'constant', constant_values=0 )
    
    return padded_image_40x_rescaled, rescaled_size

def get_best_10x_image_match_phenix_left_nomove_parallel( args ):
    #return get_best_10x_image_match_mask_phenix_left_nomove( *args )
    return get_best_10x_image_match_phenix_left_nomove( *args )

def get_best_10x_image_match_from_model_parallel( args ):
    return get_best_10x_image_match_from_model( *args )

def fit_linear_40x_to_10x_from_brute_force_matches_parallel( args ):
    return fit_linear_40x_to_10x_from_brute_force_matches( *args )

# do all matching:
def map_40x_to_10x_files_with_10xtile_mapping( list_of_files_40x, well_phenix, index_file_phenix, 
                                                list_of_files_10x, nd2_file_10x, pixel_size_40x,
                                                pixel_size_10x, output_dir_brute_force_match, output_dir_match,
                                                num_proc, overlap_ratio_final_matches = 0.5,
                                                num_40x_files_match=10, plot=False, skip_brute_force=False ):
    overlap_ratio_brute_force_initial = 1.0
    scale_factor = pixel_size_40x / pixel_size_10x

    # randomly select several 40x files --> brute force match them to all 10x tiles
    # parallelize this process
    print( "Doing initial brute force matching" )
    if not skip_brute_force:
        random_files_40x = random.sample( list_of_files_40x, k=num_40x_files_match)
        with multiprocessing.Pool( processes=num_proc) as pool:
            results = pool.map( get_best_10x_image_match_phenix_left_nomove_parallel, 
                        [(file_40x, list_of_files_10x, scale_factor, output_dir_brute_force_match,
                          overlap_ratio_brute_force_initial, plot) for file_40x in random_files_40x ])





    # based on the brute force matches fit a linear model for 40x to 10x coords
    print( "Fitting model based on brute force matches" )
    image_0_10x = read( list_of_files_10x[0] )[0]
    image_size_10x_x = np.shape( image_0_10x )[0]
    image_size_10x_y = np.shape( image_0_10x )[0]
    sites_to_xy_10x = map_10x_fields_to_xy_coords( nd2_file_10x )
    images_40x_to_xy = map_40x_fields_to_xy_coords_phenix( well_phenix, index_file_phenix )

    # need to run this as a pool with single process, otherwise plotting gets messed up (at least on mac)
    # I think this is a bug with matplotlib/mutliprocessing
    # error msg: 
    # "The process has forked and you cannot use this CoreFoundation functionality safely. You MUST exec().
    # Break on __THE_PROCESS_HAS_FORKED_AND_YOU_CANNOT_USE_THIS_COREFOUNDATION_FUNCTIONALITY___YOU_MUST_EXEC__() to debug."
    corr_thresh =0.35
    with multiprocessing.Pool( processes=1 ) as pool:
        results = pool.map( fit_linear_40x_to_10x_from_brute_force_matches_parallel,
                        [ (output_dir_brute_force_match, 
                        sites_to_xy_10x, 
                        images_40x_to_xy, 
                        pixel_size_10x, 
                        image_size_10x_x, 
                        corr_thresh,
                        plot)])
        linregress_x_40x_to_10x = results[0][0]
        linregress_y_40x_to_10x = results[0][1]


#    linregress_x_40x_to_10x, linregress_y_40x_to_10x = fit_linear_40x_to_10x_from_brute_force_matches( 
#                                                    output_dir_brute_force_match,
#                                                    sites_to_xy_10x,
#                                                    images_40x_to_xy,
#                                                    pixel_size_10x,
#                                                    image_size_10x_x,
#                                                    plot=plot)

    # use this model to get the best 10x image matches
    print( "Using model to match all 40x images" )
    files_10x_base_name = list_of_files_10x[0].split('-tile')[0] + '-'
    print( "using files_10x_base_name:", files_10x_base_name )
    #print( "image_size_10x_x", image_size_10x_x )
    #print( "pixel_size_10x", pixel_size_10x )
    tile_size_10x_x = pixel_size_10x * image_size_10x_x
    tile_size_10x_y = pixel_size_10x * image_size_10x_y
    with multiprocessing.Pool( processes=num_proc ) as pool:
        results = pool.map( get_best_10x_image_match_from_model_parallel, [( file_40x, files_10x_base_name, images_40x_to_xy, 
                                            sites_to_xy_10x, 
                                            linregress_x_40x_to_10x, linregress_y_40x_to_10x, 
                                            tile_size_10x_x, tile_size_10x_y,
                                            output_dir_match, scale_factor, overlap_ratio_final_matches, plot ) for file_40x in list_of_files_40x] )

def check_40x_to_10x_mapping( list_of_files_40x, well_phenix, index_file_phenix, 
                                                list_of_files_10x, nd2_file_10x, pixel_size_40x,
                                                pixel_size_10x, output_dir_brute_force_match, output_dir_match,
                                                num_proc, overlap_ratio_final_matches = 0.5,
                                                num_40x_files_match=10, plot=False ):
    overlap_ratio_brute_force_initial = 1.0
    scale_factor = pixel_size_40x / pixel_size_10x

    # based on the brute force matches fit a linear model for 40x to 10x coords
    print( "Fitting model based on brute force matches" )
    image_0_10x = read( list_of_files_10x[0] )[0]
    image_size_10x_x = np.shape( image_0_10x )[0]
    image_size_10x_y = np.shape( image_0_10x )[0]
    sites_to_xy_10x = map_10x_fields_to_xy_coords( nd2_file_10x )
    images_40x_to_xy = map_40x_fields_to_xy_coords_phenix( well_phenix, index_file_phenix )

    # need to run this as a pool with single process, otherwise plotting gets messed up (at least on mac)
    # I think this is a bug with matplotlib/mutliprocessing
    # error msg: 
    # "The process has forked and you cannot use this CoreFoundation functionality safely. You MUST exec().
    # Break on __THE_PROCESS_HAS_FORKED_AND_YOU_CANNOT_USE_THIS_COREFOUNDATION_FUNCTIONALITY___YOU_MUST_EXEC__() to debug."
    corr_thresh =0.35
    with multiprocessing.Pool( processes=1 ) as pool:
        results = pool.map( fit_linear_40x_to_10x_from_brute_force_matches_parallel,
                        [ (output_dir_brute_force_match, 
                        sites_to_xy_10x, 
                        images_40x_to_xy, 
                        pixel_size_10x, 
                        image_size_10x_x, 
                        corr_thresh,
                        plot)])
        linregress_x_40x_to_10x = results[0][0]
        linregress_y_40x_to_10x = results[0][1]
