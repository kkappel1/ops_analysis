import skimage
import numpy as np
from ops.io import read_stack as read
from ops.io import save_stack as save
import ops.io
from skimage import (
    color, feature, filters, measure, morphology, segmentation, util
)
# for skimage version 0.18
from skimage.registration._masked_phase_cross_correlation import cross_correlate_masked
#from skimage.feature.masked_register_translation import cross_correlate_masked
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.morphology import disk, watershed
from skimage.filters import threshold_otsu, rank
from scipy import ndimage as ndi
from ops.imports import *
import matplotlib.patches as mpatches
import random
import glob
from ops.process import Align
import ops.firesnake
from ops.firesnake import Snake
from collections import Counter
import operator
import math
from .preprocess_phenotype import *
from skimage import img_as_ubyte
from cellpose.models import Cellpose
import datetime
from scipy.stats import pearsonr



#pixel_size_40x = 0.1507
#pixel_size_10x = 0.844692184817898
#scale_factor = pixel_size_40x / pixel_size_10x
pixel_size_10x = 0.841792491782224
pixel_size_40x = 0.14949402023919043
scale_factor = pixel_size_40x / pixel_size_10x

def masked_register_translation_return_CC(
        src_image,
        target_image,
        src_mask,
        target_mask=None,
        overlap_ratio=3 / 10):

    if target_mask is None:
        target_mask = np.array(src_mask, dtype=np.bool, copy=True)

    # We need masks to be of the same size as their respective images
    for (im, mask) in [(src_image, src_mask), (target_image, target_mask)]:
        if im.shape != mask.shape:
            raise ValueError(
                "Error: image sizes must match their respective mask sizes.")

    # The mismatch in size will impact the center location of the
    # cross-correlation
    size_mismatch = np.array(target_image.shape) - np.array(src_image.shape)

    xcorr = cross_correlate_masked(target_image, src_image, target_mask,
                                   src_mask, axes=(0, 1), mode='full',
                                   overlap_ratio=overlap_ratio)

    # Generalize to the average of multiple equal maxima
    maxima = np.transpose(np.nonzero(xcorr == xcorr.max()))
    center = np.mean(maxima, axis=0)
    shifts = center - np.array(src_image.shape) + 1
    return -shifts + (size_mismatch / 2), xcorr


def get_best_10x_image_match( file_40x, list_of_files_10x, scale_factor, overlap_ratio=1. ):
    #print( "overlap ratio", overlap_ratio )
    # load the image
    image_40x = read( file_40x )
    if image_40x.ndim == 4:
        image_40x_max_proj = np.max( image_40x, axis=0)
    else:
        image_40x_max_proj = image_40x

    # rescale the image -- only the DAPI channel
    image_10x_0 = read( list_of_files_10x[0] )
    image_size_10x = np.shape( image_10x_0[0] )[0]
    padded_image_40x_rescaled, _ = rescale_40x_to_10x_and_pad( image_40x_max_proj[0], 
                                                            scale_factor, image_size_10x )
    
    # create 40x image mask (only want to align the actual image -- not the pad)
    target_mask = padded_image_40x_rescaled > 0
    
    # loop through the 10x images to find the correct alignment
    best_shift = []
    best_max_cc_mag = -1.
    best_image_match = ''
    for file_10x in list_of_files_10x:
        print( "testing", file_10x)
        image_10x = read( file_10x )
        
        # create 10x image mask (does not mask anything out, full image is valid)
        src_mask = np.ones( np.shape(image_10x[0]), dtype=bool)
        
        # get the masked correlation - use DAPI only
        # this code is essentially equivalent to masked_register_translation
        # but I want it to return the cross correlation and don't want to compute twice
        # so need to write out here
        shift, cc = masked_register_translation_return_CC( image_10x[0], 
                                                          padded_image_40x_rescaled, src_mask, target_mask,
                                                         overlap_ratio=overlap_ratio)
                                                         #overlap_ratio=10/10)
        max_cc_mag = np.abs( cc.max() )
        if max_cc_mag > best_max_cc_mag:
            best_shift = shift
            best_max_cc_mag = max_cc_mag
            best_image_match = file_10x
    
    st = skimage.transform.SimilarityTransform( translation=-1*best_shift[::-1] )
    shifted_40x_image = skimage.transform.warp( padded_image_40x_rescaled, st, preserve_range=True )
    final_shifted_40x_image = shifted_40x_image.astype(image_40x.dtype)

    # write to a file
    base_fname = file_40x.split('/')[-1].split('.tif')[0]
    #with open( 'match_files_0.25/match_{name}.txt'.format(name=base_fname), 'w') as f:
    with open( 'match_files_0.5/match_{name}.txt'.format(name=base_fname), 'w') as f:
    #with open( 'match_files/match_{name}.txt'.format(name=base_fname), 'w') as f:
        f.write( '{image} {cc_mag} {shift_x} {shift_y}\n'.format(image=best_image_match, cc_mag=best_max_cc_mag, shift_x=best_shift[0], shift_y=best_shift[1] ))
    
    return best_image_match, best_max_cc_mag, best_shift, final_shifted_40x_image


def rescale_40x_to_10x_and_pad( image_40x_to_rescale, scale_factor, image_size_10x ):
    image_40x_rescaled = skimage.transform.rescale( image_40x_to_rescale.T, 
                                                         scale_factor, anti_aliasing=False, 
                                                         multichannel=False, preserve_range=True)
    rescaled_size = np.shape( image_40x_rescaled )[0]
    pad_size = image_size_10x - rescaled_size
    padded_image_40x_rescaled = skimage.util.pad( image_40x_rescaled, 
                                                       [(0,pad_size), (0,pad_size)], 
                                                       'constant', constant_values=0 )
    
    return padded_image_40x_rescaled, rescaled_size


def map_40x_nuclei_to_10x_nuclei( nuclei_mask_40x, nuclei_10x_mask, best_shift,
                                THRESHOLD_MOST_FREQ_FRACTION=0.25, THRESHOLD_MOST_NEXT=1.5,
                                plot=False):
    
    #print( "THRESHOLD_MOST_FREQ_FRACTION", THRESHOLD_MOST_FREQ_FRACTION )
    #print( "THRESHOLD_MOST_NEXT", THRESHOLD_MOST_NEXT )
    # shrink the 40x nuclei mask by the scale factor
    image_size_10x = np.shape( nuclei_10x_mask )[0]
    padded_nuclei_mask_40x_rescaled, rescaled_size_40x = rescale_40x_to_10x_and_pad( nuclei_mask_40x, 
                                                                scale_factor, image_size_10x )

    st = skimage.transform.SimilarityTransform( translation=-1*best_shift[::-1] )
    shifted_nuclei_mask_40x_rescaled = skimage.transform.warp( padded_nuclei_mask_40x_rescaled, 
                                                              st, preserve_range=True )
    #final_shifted_nuclei_mask_40x_rescaled = shifted_nuclei_mask_40x_rescaled.astype(image_40x.dtype)
    final_shifted_nuclei_mask_40x_rescaled = shifted_nuclei_mask_40x_rescaled.astype(nuclei_mask_40x.dtype)

    # can I now loop through the regions in the final_shifted_nuclei_mask_40x_rescaled
    dict_40x_nuclei_to_10x_nuclei = {}
    for nucleus_40x in skimage.measure.regionprops(final_shifted_nuclei_mask_40x_rescaled, nuclei_10x_mask):

        # figure out which nucleus from the 10x image this corresponds to
        region_image = nucleus_40x.intensity_image
        region_pixels = region_image[ nucleus_40x.image ]
        total_num_pixels = len( region_pixels )
        # figure out what nucleus this most overlaps with?
        region_pixels_no_zero = region_pixels[region_pixels>0]

        counts = Counter( region_pixels_no_zero )
        sorted_counts = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)

        if len(sorted_counts) == 1:
            most_frequent_value = sorted_counts[0][0]
            most_frequent_count = sorted_counts[0][1]
            if most_frequent_count/total_num_pixels > THRESHOLD_MOST_FREQ_FRACTION:
                if most_frequent_value not in dict_40x_nuclei_to_10x_nuclei.values():
                    dict_40x_nuclei_to_10x_nuclei[nucleus_40x.label] = most_frequent_value
                else:
                    dict_40x_nuclei_to_10x_nuclei[nucleus_40x.label] = 0
                    index_of_bad_val = list(dict_40x_nuclei_to_10x_nuclei.keys())[list(dict_40x_nuclei_to_10x_nuclei.values()).index(most_frequent_value)]
                    dict_40x_nuclei_to_10x_nuclei[index_of_bad_val] = 0
            else:
                dict_40x_nuclei_to_10x_nuclei[nucleus_40x.label] = 0
        elif len( sorted_counts ) > 1:
            most_frequent_value = sorted_counts[0][0]
            most_frequent_count = sorted_counts[0][1]
            next_most_frequent_value = sorted_counts[1][0]
            next_most_frequent_count = sorted_counts[1][1]
            most_freq_fraction = most_frequent_count / total_num_pixels
            next_most_freq_fraction = next_most_frequent_count / total_num_pixels
            if (most_freq_fraction > THRESHOLD_MOST_FREQ_FRACTION) and ((most_freq_fraction / next_most_freq_fraction) > THRESHOLD_MOST_NEXT ):
                if most_frequent_value not in dict_40x_nuclei_to_10x_nuclei.values():
                    dict_40x_nuclei_to_10x_nuclei[nucleus_40x.label] = most_frequent_value
                else:
                    dict_40x_nuclei_to_10x_nuclei[nucleus_40x.label] = 0
                    index_of_bad_val = list(dict_40x_nuclei_to_10x_nuclei.keys())[list(dict_40x_nuclei_to_10x_nuclei.values()).index(most_frequent_value)]
                    dict_40x_nuclei_to_10x_nuclei[index_of_bad_val] = 0
            else:
                dict_40x_nuclei_to_10x_nuclei[nucleus_40x.label] = 0
        else:
            # this nucleus does not overlap with anything
            dict_40x_nuclei_to_10x_nuclei[nucleus_40x.label] = 0

    if plot:
        fig, ax = plt.subplots(1,2,figsize=(20, 10))
        ax[0].imshow(nuclei_mask_40x.T)

        #for region in skimage.measure.regionprops(final_shifted_nuclei_mask_40x_rescaled):
        for region in skimage.measure.regionprops(nuclei_mask_40x):
            #final_shifted_nuclei_mask_40x_rescaled
            # draw rectangles around cells that are mapped
            if region.label not in dict_40x_nuclei_to_10x_nuclei.keys():
                print( region.label, "not in dict_40x_nuclei_to_10x_nuclei")
                continue
            if dict_40x_nuclei_to_10x_nuclei[ region.label ] == 0:
                continue
            minc, minr, maxc, maxr = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                        fill=False, edgecolor='red', linewidth=2)
            ax[0].add_patch(rect)
            ax[0].text( region.centroid[0], region.centroid[1], region.label, color='black')

        # show the mapped region of the 10x_nuclei
        size_40x = rescaled_size_40x
        chopped_10x_mask = nuclei_10x_mask[int(best_shift[0]):int(best_shift[0])+size_40x, 
                                           int(best_shift[1]):int(best_shift[1])+size_40x]
        
        ax[1].imshow(chopped_10x_mask)
        for region in skimage.measure.regionprops(chopped_10x_mask):
            ax[1].text( region.centroid[1], region.centroid[0], 
                      region.label, color='black')
        
        
        ax[0].set_axis_off()
        ax[1].set_axis_off()

        plt.tight_layout()
        #plt.show()
        plt.clf()

    return dict_40x_nuclei_to_10x_nuclei

### taken from skimage version 0.19.0
def expand_labels( label_image, distance=1):
    distances, nearest_label_coords = ndi.distance_transform_edt(
        label_image == 0, return_indices=True
    )
    labels_out = np.zeros_like(label_image)
    dilate_mask = distances <= distance
    # build the coordinates to find nearest labels,
    # in contrast to [1] this implementation supports label arrays
    # of any dimension
    masked_nearest_label_coords = [
        dimension_indices[dilate_mask]
        for dimension_indices in nearest_label_coords
    ]
    nearest_labels = label_image[tuple(masked_nearest_label_coords)]
    labels_out[dilate_mask] = nearest_labels
    return labels_out

def segment_nuclei_40x( fname, well_num, tile_num, min_size=1000, 
                        smooth_size=5, threshold_initial_guess=500, 
                        nuclei_smooth_value=20, area_min=5000, plot=False,
                        area_max=20000, condensate_cutoff_intensity=2000,
                        THRESH_STDS=5):
    image = read( fname )
    # if z-stack, use max projection image
    if image.ndim == 4:
        image_max_proj = np.max(image, axis=0)
    else:
        image_max_proj = image

    ##################
    # segment nuclei from dapi image
    ##################
    dapi_image = image_max_proj[0]

    smoothed = rank.median( dapi_image, disk(smooth_size))
    smoothed = rank.enhance_contrast( smoothed, disk(smooth_size))
    init_guess = min( np.max( smoothed)-1, threshold_initial_guess)
    #smoothed_thresh_global = filters.threshold_li( smoothed, initial_guess=threshold_initial_guess )
    smoothed_thresh_global = filters.threshold_li( smoothed, initial_guess=init_guess )
    mask = smoothed > smoothed_thresh_global
    mask = skimage.morphology.remove_small_objects(mask, min_size=min_size)

    labeled = skimage.measure.label(mask)
    labeled = ops.process.filter_by_region(labeled, score = lambda r: r.mean_intensity, threshold = lambda x: 100, intensity_image=dapi_image) > 0

    # fill holes below minimum area
    filled = ndi.binary_fill_holes(labeled)
    difference = skimage.measure.label(filled!=labeled)
    change = ops.process.filter_by_region(difference, lambda r: r.area < area_min, 0) > 0
    labeled[change] = filled[change]

    ### the smooth value here is really important!! 
    ### using a value that is too small results in a bunch of nuclei get segmented into pieces
    nuclei = ops.process.apply_watershed(labeled, smooth=nuclei_smooth_value)

    # get rid of nuclei that are touching the edge of the image
    cleared = skimage.segmentation.clear_border(nuclei)
    final_nuclei = ops.process.filter_by_region(cleared, score=lambda r: area_min < r.area < area_max, threshold = lambda x: 100)
    
    return final_nuclei

def phenotype_TEST2_40x_image( fname, well_num, tile_num, min_size=1000, 
                        smooth_size=5, threshold_initial_guess=500, 
                        nuclei_smooth_value=20, area_min=5000, plot=False,
                        area_max=20000, condensate_cutoff_intensity=2000,
                        THRESH_STDS=5):
    image = read( fname )
    # if z-stack, use max projection image
    if image.ndim == 4:
        image_max_proj = np.max(image, axis=0)
    else:
        image_max_proj = image

    ##################
    # segment nuclei from dapi image
    ##################
    dapi_image = image_max_proj[0]

    smoothed = rank.median( dapi_image, disk(smooth_size))
    smoothed = rank.enhance_contrast( smoothed, disk(smooth_size))
    init_guess = min( np.max( smoothed)-1, threshold_initial_guess)
    #smoothed_thresh_global = filters.threshold_li( smoothed, initial_guess=threshold_initial_guess )
    smoothed_thresh_global = filters.threshold_li( smoothed, initial_guess=init_guess )
    mask = smoothed > smoothed_thresh_global
    mask = skimage.morphology.remove_small_objects(mask, min_size=min_size)

    labeled = skimage.measure.label(mask)
    labeled = ops.process.filter_by_region(labeled, score = lambda r: r.mean_intensity, threshold = lambda x: 100, intensity_image=dapi_image) > 0

    # fill holes below minimum area
    filled = ndi.binary_fill_holes(labeled)
    difference = skimage.measure.label(filled!=labeled)
    change = ops.process.filter_by_region(difference, lambda r: r.area < area_min, 0) > 0
    labeled[change] = filled[change]

    ### the smooth value here is really important!! 
    ### using a value that is too small results in a bunch of nuclei get segmented into pieces
    nuclei = ops.process.apply_watershed(labeled, smooth=nuclei_smooth_value)

    # get rid of nuclei that are touching the edge of the image
    cleared = skimage.segmentation.clear_border(nuclei)
    final_nuclei = ops.process.filter_by_region(cleared, score=lambda r: area_min < r.area < area_max, threshold = lambda x: 100)
    
    #fig, ax = plt.subplots(1,2,figsize=(10,10))
    #ax[0].imshow(dapi_image.T)
    #ax[1].imshow(final_nuclei.T)
    #plt.show()

    ##################
    gfp_image = image_max_proj[1]

    # image needs to be shifted to line up with dapi channel
    st = skimage.transform.SimilarityTransform( translation=[0,-5] )
    shifted_gfp_image = skimage.transform.warp( gfp_image, st, preserve_range=True )
    shifted_gfp_image = shifted_gfp_image.astype(image.dtype)

    ##################
    # go through the nuclei and try to find condensates, calculate relevant properties
    ##################
    
    nuclei_dict = {
        'nucleus_num': [],
        'well': [],
        'tile': [],
        'fname': [],
        'mean_GFP_intensity': [],
        'std_GFP_intensity': [],
        'num_condensates': [],
        'condensates_mask': [],
        'mean_dilute_intensity': [],
        'std_dilute_intensity': [],
        'mean_condensate_intensity': [],
        'std_condensate_intensity': [],
        'total_cond_intensity': [],
        'total_gfp_intensity': [],
        'frac_gfp_in_cond': [], 
        'total_cond_area': [],
        'total_nucleus_area': []
    }
    
    for nucleus in skimage.measure.regionprops(final_nuclei, shifted_gfp_image):
        #if nucleus.label != 3: continue
        # mean intensity
        mean_intensity = nucleus.mean_intensity
        max_intensity = np.max( nucleus.intensity_image )
        
        # standard deviation intensity
        region_image = nucleus.intensity_image
        region_pixels = region_image[ nucleus.image ]
        std_intensity = np.std( region_pixels )
        total_intensity = nucleus.mean_intensity * nucleus.area
        total_nucleus_area = nucleus.area
        
        # if intensity is below some threshold, do not look for condensates (if GFP is not actually expressed)
        if mean_intensity < 250. and max_intensity < 250.:
            # this cell does not express GFP
            num_condensates = 0
            # condensate mask - should all be false
            condensates_labeled = np.zeros( np.shape(nucleus.intensity_image))
            mean_dilute_phase_intensity = mean_intensity
            mean_condensate_intensity = np.nan
            num_condensates = 0
            std_dilute_phase_intensity = std_intensity
            std_condensate_intensity = np.nan
            total_condensate_intensity = 0
            total_condensate_area = 0
            
        else:
            if plot:
                fig, ax = plt.subplots(1,13,figsize=(10, 2))
                for i in range(13):
                    ax[i].set_axis_off()
            
            # automatically determine hole threshold
            inverted_image = -1 * nucleus.intensity_image
            dark_thresh = filters.threshold_li( inverted_image, initial_guess=-1.)
            
            hole_mask = nucleus.intensity_image < -1*dark_thresh
            if plot:
                ax[1].imshow( hole_mask, cmap='gray')
                ax[1].set_title('hole_mask', fontsize=5)
            # fill holes
            filled_holes = ndi.binary_fill_holes(hole_mask)
            difference = skimage.measure.label(filled_holes!=hole_mask)
            change = ops.process.filter_by_region(difference, lambda r: r.area < 50, 0) > 0
            filled_hole_mask = hole_mask
            filled_hole_mask[change] = filled_holes[change]
            
            
            
            # remove small objects
            MIN_SIZE_HOLE = 10.
            filled_hole_mask_filt = skimage.morphology.remove_small_objects(filled_hole_mask, min_size=MIN_SIZE_HOLE)
            num_hole_pixels = np.sum( filled_hole_mask_filt & nucleus.image )
            num_cell_pixels = np.sum( nucleus.image )
            final_filled_hole_mask_filt = np.invert( filled_hole_mask_filt )
            if num_hole_pixels > 0.75 * num_cell_pixels:
                # this is too much of the cell -- ignore holes.
                final_filled_hole_mask_filt = nucleus.image
            prelim_gfp_intensity_pixels = region_image[ final_filled_hole_mask_filt ]
            prelim_avg_gfp_intensity = np.mean( prelim_gfp_intensity_pixels )
            std_gfp_pixels = np.std( prelim_gfp_intensity_pixels )
            std_prelim_dilute_pixels = np.std( prelim_gfp_intensity_pixels[ prelim_gfp_intensity_pixels < 2.*prelim_avg_gfp_intensity])
        # first find holes - use set intensity threshold. check that holes do not account for more than X% of total cell area
        # then get prelim average intensity outside of holes
        
        

        
            if plot:
                ax[0].imshow( nucleus.intensity_image, cmap='gray', vmin=0, vmax=5000)
                ax[0].set_title(nucleus.label, fontsize=5)

                ax[2].imshow( filled_hole_mask, cmap='gray')
                ax[2].set_title('filled_hole_mask', fontsize=5)
                ax[3].imshow( filled_hole_mask_filt, cmap='gray')
                ax[3].set_title('filled_hole_mask_filt', fontsize=5)
                ax[4].imshow( final_filled_hole_mask_filt, cmap='gray')
                ax[4].set_title('final_filled_hole_mask_filt', fontsize=5)

            # find condensates - use intensity threshold relative to prelim average intensity outside holes
            #condensate_threshold = max(1.5*prelim_avg_dilute_intensity,condensate_cutoff_intensity)
            condensate_threshold = max(prelim_avg_gfp_intensity+THRESH_STDS*std_prelim_dilute_pixels,condensate_cutoff_intensity)
            condensate_mask = nucleus.intensity_image > condensate_threshold
            if plot:
                ax[5].imshow( condensate_mask, cmap='gray')
                ax[5].set_title( 'condensate_mask', fontsize=5)
            
            # fill holes in the condensates
            filled_cond = ndi.binary_fill_holes(condensate_mask)
            difference = skimage.measure.label(filled_cond!=condensate_mask)
            change = ops.process.filter_by_region(difference, lambda r: r.area < 4, 0) > 0
            filled_cond_mask = condensate_mask
            filled_cond_mask[change] = filled_cond[change]
            if plot:
                ax[6].imshow( filled_cond_mask, cmap='gray')
                ax[6].set_title( 'filled_cond_mask', fontsize=5)
            min_size_c = 4
            filled_cond_mask_nosmall = skimage.morphology.remove_small_objects(filled_cond_mask, min_size=min_size_c)
            if plot:
                ax[7].imshow( filled_cond_mask_nosmall, cmap='gray')
                ax[7].set_title( 'filled_cond_mask_nosmall', fontsize=5)
            
            # segment the condensates
            condensates_ = ops.process.apply_watershed(filled_cond_mask_nosmall, smooth=1)
            condensates = skimage.segmentation.clear_border(condensates_)
            condensates_labeled, _, _ = skimage.segmentation.relabel_sequential(condensates)
            if plot:
                ax[8].imshow( condensates_, cmap='gray')
                ax[8].set_title( 'condensates_', fontsize=5)
                ax[9].imshow( condensates, cmap='gray')
                ax[9].set_title( 'condensates', fontsize=5)
                ax[10].imshow( color.label2rgb(condensates_labeled, bg_label=0) )
                ax[10].set_title( 'condensates_labeled', fontsize=5)

            # do morphological erosion to get mean dilute and condensate intensities
            # shrink condensates
            inverse_condensates = np.invert( condensates > 0)
            NUM_SHRINK_COND_PIXELS = 5
            NUM_EXPAND_COND_PIXELS = 8
            shrunk_condensates = np.invert( expand_labels( inverse_condensates, NUM_SHRINK_COND_PIXELS) )
            
            total_condensate_intensity = np.sum( nucleus.intensity_image[condensates>0])
            total_condensate_area = np.sum( condensates>0 )
            
            
            mean_condensate_intensity = np.mean( nucleus.intensity_image[shrunk_condensates])
            std_condensate_intensity = np.std( nucleus.intensity_image[shrunk_condensates] )
            
            # expand condensates
            expanded_condensates = expand_labels( condensates, NUM_EXPAND_COND_PIXELS)
            dilute_pixel_mask = (expanded_condensates == 0) & nucleus.image & final_filled_hole_mask_filt

            mean_dilute_phase_intensity = np.mean( nucleus.intensity_image[dilute_pixel_mask])
            std_dilute_phase_intensity = np.std( nucleus.intensity_image[dilute_pixel_mask] )

            num_condensates = np.max( condensates_labeled )

            if plot:
                ax[11].imshow( shrunk_condensates, cmap='gray')
                ax[11].set_title( 'shrunk_condensates', fontsize=5)
                ax[12].imshow( dilute_pixel_mask, cmap='gray' )
                ax[12].set_title( 'dilute_pixel_mask', fontsize=5)
                plt.show()            
            
        # add everything to dictionary
        nuclei_dict['nucleus_num'].append( nucleus.label )
        nuclei_dict['well'].append( well_num )
        nuclei_dict['tile'].append( tile_num )
        nuclei_dict['fname'].append( fname )
        nuclei_dict['mean_GFP_intensity'].append( mean_intensity )
        nuclei_dict['std_GFP_intensity'].append( std_intensity )
        nuclei_dict['num_condensates'].append( num_condensates )
        nuclei_dict['condensates_mask'].append( condensates_labeled )
        nuclei_dict['mean_dilute_intensity'].append( mean_dilute_phase_intensity )
        nuclei_dict['std_dilute_intensity'].append( std_dilute_phase_intensity )
        nuclei_dict['mean_condensate_intensity'].append( mean_condensate_intensity )
        nuclei_dict['std_condensate_intensity'].append( std_condensate_intensity )
        nuclei_dict['total_cond_intensity'].append( total_condensate_intensity )
        nuclei_dict['total_gfp_intensity'].append( total_intensity )
        nuclei_dict['frac_gfp_in_cond'].append( total_condensate_intensity / total_intensity )
        nuclei_dict['total_cond_area'].append( total_condensate_area)
        nuclei_dict['total_nucleus_area'].append( total_nucleus_area)
        

    df_nuclei = pd.DataFrame(data=nuclei_dict)
    return df_nuclei, final_nuclei


def SBS_get_nuclei_and_cells( data, well_num, tile_num, THRESHOLD_DAPI=4000, NUCLEUS_AREA=(150, 1500),
#def SBS_get_nuclei_and_cells( data, well_num, tile_num, THRESHOLD_DAPI=4000, NUCLEUS_AREA=(200, 1500),
                            THRESHOLD_CELL=4500, THRESHOLD_STD=200):
    print( "getting SBS data for well {well_num}, tile {tile_num}".format( well_num=well_num, tile_num=tile_num) )
    ### SBS - align
    aligned = Snake._align_SBS(data, method='DAPI')
    save('aligned_10x_well{well_num}_tile{tile_num}.tif'.format( well_num=well_num, tile_num=tile_num) , aligned)

    ### SBS - laplacian of gaussian
    loged = Snake._transform_log(aligned, skip_index=0)
    save('loged_10x_well{well_num}_tile{tile_num}.tif'.format( well_num=well_num, tile_num=tile_num ), loged)
    maxed = Snake._max_filter(loged, 3, remove_index=0)
    save('maxed_10x_well{well_num}_tile{tile_num}.tif'.format( well_num=well_num, tile_num=tile_num ), maxed)

    std = Snake._compute_std(loged, remove_index=0)
    peaks = Snake._find_peaks(std)

    ### SBS - find nuclei and cells
    nuclei = Snake._segment_nuclei(data[0], THRESHOLD_DAPI,
                                   area_min=NUCLEUS_AREA[0], area_max=NUCLEUS_AREA[1])
    save('nuclei_10x_well{well_num}_tile{tile_num}.tif'.format( well_num=well_num, tile_num=tile_num ), nuclei)

    cells = Snake._segment_cells(data[0], nuclei, THRESHOLD_CELL)

    WILDCARDS = dict(well=well_num, tile=tile_num)
    df_bases = Snake._extract_bases(maxed, peaks, cells,
                            THRESHOLD_STD, wildcards=WILDCARDS)
    df_reads = Snake._call_reads(df_bases)
    #df_reads.to_csv('test_reads.csv', index=None)
    df_reads_file_name = 'reads_{well_num}_tile{tile_num}.csv'.format(well_num=well_num, tile_num=tile_num)
    if df_reads is not None:
        df_reads.to_csv(df_reads_file_name, index=None)
        df_reads = pd.read_csv(df_reads_file_name)
        df_cells = Snake._call_cells(df_reads)
    else:
        print( "Problem well_num: ", well_num, "Problem tile_num", tile_num, df_bases.shape[0] )
        df_reads = pd.DataFrame()
        df_reads.to_csv(df_reads_file_name, index=None)
        df_cells = pd.DataFrame()

    return nuclei, cells, df_bases, df_reads, df_cells, well_num, tile_num


def plot_nuclei( dapi_image, nuclei, save_name='' ):
    image_label_overlay = color.label2rgb(nuclei, bg_label=0)

    fig, ax = plt.subplots(1,2,figsize=(20, 10))
    ax[1].imshow(image_label_overlay)

    for region in skimage.measure.regionprops(nuclei):
        # draw rectangles around cells
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                    fill=False, edgecolor='red', linewidth=0.5)
        ax[1].add_patch(rect)
        ax[1].text( region.centroid[1], region.centroid[0], region.label, color='black')

    ax[1].set_axis_off()
    ax[0].imshow(dapi_image, cmap='gray')
    ax[0].set_axis_off()
    plt.tight_layout()
    ax[0].set_title( "DAPI" )
    ax[1].set_title( "Detect nuclei" )
    if len(save_name) > 0:
        plt.savefig( save )
    plt.clf()
    #plt.show()

def plot_cells( cells, nuclei, save_name= ''):
    cells_image = color.label2rgb( cells, bg_label=0)
    fig, ax = plt.subplots(1,1, figsize = (20,20))
    ax.imshow( nuclei > 0, cmap='gray')
    ax.imshow( cells_image, alpha = 0.5)
    for region in skimage.measure.regionprops(cells):
        ax.text( region.centroid[1], region.centroid[0], region.label, color='black', fontsize='x-small')

    if len(save_name) > 0:
        plt.savefig( save_name )
    plt.clf()
    #plt.show()



def filter_reads( df_reads, true_cell_barcodes, QUALITY_THRESHOLD=0.05 ):
    # ok actually though, I want to loop through the reads and find out which ones are really valid
    # then assign them back to cells

    # filter garbage
    # condensates get read out as GGG reads -- get rid of these
    df_reads_noGGG = df_reads[df_reads['barcode'] != 'GGGGGGGG']
    df_reads_filt = df_reads_noGGG[df_reads_noGGG['Q_min'] > QUALITY_THRESHOLD]
    df_accurate_reads = df_reads_filt[df_reads_filt['barcode'].isin( true_cell_barcodes)]
    print( "num acc reads", df_accurate_reads.shape[0])
    if df_accurate_reads.shape[0] < 1:
        df_cells_filt = pd.DataFrame()
    else:
        # loop through the cells and assign accurate barcodes (?)
        # assign cells using only filtered barcodes:
        df_cells_filt = Snake._call_cells(df_accurate_reads)
    return df_accurate_reads, df_cells_filt

def filter_reads_binary( df_reads, true_binary_cell_barcodes, QUALITY_THRESHOLD=0.05 ):
    # ok actually though, I want to loop through the reads and find out which ones are really valid
    # then assign them back to cells

    # filter garbage
    # condensates get read out as GGG reads -- get rid of these
    df_reads_noGGG = df_reads[df_reads['barcode'] != 'GGGGGGGG']
    df_reads_filt = df_reads_noGGG[df_reads_noGGG['Q_min'] > QUALITY_THRESHOLD]
    df_accurate_reads = df_reads_filt[df_reads_filt['binary_barcode'].isin( true_binary_cell_barcodes)]

    if df_accurate_reads.shape[0] < 1:
        df_cells_filt = pd.DataFrame()
    else:
        # loop through the cells and assign accurate barcodes (?)
        # assign cells using only filtered barcodes:
        df_cells_filt = Snake._call_cells(df_accurate_reads)
    return df_accurate_reads, df_cells_filt


def plot_reads_on_cells( df_reads, nuclei, cells, true_cell_barcodes, save_name='' ):
    # make a mask that shows read locations
    read_mask = np.zeros( np.shape(nuclei))
    true_cell_barcodes_to_num = {}
    bc_num = 1
    for barcode in true_cell_barcodes:
        true_cell_barcodes_to_num[ barcode ] = bc_num
        bc_num += 1

    for index, row in df_reads.iterrows():
        i = row['i']
        j = row['j']
        if row['barcode'] not in true_cell_barcodes_to_num.keys():
            read_mask[i,j] = 99999
            read_mask[i+1,j] = 99999
            read_mask[i-1,j] = 99999
            read_mask[i,j+1] = 99999
            read_mask[i,j-1] = 99999

            read_mask[i+1,j+1] = 99999
            read_mask[i-1,j-1] = 99999
            read_mask[i+1,j-1] = 99999
            read_mask[i-1,j+1] = 99999
        else:
            read_mask[i,j] = true_cell_barcodes_to_num[ row['barcode'] ]
            read_mask[i+1,j] = true_cell_barcodes_to_num[ row['barcode'] ]
            read_mask[i-1,j] = true_cell_barcodes_to_num[ row['barcode'] ]
            read_mask[i,j+1] = true_cell_barcodes_to_num[ row['barcode'] ]
            read_mask[i,j-1] = true_cell_barcodes_to_num[ row['barcode'] ]

            read_mask[i+1,j+1] = true_cell_barcodes_to_num[ row['barcode'] ]
            read_mask[i-1,j-1] = true_cell_barcodes_to_num[ row['barcode'] ]
            read_mask[i+1,j-1] = true_cell_barcodes_to_num[ row['barcode'] ]
            read_mask[i-1,j+1] = true_cell_barcodes_to_num[ row['barcode'] ]

    read_mask_image = color.label2rgb( read_mask,
                                      colors = ['white','blue','green', 'red', 'olive',
                                                'cyan','magenta','lime','orange'])
    plt.figure( figsize=(10,10))
    plt.imshow( read_mask_image, cmap="Paired", interpolation='none' )
    plt.imshow( cells, alpha = 0.5, cmap='gray')
    plt.imshow( nuclei, alpha = 0.5, cmap='gray')

    for index, row in df_reads.iterrows():
        i = row['i']
        j = row['j']
        plt.text(j,i,row['barcode'], fontsize=2)

    if len(save_name) > 0:
        plt.savefig( save_name )

    #plt.show()
    plt.clf()

def get_nucleoli_mask( nucleus, cutoff_intensity=200,
                plot=True):
    
    #if nucleus.label != 3: continue

    # standard deviation intensity
    region_image = nucleus.intensity_image
    region_pixels = region_image[ nucleus.image ]
    std_intensity = np.std( region_pixels )
    total_intensity = nucleus.mean_intensity * nucleus.area
    total_nucleus_area = nucleus.area

    prelim_gfp_intensity_pixels = region_image[ nucleus.image ]
    prelim_avg_gfp_intensity = np.mean( prelim_gfp_intensity_pixels )
    std_gfp_pixels = np.std( prelim_gfp_intensity_pixels )
    std_prelim_dilute_pixels = np.std( 
        prelim_gfp_intensity_pixels[ prelim_gfp_intensity_pixels < 2.*prelim_avg_gfp_intensity])


    condensate_threshold = max(prelim_avg_gfp_intensity+0.9*std_prelim_dilute_pixels,
                           cutoff_intensity)
    print( "condensate_threshold", condensate_threshold )
    print( "prelim_avg_gfp_intensity", prelim_avg_gfp_intensity )
    print( "std_gfp_pixels", std_gfp_pixels)
    print( "std_prelim_dilute_pixels", std_prelim_dilute_pixels )

    condensate_mask = nucleus.intensity_image > condensate_threshold

    # then fill the holes
    filled_holes_nucleoli = ndi.binary_fill_holes( condensate_mask )
    
    if plot:
        fig2, ax2 = plt.subplots(1,3,figsize=(5, 5))
        for i in range(3):
            ax2[i].set_axis_off()
        ax2[0].imshow( nucleus.intensity_image, cmap='gray', vmin=0, vmax=10000)
        ax2[0].set_title(nucleus.label, fontsize=8)
        ax2[1].imshow( condensate_mask, cmap='gray')
        ax2[1].set_title( 'condensate_mask', fontsize=8)
        ax2[2].imshow( filled_holes_nucleoli, cmap='gray')
        ax2[2].set_title( 'filled_holes', fontsize=8)
        plt.show()
    
    return filled_holes_nucleoli

def get_nucleoli_mask_explicit( image, mean_intensity, 
                               area, 
                               intensity_image,
                               label,
                               cutoff_intensity=200,
                               plot=True):

    # standard deviation intensity
    region_image = intensity_image
    region_pixels = region_image[ image ]
    std_intensity = np.std( region_pixels )
    total_intensity = mean_intensity * area
    total_nucleus_area = area

    prelim_gfp_intensity_pixels = region_image[ image ]
    prelim_avg_gfp_intensity = np.mean( prelim_gfp_intensity_pixels )
    std_gfp_pixels = np.std( prelim_gfp_intensity_pixels )
    std_prelim_dilute_pixels = np.std( 
        prelim_gfp_intensity_pixels[ prelim_gfp_intensity_pixels < 2.*prelim_avg_gfp_intensity])


    condensate_threshold = max(prelim_avg_gfp_intensity+0.9*std_prelim_dilute_pixels,
                           cutoff_intensity)
#     print( "condensate_threshold", condensate_threshold )
#     print( "prelim_avg_gfp_intensity", prelim_avg_gfp_intensity )
#     print( "std_gfp_pixels", std_gfp_pixels)
#     print( "std_prelim_dilute_pixels", std_prelim_dilute_pixels )

    condensate_mask = intensity_image > condensate_threshold

    # then fill the holes
    filled_holes_nucleoli = ndi.binary_fill_holes( condensate_mask )
    
    if plot:
        fig2, ax2 = plt.subplots(1,3,figsize=(5, 5))
        for i in range(3):
            ax2[i].set_axis_off()
        ax2[0].imshow( intensity_image, cmap='gray', vmin=0, vmax=10000)
        ax2[0].set_title( label, fontsize=8)
        ax2[1].imshow( condensate_mask, cmap='gray')
        ax2[1].set_title( 'condensate_mask', fontsize=8)
        ax2[2].imshow( filled_holes_nucleoli, cmap='gray')
        ax2[2].set_title( 'filled_holes', fontsize=8)
        plt.show()
    
    return filled_holes_nucleoli

def prepare_png_cellpose( dapi_image, save_png=False ):

    blank = np.zeros_like( dapi_image ) 
    dapi_upper = np.percentile( dapi_image, 99.5 )
    dapi_image = dapi_image / dapi_upper
    dapi_image[dapi_image >1] = 1
    red, green, blue = img_as_ubyte( blank), img_as_ubyte( blank ), img_as_ubyte( dapi_image )

    rgb_img = np.array([red, green, blue]).transpose([1,2,0])
    if save_png:
        skimage.io.imsave( 'cellpose_input.png', rgb_img )
    return rgb_img

def segment_nuclei_phenotype_cellpose_from_model( model_nuclei, rgb, diameter, gpu=False ):
    nuclei, _, _, _ = model_nuclei.eval( rgb, channels=[3,0], diameter=diameter )

    nuclei = skimage.segmentation.clear_border( nuclei )

    return nuclei
    

def segment_nuclei_phenotype_cellpose( rgb, diameter, gpu=False ):

    model_nuclei = Cellpose( model_type='nuclei', gpu=gpu )
    nuclei, _, _, _ = model_nuclei.eval( rgb, channels=[3,0], diameter=diameter )

    nuclei = skimage.segmentation.clear_border( nuclei )

    #save( 'nuclei_cellpose.tif', nuclei )
    return nuclei 

def segment_nuclei_phenotype( dapi_image,
                             threshold_initial_guess,
                             smooth_size,
                             nuclei_smooth_value,
                             min_size,
                             area_min,
                             area_max,
                             plot=True ):

    # these are the 2 lines that always give a warning "bad rank filter performance"
    smoothed = rank.median( dapi_image, disk(smooth_size))
    smoothed = rank.enhance_contrast( smoothed, disk(smooth_size))
    init_guess = min( np.max( smoothed)-1, threshold_initial_guess)
    smoothed_thresh_global = filters.threshold_li( smoothed, initial_guess=init_guess )
    mask = smoothed > smoothed_thresh_global
    mask = skimage.morphology.remove_small_objects(mask, min_size=min_size)

    labeled = skimage.measure.label(mask)
    labeled = ops.process.filter_by_region(labeled,
                                           score = lambda r: r.mean_intensity,
                                           threshold = lambda x: 100, intensity_image=dapi_image) > 0

    # fill holes below minimum area
    filled = ndi.binary_fill_holes(labeled)
    difference = skimage.measure.label(filled!=labeled)
    change = ops.process.filter_by_region(difference, lambda r: r.area < area_min, 0) > 0
    labeled[change] = filled[change]

    ### the smooth value here is really important!!
    ### using a value that is too small results in a bunch of nuclei get segmented into pieces
    nuclei = ops.process.apply_watershed(labeled, smooth=nuclei_smooth_value)

    # get rid of nuclei that are touching the edge of the image
    cleared = skimage.segmentation.clear_border(nuclei)
    final_nuclei = ops.process.filter_by_region(cleared,
                                                score=lambda r: area_min < r.area < area_max,
                                                threshold = lambda x: 100)

    if plot:
        fig, ax = plt.subplots(1,2,figsize=(10,10))
        ax[0].imshow(dapi_image.T)
        ax[1].imshow(final_nuclei.T)
        #plt.savefig( 'test_view_nuclei.pdf' )
        plt.show()


    #save( 'dapi_img.tif', dapi_image )
    #save( 'final_nuclei_test.tif', final_nuclei )
    return final_nuclei

def get_condensate_properties( condensates, image, intensity_image, label,
                              final_filled_hole_mask_filt, plot=True, condensates_for_dilute=None):
    condensates_labeled, _, _ = skimage.segmentation.relabel_sequential(condensates)

    # do morphological erosion to get mean dilute and condensate intensities
    # shrink condensates
    inverse_condensates = np.invert( condensates > 0)
    NUM_SHRINK_COND_PIXELS = 2
    #NUM_SHRINK_COND_PIXELS = 5
    NUM_EXPAND_COND_PIXELS = 8
    shrunk_condensates = np.invert( expand_labels( inverse_condensates, NUM_SHRINK_COND_PIXELS) )

    total_condensate_intensity = np.sum( intensity_image[condensates>0])
    total_condensate_area = np.sum( condensates>0 )


    mean_condensate_intensity = np.mean( intensity_image[shrunk_condensates])
    mean_condensate_intensity_no_shrink = total_condensate_intensity / total_condensate_area

    std_condensate_intensity = np.std( intensity_image[shrunk_condensates] )
    std_condensate_intensity_no_shrink = np.std( intensity_image[condensates>0] )

    # expand condensates
    if condensates_for_dilute is None:
        condensates_for_dilute = condensates
    expanded_condensates = expand_labels( condensates_for_dilute, NUM_EXPAND_COND_PIXELS)
    dilute_pixel_mask = (expanded_condensates == 0) & image & final_filled_hole_mask_filt

    mean_dilute_phase_intensity = np.mean( intensity_image[dilute_pixel_mask])
    std_dilute_phase_intensity = np.std( intensity_image[dilute_pixel_mask] )

    num_condensates = np.max( condensates_labeled )

    # get the size distribution of the condensates
    properties_cond = skimage.measure.regionprops_table( condensates_labeled, intensity_image,
                                        properties=('label','mean_intensity','area','eccentricity'))
    mean_condensate_area = np.mean( properties_cond['area'] )
    std_condensate_area = np.std( properties_cond['area'] )
    mean_condensate_eccentricity = np.mean( properties_cond['eccentricity'] )
    std_condensate_eccentricity = np.std( properties_cond['eccentricity'] )



    if plot:

        fig, ax = plt.subplots(1,5,figsize=(10, 2))
        for i in range(5):
            ax[i].set_axis_off()
        ax[0].imshow( condensates, cmap='gray')
        ax[0].set_title( 'condensates', fontsize=5)
        ax[1].imshow( color.label2rgb(condensates_labeled, bg_label=0) )
        ax[1].set_title( 'condensates_labeled', fontsize=5)
        ax[2].imshow( shrunk_condensates, cmap='gray')
        ax[2].set_title( 'shrunk_condensates', fontsize=5)
        ax[3].imshow( dilute_pixel_mask, cmap='gray' )
        ax[3].set_title( 'dilute_pixel_mask', fontsize=5)
        #plt.savefig( f'view_2_condensates_{label}.pdf' )

        fig2, ax2 = plt.subplots(1,2,figsize=(5, 5))
        for i in range(2):
            ax2[i].set_axis_off()
        ax2[0].imshow( intensity_image, cmap='gray', vmin=0, vmax=5000)
        ax2[0].set_title( label, fontsize=5)
        ax2[1].imshow( color.label2rgb(condensates_labeled, bg_label=0) )

        plt.show()
        plt.clf()

    condensates_properties_dict = {
        'total_condensate_intensity': total_condensate_intensity,
        'total_condensate_area': total_condensate_area,
        'mean_condensate_intensity': mean_condensate_intensity,
        'mean_condensate_intensity_no_shrink': mean_condensate_intensity_no_shrink,
        'std_condensate_intensity': std_condensate_intensity,
        'std_condensate_intensity_no_shrink': std_condensate_intensity_no_shrink,
        'mean_dilute_phase_intensity': mean_dilute_phase_intensity,
        'std_dilute_phase_intensity': std_dilute_phase_intensity,
        'num_condensates': num_condensates,
        'mean_condensate_area': mean_condensate_area,
        'std_condensate_area': std_condensate_area,
        'mean_condensate_eccentricity': mean_condensate_eccentricity,
        'std_condensate_eccentricity': std_condensate_eccentricity,
    }

    return condensates_labeled, condensates_properties_dict

def get_condensates_general_explicit( image,
                                     mean_intensity,
                                     area,
                                     intensity_image,
                                     label,
                                     condensate_cutoff_intensity, 
                                     THRESH_STDS,
                                     plot=True,
                                     save_file=''):

    # standard deviation intensity
    region_image = intensity_image
    region_pixels = region_image[ image ]
    std_intensity = np.std( region_pixels )
    total_intensity = mean_intensity * area
    total_nucleus_area = area

    # automatically determine hole threshold
    inverted_image = -1 * intensity_image
    try:
        dark_thresh = filters.threshold_li( inverted_image, initial_guess=-1.)
    except:
        print( "error with threshold li" )
        print( "save file name:", save_file )
        print( np.min( inverted_image), np.max(inverted_image) )
        dark_thresh = filters.threshold_li( inverted_image, initial_guess=np.max(inverted_image)-0.5)
        #dark_thresh = filters.threshold_li( inverted_image, initial_guess=np.max(inverted_image)-1.)

    hole_mask = intensity_image < -1*dark_thresh

    # fill holes
    filled_holes = ndi.binary_fill_holes(hole_mask)
    difference = skimage.measure.label(filled_holes!=hole_mask)
    change = ops.process.filter_by_region(difference, lambda r: r.area < 50, 0) > 0
    filled_hole_mask = hole_mask
    filled_hole_mask[change] = filled_holes[change]


    # remove small objects
    MIN_SIZE_HOLE = 10.
    filled_hole_mask_filt = skimage.morphology.remove_small_objects(filled_hole_mask, min_size=MIN_SIZE_HOLE)
    num_hole_pixels = np.sum( filled_hole_mask_filt & image )
    num_cell_pixels = np.sum( image )
    final_filled_hole_mask_filt = np.invert( filled_hole_mask_filt )


    num_hole_pixels = np.sum( filled_hole_mask_filt & image )
    num_cell_pixels = np.sum( image )
    final_filled_hole_mask_filt = np.invert( filled_hole_mask_filt )
    if num_hole_pixels > 0.75 * num_cell_pixels:
        # this is too much of the cell -- ignore holes.
        final_filled_hole_mask_filt = image
    prelim_gfp_intensity_pixels = region_image[ final_filled_hole_mask_filt ]
    prelim_avg_gfp_intensity = np.mean( prelim_gfp_intensity_pixels )
    std_gfp_pixels = np.std( prelim_gfp_intensity_pixels )
    prelim_dilute_pixels = prelim_gfp_intensity_pixels[ prelim_gfp_intensity_pixels < 2.*prelim_avg_gfp_intensity]
    mean_prelim_dilute_pixels = np.mean( prelim_dilute_pixels )
    std_prelim_dilute_pixels = np.std( prelim_gfp_intensity_pixels[ prelim_gfp_intensity_pixels < 2.*prelim_avg_gfp_intensity])
    # first find holes - use set intensity threshold. check that holes do not account for more than X% of total cell area
    # then get prelim average intensity outside of holes


    # find condensates - use intensity threshold relative to prelim average intensity outside holes
    #condensate_threshold = max(1.5*prelim_avg_dilute_intensity,condensate_cutoff_intensity)
    #condensate_threshold = max(prelim_avg_gfp_intensity+THRESH_STDS*std_prelim_dilute_pixels,
    condensate_threshold = max(mean_prelim_dilute_pixels+THRESH_STDS*std_prelim_dilute_pixels,
                               condensate_cutoff_intensity)
    condensate_mask = intensity_image > condensate_threshold


    # fill holes in the condensates
    filled_cond = ndi.binary_fill_holes(condensate_mask)
    difference = skimage.measure.label(filled_cond!=condensate_mask)
    change = ops.process.filter_by_region(difference, lambda r: r.area < 4, 0) > 0
    filled_cond_mask = condensate_mask
    filled_cond_mask[change] = filled_cond[change]

    min_size_c = 4
    filled_cond_mask_nosmall = skimage.morphology.remove_small_objects(filled_cond_mask, min_size=min_size_c)


    # segment the condensates
    condensates_ = ops.process.apply_watershed(filled_cond_mask_nosmall, smooth=1)
    # get rid of "condensates" that touch the edge of the mask - this is necessary to get rid of 
    # things that are not really inside the cell
    mask_border = skimage.segmentation.expand_labels( np.invert( image ) )
    mask_border = np.invert( mask_border )
    # clear any condensates touching the border of the image
    condensates = skimage.segmentation.clear_border(condensates_)
    # clear any condensates touching the border of the mask
    condensates = skimage.segmentation.clear_border(condensates, mask=mask_border)

    # optionally save the condensates to a file
    if save_file != '':
        save( save_file, condensates )


#    # test another way of segmenting condensates
#    # try to threshold the image into foreground and background?
#    intensity_image_hist = np.histogram( intensity_image[ intensity_image > 0], bins=256 )
#    threshold_otsu = skimage.filters.threshold_otsu(  hist=intensity_image_hist[0], nbins=256 )
#    #print( intensity_image_hist[1][threshold_otsu], condensate_threshold )
#
#
#    # add a more conservative measure of the average diltue intensity 
#    # if the threshold standard deviations for identifying condensates = 0 (so super liberal definition of condensates)
#    # the call the remaining pixels (except "hole" pixels) the dilute phase -- get the mean and standard deviation of these pixels
#    # this will be the "conservative" dilute phase intensity
#    liberal_condensate_threshold = max( mean_prelim_dilute_pixels+1.0*std_prelim_dilute_pixels, condensate_cutoff_intensity )
#    conservative_dilute_mean = np.mean(prelim_gfp_intensity_pixels[prelim_gfp_intensity_pixels < liberal_condensate_threshold])
#    #conservative_dilute_mean = np.mean(prelim_gfp_intensity_pixels[prelim_gfp_intensity_pixels < mean_prelim_dilute_pixels])

    # do GLCM calculation
    scale16to8 = np.power(2,8) / np.power(2,16)
    scaled_intensity_image = intensity_image * scale16to8
    sclaed_intensity_image = np.round( scaled_intensity_image )
    scaled_intensity_image = scaled_intensity_image.astype( "uint8" )

    # remove the 0th row and 0th column of the co-occurrence matrix to get rid of the values for the masked pixels
    glcm = skimage.feature.greycomatrix( scaled_intensity_image, distances=[5], 
            angles=[0], levels=256, symmetric=True, normed=True )[1:, 1:, :, :]
    glcm_dissim = skimage.feature.greycoprops( glcm, 'dissimilarity' )[0,0]
    glcm_contrast = skimage.feature.greycoprops( glcm, 'contrast' )[0,0]
    glcm_homogeneity = skimage.feature.greycoprops( glcm, 'homogeneity' )[0,0]
    glcm_correlation = skimage.feature.greycoprops( glcm, 'correlation' )[0,0]
    glcm_energy = skimage.feature.greycoprops( glcm, 'energy' )[0,0]

    if plot:
        fig, ax = plt.subplots(1,13,figsize=(10, 2))
        for i in range(13):
            ax[i].set_axis_off()

        #ax[0].imshow( intensity_image, cmap='gray', vmin=0, vmax=1500)
        ax[0].imshow( intensity_image, cmap='gray', vmin=0, vmax=3000, interpolation='none')
        ax[0].set_title( label, fontsize=5)
        ax[1].imshow( hole_mask, cmap='gray',interpolation='none')
        ax[1].set_title('hole_mask', fontsize=5)

        ax[2].imshow( filled_hole_mask, cmap='gray', interpolation='none')
        ax[2].set_title('filled_hole_mask', fontsize=5)
        ax[3].imshow( filled_hole_mask_filt, cmap='gray', interpolation='none')
        ax[3].set_title('filled_hole_mask_filt', fontsize=5)
        ax[4].imshow( final_filled_hole_mask_filt, cmap='gray', interpolation='none')
        ax[4].set_title('final_filled_hole_mask_filt', fontsize=5)
        ax[5].imshow( condensate_mask, cmap='gray', interpolation='none')
        ax[5].set_title( 'condensate_mask', fontsize=5)
        ax[6].imshow( filled_cond_mask, cmap='gray', interpolation='none')
        ax[6].set_title( 'filled_cond_mask', fontsize=5)
        ax[7].imshow( filled_cond_mask_nosmall, cmap='gray', interpolation='none')
        ax[7].set_title( 'filled_cond_mask_nosmall', fontsize=5)
        ax[8].imshow( condensates_, cmap='gray', interpolation='none')
        ax[8].set_title( 'condensates_', fontsize=5)
        ax[9].imshow( color.label2rgb(condensates, bg_label=0), cmap='gray', interpolation='none')
        #ax[9].imshow( condensates, cmap='gray', interpolation='none')
        ax[9].set_title( 'condensates', fontsize=5)

#        ax[10].set_title( 'glcm dissim %0.2f' %(glcm_dissim) )
#        ax[10].imshow( intensity_image, cmap='gray', vmin=0, vmax=3000, interpolation='none')
#        blobs = skimage.feature.blob_log( intensity_image, min_sigma=1, max_sigma=2, num_sigma=2, threshold=0.0025 )
#        #blobs = skimage.feature.blob_log( intensity_image, min_sigma=1, max_sigma=10, threshold=0.0025 )
#        #blobs = skimage.feature.blob_log( intensity_image, min_sigma=1, max_sigma=10, threshold=0.005 )
#        for blob in blobs:
#            y, x, r = blob
#            c = plt.Circle((x,y), r, color='red', linewidth=1, fill=False )
#            ax[10].add_patch( c )
#        # find local peaks?
#        peaks = skimage.feature.peak_local_max( intensity_image, threshold_abs=condensate_threshold, 
#                footprint = np.ones((20,20)), labels=image )
#        ax[11].imshow( intensity_image, cmap='gray', vmin=0, vmax=3000, interpolation='none')
#        for peak in peaks:
#            y, x = peak
#            c = plt.Circle((x,y), 2, color='red', linewidth=0.5, fill=False )
#            ax[11].add_patch( c )
#
#        ax[12].imshow( intensity_image > intensity_image_hist[1][threshold_otsu], cmap='gray', interpolation='none' )
#        #ax[12].imshow( final_filled_hole_mask_filt & (intensity_image < conservative_dilute_mean) )


        #plt.savefig('test_view_condensates.pdf')
#        plt.savefig(f'test_view_condensates_{label}.pdf')
        plt.show()

    #############
    # calculate condensate properties
    #############
    # for computing dilute phase: exclude all condensates including those that touch the edges of the cell 
    # (which are excluded from the final actual condensates)
    condensates_labeled, condensates_properties_dict = get_condensate_properties( condensates,
                                                            image, intensity_image, label,
                                                            final_filled_hole_mask_filt, plot=plot,
                                                            condensates_for_dilute=condensates_)

    # add the glcm metrics to the condensate properties dict 
    condensates_properties_dict['glcm_dissim'] = glcm_dissim
    condensates_properties_dict['glcm_contrast'] = glcm_contrast
    condensates_properties_dict['glcm_homogeneity'] = glcm_homogeneity
    condensates_properties_dict['glcm_correlation'] = glcm_correlation
    condensates_properties_dict['glcm_energy'] = glcm_energy

    return (condensates, condensates_labeled, condensates_properties_dict)


def get_condensates_nucleoli_explicit(image,
                                     mean_intensity,
                                     area,
                                     intensity_image,
                                     label,
                                     condensate_cutoff_intensity, plot=True, save_file=''):
    condensates = get_nucleoli_mask_explicit( image, mean_intensity, area,
                                            intensity_image, label,
                                             condensate_cutoff_intensity, plot)
    # segment the condensates
    #print( np.unique(condensates ))
    condensates = ops.process.apply_watershed(condensates, smooth=8)
    #print( np.unique(condensates) )

    if save_file != '':
        save( save_file, condensates )

    # there shouldn't be any holes that we need to exclude from dilute phase concentration calculations
    final_filled_hole_mask_filt = image

    condensates_labeled, condensates_properties_dict = get_condensate_properties( condensates,
                                                            image, intensity_image, label,
                                                            final_filled_hole_mask_filt, plot=plot)

    return (condensates, condensates_labeled, condensates_properties_dict)

def get_condensate_overlap( condensates_1, condensates_2, plot=True ):
    overlap_1_2 = (condensates_1 > 0) & (condensates_2 > 0)

    # fractional overlap: fraction of GFP condensate area that overlaps with NPM
    # GFP condensate pixels that overlap with NPM / total GFP condensate pixels
    total_overlap_area = np.sum( overlap_1_2 > 0)
    fraction_1_2_overlap = total_overlap_area / np.sum( condensates_1 > 0 )

    if plot:
        fig2, ax2 = plt.subplots(1,3,figsize=(5, 5))
        for i in range(3):
            ax2[i].set_axis_off()
        ax2[0].imshow( color.label2rgb(condensates_1, bg_label=0))
        ax2[1].imshow( color.label2rgb(condensates_2, bg_label=0))
        #ax2[2].imshow( region_image_coilin, cmap='Reds', vmin=0, vmax=5000, alpha=0.5)
        #ax2[2].imshow( region_image_GFP, cmap='Greens', vmin=0, vmax=5000, alpha=0.5)
        ax2[2].imshow( color.label2rgb(overlap_1_2, bg_label=0))
        plt.show()
        print( total_overlap_area, fraction_1_2_overlap )

    return total_overlap_area, fraction_1_2_overlap

def phenotype_phenix_4channel_NPM_coilin( fname_dapi, fname_gfp, fname_npm, fname_coilin,
                                        ffc_dapi, ffc_gfp, ffc_npm, ffc_coilin,
                                         well_num, tile_num, 
                                         file_save_dir,
                                         min_size=1000,
                        smooth_size=5, threshold_initial_guess=500,
                        nuclei_smooth_value=20, area_min=5000, plot=False,
                        area_max=20000, condensate_cutoff_intensity=2000,
                        THRESH_STDS=5, cellpose=True, cellpose_diameter=87, 
                        nuclei_mask_file=''):
    ##################
    # segment nuclei from dapi image
    ##################
    dapi_image = read( fname_dapi )
    gfp_image = read( fname_gfp )
    npm_image = read( fname_npm )
    coilin_image = read( fname_coilin )

    # apply flatfield correction
    dapi_image = apply_flatfield_correction( dapi_image, ffc_dapi )
    gfp_image = apply_flatfield_correction( gfp_image, ffc_gfp )
    npm_image = apply_flatfield_correction( npm_image, ffc_npm )
    coilin_image = apply_flatfield_correction( coilin_image, ffc_coilin )

    if nuclei_mask_file != '':
        final_nuclei = read( nuclei_mask_file )
    else:
    
        if cellpose:
            dapi_rgb = prepare_png_cellpose( dapi_image )
            final_nuclei = segment_nuclei_phenotype_cellpose( dapi_rgb, cellpose_diameter )
        else:
            final_nuclei = segment_nuclei_phenotype( dapi_image, threshold_initial_guess, smooth_size,
                        nuclei_smooth_value, min_size, area_min, area_max, plot)

    ##################
    # go through the nuclei and try to find condensates, calculate relevant properties
    ##################

    nuclei_dict = {
        'nucleus_num': [],
        'well': [],
        'tile': [],
        'fname_dapi': [],
        'total_nucleus_area': [],
        'mean_dapi_intensity': [],

        'mean_GFP_intensity_GFP': [],
        'std_GFP_intensity_GFP': [],
        'num_condensates_GFP': [],
        'mean_dilute_intensity_GFP': [],
        'std_dilute_intensity_GFP': [],
        'mean_condensate_intensity_GFP': [],
        'mean_condensate_intensity_no_shrink_GFP': [],
        'std_condensate_intensity_GFP': [],
        'std_condensate_intensity_no_shrink_GFP': [],
        'total_cond_intensity_GFP': [],
        'total_gfp_intensity_GFP': [],
        'frac_gfp_in_cond_GFP': [],
        'total_cond_area_GFP': [],
        'mean_condensate_area': [],
        'std_condensate_area': [],
        'mean_condensate_eccentricity': [],
        'std_condensate_eccentricity': [],
        'glcm_energy': [],
        'glcm_correlation': [],
        'glcm_dissim': [],
        'glcm_homogeneity': [],
        'glcm_contrast': [],


        'mean_GFP_intensity_npm': [],
        'std_GFP_intensity_npm': [],
        'num_condensates_npm': [],
        'mean_dilute_intensity_npm': [],
        'std_dilute_intensity_npm': [],
        'mean_condensate_intensity_npm': [],
        'mean_condensate_intensity_no_shrink_npm': [],
        'std_condensate_intensity_npm': [],
        'std_condensate_intensity_no_shrink_npm': [],
        'total_cond_intensity_npm': [],
        'total_gfp_intensity_npm': [],
        'frac_gfp_in_cond_npm': [],
        'total_cond_area_npm': [],
        'mean_condensate_area_npm': [],
        'std_condensate_area_npm': [],
        'mean_condensate_eccentricity_npm': [],
        'std_condensate_eccentricity_npm': [],


        'mean_GFP_intensity_coilin': [],
        'std_GFP_intensity_coilin': [],
        'num_condensates_coilin': [],
        'mean_dilute_intensity_coilin': [],
        'std_dilute_intensity_coilin': [],
        'mean_condensate_intensity_coilin': [],
        'mean_condensate_intensity_no_shrink_coilin': [],
        'std_condensate_intensity_coilin': [],
        'std_condensate_intensity_no_shrink_coilin': [],
        'total_cond_intensity_coilin': [],
        'total_gfp_intensity_coilin': [],
        'frac_gfp_in_cond_coilin': [],
        'total_cond_area_coilin': [],
        'mean_condensate_area_coilin': [],
        'std_condensate_area_coilin': [],
        'mean_condensate_eccentricity_coilin': [],
        'std_condensate_eccentricity_coilin': [],
        'glcm_energy_coilin': [],
        'glcm_correlation_coilin': [],
        'glcm_dissim_coilin': [],
        'glcm_homogeneity_coilin': [],
        'glcm_contrast_coilin': [],

        'overlap_area_gfp_npm': [],
        'fraction_gfp_npm_overlap': [],
        'overlap_area_gfp_coilin': [],
        'fraction_gfp_coilin_overlap': [],
        'overlap_area_coilin_npm': [],
        'fraction_coilin_npm_overlap': [],

        'cell_img_gfp_file': [],
        'cell_img_dapi_file': [],
        'cell_img_npm_file': [],
        'cell_img_coilin_file': [],
        'cell_img_mask_file': [],
        'corr_GFP_coilin': [],
        'corr_GFP_npm': [],
        'corr_npm_coilin': [],
        'corr_GFP_dapi': [],

    }
    num_nuclei = len(np.unique(final_nuclei) ) -1
    if num_nuclei < 1:
        df_nuclei = pd.DataFrame(data=nuclei_dict)
        return df_nuclei, final_nuclei


    properties_dapi = skimage.measure.regionprops_table( final_nuclei, intensity_image=dapi_image,
                                        properties=('label','mean_intensity','intensity_image'))
    properties_gfp = skimage.measure.regionprops_table( final_nuclei, intensity_image=gfp_image,
                                        properties=('label','mean_intensity','max_intensity','intensity_image',
                                                   'image','area','bbox'))
    properties_npm = skimage.measure.regionprops_table( final_nuclei, intensity_image=npm_image,
                                        properties=('label','mean_intensity','max_intensity','intensity_image',
                                                   'image','area','bbox'))
    properties_coilin = skimage.measure.regionprops_table( final_nuclei, intensity_image=coilin_image,
                                        properties=('label','mean_intensity','max_intensity','intensity_image',
                                                   'image','area','bbox'))

    for nucleus_index in range( len( properties_gfp['label'])):
        nucleus_image = properties_gfp['image'][nucleus_index]
        nucleus_area = properties_gfp['area'][nucleus_index]
        nucleus_label = properties_gfp['label'][nucleus_index]
        mean_dapi_intensity = properties_dapi['mean_intensity'][nucleus_index]

        region_image_dapi = properties_dapi['intensity_image'][nucleus_index]
        region_pixels_dapi = region_image_dapi[nucleus_image]

        mean_intensity_GFP = properties_gfp['mean_intensity'][nucleus_index]
        max_intensity_GFP = properties_gfp['max_intensity'][nucleus_index]
        region_image_GFP = properties_gfp['intensity_image'][nucleus_index]
        region_pixels_GFP = region_image_GFP[nucleus_image]
        std_intensity_GFP = np.std( region_pixels_GFP )
        total_intensity_GFP = mean_intensity_GFP * nucleus_area

        mean_intensity_npm = properties_npm['mean_intensity'][nucleus_index]
        max_intensity_npm = properties_npm['max_intensity'][nucleus_index]
        region_image_npm = properties_npm['intensity_image'][nucleus_index]
        region_pixels_npm = region_image_npm[nucleus_image]
        std_intensity_npm = np.std( region_pixels_npm )
        total_intensity_npm = mean_intensity_npm * nucleus_area

        mean_intensity_coilin = properties_coilin['mean_intensity'][nucleus_index]
        max_intensity_coilin = properties_coilin['max_intensity'][nucleus_index]
        region_image_coilin = properties_coilin['intensity_image'][nucleus_index]
        region_pixels_coilin = region_image_coilin[nucleus_image]
        std_intensity_coilin = np.std( region_pixels_coilin )
        total_intensity_coilin = mean_intensity_coilin * nucleus_area

        correlation_GFP_dapi = pearsonr( region_pixels_GFP, region_pixels_dapi )[0]
        correlation_GFP_coilin = pearsonr( region_pixels_GFP, region_pixels_coilin )[0]
        correlation_GFP_npm = pearsonr( region_pixels_GFP, region_pixels_npm )[0]
        correlation_npm_coilin = pearsonr( region_pixels_npm, region_pixels_coilin )[0]

        bbox = (properties_gfp['bbox-0'][nucleus_index],
                properties_gfp['bbox-1'][nucleus_index],
                properties_gfp['bbox-2'][nucleus_index],
                properties_gfp['bbox-3'][nucleus_index])

        unmasked_region_gfp_image = gfp_image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        unmasked_region_dapi_image = dapi_image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        unmasked_region_npm_image = npm_image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        unmasked_region_coilin_image = coilin_image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        mask_region_nucleus = properties_gfp['image'][nucleus_index]


        output_fname_dapi = '{file_save_dir}/well{well_num}_field{tile_num}_cell{cell_num}_DAPI.tif'.format(
            file_save_dir=file_save_dir,
            cell_num=nucleus_label, well_num=well_num, tile_num=tile_num)

        output_fname_gfp = '{file_save_dir}/well{well_num}_field{tile_num}_cell{cell_num}_GFP.tif'.format(
            file_save_dir=file_save_dir,
            cell_num=nucleus_label, well_num=well_num, tile_num=tile_num)

        output_fname_npm = '{file_save_dir}/well{well_num}_field{tile_num}_cell{cell_num}_npm.tif'.format(
            file_save_dir=file_save_dir,
            cell_num=nucleus_label, well_num=well_num, tile_num=tile_num)

        output_fname_coilin = '{file_save_dir}/well{well_num}_field{tile_num}_cell{cell_num}_coilin.tif'.format(
            file_save_dir=file_save_dir,
            cell_num=nucleus_label, well_num=well_num, tile_num=tile_num)

        output_fname_condensates = '{file_save_dir}/well{well_num}_field{tile_num}_cell{cell_num}_condensates.tif'.format(
            file_save_dir=file_save_dir,
            cell_num=nucleus_label, well_num=well_num, tile_num=tile_num)

        output_fname_condensates_coilin = '{file_save_dir}/well{well_num}_field{tile_num}_cell{cell_num}_coilin_condensates.tif'.format(
            file_save_dir=file_save_dir,
            cell_num=nucleus_label, well_num=well_num, tile_num=tile_num)

        output_fname_condensates_npm = '{file_save_dir}/well{well_num}_field{tile_num}_cell{cell_num}_npm_condensates.tif'.format(
            file_save_dir=file_save_dir,
            cell_num=nucleus_label, well_num=well_num, tile_num=tile_num)

        output_fname_mask = '{file_save_dir}/well{well_num}_field{tile_num}_cell{cell_num}_mask.npy'.format(
            file_save_dir=file_save_dir,
            cell_num=nucleus_label, well_num=well_num, tile_num=tile_num)

        save( output_fname_dapi, unmasked_region_dapi_image, compress=1)
        save( output_fname_gfp, unmasked_region_gfp_image, compress=1)
        save( output_fname_npm, unmasked_region_npm_image, compress=1)
        save( output_fname_coilin, unmasked_region_coilin_image, compress=1)
        np.save( output_fname_mask, mask_region_nucleus )


        ############
        # Get GFP condensates
        ############

        # if intensity is below some threshold, do not look for condensates (if GFP is not actually expressed)
#         if mean_intensity_GFP < 250. and max_intensity_GFP < 250.:
#             # this cell does not express GFP
#             num_condensates_GFP = 0
#             # condensate mask - should all be false
#             condensates_labeled_GFP = np.zeros( np.shape(nucleus.intensity_image))
#             mean_dilute_phase_intensity_GFP = mean_intensity
#             mean_condensate_intensity_GFP = np.nan
#             num_condensates_GFP = 0
#             std_dilute_phase_intensity_GFP = std_intensity
#             std_condensate_intensity_GFP = np.nan
#             total_condensate_intensity_GFP = 0
#             total_condensate_area_GFP = 0

#         else:
        # get condensates
        (condensates_GFP, condensates_labeled_GFP,
         condensates_properties_dict_GFP) = get_condensates_general_explicit( nucleus_image,
                                                                        mean_intensity_GFP,
                                                                        nucleus_area,
                                                                        region_image_GFP,
                                                                        nucleus_label,
                                                                        condensate_cutoff_intensity,
                                                                        THRESH_STDS,
                                                                        plot=plot,
                                                                        save_file=output_fname_condensates )


        ##############
        # Get NPM condensates
        ##############
        (condensates_npm, condensates_labeled_npm,
         condensates_properties_dict_npm) = get_condensates_nucleoli_explicit( nucleus_image,
                                                                            mean_intensity_npm,
                                                                            nucleus_area,
                                                                            region_image_npm,
                                                                            nucleus_label,
                                                                            condensate_cutoff_intensity,
                                                                            plot=plot,
                                                                            save_file=output_fname_condensates_npm )

        ############
        # Get coilin condensates
        ############
        (condensates_coilin, condensates_labeled_coilin,
             condensates_properties_dict_coilin) = get_condensates_general_explicit( nucleus_image,
                                                                            mean_intensity_coilin,
                                                                            nucleus_area,
                                                                            region_image_coilin,
                                                                            nucleus_label,
                                                                            condensate_cutoff_intensity,
                                                                            THRESH_STDS,
                                                                            plot=plot,
                                                                            save_file=output_fname_condensates_coilin )

        ############
        # Get the overlap between GFP and other condensates
        ############

        overlap_area_gfp_npm, fraction_gfp_npm_overlap = get_condensate_overlap( condensates_GFP,
                                                                                condensates_npm, plot=plot )
        overlap_area_gfp_coilin, fraction_gfp_coilin_overlap = get_condensate_overlap( condensates_GFP,
                                                                                     condensates_coilin, plot=plot)
        overlap_area_coilin_npm, fraction_coilin_npm_overlap = get_condensate_overlap( condensates_coilin,
                                                                                     condensates_npm, plot=plot )


        ############
        # add everything to dictionary
        ############

        nuclei_dict['nucleus_num'].append(nucleus_label)
        nuclei_dict['well'].append( well_num )
        nuclei_dict['tile'].append( tile_num )
        nuclei_dict['fname_dapi'].append( fname_dapi)
        nuclei_dict['total_nucleus_area'].append( nucleus_area )
        nuclei_dict['mean_dapi_intensity'].append( mean_dapi_intensity )

        nuclei_dict['mean_GFP_intensity_GFP'].append( mean_intensity_GFP )
        nuclei_dict['std_GFP_intensity_GFP'].append( std_intensity_GFP )
        nuclei_dict['num_condensates_GFP'].append( condensates_properties_dict_GFP['num_condensates'])
        nuclei_dict['mean_dilute_intensity_GFP'].append(
            condensates_properties_dict_GFP['mean_dilute_phase_intensity'])
        nuclei_dict['std_dilute_intensity_GFP'].append(
            condensates_properties_dict_GFP['std_dilute_phase_intensity'])
        nuclei_dict['mean_condensate_intensity_GFP'].append(
            condensates_properties_dict_GFP['mean_condensate_intensity'])
        nuclei_dict['mean_condensate_intensity_no_shrink_GFP'].append(
            condensates_properties_dict_GFP['mean_condensate_intensity_no_shrink'])
        nuclei_dict['std_condensate_intensity_GFP'].append(
            condensates_properties_dict_GFP['std_condensate_intensity'])
        nuclei_dict['std_condensate_intensity_no_shrink_GFP'].append(
            condensates_properties_dict_GFP['std_condensate_intensity_no_shrink'])
        nuclei_dict['total_cond_intensity_GFP'].append(
            condensates_properties_dict_GFP['total_condensate_intensity'])
        nuclei_dict['total_gfp_intensity_GFP'].append( total_intensity_GFP)
        nuclei_dict['frac_gfp_in_cond_GFP'].append(
            condensates_properties_dict_GFP['total_condensate_intensity'] / total_intensity_GFP )
        nuclei_dict['total_cond_area_GFP'].append(
            condensates_properties_dict_GFP['total_condensate_area'])
        nuclei_dict['mean_condensate_area'].append(
                condensates_properties_dict_GFP['mean_condensate_area'])
        nuclei_dict['std_condensate_area'].append(
                condensates_properties_dict_GFP['std_condensate_area'])
        nuclei_dict['mean_condensate_eccentricity'].append(
                condensates_properties_dict_GFP['mean_condensate_eccentricity'])
        nuclei_dict['std_condensate_eccentricity'].append(
                condensates_properties_dict_GFP['std_condensate_eccentricity'])
        nuclei_dict['glcm_contrast'].append(
                condensates_properties_dict_GFP['glcm_contrast'])
        nuclei_dict['glcm_dissim'].append(
                condensates_properties_dict_GFP['glcm_dissim'])
        nuclei_dict['glcm_energy'].append(
                condensates_properties_dict_GFP['glcm_energy'])
        nuclei_dict['glcm_correlation'].append(
                condensates_properties_dict_GFP['glcm_correlation'])
        nuclei_dict['glcm_homogeneity'].append(
                condensates_properties_dict_GFP['glcm_homogeneity'])


        nuclei_dict['mean_GFP_intensity_npm'].append( mean_intensity_npm )
        nuclei_dict['std_GFP_intensity_npm'].append( std_intensity_npm )
        nuclei_dict['num_condensates_npm'].append( condensates_properties_dict_npm['num_condensates'])
        nuclei_dict['mean_dilute_intensity_npm'].append(
            condensates_properties_dict_npm['mean_dilute_phase_intensity'])
        nuclei_dict['std_dilute_intensity_npm'].append(
            condensates_properties_dict_npm['std_dilute_phase_intensity'])
        nuclei_dict['mean_condensate_intensity_npm'].append(
            condensates_properties_dict_npm['mean_condensate_intensity'])
        nuclei_dict['mean_condensate_intensity_no_shrink_npm'].append(
            condensates_properties_dict_npm['mean_condensate_intensity_no_shrink'])
        nuclei_dict['std_condensate_intensity_npm'].append(
            condensates_properties_dict_npm['std_condensate_intensity'])
        nuclei_dict['std_condensate_intensity_no_shrink_npm'].append(
            condensates_properties_dict_npm['std_condensate_intensity_no_shrink'])
        nuclei_dict['total_cond_intensity_npm'].append(
            condensates_properties_dict_npm['total_condensate_intensity'])
        nuclei_dict['total_gfp_intensity_npm'].append( total_intensity_npm )
        nuclei_dict['frac_gfp_in_cond_npm'].append(
            condensates_properties_dict_npm['total_condensate_intensity'] / total_intensity_npm )
        nuclei_dict['total_cond_area_npm'].append(
            condensates_properties_dict_npm['total_condensate_area'])
        nuclei_dict['mean_condensate_area_npm'].append(
                condensates_properties_dict_npm['mean_condensate_area'])
        nuclei_dict['std_condensate_area_npm'].append(
                condensates_properties_dict_npm['std_condensate_area'])
        nuclei_dict['mean_condensate_eccentricity_npm'].append(
                condensates_properties_dict_npm['mean_condensate_eccentricity'])
        nuclei_dict['std_condensate_eccentricity_npm'].append(
                condensates_properties_dict_npm['std_condensate_eccentricity'])


        nuclei_dict['mean_GFP_intensity_coilin'].append( mean_intensity_coilin )
        nuclei_dict['std_GFP_intensity_coilin'].append( std_intensity_coilin )
        nuclei_dict['num_condensates_coilin'].append( condensates_properties_dict_coilin['num_condensates'])
        nuclei_dict['mean_dilute_intensity_coilin'].append(
            condensates_properties_dict_coilin['mean_dilute_phase_intensity'])
        nuclei_dict['std_dilute_intensity_coilin'].append(
            condensates_properties_dict_coilin['std_dilute_phase_intensity'])
        nuclei_dict['mean_condensate_intensity_coilin'].append(
            condensates_properties_dict_coilin['mean_condensate_intensity'])
        nuclei_dict['mean_condensate_intensity_no_shrink_coilin'].append(
            condensates_properties_dict_coilin['mean_condensate_intensity_no_shrink'])
        nuclei_dict['std_condensate_intensity_coilin'].append(
            condensates_properties_dict_coilin['std_condensate_intensity'])
        nuclei_dict['std_condensate_intensity_no_shrink_coilin'].append(
            condensates_properties_dict_coilin['std_condensate_intensity_no_shrink'])
        nuclei_dict['total_cond_intensity_coilin'].append(
            condensates_properties_dict_coilin['total_condensate_intensity'])
        nuclei_dict['total_gfp_intensity_coilin'].append( total_intensity_coilin )
        nuclei_dict['frac_gfp_in_cond_coilin'].append(
            condensates_properties_dict_coilin['total_condensate_intensity'] / total_intensity_coilin )
        nuclei_dict['total_cond_area_coilin'].append(
            condensates_properties_dict_coilin['total_condensate_area'])
        nuclei_dict['mean_condensate_area_coilin'].append(
                condensates_properties_dict_coilin['mean_condensate_area'])
        nuclei_dict['std_condensate_area_coilin'].append(
                condensates_properties_dict_coilin['std_condensate_area'])
        nuclei_dict['mean_condensate_eccentricity_coilin'].append(
                condensates_properties_dict_coilin['mean_condensate_eccentricity'])
        nuclei_dict['std_condensate_eccentricity_coilin'].append(
                condensates_properties_dict_coilin['std_condensate_eccentricity'])
        nuclei_dict['glcm_contrast_coilin'].append(
                condensates_properties_dict_coilin['glcm_contrast'])
        nuclei_dict['glcm_dissim_coilin'].append(
                condensates_properties_dict_coilin['glcm_dissim'])
        nuclei_dict['glcm_energy_coilin'].append(
                condensates_properties_dict_coilin['glcm_energy'])
        nuclei_dict['glcm_correlation_coilin'].append(
                condensates_properties_dict_coilin['glcm_correlation'])
        nuclei_dict['glcm_homogeneity_coilin'].append(
                condensates_properties_dict_coilin['glcm_homogeneity'])

        nuclei_dict['overlap_area_gfp_npm'].append( overlap_area_gfp_npm )
        nuclei_dict['overlap_area_gfp_coilin'].append( overlap_area_gfp_coilin )
        nuclei_dict['overlap_area_coilin_npm'].append( overlap_area_coilin_npm )
        nuclei_dict['fraction_gfp_npm_overlap'].append( fraction_gfp_npm_overlap )
        nuclei_dict['fraction_gfp_coilin_overlap'].append( fraction_gfp_coilin_overlap )
        nuclei_dict['fraction_coilin_npm_overlap'].append( fraction_coilin_npm_overlap )

        nuclei_dict['cell_img_gfp_file'].append( output_fname_gfp )
        nuclei_dict['cell_img_npm_file'].append( output_fname_npm )
        nuclei_dict['cell_img_coilin_file'].append( output_fname_coilin )
        nuclei_dict['cell_img_dapi_file'].append( output_fname_dapi )
        nuclei_dict['cell_img_mask_file'].append( output_fname_mask )
        nuclei_dict['corr_GFP_npm'].append( correlation_GFP_npm )
        nuclei_dict['corr_GFP_coilin'].append( correlation_GFP_coilin )
        nuclei_dict['corr_npm_coilin'].append( correlation_npm_coilin )
        nuclei_dict['corr_GFP_dapi'].append( correlation_GFP_dapi )


    df_nuclei = pd.DataFrame(data=nuclei_dict)
    return df_nuclei, final_nuclei

def phenotype_phenix_4channel_PML_SRRM2( fname_dapi, fname_gfp, fname_pml, fname_srrm2,
                                         ffc_dapi, ffc_gfp, ffc_pml, ffc_srrm2,
                                         well_num, tile_num, 
                                         file_save_dir,
                                         min_size=1000, 
                        smooth_size=5, threshold_initial_guess=500, 
                        nuclei_smooth_value=20, area_min=5000, plot=False,
                        area_max=20000, condensate_cutoff_intensity=2000,
                        PML_condensate_cutoff_intensity=1000,
                        THRESH_STDS=5,cellpose=True,cellpose_diameter=87, nuclei_mask_file=''):
    ##################
    # segment nuclei from dapi image
    ##################
    dapi_image = read( fname_dapi )
    gfp_image = read( fname_gfp )
    pml_image = read( fname_pml )
    srrm2_image = read( fname_srrm2 )

    # apply flatfield correction
    dapi_image = apply_flatfield_correction( dapi_image, ffc_dapi )
    gfp_image = apply_flatfield_correction( gfp_image, ffc_gfp )
    pml_image = apply_flatfield_correction( pml_image, ffc_pml )
    srrm2_image = apply_flatfield_correction( srrm2_image, ffc_srrm2 )
    
    if nuclei_mask_file != '':
        final_nuclei = read( nuclei_mask_file )
    else:
    
        if cellpose:
            dapi_rgb = prepare_png_cellpose( dapi_image )
            final_nuclei = segment_nuclei_phenotype_cellpose( dapi_rgb, cellpose_diameter )
        else:
            final_nuclei = segment_nuclei_phenotype( dapi_image, threshold_initial_guess, smooth_size,
                        nuclei_smooth_value, min_size, area_min, area_max, plot)


    ##################
    # go through the nuclei and try to find condensates, calculate relevant properties
    ##################
    
    nuclei_dict = { 
        'nucleus_num': [],
        'well': [],
        'tile': [],
        'fname_dapi': [],
        'total_nucleus_area': [],
        'mean_dapi_intensity': [],
        
        'mean_GFP_intensity_GFP': [],
        'std_GFP_intensity_GFP': [],
        'num_condensates_GFP': [],
        'mean_dilute_intensity_GFP': [],
        'std_dilute_intensity_GFP': [],
        'mean_condensate_intensity_GFP': [],
        'mean_condensate_intensity_no_shrink_GFP': [],
        'std_condensate_intensity_GFP': [],
        'std_condensate_intensity_no_shrink_GFP': [],
        'total_cond_intensity_GFP': [],
        'total_gfp_intensity_GFP': [],
        'frac_gfp_in_cond_GFP': [],
        'total_cond_area_GFP': [],
        'mean_condensate_area': [],
        'std_condensate_area': [],
        'mean_condensate_eccentricity': [],
        'std_condensate_eccentricity': [],
        'glcm_energy': [],
        'glcm_correlation': [],
        'glcm_dissim': [],
        'glcm_homogeneity': [],
        'glcm_contrast': [],
        
        'mean_GFP_intensity_pml': [],
        'std_GFP_intensity_pml': [],
        'num_condensates_pml': [],
        'mean_dilute_intensity_pml': [],
        'std_dilute_intensity_pml': [],
        'mean_condensate_intensity_pml': [],
        'mean_condensate_intensity_no_shrink_pml': [],
        'std_condensate_intensity_pml': [],
        'std_condensate_intensity_no_shrink_pml': [],
        'total_cond_intensity_pml': [],
        'total_gfp_intensity_pml': [],
        'frac_gfp_in_cond_pml': [],
        'total_cond_area_pml': [],
        'mean_condensate_area_pml': [],
        'std_condensate_area_pml': [],
        'mean_condensate_eccentricity_pml': [],
        'std_condensate_eccentricity_pml': [],
        'glcm_energy_pml': [],
        'glcm_correlation_pml': [],
        'glcm_dissim_pml': [],
        'glcm_homogeneity_pml': [],
        'glcm_contrast_pml': [],
        
        'mean_GFP_intensity_srrm2': [],
        'std_GFP_intensity_srrm2': [],
        'num_condensates_srrm2': [],
        'mean_dilute_intensity_srrm2': [],
        'std_dilute_intensity_srrm2': [],
        'mean_condensate_intensity_srrm2': [],
        'mean_condensate_intensity_no_shrink_srrm2': [],
        'std_condensate_intensity_srrm2': [],
        'std_condensate_intensity_no_shrink_srrm2': [],
        'total_cond_intensity_srrm2': [],
        'total_gfp_intensity_srrm2': [],
        'frac_gfp_in_cond_srrm2': [],
        'total_cond_area_srrm2': [],
        'mean_condensate_area_srrm2': [],
        'std_condensate_area_srrm2': [],
        'mean_condensate_eccentricity_srrm2': [],
        'std_condensate_eccentricity_srrm2': [],
        'glcm_energy_srrm2': [],
        'glcm_correlation_srrm2': [],
        'glcm_dissim_srrm2': [],
        'glcm_homogeneity_srrm2': [],
        'glcm_contrast_srrm2': [],
        
        'overlap_area_gfp_pml': [],
        'fraction_gfp_pml_overlap': [],
        'overlap_area_gfp_srrm2': [],
        'fraction_gfp_srrm2_overlap': [],
        'overlap_area_srrm2_pml': [],
        'fraction_srrm2_pml_overlap': [],

        'cell_img_gfp_file': [],
        'cell_img_dapi_file': [],
        'cell_img_pml_file': [],
        'cell_img_srrm2_file': [],
        'cell_img_mask_file': [],
        'corr_GFP_pml': [],
        'corr_GFP_srrm2': [],
        'corr_pml_srrm2': [],
        'corr_GFP_dapi': [],
        
    }
    num_nuclei = len(np.unique(final_nuclei) ) -1
    if num_nuclei < 1:
        df_nuclei = pd.DataFrame(data=nuclei_dict)
        return df_nuclei, final_nuclei
    
    properties_dapi = skimage.measure.regionprops_table( final_nuclei, intensity_image=dapi_image,
                                        properties=('label','mean_intensity','intensity_image'))
    properties_gfp = skimage.measure.regionprops_table( final_nuclei, intensity_image=gfp_image, 
                                        properties=('label','mean_intensity','max_intensity','intensity_image',
                                                   'image','area','bbox'))
    properties_pml = skimage.measure.regionprops_table( final_nuclei, intensity_image=pml_image, 
                                        properties=('label','mean_intensity','max_intensity','intensity_image',
                                                   'image','area','bbox'))
    properties_srrm2 = skimage.measure.regionprops_table( final_nuclei, intensity_image=srrm2_image, 
                                        properties=('label','mean_intensity','max_intensity','intensity_image',
                                                   'image','area','bbox'))
    
    for nucleus_index in range( len( properties_gfp['label'])):
        nucleus_image = properties_gfp['image'][nucleus_index]
        nucleus_area = properties_gfp['area'][nucleus_index]
        nucleus_label = properties_gfp['label'][nucleus_index]
        mean_dapi_intensity = properties_dapi['mean_intensity'][nucleus_index]
    
        region_image_dapi = properties_dapi['intensity_image'][nucleus_index]
        region_pixels_dapi = region_image_dapi[nucleus_image]

        mean_intensity_GFP = properties_gfp['mean_intensity'][nucleus_index]
        max_intensity_GFP = properties_gfp['max_intensity'][nucleus_index]
        region_image_GFP = properties_gfp['intensity_image'][nucleus_index]
        region_pixels_GFP = region_image_GFP[nucleus_image]
        std_intensity_GFP = np.std( region_pixels_GFP )
        total_intensity_GFP = mean_intensity_GFP * nucleus_area
        
        mean_intensity_pml = properties_pml['mean_intensity'][nucleus_index]
        max_intensity_pml = properties_pml['max_intensity'][nucleus_index]
        region_image_pml = properties_pml['intensity_image'][nucleus_index]
        region_pixels_pml = region_image_pml[nucleus_image]
        std_intensity_pml = np.std( region_pixels_pml )
        total_intensity_pml = mean_intensity_pml * nucleus_area
        
        mean_intensity_srrm2 = properties_srrm2['mean_intensity'][nucleus_index]
        max_intensity_srrm2 = properties_srrm2['max_intensity'][nucleus_index]
        region_image_srrm2 = properties_srrm2['intensity_image'][nucleus_index]
        region_pixels_srrm2 = region_image_srrm2[nucleus_image]
        std_intensity_srrm2 = np.std( region_pixels_srrm2 )
        total_intensity_srrm2 = mean_intensity_srrm2 * nucleus_area

        correlation_GFP_pml = pearsonr( region_pixels_GFP, region_pixels_pml )[0]
        correlation_GFP_srrm2 = pearsonr( region_pixels_GFP, region_pixels_srrm2 )[0]
        correlation_pml_srrm2 = pearsonr( region_pixels_pml, region_pixels_srrm2 )[0]
        correlation_GFP_dapi = pearsonr( region_pixels_GFP, region_pixels_dapi )[0]


        bbox = (properties_gfp['bbox-0'][nucleus_index],
                properties_gfp['bbox-1'][nucleus_index],
                properties_gfp['bbox-2'][nucleus_index],
                properties_gfp['bbox-3'][nucleus_index])

        unmasked_region_gfp_image = gfp_image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        unmasked_region_dapi_image = dapi_image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        unmasked_region_srrm2_image = srrm2_image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        unmasked_region_pml_image = pml_image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        mask_region_nucleus = properties_gfp['image'][nucleus_index]
        
        output_fname_dapi = '{file_save_dir}/well{well_num}_field{tile_num}_cell{cell_num}_DAPI.tif'.format(
            file_save_dir=file_save_dir,
            cell_num=nucleus_label, well_num=well_num, tile_num=tile_num)

        output_fname_gfp = '{file_save_dir}/well{well_num}_field{tile_num}_cell{cell_num}_GFP.tif'.format(
            file_save_dir=file_save_dir,
            cell_num=nucleus_label, well_num=well_num, tile_num=tile_num)

        output_fname_pml = '{file_save_dir}/well{well_num}_field{tile_num}_cell{cell_num}_pml.tif'.format(
            file_save_dir=file_save_dir,
            cell_num=nucleus_label, well_num=well_num, tile_num=tile_num)

        output_fname_srrm2 = '{file_save_dir}/well{well_num}_field{tile_num}_cell{cell_num}_srrm2.tif'.format(
            file_save_dir=file_save_dir,
            cell_num=nucleus_label, well_num=well_num, tile_num=tile_num)

        output_fname_condensates = '{file_save_dir}/well{well_num}_field{tile_num}_cell{cell_num}_condensates.tif'.format(
            file_save_dir=file_save_dir,
            cell_num=nucleus_label, well_num=well_num, tile_num=tile_num)

        output_fname_condensates_srrm2 = '{file_save_dir}/well{well_num}_field{tile_num}_cell{cell_num}_srrm2_condensates.tif'.format(
            file_save_dir=file_save_dir,
            cell_num=nucleus_label, well_num=well_num, tile_num=tile_num)

        output_fname_condensates_pml = '{file_save_dir}/well{well_num}_field{tile_num}_cell{cell_num}_pml_condensates.tif'.format(
            file_save_dir=file_save_dir,
            cell_num=nucleus_label, well_num=well_num, tile_num=tile_num)

        output_fname_mask = '{file_save_dir}/well{well_num}_field{tile_num}_cell{cell_num}_mask.npy'.format(
            file_save_dir=file_save_dir,
            cell_num=nucleus_label, well_num=well_num, tile_num=tile_num)

        save( output_fname_dapi, unmasked_region_dapi_image, compress=1)
        save( output_fname_gfp, unmasked_region_gfp_image, compress=1)
        save( output_fname_pml, unmasked_region_pml_image, compress=1)
        save( output_fname_srrm2, unmasked_region_srrm2_image, compress=1)
        np.save( output_fname_mask, mask_region_nucleus )


        ############
        # Get GFP condensates
        ############

        # if intensity is below some threshold, do not look for condensates (if GFP is not actually expressed)
#         if mean_intensity_GFP < 250. and max_intensity_GFP < 250.:
#             # this cell does not express GFP
#             num_condensates_GFP = 0
#             # condensate mask - should all be false
#             condensates_labeled_GFP = np.zeros( np.shape(nucleus.intensity_image))
#             mean_dilute_phase_intensity_GFP = mean_intensity
#             mean_condensate_intensity_GFP = np.nan
#             num_condensates_GFP = 0
#             std_dilute_phase_intensity_GFP = std_intensity
#             std_condensate_intensity_GFP = np.nan
#             total_condensate_intensity_GFP = 0
#             total_condensate_area_GFP = 0

#         else:
        # get condensates 
        (condensates_GFP, condensates_labeled_GFP, 
         condensates_properties_dict_GFP) = get_condensates_general_explicit( nucleus_image,
                                                                        mean_intensity_GFP,
                                                                        nucleus_area,
                                                                        region_image_GFP,
                                                                        nucleus_label,
                                                                        condensate_cutoff_intensity, 
                                                                        THRESH_STDS,
                                                                        plot=plot,
                                                                        save_file=output_fname_condensates )

            
        ##############
        # Get PML condensates
        ##############
        (condensates_pml, condensates_labeled_pml, 
         condensates_properties_dict_pml) = get_condensates_general_explicit( nucleus_image,
                                                                            mean_intensity_pml,
                                                                            nucleus_area,
                                                                            region_image_pml,
                                                                            nucleus_label,
                                                                            PML_condensate_cutoff_intensity, 
                                                                            THRESH_STDS,
                                                                            plot=plot,
                                                                            save_file=output_fname_condensates_pml )
        
        ############
        # Get srrm2 condensates
        ############
        (condensates_srrm2, condensates_labeled_srrm2, 
             condensates_properties_dict_srrm2) = get_condensates_general_explicit( nucleus_image,
                                                                            mean_intensity_srrm2,
                                                                            nucleus_area,
                                                                            region_image_srrm2,
                                                                            nucleus_label,
                                                                            condensate_cutoff_intensity, 
                                                                            THRESH_STDS,
                                                                            plot=plot,
                                                                            save_file=output_fname_condensates_srrm2 )
        
        ############
        # Get the overlap between GFP and other condensates
        ############
        
        overlap_area_gfp_pml, fraction_gfp_pml_overlap = get_condensate_overlap( condensates_GFP, 
                                                                                condensates_pml, plot=plot )
        overlap_area_gfp_srrm2, fraction_gfp_srrm2_overlap = get_condensate_overlap( condensates_GFP, 
                                                                                     condensates_srrm2, plot=plot)
        overlap_area_srrm2_pml, fraction_srrm2_pml_overlap = get_condensate_overlap( condensates_srrm2,
                                                                                     condensates_pml, plot=plot )
    
        ############
        # add everything to dictionary
        ############
        
        nuclei_dict['nucleus_num'].append(nucleus_label)
        nuclei_dict['well'].append( well_num )
        nuclei_dict['tile'].append( tile_num )
        nuclei_dict['fname_dapi'].append( fname_dapi)
        nuclei_dict['total_nucleus_area'].append( nucleus_area )
        nuclei_dict['mean_dapi_intensity'].append( mean_dapi_intensity )
        
        nuclei_dict['mean_GFP_intensity_GFP'].append( mean_intensity_GFP )
        nuclei_dict['std_GFP_intensity_GFP'].append( std_intensity_GFP )
        nuclei_dict['num_condensates_GFP'].append( condensates_properties_dict_GFP['num_condensates'])
        nuclei_dict['mean_dilute_intensity_GFP'].append( 
            condensates_properties_dict_GFP['mean_dilute_phase_intensity'])
        nuclei_dict['std_dilute_intensity_GFP'].append(
            condensates_properties_dict_GFP['std_dilute_phase_intensity'])
        nuclei_dict['mean_condensate_intensity_GFP'].append(
            condensates_properties_dict_GFP['mean_condensate_intensity'])
        nuclei_dict['mean_condensate_intensity_no_shrink_GFP'].append( 
            condensates_properties_dict_GFP['mean_condensate_intensity_no_shrink'])
        nuclei_dict['std_condensate_intensity_GFP'].append(
            condensates_properties_dict_GFP['std_condensate_intensity'])
        nuclei_dict['std_condensate_intensity_no_shrink_GFP'].append( 
            condensates_properties_dict_GFP['std_condensate_intensity_no_shrink'])
        nuclei_dict['total_cond_intensity_GFP'].append(
            condensates_properties_dict_GFP['total_condensate_intensity'])
        nuclei_dict['total_gfp_intensity_GFP'].append( total_intensity_GFP)
        nuclei_dict['frac_gfp_in_cond_GFP'].append( 
            condensates_properties_dict_GFP['total_condensate_intensity'] / total_intensity_GFP )
        nuclei_dict['total_cond_area_GFP'].append( 
            condensates_properties_dict_GFP['total_condensate_area'])
        nuclei_dict['mean_condensate_area'].append(
                condensates_properties_dict_GFP['mean_condensate_area'])
        nuclei_dict['std_condensate_area'].append(
                condensates_properties_dict_GFP['std_condensate_area'])
        nuclei_dict['mean_condensate_eccentricity'].append(
                condensates_properties_dict_GFP['mean_condensate_eccentricity'])
        nuclei_dict['std_condensate_eccentricity'].append(
                condensates_properties_dict_GFP['std_condensate_eccentricity'])
        nuclei_dict['glcm_contrast'].append(
                condensates_properties_dict_GFP['glcm_contrast'])
        nuclei_dict['glcm_dissim'].append(
                condensates_properties_dict_GFP['glcm_dissim'])
        nuclei_dict['glcm_energy'].append(
                condensates_properties_dict_GFP['glcm_energy'])
        nuclei_dict['glcm_correlation'].append(
                condensates_properties_dict_GFP['glcm_correlation'])
        nuclei_dict['glcm_homogeneity'].append(
                condensates_properties_dict_GFP['glcm_homogeneity'])
        

        nuclei_dict['mean_GFP_intensity_pml'].append( mean_intensity_pml )
        nuclei_dict['std_GFP_intensity_pml'].append( std_intensity_pml )
        nuclei_dict['num_condensates_pml'].append( condensates_properties_dict_pml['num_condensates'])
        nuclei_dict['mean_dilute_intensity_pml'].append( 
            condensates_properties_dict_pml['mean_dilute_phase_intensity'])
        nuclei_dict['std_dilute_intensity_pml'].append(
            condensates_properties_dict_pml['std_dilute_phase_intensity'])
        nuclei_dict['mean_condensate_intensity_pml'].append(
            condensates_properties_dict_pml['mean_condensate_intensity'])
        nuclei_dict['mean_condensate_intensity_no_shrink_pml'].append( 
            condensates_properties_dict_pml['mean_condensate_intensity_no_shrink'])
        nuclei_dict['std_condensate_intensity_pml'].append(
            condensates_properties_dict_pml['std_condensate_intensity'])
        nuclei_dict['std_condensate_intensity_no_shrink_pml'].append( 
            condensates_properties_dict_pml['std_condensate_intensity_no_shrink'])
        nuclei_dict['total_cond_intensity_pml'].append(
            condensates_properties_dict_pml['total_condensate_intensity'])
        nuclei_dict['total_gfp_intensity_pml'].append( total_intensity_pml )
        nuclei_dict['frac_gfp_in_cond_pml'].append( 
            condensates_properties_dict_pml['total_condensate_intensity'] / total_intensity_pml )
        nuclei_dict['total_cond_area_pml'].append( 
            condensates_properties_dict_pml['total_condensate_area'])
        nuclei_dict['mean_condensate_area_pml'].append(
                condensates_properties_dict_pml['mean_condensate_area'])
        nuclei_dict['std_condensate_area_pml'].append(
                condensates_properties_dict_pml['std_condensate_area'])
        nuclei_dict['mean_condensate_eccentricity_pml'].append(
                condensates_properties_dict_pml['mean_condensate_eccentricity'])
        nuclei_dict['std_condensate_eccentricity_pml'].append(
                condensates_properties_dict_pml['std_condensate_eccentricity'])
        nuclei_dict['glcm_contrast_pml'].append(
                condensates_properties_dict_pml['glcm_contrast'])
        nuclei_dict['glcm_dissim_pml'].append(
                condensates_properties_dict_pml['glcm_dissim'])
        nuclei_dict['glcm_energy_pml'].append(
                condensates_properties_dict_pml['glcm_energy'])
        nuclei_dict['glcm_correlation_pml'].append(
                condensates_properties_dict_pml['glcm_correlation'])
        nuclei_dict['glcm_homogeneity_pml'].append(
                condensates_properties_dict_pml['glcm_homogeneity'])
        
        
        nuclei_dict['mean_GFP_intensity_srrm2'].append( mean_intensity_srrm2 )
        nuclei_dict['std_GFP_intensity_srrm2'].append( std_intensity_srrm2 )
        nuclei_dict['num_condensates_srrm2'].append( condensates_properties_dict_srrm2['num_condensates'])
        nuclei_dict['mean_dilute_intensity_srrm2'].append( 
            condensates_properties_dict_srrm2['mean_dilute_phase_intensity'])
        nuclei_dict['std_dilute_intensity_srrm2'].append(
            condensates_properties_dict_srrm2['std_dilute_phase_intensity'])
        nuclei_dict['mean_condensate_intensity_srrm2'].append(
            condensates_properties_dict_srrm2['mean_condensate_intensity'])
        nuclei_dict['mean_condensate_intensity_no_shrink_srrm2'].append( 
            condensates_properties_dict_srrm2['mean_condensate_intensity_no_shrink'])
        nuclei_dict['std_condensate_intensity_srrm2'].append(
            condensates_properties_dict_srrm2['std_condensate_intensity'])
        nuclei_dict['std_condensate_intensity_no_shrink_srrm2'].append( 
            condensates_properties_dict_srrm2['std_condensate_intensity_no_shrink'])
        nuclei_dict['total_cond_intensity_srrm2'].append(
            condensates_properties_dict_srrm2['total_condensate_intensity'])
        nuclei_dict['total_gfp_intensity_srrm2'].append( total_intensity_srrm2 )
        nuclei_dict['frac_gfp_in_cond_srrm2'].append( 
            condensates_properties_dict_srrm2['total_condensate_intensity'] / total_intensity_srrm2 )
        nuclei_dict['total_cond_area_srrm2'].append( 
            condensates_properties_dict_srrm2['total_condensate_area'])
        nuclei_dict['mean_condensate_area_srrm2'].append(
                condensates_properties_dict_srrm2['mean_condensate_area'])
        nuclei_dict['std_condensate_area_srrm2'].append(
                condensates_properties_dict_srrm2['std_condensate_area'])
        nuclei_dict['mean_condensate_eccentricity_srrm2'].append(
                condensates_properties_dict_srrm2['mean_condensate_eccentricity'])
        nuclei_dict['std_condensate_eccentricity_srrm2'].append(
                condensates_properties_dict_srrm2['std_condensate_eccentricity'])
        nuclei_dict['glcm_contrast_srrm2'].append(
                condensates_properties_dict_srrm2['glcm_contrast'])
        nuclei_dict['glcm_dissim_srrm2'].append(
                condensates_properties_dict_srrm2['glcm_dissim'])
        nuclei_dict['glcm_energy_srrm2'].append(
                condensates_properties_dict_srrm2['glcm_energy'])
        nuclei_dict['glcm_correlation_srrm2'].append(
                condensates_properties_dict_srrm2['glcm_correlation'])
        nuclei_dict['glcm_homogeneity_srrm2'].append(
                condensates_properties_dict_srrm2['glcm_homogeneity'])
        
        nuclei_dict['overlap_area_gfp_pml'].append( overlap_area_gfp_pml )
        nuclei_dict['overlap_area_gfp_srrm2'].append( overlap_area_gfp_srrm2 )
        nuclei_dict['overlap_area_srrm2_pml'].append( overlap_area_srrm2_pml )
        nuclei_dict['fraction_gfp_pml_overlap'].append( fraction_gfp_pml_overlap )
        nuclei_dict['fraction_gfp_srrm2_overlap'].append( fraction_gfp_srrm2_overlap )
        nuclei_dict['fraction_srrm2_pml_overlap'].append( fraction_srrm2_pml_overlap )

        nuclei_dict['cell_img_gfp_file'].append( output_fname_gfp )
        nuclei_dict['cell_img_pml_file'].append( output_fname_pml )
        nuclei_dict['cell_img_srrm2_file'].append( output_fname_srrm2 )
        nuclei_dict['cell_img_dapi_file'].append( output_fname_dapi )
        nuclei_dict['cell_img_mask_file'].append( output_fname_mask )
        nuclei_dict['corr_GFP_pml'].append( correlation_GFP_pml )
        nuclei_dict['corr_GFP_srrm2'].append( correlation_GFP_srrm2 )
        nuclei_dict['corr_pml_srrm2'].append( correlation_pml_srrm2 )
        nuclei_dict['corr_GFP_dapi'].append( correlation_GFP_dapi )


    df_nuclei = pd.DataFrame(data=nuclei_dict)
    return df_nuclei, final_nuclei

def prelim_phenotype_phenix_2channel_write_img_files( fname_dapi, fname_gfp,
                                         ffc_dapi, ffc_gfp,
                                         well_num, tile_num,
                                         file_save_dir, min_size=1000, 
                        smooth_size=5, threshold_initial_guess=500, 
                        nuclei_smooth_value=20, area_min=5000, plot=False,
                        area_max=20000, condensate_cutoff_intensity=2000,
                        THRESH_STDS=5,cellpose=True,cellpose_diameter=87, nuclei_mask_file=''):
    ##################
    # segment nuclei from dapi image
    ##################
    dapi_image = read( fname_dapi )
    gfp_image = read( fname_gfp )

    #save( f'preffc_dapi_well{well_num}_field{tile_num}.tif', dapi_image )
    #save( f'preffc_gfp_well{well_num}_field{tile_num}.tif', gfp_image )

    dapi_image = apply_flatfield_correction( dapi_image, ffc_dapi )
    gfp_image = apply_flatfield_correction( gfp_image, ffc_gfp )

    #save( f'ffc_dapi_well{well_num}_field{tile_num}.tif', dapi_image )
    #save( f'ffc_gfp_well{well_num}_field{tile_num}.tif', gfp_image )

    if nuclei_mask_file != '':
        # get the nuclei mask from file
        final_nuclei = read( nuclei_mask_file )

    else:

        dapi_rgb = prepare_png_cellpose( dapi_image )
        #diameter = 87
    
        #final_nuclei = read( 'nuclei_masks_plate_10_well_C2_test_20230320/nuclei_mask_r03c02f03p01-ch1sk1fk1fl1.tif' )
        #start_time = datetime.datetime.now()
        final_nuclei = segment_nuclei_phenotype_cellpose( dapi_rgb, cellpose_diameter )
        #end_time = datetime.datetime.now()
        #print( "time segmenting nuclei:", end_time - start_time )
        #final_nuclei = segment_nuclei_phenotype( dapi_image, threshold_initial_guess, smooth_size,
        #                            nuclei_smooth_value, min_size, area_min, area_max, plot)


    ##################
    # go through the nuclei and try to find condensates, calculate relevant properties
    ##################
    
    nuclei_dict = { 
        'nucleus_num': [],
        'well': [],
        'tile': [],
        'fname_dapi': [],
        'total_nucleus_area': [],
        'mean_dapi_intensity': [],
        
        'mean_GFP_intensity_GFP': [],
        'std_GFP_intensity_GFP': [],
        'num_condensates_GFP': [],
        'mean_dilute_intensity_GFP': [],
        'std_dilute_intensity_GFP': [],
        'mean_condensate_intensity_GFP': [],
        'mean_condensate_intensity_no_shrink_GFP': [],
        'std_condensate_intensity_GFP': [],
        'std_condensate_intensity_no_shrink_GFP': [],
        'total_cond_intensity_GFP': [],
        'total_gfp_intensity_GFP': [],
        'frac_gfp_in_cond_GFP': [],
        'total_cond_area_GFP': [],
        'mean_condensate_area': [],
        'std_condensate_area': [],
        'mean_condensate_eccentricity': [],
        'std_condensate_eccentricity': [],
        'glcm_energy': [],
        'glcm_correlation': [],
        'glcm_dissim': [],
        'glcm_homogeneity': [],
        'glcm_contrast': [],
        'cell_img_gfp_file': [],
        'cell_img_dapi_file': [],
        'cell_img_mask_file': [],
        
    }

    num_nuclei = len(np.unique(final_nuclei) ) -1
    if num_nuclei < 1:
        df_nuclei = pd.DataFrame(data=nuclei_dict)
        return df_nuclei, final_nuclei

    properties_dapi = skimage.measure.regionprops_table( final_nuclei, intensity_image=dapi_image,
                                        properties=('label','mean_intensity'))

    #print( well_num, tile_num, fname_dapi, fname_gfp )
    properties_gfp = skimage.measure.regionprops_table( final_nuclei, intensity_image=gfp_image, 
                                        properties=('label','mean_intensity','max_intensity','intensity_image',
                                                   'image','area','bbox'))
    
    for nucleus_index in range( len( properties_gfp['label'])):
        nucleus_image = properties_gfp['image'][nucleus_index]
        nucleus_area = properties_gfp['area'][nucleus_index]
        nucleus_label = properties_gfp['label'][nucleus_index]
        mean_dapi_intensity = properties_dapi['mean_intensity'][nucleus_index]
    
        mean_intensity_GFP = properties_gfp['mean_intensity'][nucleus_index]
        max_intensity_GFP = properties_gfp['max_intensity'][nucleus_index]
        region_image_GFP = properties_gfp['intensity_image'][nucleus_index]
        region_pixels_GFP = region_image_GFP[nucleus_image]
        std_intensity_GFP = np.std( region_pixels_GFP )
        total_intensity_GFP = mean_intensity_GFP * nucleus_area

        # write images to files
        # the dapi image: nucleus_image = properties_gfp['image'][nucleus_index] (this is already masked though)
        # the GFP image: properties_gfp['intensity_image'][nucleus_index] (this is already masked though)

        #bbox = properties_gfp['bbox'][nucleus_index]
        bbox = (properties_gfp['bbox-0'][nucleus_index],
                properties_gfp['bbox-1'][nucleus_index],
                properties_gfp['bbox-2'][nucleus_index],
                properties_gfp['bbox-3'][nucleus_index])


        # don't worry about padding for now 
        # pad to a constant size
        #region_size_x = bbox[2] - bbox[0]
        #region_size_y = bbox[3] - bbox[1]
        #pad_x_size = max(0, 185 - np.shape( unmasked_gfp_image)[0] )
        #pad_y_size = max(0, 185 - np.shape( unmasked_gfp_image)[1] )

        unmasked_region_gfp_image = gfp_image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        unmasked_region_dapi_image = dapi_image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        mask_region_nucleus = properties_gfp['image'][nucleus_index]

        # save to file, then read the file back in and check some properties to confirm that they exactly match
        output_fname_dapi = '{file_save_dir}/well{well_num}_field{tile_num}_cell{cell_num}_DAPI.tif'.format( 
            file_save_dir=file_save_dir, 
            cell_num=nucleus_label, well_num=well_num, tile_num=tile_num)

        output_fname_gfp = '{file_save_dir}/well{well_num}_field{tile_num}_cell{cell_num}_GFP.tif'.format( 
            file_save_dir=file_save_dir, 
            cell_num=nucleus_label, well_num=well_num, tile_num=tile_num)

        output_fname_condensates = '{file_save_dir}/well{well_num}_field{tile_num}_cell{cell_num}_condensates.tif'.format( 
            file_save_dir=file_save_dir, 
            cell_num=nucleus_label, well_num=well_num, tile_num=tile_num)

        output_fname_mask = '{file_save_dir}/well{well_num}_field{tile_num}_cell{cell_num}_mask.npy'.format( 
        #output_fname_mask = '{file_save_dir}/well{well_num}_field{tile_num}_cell{cell_num}_mask.tif'.format( 
            file_save_dir=file_save_dir, 
            cell_num=nucleus_label, well_num=well_num, tile_num=tile_num)


        save( output_fname_dapi, unmasked_region_dapi_image, compress=1)
        save( output_fname_gfp, unmasked_region_gfp_image, compress=1)
        np.save( output_fname_mask, mask_region_nucleus )
        # some quick checks to see if this works as expected
        ##save( output_fname_dapi, unmasked_region_dapi_image)
        ##save( output_fname_gfp, unmasked_region_gfp_image)
        ##save( output_fname_mask, mask_region_nucleus)
        ##save( output_fname_mask, mask_region_nucleus, compress=1)

        #test_gfp = read( output_fname_gfp )
        #test_mask = np.load( output_fname_mask)
        ##test_mask = read( output_fname_mask)

        ##print( "masked region before save",  unmasked_region_gfp_image[mask_region_nucleus] )

        ##print( "shape_gfp", np.shape( test_gfp))
        ##print( "shape mask", np.shape(test_mask))
        ##print( "min/max test mask", np.min( test_mask), np.max( test_mask ) )
        ##print( "all values", np.unique( test_mask ))
        ##print( test_gfp )
        ##print( test_mask )
        ##print( test_gfp[test_mask] )
        ##print( np.mean(test_gfp[test_mask]), mean_intensity_GFP )
        ##print( np.mean(test_gfp[test_mask]) == mean_intensity_GFP)
        #if np.mean(test_gfp[test_mask]) != mean_intensity_GFP: 
        #    print( "not equal" )


        ############
        # Get GFP condensates
        ############

        # if intensity is below some threshold, do not look for condensates (if GFP is not actually expressed)
#         if mean_intensity_GFP < 250. and max_intensity_GFP < 250.:
#             # this cell does not express GFP
#             num_condensates_GFP = 0
#             # condensate mask - should all be false
#             condensates_labeled_GFP = np.zeros( np.shape(nucleus.intensity_image))
#             mean_dilute_phase_intensity_GFP = mean_intensity
#             mean_condensate_intensity_GFP = np.nan
#             num_condensates_GFP = 0
#             std_dilute_phase_intensity_GFP = std_intensity
#             std_condensate_intensity_GFP = np.nan
#             total_condensate_intensity_GFP = 0
#             total_condensate_area_GFP = 0

#         else:
        #print( nucleus_image )
        # get condensates 
        (condensates_GFP, condensates_labeled_GFP, 
         condensates_properties_dict_GFP) = get_condensates_general_explicit( nucleus_image,
                                                                        mean_intensity_GFP,
                                                                        nucleus_area,
                                                                        region_image_GFP,
                                                                        nucleus_label,
                                                                        condensate_cutoff_intensity, 
                                                                        THRESH_STDS,
                                                                        plot=plot,
                                                                        save_file=output_fname_condensates)

            
        ############
        # add everything to dictionary
        ############
        
        nuclei_dict['nucleus_num'].append(nucleus_label)
        nuclei_dict['well'].append( well_num )
        nuclei_dict['tile'].append( tile_num )
        nuclei_dict['fname_dapi'].append( fname_dapi)
        nuclei_dict['total_nucleus_area'].append( nucleus_area )
        nuclei_dict['mean_dapi_intensity'].append( mean_dapi_intensity )
        
        nuclei_dict['mean_GFP_intensity_GFP'].append( mean_intensity_GFP )
        nuclei_dict['std_GFP_intensity_GFP'].append( std_intensity_GFP )
        nuclei_dict['num_condensates_GFP'].append( condensates_properties_dict_GFP['num_condensates'])
        nuclei_dict['mean_dilute_intensity_GFP'].append( 
            condensates_properties_dict_GFP['mean_dilute_phase_intensity'])
        nuclei_dict['std_dilute_intensity_GFP'].append(
            condensates_properties_dict_GFP['std_dilute_phase_intensity'])
        nuclei_dict['mean_condensate_intensity_GFP'].append(
            condensates_properties_dict_GFP['mean_condensate_intensity'])
        nuclei_dict['mean_condensate_intensity_no_shrink_GFP'].append( 
            condensates_properties_dict_GFP['mean_condensate_intensity_no_shrink'])
        nuclei_dict['std_condensate_intensity_GFP'].append(
            condensates_properties_dict_GFP['std_condensate_intensity'])
        nuclei_dict['std_condensate_intensity_no_shrink_GFP'].append( 
            condensates_properties_dict_GFP['std_condensate_intensity_no_shrink'])
        nuclei_dict['total_cond_intensity_GFP'].append(
            condensates_properties_dict_GFP['total_condensate_intensity'])
        nuclei_dict['total_gfp_intensity_GFP'].append( total_intensity_GFP)
        nuclei_dict['frac_gfp_in_cond_GFP'].append( 
            condensates_properties_dict_GFP['total_condensate_intensity'] / total_intensity_GFP )
        nuclei_dict['total_cond_area_GFP'].append( 
            condensates_properties_dict_GFP['total_condensate_area'])
        nuclei_dict['mean_condensate_area'].append(
                condensates_properties_dict_GFP['mean_condensate_area'])
        nuclei_dict['std_condensate_area'].append(
                condensates_properties_dict_GFP['std_condensate_area'])
        nuclei_dict['mean_condensate_eccentricity'].append(
                condensates_properties_dict_GFP['mean_condensate_eccentricity'])
        nuclei_dict['std_condensate_eccentricity'].append(
                condensates_properties_dict_GFP['std_condensate_eccentricity'])
        nuclei_dict['glcm_contrast'].append( 
                condensates_properties_dict_GFP['glcm_contrast'])
        nuclei_dict['glcm_dissim'].append( 
                condensates_properties_dict_GFP['glcm_dissim'])
        nuclei_dict['glcm_energy'].append( 
                condensates_properties_dict_GFP['glcm_energy'])
        nuclei_dict['glcm_correlation'].append( 
                condensates_properties_dict_GFP['glcm_correlation'])
        nuclei_dict['glcm_homogeneity'].append( 
                condensates_properties_dict_GFP['glcm_homogeneity'])
        nuclei_dict['cell_img_gfp_file'].append( output_fname_gfp )
        nuclei_dict['cell_img_dapi_file'].append( output_fname_dapi )
        nuclei_dict['cell_img_mask_file'].append( output_fname_mask )
        

    df_nuclei = pd.DataFrame(data=nuclei_dict)
    return df_nuclei, final_nuclei
    

def phenotype_phenix_2channel( fname_dapi, fname_gfp,
                                         well_num, tile_num, min_size=1000, 
                        smooth_size=5, threshold_initial_guess=500, 
                        nuclei_smooth_value=20, area_min=5000, plot=False,
                        area_max=20000, condensate_cutoff_intensity=2000,
                        THRESH_STDS=5):
    ##################
    # segment nuclei from dapi image
    ##################
    dapi_image = read( fname_dapi )
    gfp_image = read( fname_gfp )
    
    final_nuclei = segment_nuclei_phenotype( dapi_image, threshold_initial_guess, smooth_size,
                                nuclei_smooth_value, min_size, area_min, area_max, plot)


    ##################
    # go through the nuclei and try to find condensates, calculate relevant properties
    ##################
    
    nuclei_dict = { 
        'nucleus_num': [],
        'well': [],
        'tile': [],
        'fname_dapi': [],
        'total_nucleus_area': [],
        'mean_dapi_intensity': [],
        
        'mean_GFP_intensity_GFP': [],
        'std_GFP_intensity_GFP': [],
        'num_condensates_GFP': [],
        'mean_dilute_intensity_GFP': [],
        'std_dilute_intensity_GFP': [],
        'mean_condensate_intensity_GFP': [],
        'mean_condensate_intensity_no_shrink_GFP': [],
        'std_condensate_intensity_GFP': [],
        'std_condensate_intensity_no_shrink_GFP': [],
        'total_cond_intensity_GFP': [],
        'total_gfp_intensity_GFP': [],
        'frac_gfp_in_cond_GFP': [],
        'total_cond_area_GFP': [],
        'mean_condensate_area': [],
        'std_condensate_area': [],
        'mean_condensate_eccentricity': [],
        'std_condensate_eccentricity': [],
    }

    num_nuclei = len(np.unique(final_nuclei) ) -1
    if num_nuclei < 1:
        df_nuclei = pd.DataFrame(data=nuclei_dict)
        return df_nuclei, final_nuclei

    properties_dapi = skimage.measure.regionprops_table( final_nuclei, intensity_image=dapi_image,
                                        properties=('label','mean_intensity'))

    #print( well_num, tile_num, fname_dapi, fname_gfp )
    properties_gfp = skimage.measure.regionprops_table( final_nuclei, intensity_image=gfp_image, 
                                        properties=('label','mean_intensity','max_intensity','intensity_image',
                                                   'image','area'))
    
    for nucleus_index in range( len( properties_gfp['label'])):
        nucleus_image = properties_gfp['image'][nucleus_index]
        nucleus_area = properties_gfp['area'][nucleus_index]
        nucleus_label = properties_gfp['label'][nucleus_index]
        mean_dapi_intensity = properties_dapi['mean_intensity'][nucleus_index]
    
        mean_intensity_GFP = properties_gfp['mean_intensity'][nucleus_index]
        max_intensity_GFP = properties_gfp['max_intensity'][nucleus_index]
        region_image_GFP = properties_gfp['intensity_image'][nucleus_index]
        region_pixels_GFP = region_image_GFP[nucleus_image]
        std_intensity_GFP = np.std( region_pixels_GFP )
        total_intensity_GFP = mean_intensity_GFP * nucleus_area
        
        ############
        # Get GFP condensates
        ############

        # if intensity is below some threshold, do not look for condensates (if GFP is not actually expressed)
#         if mean_intensity_GFP < 250. and max_intensity_GFP < 250.:
#             # this cell does not express GFP
#             num_condensates_GFP = 0
#             # condensate mask - should all be false
#             condensates_labeled_GFP = np.zeros( np.shape(nucleus.intensity_image))
#             mean_dilute_phase_intensity_GFP = mean_intensity
#             mean_condensate_intensity_GFP = np.nan
#             num_condensates_GFP = 0
#             std_dilute_phase_intensity_GFP = std_intensity
#             std_condensate_intensity_GFP = np.nan
#             total_condensate_intensity_GFP = 0
#             total_condensate_area_GFP = 0

#         else:
        # get condensates 
        (condensates_GFP, condensates_labeled_GFP, 
         condensates_properties_dict_GFP) = get_condensates_general_explicit( nucleus_image,
                                                                        mean_intensity_GFP,
                                                                        nucleus_area,
                                                                        region_image_GFP,
                                                                        nucleus_label,
                                                                        condensate_cutoff_intensity, 
                                                                        THRESH_STDS,
                                                                        plot=plot )

            
        ############
        # add everything to dictionary
        ############
        
        nuclei_dict['nucleus_num'].append(nucleus_label)
        nuclei_dict['well'].append( well_num )
        nuclei_dict['tile'].append( tile_num )
        nuclei_dict['fname_dapi'].append( fname_dapi)
        nuclei_dict['total_nucleus_area'].append( nucleus_area )
        nuclei_dict['mean_dapi_intensity'].append( mean_dapi_intensity )
        
        nuclei_dict['mean_GFP_intensity_GFP'].append( mean_intensity_GFP )
        nuclei_dict['std_GFP_intensity_GFP'].append( std_intensity_GFP )
        nuclei_dict['num_condensates_GFP'].append( condensates_properties_dict_GFP['num_condensates'])
        nuclei_dict['mean_dilute_intensity_GFP'].append( 
            condensates_properties_dict_GFP['mean_dilute_phase_intensity'])
        nuclei_dict['std_dilute_intensity_GFP'].append(
            condensates_properties_dict_GFP['std_dilute_phase_intensity'])
        nuclei_dict['mean_condensate_intensity_GFP'].append(
            condensates_properties_dict_GFP['mean_condensate_intensity'])
        nuclei_dict['mean_condensate_intensity_no_shrink_GFP'].append( 
            condensates_properties_dict_GFP['mean_condensate_intensity_no_shrink'])
        nuclei_dict['std_condensate_intensity_GFP'].append(
            condensates_properties_dict_GFP['std_condensate_intensity'])
        nuclei_dict['std_condensate_intensity_no_shrink_GFP'].append( 
            condensates_properties_dict_GFP['std_condensate_intensity_no_shrink'])
        nuclei_dict['total_cond_intensity_GFP'].append(
            condensates_properties_dict_GFP['total_condensate_intensity'])
        nuclei_dict['total_gfp_intensity_GFP'].append( total_intensity_GFP)
        nuclei_dict['frac_gfp_in_cond_GFP'].append( 
            condensates_properties_dict_GFP['total_condensate_intensity'] / total_intensity_GFP )
        nuclei_dict['total_cond_area_GFP'].append( 
            condensates_properties_dict_GFP['total_condensate_area'])
        nuclei_dict['mean_condensate_area'].append( 
                condensates_properties_dict_GFP['mean_condensate_area'])
        nuclei_dict['std_condensate_area'].append( 
                condensates_properties_dict_GFP['std_condensate_area'])
        nuclei_dict['mean_condensate_eccentricity'].append(
                condensates_properties_dict_GFP['mean_condensate_eccentricity'])
        nuclei_dict['std_condensate_eccentricity'].append(
                condensates_properties_dict_GFP['std_condensate_eccentricity'])

    df_nuclei = pd.DataFrame(data=nuclei_dict)
    return df_nuclei, final_nuclei


def plot_example_cells_many( cells_with_barcode, title, save_name, min_intensity=100, vmax=2500 ):
    # these values need to match what was used when I ran phenotype_phenix...
    min_size = 1000
    smooth_size = 5
    threshold_initial_guess = 500
    nuclei_smooth_value = 20
    #area_min = 5000
    area_min=2500
    plot=False
    area_max = 20000
    condensate_cutoff_intensity = 1000
    THRESH_STDS =3

    NUM_IMGS = 40
    num_cols = 5
    #fig, ax = plt.subplots(1, NUM_IMGS, figsize=(7,2))
    fig, ax = plt.subplots(int(NUM_IMGS/num_cols), num_cols, figsize=(num_cols,1.*(NUM_IMGS/num_cols)))
    num = 0

#     # omit the cell if it does not express.
#     median = np.median(cells_with_barcode['mean_GFP_intensity_GFP'])
#     std = np.std(cells_with_barcode['mean_GFP_intensity_GFP'])
#     #median_num_condensates = np.median( in_range['num_condensates'])
#     #std_num_condensates = np.median( in_range['num_condensates'])
#     median_num_condensates = np.median( cells_with_barcode[cells_with_barcode['mean_GFP_intensity_GFP']>200]['num_condensates'])
#     std_num_condensates = np.std( cells_with_barcode[cells_with_barcode['mean_GFP_intensity_GFP']>200]['num_condensates'])

    # try to plot "median" cells

    img_num = 0
    #ax[3].set_title(barcode_to_name_full[barcode],fontsize=6, color='blue' )
    #ax[3].set_title( title, fontsize=6 )
    plt.suptitle( title, fontsize=6)
    dict_40x_nuclei_masks = {}
    for index, row in cells_with_barcode.iterrows():
        if img_num >(NUM_IMGS-1): break
        if row['mean_GFP_intensity_GFP'] < min_intensity: continue

        row_index = int(img_num / num_cols)
        col_index = img_num - row_index*num_cols
        #print( img_num, row_index, col_index )

        fname_dapi = row['fname_dapi']
        fname_gfp = fname_dapi.replace('ch1','ch2')
        tile_num = row['tile']
        well_num = row['well']

        dapi_image = read( fname_dapi )
        gfp_image = read( fname_gfp )

        if fname_dapi in dict_40x_nuclei_masks.keys():
            nucleus_mask = dict_40x_nuclei_masks[ fname_dapi ]
        else:
            nucleus_mask = segment_nuclei_phenotype( dapi_image, threshold_initial_guess,
                                                smooth_size, nuclei_smooth_value, min_size,
                                                area_min, area_max, plot )
            dict_40x_nuclei_masks[ fname_dapi ] = nucleus_mask

        nucleus_num = row['nucleus_num']

        # save the shifted, scaled down 40x image
        for nucleus in skimage.measure.regionprops(nucleus_mask, gfp_image):
            if nucleus.label != nucleus_num: continue

            # alternatively pad image
            pad_x_size = max(0, 185 - np.shape(nucleus.intensity_image)[0])
            pad_y_size = max(0, 185 - np.shape(nucleus.intensity_image)[1])
            padded_image = np.pad( nucleus.intensity_image, [(int(pad_x_size/2),int(pad_x_size/2)), (int(pad_y_size/2),int(pad_y_size/2))],
                                            'constant', constant_values=0)

            ax[row_index][col_index].set_axis_off()
            ax[row_index][col_index].imshow( padded_image, cmap='gray', vmin=0, vmax=vmax )
            img_num += 1

    # turn all axes off
    for i in range( int(NUM_IMGS/num_cols)):
        for j in range(num_cols):
            ax[i][j].set_axis_off()

    #plt.tight_layout()
    plt.savefig(save_name,dpi=300,facecolor='white')
    #plt.show()
    plt.clf()

def plot_example_cells_4channel_many( cells_with_barcode, title, save_name, min_intensity=100, vmax=2500 ):
    # these values need to match what was used when I ran phenotype_phenix...
    min_size = 1000
    smooth_size = 5
    threshold_initial_guess = 500
    nuclei_smooth_value = 20
    #area_min = 5000
    area_min=2500
    plot=False
    area_max = 20000
    condensate_cutoff_intensity = 1000
    THRESH_STDS =3

    NUM_IMGS = 40
    num_cols = 5
    #fig, ax = plt.subplots(1, NUM_IMGS, figsize=(7,2))
    fig, ax = plt.subplots(int(NUM_IMGS/num_cols), num_cols, figsize=(num_cols,1.*(NUM_IMGS/num_cols)))
    num = 0

#     # omit the cell if it does not express.
#     median = np.median(cells_with_barcode['mean_GFP_intensity_GFP'])
#     std = np.std(cells_with_barcode['mean_GFP_intensity_GFP'])
#     #median_num_condensates = np.median( in_range['num_condensates'])
#     #std_num_condensates = np.median( in_range['num_condensates'])
#     median_num_condensates = np.median( cells_with_barcode[cells_with_barcode['mean_GFP_intensity_GFP']>200]['num_condensates'])
#     std_num_condensates = np.std( cells_with_barcode[cells_with_barcode['mean_GFP_intensity_GFP']>200]['num_condensates'])

    # try to plot "median" cells

    img_num = 0
    #ax[3].set_title(barcode_to_name_full[barcode],fontsize=6, color='blue' )
    #ax[3].set_title( title, fontsize=6 )
    plt.suptitle( title, fontsize=6)
    dict_40x_nuclei_masks = {}
    for index, row in cells_with_barcode.iterrows():
        if img_num >(NUM_IMGS-1): break
        if row['mean_GFP_intensity_GFP'] < min_intensity: continue

        row_index = int(img_num / num_cols)
        col_index = img_num - row_index*num_cols
        #print( img_num, row_index, col_index )

        fname_dapi = row['fname_dapi']
        fname_gfp = fname_dapi.replace('ch1','ch2')
        fname_568 = fname_dapi.replace('ch1','ch3')
        fname_647 = fname_dapi.replace('ch1','ch4')
        tile_num = row['tile']
        well_num = row['well']

        dapi_image = read( fname_dapi )
        gfp_image = read( fname_gfp )
        image_568 = read( fname_568 )
        image_647 = read( fname_647 )

        if fname_dapi in dict_40x_nuclei_masks.keys():
            nucleus_mask = dict_40x_nuclei_masks[ fname_dapi ]
        else:
            nucleus_mask = segment_nuclei_phenotype( dapi_image, threshold_initial_guess,
                                                smooth_size, nuclei_smooth_value, min_size,
                                                area_min, area_max, plot )
            dict_40x_nuclei_masks[ fname_dapi ] = nucleus_mask

        nucleus_num = row['nucleus_num']

        # save the shifted, scaled down 40x image
        for nucleus in skimage.measure.regionprops(nucleus_mask, gfp_image):
            if nucleus.label != nucleus_num: continue

            # alternatively pad image
            pad_x_size = max(0, 185 - np.shape(nucleus.intensity_image)[0])
            pad_y_size = max(0, 185 - np.shape(nucleus.intensity_image)[1])
            padded_image = np.pad( nucleus.intensity_image, [(int(pad_x_size/2),int(pad_x_size/2)), (int(pad_y_size/2),int(pad_y_size/2))],
                                            'constant', constant_values=0)

            ax[row_index][col_index].set_axis_off()
            ax[row_index][col_index].imshow( padded_image, cmap='Greens', vmin=0, vmax=vmax, alpha=0.5 )
        for nucleus in skimage.measure.regionprops(nucleus_mask, image_568):
            if nucleus.label != nucleus_num: continue
            # alternatively pad image
            pad_x_size = max(0, 185 - np.shape(nucleus.intensity_image)[0])
            pad_y_size = max(0, 185 - np.shape(nucleus.intensity_image)[1])
            padded_image = np.pad( nucleus.intensity_image, [(int(pad_x_size/2),int(pad_x_size/2)), (int(pad_y_size/2),int(pad_y_size/2))],
                                            'constant', constant_values=0)

            ax[row_index][col_index].set_axis_off()
            ax[row_index][col_index].imshow( padded_image, cmap='Oranges', vmin=0, vmax=vmax, alpha=0.5 )

        for nucleus in skimage.measure.regionprops(nucleus_mask, image_647):
            if nucleus.label != nucleus_num: continue
            # alternatively pad image
            pad_x_size = max(0, 185 - np.shape(nucleus.intensity_image)[0])
            pad_y_size = max(0, 185 - np.shape(nucleus.intensity_image)[1])
            padded_image = np.pad( nucleus.intensity_image, [(int(pad_x_size/2),int(pad_x_size/2)), (int(pad_y_size/2),int(pad_y_size/2))],
                                            'constant', constant_values=0)

            ax[row_index][col_index].set_axis_off()
            ax[row_index][col_index].imshow( padded_image, cmap='Purples', vmin=0, vmax=vmax, alpha=0.5 )


            img_num += 1

    # turn all axes off
    for i in range( int(NUM_IMGS/num_cols)):
        for j in range(num_cols):
            ax[i][j].set_axis_off()

    #plt.tight_layout()
    plt.savefig(save_name,dpi=300,facecolor='white')
    #plt.show()
    plt.clf()


