import numpy as np
from ops.io import read_stack as read
from ops.io import save_stack as save
import skimage
import skimage.filters


def get_flatfield_correction( list_of_files ):
    # get the average intensity for each pixel (try mean initially)
    img1 = read( list_of_files[0] )
    total_pixel_intensity = np.zeros( img1.shape )
    for f in list_of_files:
        img = read( f )
        total_pixel_intensity += img
    #save( 'total_pixel_int.tif', total_pixel_intensity )
    mean_pixel_intensity = total_pixel_intensity / len( list_of_files )
    # smooth with a gaussian filter
    smooth_value = mean_pixel_intensity.shape[0]/10.
    smooth_mean_pixel_intensity = skimage.filters.gaussian( mean_pixel_intensity, sigma=smooth_value)

    # normalize so that the maximum average intensity=1
    smooth_mean_pixel_intensity_norm = smooth_mean_pixel_intensity / np.max(smooth_mean_pixel_intensity)

    # return this matrix 
    return smooth_mean_pixel_intensity_norm


def apply_flatfield_correction( img, correction_matrix ):
    # need to take into account any saturated pixels
    corrected = img / correction_matrix
    saturated_pixels = corrected >= 65535
    corrected[ saturated_pixels ] = 65535
    corrected = corrected.astype( img.dtype )
    return corrected 


def apply_bleedthrough_correction( img, correction_image, bleedthrough_factor ):
    if bleedthrough_factor == 0.:
        # no correction to be done
        return img

    intensity_to_subtract = correction_image * bleedthrough_factor
    corrected_img = img - intensity_to_subtract

    # there should not be any values below zero
    negative_pixels = corrected_img < 0.
    corrected_img[ negative_pixels ] = 0.

    corrected_img = corrected_img.astype( img.dtype )
    return corrected_img 

    pml_image = apply_bleedthrough_correction( pml_image, correction_image=gfp_image, 
            bleedthrough_factor=pml_gfp_bleed_factor )
