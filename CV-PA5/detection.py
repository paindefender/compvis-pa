import numpy as np
from skimage import feature, data, color, exposure, io
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.filters import gaussian
from scipy import signal
from scipy.ndimage import interpolation
import math

def hog_feature(image, pixel_per_cell = 8):
    ''' 
    Compute hog feature for a given image.
    
    Hint: use the hog function provided by skimage
    
    Args:
        image: an image with object that we want to detect
        pixel_per_cell: number of pixels in each cell, an argument for hog descriptor
        
    Returns:
        hogFeature: a vector of hog representation
        hogImage: an image representation of hog provided by skimage
    '''
    #####################################
    #       START YOUR CODE HERE        #
    #####################################   
    hogFeature, hogImage = feature.hog(image, pixels_per_cell=(pixel_per_cell, pixel_per_cell), visualize=True)
    ######################################
    #        END OF YOUR CODE            #
    ######################################
    return (hogFeature, hogImage)

def sliding_window(image, base_score, stepSize, windowSize, pixel_per_cell=8):
    ''' A sliding window that checks each different location in the image, 
        and finds which location has the highest hog score. The hog score is computed
        as the dot product between hog feature of the sliding window and the hog feature
        of the template. It generates a response map where each location of the
        response map is a corresponding score. And you will need to resize the response map
        so that it has the same shape as the image.
    
    Args:
        image: an np array of size (h,w)
        base_score: hog representation of the object you want to find, an array of size (m,)
        stepSize: an int of the step size to move the window
        windowSize: a pair of ints that is the height and width of the window
    Returns:
        max_score: float of the highest hog score 
        maxr: int of row where the max_score is found
        maxc: int of column where the max_score is found
        response_map: an np array of size (h,w)
    '''
    # slide a window across the image
    (max_score, maxr, maxc) = (0,0,0)
    winH, winW = windowSize
    H,W = image.shape
    pad_image = np.lib.pad(image, ((winH//2,winH-winH//2),(winW//2, winW-winW//2)), mode='constant')
    response_map = np.zeros((H//stepSize+1, W//stepSize+1))
    
    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    for r in range(0, H + 1, stepSize): # rows
        for c in range(0, W + 1, stepSize): # columns
            hogFeature = feature.hog(pad_image[r:r+winH, c:c+winW], pixels_per_cell=(pixel_per_cell, pixel_per_cell))
            h_score = hogFeature.dot(base_score)
            response_map[r//stepSize, c//stepSize] = h_score
            if h_score > max_score:
                max_score = h_score
                maxr = r
                maxc = c
    response_map = resize(response_map, image.shape[0:2])
    maxr = maxr - winH//2
    maxc = maxc - winW//2
    ######################################
    #        END OF YOUR CODE            #
    ######################################
    
    
    return (max_score, maxr, maxc, response_map)


def pyramid(image, scale=0.9, minSize=(200, 100)):
    '''
    Generate image pyramid using the given image and scale.
    Reducing the size of the image until one of the height or
    width reaches the minimum limit. In the ith iteration, 
    the image is resized to scale^i of the original image.
    
    Args:
        image: np array of (h,w), an image to scale
        scale: float of how much to rescale the image each time
        minSize: pair of ints showing the minimum height and width
        
    Returns:
        images: a list containing pair of 
            (the current scale of the image, resized image)
    '''
    # yield the original image
    images = []
    current_scale = 1.0
    images.append((current_scale, image))
    # keep looping over the pyramid
    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    while(current_scale*scale*image.shape[0] > minSize[0] and current_scale*scale*image.shape[1] > minSize[1]):
        current_scale *= scale
        images.append((current_scale, rescale(image, current_scale)))
    ######################################
    #        END OF YOUR CODE            #
    ######################################
    return images

def pyramid_score(image, base_score, shape, stepSize=20, scale = 0.9, pixel_per_cell = 8):
    '''
    Calculate the maximum score found in the image pyramid using sliding window.
    
    Args:
        image: np array of (h,w)
        base_score: the hog representation of the object you want to detect
        shape: shape of window you want to use for the sliding_window
        
    Returns:
        max_score: float of the highest hog score 
        maxr: int of row where the max_score is found
        maxc: int of column where the max_score is found
        max_scale: float of scale when the max_score is found
        max_response_map: np array of the response map when max_score is found
    '''
    max_score = 0
    maxr = 0
    maxc = 0
    max_scale = 1.0
    max_response_map =np.zeros(image.shape)
    images = pyramid(image, scale)
    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    for cur_scale, image in images:
        cur_score, cur_r, cur_c, cur_map = sliding_window(image, base_score, stepSize, shape, pixel_per_cell=pixel_per_cell)
        if cur_score > max_score:
            max_score = cur_score
            maxr = cur_r
            maxc = cur_c
            max_scale = cur_scale
            max_response_map = cur_map
    ######################################
    #        END OF YOUR CODE            #
    ######################################
    return max_score, maxr, maxc, max_scale, max_response_map




            

    