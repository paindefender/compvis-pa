import numpy as np
from skimage import filters
from skimage.util.shape import view_as_blocks
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import convolve
#import random
from utils import pad, unpad


def harris_corners(img, window_size=3, k=0.04):
    """
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).

    Hint:
        You may use the function scipy.ndimage.filters.convolve for convolution, 
        which is already imported above
        
    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    """

    H, W = img.shape
    window = np.ones((window_size, window_size))

    response = np.zeros((H, W))

    # step 1 in Harris corner detection: compute image derivatives
    dx = filters.sobel_v(img)  
    dy = filters.sobel_h(img)

    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    
    dx2 = np.multiply(dx, dx)
    dy2 = np.multiply(dy, dy)
    dxy = np.multiply(dx, dy)

    dx2 = convolve(dx2, window)
    dy2 = convolve(dy2, window)
    dxy = convolve(dxy, window)

    response = (np.multiply(dx2, dy2) - dxy**2) - k*(dx2+dy2)**2 # vectorized i guess
    
    ######################################
    #        END OF YOUR CODE            #
    ######################################

    return response


def simple_descriptor(patch):
    """
    Describe the patch by normalizing the image values into a standard 
    normal distribution (having mean of 0 and standard deviation of 1) 
    and then flattening into a 1D array. 
    
    The normalization will make the descriptor more robust to change 
    in lighting condition.
    
    Hint:
        If a denominator is zero, divide by 1 instead.
    
    Args:
        patch: grayscale image patch of shape (h, w)
    
    Returns:
        feature: 1D array of shape (h * w)
    """
    feature = []
    
    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    
    std = np.std(patch)
    if std == 0:
        std = 1
    mean = np.mean(patch)
    normalize = lambda x: (x-mean)/std
    feature = normalize(patch).flatten()

    ######################################
    #        END OF YOUR CODE            #
    ######################################
    return feature


def describe_keypoints(image, keypoints, desc_func, patch_size=16):
    """
    Args:
        image: grayscale image of shape (H, W)
        keypoints: 2D array containing a keypoint (y, x) in each row
        desc_func: function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: size of a square patch at each keypoint
                
    Returns:
        desc: array of features describing the keypoints
    """

    image.astype(np.float32)
    desc = []

    for i, kp in enumerate(keypoints):
        y, x = kp
        patch = image[y-(patch_size//2):y+((patch_size+1)//2),
                      x-(patch_size//2):x+((patch_size+1)//2)]
        desc.append(desc_func(patch))
    return np.array(desc)


def match_descriptors(desc1, desc2, threshold=0.5):
    """
    Match the feature descriptors by finding distances between them. A match is formed 
    when the distance to the closest vector is much smaller than the distance to the 
    second-closest, that is, the ratio of the distances should be smaller
    than the threshold. Return the matches as pairs of vector indices.
    
    Args:
        desc1: an array of shape (M, P) holding descriptors of size P about M keypoints
        desc2: an array of shape (N, P) holding descriptors of size P about N keypoints
        
    Returns:
        matches: an array of shape (Q, 2) where each row holds the indices of one pair 
        of matching descriptors
    """
    matches = []
    
    N = desc1.shape[0]
    dists = cdist(desc1, desc2)

    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    for i in range(dists.shape[0]):
        ids = np.argpartition(dists[i], 1)[:2]
        ratio = dists[i][ids][0]/dists[i][ids][1]
        if ratio < threshold:
            matches.append([i, ids[0]])
    matches = np.array(matches)
    ######################################
    #        END OF YOUR CODE            #
    ######################################
    
    return matches


def fit_affine_matrix(p1, p2):
    """ Fit affine matrix such that p2 * H = p1 
    
    Hint:
        You can use np.linalg.lstsq function to solve the problem. 
        
    Args:
        p1: an array of shape (M, P)
        p2: an array of shape (M, P)
        
    Return:
        H: a matrix of shape (P * P) that transform p2 to p1.
    """

    assert (p1.shape[0] == p2.shape[0]),\
        'Different number of points in p1 and p2'
    p1 = pad(p1)
    p2 = pad(p2)
    
    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    
    H = np.linalg.lstsq(p2, p1, rcond=None)[0]
    
    ######################################
    #        END OF YOUR CODE            #
    ######################################

    # Sometimes numerical issues cause least-squares to produce the last
    # column which is not exactly [0, 0, 1]
    H[:,2] = np.array([0, 0, 1])
    return H


def ransac(keypoints1, keypoints2, matches, n_iters=200, threshold=20):
    """
    Use RANSAC to find a robust affine transformation

        1. Select random set of matches
        2. Compute affine transformation matrix
        3. Compute inliers
        4. Keep the largest set of inliers
        5. Re-compute least-squares estimate on all of the inliers

    Args:
        keypoints1: M1 x 2 matrix, each row is a point
        keypoints2: M2 x 2 matrix, each row is a point
        matches: N x 2 matrix, each row represents a match
            [index of keypoint1, index of keypoint 2]
        n_iters: the number of iterations RANSAC will run
        threshold: the number of threshold to find inliers

    Returns:
        H: a robust estimation of affine transformation from keypoints2 to
        keypoints 1
    """
    N = matches.shape[0]
    n_samples = int(N * 0.2)

    matched1 = pad(keypoints1[matches[:,0]])
    matched2 = pad(keypoints2[matches[:,1]])

    max_inliers = np.zeros(N)
    n_inliers = 0

    # RANSAC iteration start
    
    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    for i in range(n_iters):
        sample = np.random.choice(N, n_samples, replace=False)
        p1 = matched1[sample]
        p2 = matched2[sample]
        
        affine = np.linalg.lstsq(p2, p1, rcond=None)[0]
        affine[:,2] = np.array([0, 0, 1])
        inliers = np.zeros(N)

        for j in range(N):
            if np.linalg.norm(matched1[j] - matched2[j] @ affine) < threshold:
                inliers[j] += 1
        current_inliers = np.sum(inliers)

        if current_inliers > n_inliers:
            n_inliers = current_inliers
            max_inliers = np.copy(inliers).astype(bool)

    p1 = matched1[max_inliers]
    p2 = matched2[max_inliers]
    H = np.linalg.lstsq(p2, p1, rcond=None)[0]
    H[:,2] = np.array([0, 0, 1])
    ######################################
    #        END OF YOUR CODE            #
    ######################################
    
    return H, matches[max_inliers]
    
