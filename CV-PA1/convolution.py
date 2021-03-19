import numpy as np


def conv_naive(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    h_padding = (Hk-1)//2
    v_padding = (Wk-1)//2
    image = np.pad(image, ((v_padding, v_padding),(h_padding, h_padding)), 'constant')
    for ih in range(Hi):
        for iw in range(Wi):
            for kh in range(Hk):
                for kw in range(Wk):
                    out[ih, iw] += image[ih+kh, iw+kw] * kernel[Hk - 1 - kh, Wk - 1 - kw] 
    ######################################
    #        END OF YOUR CODE            #
    ######################################

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Example: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W)
        pad_width: width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """

    H, W = image.shape
    out = None

    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    out = np.pad(image, ((pad_height, pad_height),(pad_width, pad_width)), 'constant')
    ######################################
    #        END OF YOUR CODE            #
    ######################################
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    h_padding = (Hk-1)//2
    v_padding = (Wk-1)//2
    image = zero_pad(image, v_padding, h_padding) # pad
    kernel = np.flipud(np.fliplr(kernel)) # flip h/v
    for h in range(Hi):
        for w in range(Wi):
            out[h, w] = np.sum(np.multiply(kernel, image[h : h + Hk, w : w + Wk]))
    ######################################
    #        END OF YOUR CODE            #
    ######################################

    return out

