import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import pyramid_gaussian


def convert_to_YIQ(img):
    """RGB to YIQ matrix
    courtesy of http://www.cs.rit.edu/~ncs/color/t_convert.html"""
    assert 0 <= np.max(img) <= 1
    m = np.array([[ 0.299,  0.587,  0.114],
                  [ 0.596, -0.275, -0.321],
                  [ 0.212, -0.523,  0.311]])
    return np.einsum('ij,klj->kli', m, img)


def convert_to_RGB(img):
    """YIQ to RGB matrix
    courtesy of http://www.cs.rit.edu/~ncs/color/t_convert.html"""
    m = np.array([[ 1.,  0.956,  0.621],
                  [ 1., -0.272, -0.647],
                  [ 1., -1.105,  1.702]])
    return np.einsum('ij,klj->kli', m, img)


def remap_luminance(A, Ap, B):
    # single channel only
    assert(len(A.shape) == len(Ap.shape) == len(B.shape) == 2)

    m_A = np.mean(A)
    m_B = np.mean(B)
    s_A = np.std(A)
    s_B = np.std(B)

    A_remap  = (s_B/s_A) * ( A - m_A) + m_B
    Ap_remap = (s_B/s_A) * (Ap - m_A) + m_B

    return A_remap, Ap_remap

def compute_gaussian_pyramid(img, min_size):
    h, w = img.shape[0:2]
    curr_size = np.min([h, w])
    levels = 0

    while curr_size > min_size:
        curr_size = np.floor(curr_size/2.)
        levels += 1

    img_pyr = list(pyramid_gaussian(img, max_layer=levels))

    img_pyr.reverse() # smallest to largest

    assert np.min(img_pyr[1].shape[:2]) > min_size
    assert np.min(img_pyr[1].shape[:2]) <= np.min(img_pyr[-1].shape[:2])

    return img_pyr


def initialize_Bp(B_pyr, init_rand=True):
    Bp_pyr = [list([]) for _ in xrange(len(B_pyr))]

    for level in range(len(B_pyr)):
        if init_rand:
            # initialize randomly
            level_shape = B_pyr[level].shape
            Bp_pyr[level] = np.random.rand(np.product(level_shape)).reshape(level_shape)
        else:
            # initialize with correct answer
            Bp_pyr[level] = B_pyr[level].copy()

    return Bp_pyr


def SaveFigureAsImage(fileName,fig=None,**kwargs):
    ''' Save a Matplotlib figure as an image without borders or frames.
        Courtesy of: http://robotics.usc.edu/~ampereir/wordpress/?p=626
        Args:
            fileName (str): String that ends in .png etc.

            fig (Matplotlib figure instance): figure you want to save as the image
        Keyword Args:
            orig_size (tuple): width, height of the original image used to maintain
            aspect ratio.
    '''
    fig_size = fig.get_size_inches()
    w,h = fig_size[0], fig_size[1]
    fig.patch.set_alpha(0)
    if kwargs.has_key('orig_size'): # Aspect ratio scaling if required
        w,h = kwargs['orig_size']
        w2,h2 = fig_size[0],fig_size[1]
        fig.set_size_inches([(w2/w)*w,(w2/w)*h])
        fig.set_dpi((w2/w)*fig.get_dpi())
    a=fig.gca()
    a.set_frame_on(False)
    a.set_xticks([]); a.set_yticks([])
    plt.axis('off')
    plt.xlim(0,h); plt.ylim(w,0)
    fig.savefig(fileName, transparent=True, bbox_inches='tight', pad_inches=0, interpolation='nearest')