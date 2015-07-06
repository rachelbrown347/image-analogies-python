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
    m_A = np.mean(A)
    m_B = np.mean(B)
    s_A = np.std(A)
    s_B = np.std(B)

    A_remap  = (m_B/m_A) * ( A - m_A) + m_B
    Ap_remap = (m_B/m_A) * (Ap - m_A) + m_B

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


def initialize_Bp(Ap_pyr, init_rand=True):
    Bp_pyr = [list([]) for _ in xrange(len(Ap_pyr))]

    # set all to nan
    for level in range(len(Ap_pyr)):
        Bp_pyr[level] = np.nan * Ap_pyr[level]

    if init_rand:
        for level in [0, 1]:
            level_shape = Ap_pyr[level].shape
            Bp_pyr[level] = np.random.rand(np.product(level_shape)).reshape(level_shape)
    else:
        # initialize with correct answer
        for level in [0, 1]:
            Bp_pyr[level] = Ap_pyr[level]

    return Bp_pyr