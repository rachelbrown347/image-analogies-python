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


def remap_luminance(A, Ap_list, B):
    # single channel only
    assert(len(A.shape) == len(Ap_list[0].shape) == len(B.shape) == 2)

    m_A = np.mean(A)
    m_B = np.mean(B)
    s_A = np.std(A)
    s_B = np.std(B)

    A_remap = (s_B/s_A) * ( A - m_A) + m_B

    Ap_remap_list = []
    for Ap in Ap_list:
        Ap_remap_list.append((s_B/s_A) * (Ap - m_A) + m_B)

    return A_remap, Ap_remap_list


def compress_values(A, B, ratio):
    return ratio * A, ratio * B


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


def pad_img_pair(img_sm, img_lg, c):
    return [np.pad(img_sm, c.padding_sm, mode='symmetric'),
        np.pad(img_lg, c.padding_lg, mode='symmetric')]

def px2ix(pxs, w):
    rows, cols = pxs[0], pxs[1]
    return (rows * w + cols).astype(int)


def ix2px(ixs, w):
    cols = ixs % w
    rows = (ixs - cols) // w
    return np.array([rows, cols])


def Ap_ix2px(ixs, h, w):
    pxs = ix2px(ixs, w)
    rows, cols = pxs[0], pxs[1]
    img_nums = (np.floor(rows/h)).astype(int)
    img_ixs = ixs - img_nums * h * w
    return ix2px(img_ixs, w), img_nums


def Ap_px2ix(pxs, img_nums, h, w):
    rows, cols = pxs[0], pxs[1]
    return (((h * img_nums) + rows) * w + cols).astype(int)


def savefig_noborder(fileName, fig):
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig(fileName, bbox_inches='tight', pad_inches=0)

