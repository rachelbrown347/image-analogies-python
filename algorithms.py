from itertools import product

import numpy as np
from numpy.linalg import norm
import pyflann as pf
from sklearn.feature_extraction.image import extract_patches_2d

from img_preprocess import px2ix, pad_img_pair, Ap_ix2px, Ap_px2ix


def compute_feature_array(im_pyr, c, full_feat):
    # features will be organized like this:
    # sm_imA, lg_imA (channels shuffled C-style)

    # create a list of features for each pyramid level
    # level 0 is empty for indexing alignment
    im_features = [[]]

    # pad each pyramid level to avoid edge problems
    for level in range(1, len(im_pyr)):
        padded_sm, padded_lg = pad_img_pair(im_pyr[level - 1], im_pyr[level], c)

        patches_sm = extract_patches_2d(padded_sm, (c.n_sm, c.n_sm))
        patches_lg = extract_patches_2d(padded_lg, (c.n_lg, c.n_lg))

        assert(patches_sm.shape[0] == im_pyr[level - 1].shape[0] * im_pyr[level - 1].shape[1])
        assert(patches_lg.shape[0] == im_pyr[level    ].shape[0] * im_pyr[level    ].shape[1])

        # discard second half of larger feature vector
        if not full_feat:
            patches_lg = patches_lg.reshape(patches_lg.shape[0], -1)[:, :c.num_ch * c.n_half]

        # concatenate small and large patches
        level_features = []
        imh, imw = im_pyr[level].shape[:2]
        for row in range(imh):
            for col in range(imw):
                level_features.append(np.hstack([
                    patches_sm[np.floor(row/2.) * np.ceil(imw/2.) + np.floor(col/2.)].flatten(),
                    patches_lg[row * imw + col].flatten()
                ]))

        assert(len(level_features) == imh * imw)

        # final feature array is n_pixels by f_length
        im_features.append(np.vstack(level_features))
    return im_features


def create_index(A_pyr, Ap_pyr_list, c):
    A_feat  = compute_feature_array(A_pyr,  c, full_feat=True)
    Ap_feat_list = []
    for Ap_pyr in Ap_pyr_list:
        Ap_feat_list.append(compute_feature_array(Ap_pyr, c, full_feat=False))

    flann = [pf.FLANN() for _ in xrange(c.max_levels)]
    flann_params = [list([]) for _ in xrange(c.max_levels)]
    As = [list([]) for _ in xrange(c.max_levels)]
    As_size = [list([]) for _ in xrange(c.max_levels)]

    for level in range(1, c.max_levels):
        print('Building index for level %d out of %d' % (level, c.max_levels - 1))
        As_list = []
        for Ap_feat in Ap_feat_list:
            As_list.append(np.hstack([A_feat[level], Ap_feat[level]]))

        As[level] = np.vstack(As_list)
        As_size[level] = As[level].shape
        flann_params[level] = flann[level].build_index(As[level], algorithm='kdtree')
    return flann, flann_params, As, As_size


def best_approximate_match(flann, params, BBp_feat):
    result, dists = flann.nn_index(BBp_feat, 1, checks=params['checks'])
    return result[0]


def extract_pixel_feature((im_sm_padded, im_lg_padded), (row, col), c, full_feat):
    # first extract full feature vector
    # since the images are padded, we need to add the padding to our indexing
    px_feat = np.hstack([im_sm_padded[np.floor(row/2.) : np.floor(row/2.) + 2 * c.pad_sm + 1, \
                                      np.floor(col/2.) : np.floor(col/2.) + 2 * c.pad_sm + 1].flatten(),
                         im_lg_padded[row : row + 2 * c.pad_lg + 1,
                                      col : col + 2 * c.pad_lg + 1].flatten()])
    if full_feat:
        return px_feat
    else:
        # only keep c.n_half pixels from second level
        return px_feat[:c.num_ch * ((c.n_sm * c.n_sm) + c.n_half)]


def best_coherence_match(As, (A_h, A_w), BBp_feat, s, im, px, Bp_w, c):
    assert(len(s) >= 1)

    row, col = px

    # construct iterables
    rs = []
    ims = []
    prs = []
    rows = np.arange(np.max([0, row - c.pad_lg]), row + 1, dtype=int)
    cols = np.arange(np.max([0, col - c.pad_lg]), np.min([Bp_w, col + c.pad_lg + 1]), dtype=int)

    for r_coord in product(rows, cols):
        # discard anything after current pixel
        if px2ix(r_coord, Bp_w) < px2ix(px, Bp_w):
            # p_r = s(r) + (q - r)

            # pr is an index in a given image Ap_list[img_num]
            pr = s[px2ix(r_coord, Bp_w)] + px - r_coord

            # i is a list of image nums for each pixel in Bp
            img_nums = im[px2ix(r_coord, Bp_w)]

            # discard anything outside the bounds of A/Ap lg
            if 0 <= pr[0] < A_h and 0 <= pr[1] < A_w:
                rs.append(np.array(r_coord))
                ims.append(img_nums)
                prs.append(Ap_px2ix(pr, img_nums, A_h, A_w))


    if not rs:
        # no good coherence match
        return (-1, -1), 0, (0, 0)

    rix = np.argmin(norm(As[np.array(prs)] - BBp_feat, ord=2, axis=1))
    r_star = rs[rix]
    i_star = ims[rix]
    # s[r_star] + (q - r-star)
    return s[px2ix(r_star, Bp_w)] + px - r_star, i_star, r_star


def compute_distance(AAp_p, BBp_q, weights):
    assert(AAp_p.shape == BBp_q.shape == weights.shape)
    return norm((AAp_p - BBp_q) * weights, ord=2)**2

