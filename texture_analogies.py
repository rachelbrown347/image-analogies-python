from itertools import product
import pyflann as pf
import numpy as np
from numpy.linalg import norm
from sklearn.feature_extraction.image import extract_patches_2d

from img_preprocess import px2ix


def pad_img_pair(img_sm, img_lg, c):
    return [np.pad(img_sm, c.padding_sm, mode='symmetric'),
        np.pad(img_lg, c.padding_lg, mode='symmetric')]


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


def create_index(A_pyr, Ap_pyr, c):
    A_feat  = compute_feature_array(A_pyr,  c, full_feat=True)
    Ap_feat = compute_feature_array(Ap_pyr, c, full_feat=False)

    flann = [pf.FLANN() for _ in xrange(c.max_levels)]
    flann_params = [list([]) for _ in xrange(c.max_levels)]
    As = [list([]) for _ in xrange(c.max_levels)]
    As_size = [list([]) for _ in xrange(c.max_levels)]
    for level in range(1, c.max_levels):
        print('Building index for level %d out of %d' % (level, c.max_levels - 1))
        As[level] = np.hstack([A_feat[level], Ap_feat[level]])
        As_size[level] = As[level].shape
        flann_params[level] = flann[level].build_index(As[level], algorithm='kmeans')
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


def best_coherence_match_orig(A_pd, Ap_pd, BBp_feat, s, (row, col, Bp_w), c):
    assert(len(s) >= 1)

    # Handle edge cases
    row_min = np.max([0, row - c.pad_lg])
    row_max = row + 1
    col_min = np.max([0, col - c.pad_lg])
    col_max = np.min([Bp_w, col + c.pad_lg + 1])

    min_sum = float('inf')
    r_star = (np.nan, np.nan)
    for r_row in np.arange(row_min, row_max, dtype=int):
        col_end = col if r_row == row else col_max
        for r_col in np.arange(col_min, col_end, dtype=int):
            s_ix = r_row * Bp_w + r_col

            # p = s(r) + (q - r)
            p_r = np.array(s[s_ix]) + np.array([row, col]) - np.array([r_row, r_col])

            # check that p_r is inside the bounds of A/Ap lg
            A_h, A_w = A_pd[1].shape[:2] - 2 * c.pad_lg

            if 0 <= p_r[0] < A_h and 0 <= p_r[1] < A_w:
                AAp_feat = np.hstack([extract_pixel_feature( A_pd, p_r, c, full_feat=True),
                                      extract_pixel_feature(Ap_pd, p_r, c, full_feat=False)])

                assert(AAp_feat.shape == BBp_feat.shape)

                new_sum = norm(AAp_feat - BBp_feat, ord=2)**2

                if new_sum <= min_sum:
                    min_sum = new_sum
                    r_star = np.array([r_row, r_col])
    if np.isnan(r_star).any():
        return (-1, -1), (0, 0)

    # s[r_star] + (q - r_star)
    return tuple(s[r_star[0] * Bp_w + r_star[1]] + (np.array([row, col]) - r_star)), tuple(r_star)


def best_coherence_match(As, A_shape, BBp_feat, s, (row, col, Bp_w), c):
    assert(len(s) >= 1)

    A_h, A_w = A_shape[:2]

    # construct iterables
    rs = []
    prs = []
    rows = np.arange(np.max([0, row - c.pad_lg]), row + 1, dtype=int)
    cols = np.arange(np.max([0, col - c.pad_lg]), np.min([Bp_w, col + c.pad_lg + 1]), dtype=int)

    for r_coord in product(rows, cols):
        # discard anything after current pixel
        if px2ix(r_coord, Bp_w) < px2ix((row, col), Bp_w):
            # p_r = s(r) + (q - r)
            pr = s[px2ix(r_coord, Bp_w)] + np.array([row, col]) - r_coord

            # discard anything outside the bounds of A/Ap lg
            if 0 <= pr[0] < A_h and 0 <= pr[1] < A_w:
                rs.append(r_coord)
                prs.append(px2ix(pr, A_w))

    if not rs:
        # no good coherence match
        return (-1, -1), (0, 0)

    rix = np.argmin(norm(As[np.array(prs)] - BBp_feat, ord=2, axis=1))
    r_star = rs[rix]
    # s[r_star] + (q - r-star)
    return tuple(s[px2ix(r_star, Bp_w)] + np.array([row, col]) - r_star), r_star


def compute_distance(AAp_p, BBp_q, weights):
    assert(AAp_p.shape == BBp_q.shape == weights.shape)
    return norm((AAp_p - BBp_q) * weights, ord=2)**2

