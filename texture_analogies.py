import pyflann as pf
import numpy as np
from numpy.linalg import norm
from sklearn.feature_extraction.image import extract_patches_2d


def create_index(A_pyr, Ap_pyr, c):
    A_feat  = compute_feature_array(A_pyr,  c, full_feat=True)
    Ap_feat = compute_feature_array(Ap_pyr, c, full_feat=False)

    max_levels = len(A_pyr)

    flann = [pf.FLANN() for _ in xrange(max_levels)]
    flann_params = [list([]) for _ in xrange(max_levels)]
    As_size = [list([]) for _ in xrange(max_levels)]
    for level in range(1, max_levels):
        As = np.hstack([A_feat[level], Ap_feat[level]])
        As_size[level] = As.shape
        flann_params[level] = flann[level].build_index(As[level],
                                                       algorithm='autotuned',
                                                       target_precision=0.9,
                                                       sample_fraction=0.5)
    return flann, flann_params, As_size


def compute_feature_array(im_pyr, c, full_feat):
    # features will be organized like this:
    # sm_imA, lg_imA (channels shuffled C-style)

    # create a list of features for each pyramid level
    # level 0 is empty for indexing alignment
    im_features = [[]]

    # pad each pyramid level to avoid edge problems
    for level in range(1, len(im_pyr)):
        padded_sm = np.pad(im_pyr[level - 1], c.padding_sm, mode='symmetric')
        padded_lg = np.pad(im_pyr[level    ], c.padding_lg, mode='symmetric')

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


def extract_pixel_feature((im_sm_padded, im_lg_padded), (row, col), c, full_feat):
    # Extract B/Bp Feature Vector

    Bp_feat = np.hstack([im_sm_padded[np.floor(row/2.) : np.floor(row/2.) + 2 * c.pad_sm + 1, \
                               np.floor(col/2.) : np.floor(col/2.) + 2 * c.pad_sm + 1].flatten(),
                         im_lg_padded[row : row + c.pad_lg + 1, col : col + 2 * c.pad_lg + 1].flatten()])

    if full_feat:
        return Bp_feat
    else:
        # we pull the first pad + 1 rows and discard the last pad + 1 columns of the last row
        return Bp_feat[:-(c.num_ch * (c.pad_lg + 1))]


def best_approximate_match(flann, params, BBp_feat, num_px):
    result, dists = flann.nn_index(BBp_feat, 1, checks=params['checks'])
    assert(result[0] < num_px)
    return result[0]


def extract_coherence_neighborhood(ix_img, row, col, c):
    imh, imw = ix_img.shape

    # search through all synthesized pixels within most recent half-neighborhood
    row_start = 0 if row - c.pad_lg <= 0 else row - c.pad_lg
    col_start = 0 if col - c.pad_lg <= 0 else col - c.pad_lg
    col_end = imw if col + c.pad_lg >= imw else col + c.pad_lg
    neigh_end = col - imw if col - imw > - c.pad_lg - 1 else - c.pad_lg - 1
    set_r = ix_img[row_start: row + 1, col_start: col_end + 1].flatten()[:neigh_end]

    return set_r


def best_coherence_match(A_pd, Ap_pd, BBp_feat, s, (row, col), c):
    assert(np.sum(~np.isnan(s)) >= 1)

    min_sum = float('inf')
    r_star = np.nan
    for r_row in range(row - c.pad_lg, row + 1):
        for r_col in range(col - c.pad_lg, col + c.pad_lg + 1):
            # p = s(r) + (q - r)
            p_r = s[r_row, r_col] + (row, col) - (r_row, r_col)
            A_feat = extract_pixel_feature(A_pd, p_r, c, full_feat=True)
            Ap_feat = extract_pixel_feature(Ap_pd, p_r, c, full_feat=False)
            AAp_feat = np.hstack([A_feat, Ap_feat])

            new_sum = norm(AAp_feat - BBp_feat, ord=2)
            if new_sum < min_sum:
                min_sum = new_sum
                r_star = (r_row, r_col)
    if np.isnan(r_star):
        print((row,col), (r_row, r_col), p_r)

    return s[r_star[0], r_star[1]] + ((row, col) - r_star)


    # set_p = (s[set_r] + (q - set_r)).astype(int)
    #
    # #r_star = np.argmin(norm(As[set_p, :] - BBp_feat, ord=2, axis=1))
    #
    # min_sum = float('inf')
    # r_star = np.nan
    # for rix, p in enumerate(set_p):
    #     # only use p's that are inside the image
    #     if p < As.shape[0]:
    #         new_sum = norm(As[p, :] - BBp_feat, ord=2)
    #         if new_sum < min_sum:
    #             min_sum = new_sum
    #             r_star = rix
    #
    # if np.isnan(r_star):
    #     print(row, col, set_r, s[set_r], set_p)
    #     return -1 # blue
    #
    # if s[r_star] + (q - r_star) > As.shape[0]:
    #     print(row, col, set_r, s[set_r], set_p)
    #
    # return s[r_star] + (q - r_star)


def compute_distance(AAp_p, BBp_q, weights):
    diff = (AAp_p - BBp_q) * weights
    return np.sum((np.abs(diff))**2)

