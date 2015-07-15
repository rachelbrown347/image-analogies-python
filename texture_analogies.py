import pyflann as pf
import numpy as np
from numpy.linalg import norm
from sklearn.feature_extraction.image import extract_patches_2d


def pad_img_pair(img_sm, img_lg, c, noise=False):
    if noise:
        assert(len(img_sm.shape) == 2)
        pd_shape_sm = img_sm.shape[:2] + 2 * c.pad_sm
        pd_shape_lg = img_lg.shape[:2] + 2 * c.pad_lg
        pd_sm = np.random.rand(np.product(pd_shape_sm)).reshape(pd_shape_sm)
        pd_lg = np.random.rand(np.product(pd_shape_lg)).reshape(pd_shape_lg)
        pd_sm[c.pad_sm:-c.pad_sm, c.pad_sm:-c.pad_sm] = img_sm
        pd_lg[c.pad_lg:-c.pad_lg, c.pad_lg:-c.pad_lg] = img_lg
        return [pd_sm, pd_lg]
    else:
        return [np.pad(img_sm, c.padding_sm, mode='symmetric'),
                np.pad(img_lg, c.padding_lg, mode='symmetric')]


def pad_img(img, padding, noise=False):
    if noise:
        assert(len(img.shape) == 2)
        pd_shape = img.shape[:2] + 2 * padding
        pd_img = np.random.rand(np.product(pd_shape)).reshape(pd_shape)
        pd_img[padding:-padding, padding:-padding] = img
        return pd_img
    else:
        return np.pad(img, padding, mode='symmetric')


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

    max_levels = len(A_pyr)

    flann = [pf.FLANN() for _ in xrange(max_levels)]
    flann_params = [list([]) for _ in xrange(max_levels)]
    As_size = [list([]) for _ in xrange(max_levels)]
    for level in range(1, max_levels):
        print('Building index for level %d out of %d' % (level, max_levels - 1))
        As = np.hstack([A_feat[level], Ap_feat[level]])
        As_size[level] = As.shape
        flann_params[level] = flann[level].build_index(As, algorithm='kmeans')
    return flann, flann_params, As_size


def best_approximate_match(flann, params, BBp_feat):
    result, dists = flann.nn_index(BBp_feat, 1, checks=params['checks'])
    return result[0]


def extract_pixel_feature((im_sm_padded, im_lg_padded), (row, col), c, full_feat):
    # Single channel only
    assert(len(im_sm_padded.shape) == 2)

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


def best_coherence_match(A_pd, Ap_pd, BBp_feat, s, (row, col, Bp_w), c):
    assert(len(s) >= 1)

    # Handle edge cases
    row_min = np.max([0, row - c.pad_lg])
    row_max = row + 1
    col_min = np.max([0, col - c.pad_lg])
    col_max = np.min([Bp_w, col + c.pad_lg + 1])

    min_sum = float('inf')
    r_star = None
    for r_row in np.arange(row_min, row_max, dtype=int):
        col_end = col if r_row == row else col_max
        for r_col in np.arange(col_min, col_end, dtype=int):
            s_ix = r_row * Bp_w + r_col

            # p = s(r) + (q - r)
            p_r = np.array(s[s_ix]) + np.array([row, col]) - np.array([r_row, r_col])

            # check that p_r is inside the bounds of A/Ap lg
            A_h, A_w = A_pd[1].shape - 2 * c.pad_lg

            if 0 <= p_r[0] < A_h and 0 <= p_r[1] < A_w:
                AAp_feat = np.hstack([extract_pixel_feature( A_pd, p_r, c, full_feat=True),
                                      extract_pixel_feature(Ap_pd, p_r, c, full_feat=False)])

                assert(AAp_feat.shape == BBp_feat.shape)

                new_sum = norm(AAp_feat - BBp_feat, ord=2)**2

                if new_sum <= min_sum:
                    min_sum = new_sum
                    r_star = np.array([r_row, r_col])
    if r_star == None:
        return (-1, -1), (0, 0)

    # s[r_star] + (q - r_star)
    return tuple(s[r_star[0] * Bp_w + r_star[1]] + (np.array([row, col]) - r_star)), tuple(r_star)


def compute_distance(AAp_p, BBp_q, weights):
    assert(AAp_p.shape == BBp_q.shape == weights.shape)
    return norm((AAp_p - BBp_q) * weights, ord=2)**2

