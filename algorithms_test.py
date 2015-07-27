import numpy as np
import matplotlib.pyplot as plt

from img_preprocess import convert_to_YIQ, compute_gaussian_pyramid, pad_img_pair
from algorithms import compute_feature_array, extract_pixel_feature, best_coherence_match, \
                              best_coherence_match_orig, create_index
import config as c


def test_compute_feature_array():
    # Make and test 2D image

    sm = 0.5 * np.ones((4, 5))
    sm[0, 0] = 0
    lg = 0.3 * np.ones((7, 10))
    lg[0, 0] = 1

    # First test a full feature

    c.num_ch, c.padding_sm, c.padding_lg, c.weights = c.setup_vars(lg)
    feat = compute_feature_array([sm, lg], c, full_feat=True)

    sm_0 = np.array([[  0,   0, 0.5],
                     [  0,   0, 0.5],
                     [0.5, 0.5, 0.5]])

    lg_0 = np.array([[0.3, 0.3, 0.3, 0.3, 0.3],
                     [0.3,   1,   1, 0.3, 0.3],
                     [0.3,   1,   1, 0.3, 0.3],
                     [0.3, 0.3, 0.3, 0.3, 0.3],
                     [0.3, 0.3, 0.3, 0.3, 0.3]])

    correct_feat_0 = np.hstack([sm_0.flatten(), lg_0.flatten()])

    assert(len(feat) == 2)
    assert(feat[0] == [])
    assert(feat[1].shape == (7 * 10, 9 + 25))
    assert(np.allclose(feat[1][0], correct_feat_0))

    # Next test a half feature

    feat = compute_feature_array([sm, lg], c, full_feat=False)

    correct_feat_0 = np.hstack([sm_0.flatten(), lg_0.flatten()[:c.n_half]])

    assert(len(feat) == 2)
    assert(feat[0] == [])
    assert(feat[1].shape == (7 * 10, 9 + c.n_half))
    assert(np.allclose(feat[1][0], correct_feat_0))

    # Make and test 3D image

    sm = 0.5 * np.ones((4, 5, 3))
    sm[0, 0, :] = 0
    lg = 0.3 * np.ones((7, 10, 3))
    lg[0, 0] = 1

    # First test a full feature

    c.num_ch, c.padding_sm, c.padding_lg, c.weights = c.setup_vars(lg)
    feat = compute_feature_array([sm, lg], c, full_feat=True)

    sm_3d_0 = np.dstack([sm_0, sm_0, sm_0])
    lg_3d_0 = np.dstack([lg_0, lg_0, lg_0])
    correct_feat_0 = np.hstack([sm_3d_0.flatten(), lg_3d_0.flatten()])

    assert(len(feat) == 2)
    assert(feat[0] == [])
    assert(feat[1].shape == (7 * 10, (9 + 25) * 3))
    assert(np.allclose(feat[1][0], correct_feat_0))

    # Next test a half feature

    feat = compute_feature_array([sm, lg], c, full_feat=False)

    correct_feat_0 = np.hstack([sm_3d_0.flatten(), lg_3d_0.flatten()[:c.n_half * 3]])

    assert(len(feat) == 2)
    assert(feat[0] == [])
    assert(feat[1].shape == (7 * 10, (9 + c.n_half) * 3))
    assert(np.allclose(feat[1][0], correct_feat_0))


def test_extract_pixel_feature():
    # Make and test 2D image
    sm = 0.5 * np.ones((4, 5))
    sm[0, 0] = 0
    lg = 0.3 * np.ones((7, 10))
    lg[0, 0] = 1

    c.num_ch, c.padding_sm, c.padding_lg, c.weights = c.setup_vars(lg)

    sm_0 = np.array([[  0,   0, 0.5],
                     [  0,   0, 0.5],
                     [0.5, 0.5, 0.5]])

    lg_0 = np.array([[0.3, 0.3, 0.3, 0.3, 0.3],
                     [0.3,   1,   1, 0.3, 0.3],
                     [0.3,   1,   1, 0.3, 0.3],
                     [0.3, 0.3, 0.3, 0.3, 0.3],
                     [0.3, 0.3, 0.3, 0.3, 0.3]])

    im_padded = pad_img_pair(sm, lg, c)

    # First test full feature

    correct_feat_0_0 = np.hstack([sm_0.flatten(), lg_0.flatten()])
    feat = extract_pixel_feature(im_padded, (0, 0), c, full_feat=True)
    assert(np.allclose(feat, correct_feat_0_0))

    # Now test half feature

    correct_feat_0_0 = np.hstack([sm_0.flatten(), lg_0.flatten()[:c.n_half]])
    feat = extract_pixel_feature(im_padded, (0, 0), c, full_feat=False)
    assert(np.allclose(feat, correct_feat_0_0))


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


def test_best_coherence_match():
    # make A_pd, Ap_pd, BBp_feat, s
    A_orig  = plt.imread('./test_images/test_best_coherence_match_A.jpg')
    Ap_orig = plt.imread('./test_images/test_best_coherence_match_Ap.jpg')

    A  = convert_to_YIQ( A_orig/255.)[:, :, 0]
    Ap = convert_to_YIQ(Ap_orig/255.)[:, :, 0]

    A_pyr  = compute_gaussian_pyramid( A, min_size=3)
    Ap_pyr = compute_gaussian_pyramid(Ap, min_size=3)

    imh, imw = A.shape[:2]

    c.num_ch, c.padding_sm, c.padding_lg, c.weights = c.setup_vars(A)
    c.max_levels = len(A_pyr)

    A_pd  = pad_img_pair( A_pyr[-2],  A_pyr[-1], c)
    Ap_pd = pad_img_pair(Ap_pyr[-2], Ap_pyr[-1], c)

    flann, flann_params, As, As_size = create_index(A_pyr, Ap_pyr, c)

    # BBp_feat cases: all corners and middle
    indices = [(1, 1),
               (1, imw - 1),
               (imh - 1, 1),
               (imh - 1, imw - 1),
               (np.floor(imh/2.).astype(int), np.floor(imw/2.).astype(int))]

    for row, col in indices:
        num_px = row * imw + col
        s_rows = np.random.random_integers(num_px, size=num_px) - 1
        s_cols = np.random.random_integers(num_px, size=num_px) - 1
        s = [(rr, cc) for rr, cc in zip(s_rows, s_cols)]
        s[(row - 1) * imw + col - 1] = (row - 1, col - 1)

        Bs_feat = np.hstack([extract_pixel_feature( A_pd, (row, col), c, full_feat=True),
                             extract_pixel_feature(Ap_pd, (row, col), c, full_feat=False)])

        p_coh_orig, r_star_orig = best_coherence_match_orig(A_pd, Ap_pd, Bs_feat, s, (row, col, imw), c)
        p_coh_new, r_star_new = best_coherence_match(As[-1], A.shape, Bs_feat, s, (row, col, imw), c)

        try:
            assert(p_coh_orig == (row, col))
            assert(p_coh_new == (row, col))
        except:
            print('row, col, p_coh_orig, p_coh_new, s', row, col, p_coh_orig, p_coh_new, s)
            As_feat = np.hstack([extract_pixel_feature( A_pd, p_coh_orig, p_coh_new, c, full_feat=True),
                                 extract_pixel_feature(Ap_pd, p_coh_orig, p_coh_new, c, full_feat=False)])
            print('As_feat', As_feat)
            print('Bs_feat', Bs_feat)
            assert(False)