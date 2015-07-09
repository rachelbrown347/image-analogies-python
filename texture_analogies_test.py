import numpy as np
from img_preprocess import compute_gaussian_pyramid
from texture_analogies import compute_feature_array, extract_pixel_feature, extract_coherence_neighborhood

import config as c

def test_compute_features():
    # Make and test 2D image

    sm = 0.5 * np.ones((4, 5))
    sm[0, 0] = 0
    lg = 0.3 * np.ones((7, 10))
    lg[0, 0] = 1

    # First test a full feature

    c.num_ch, c.padding_sm, c.padding_lg, c.weights = c.setup_vars(lg)
    feat = compute_feature_array([sm, lg], c, full_feat=True)

    sm_0 = np.array([[0.5, 0.5, 0.5],
                     [0.5,   0, 0.5],
                     [0.5, 0.5, 0.5]])

    lg_0 = np.array([[0.3, 0.3, 0.3, 0.3, 0.3],
                     [0.3, 0.3, 0.3, 0.3, 0.3],
                     [0.3, 0.3,   1, 0.3, 0.3],
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


def test_extract_Bp_feature():
    # Make and test 2D image
    sm = 0.5 * np.ones((4, 5))
    sm[0, 0] = 0
    lg = 0.3 * np.ones((7, 10))
    lg[0, 0] = 1

    c.num_ch, c.padding_sm, c.padding_lg, c.weights = c.setup_vars(lg)

    sm_0 = np.array([[0.5, 0.5, 0.5],
                     [0.5,   0, 0.5],
                     [0.5, 0.5, 0.5]])

    lg_0 = np.array([[0.3, 0.3, 0.3, 0.3, 0.3],
                     [0.3, 0.3, 0.3, 0.3, 0.3],
                     [0.3, 0.3,   1, 0.3, 0.3],
                     [0.3, 0.3, 0.3, 0.3, 0.3],
                     [0.3, 0.3, 0.3, 0.3, 0.3]])

    correct_feat_0 = np.hstack([sm_0.flatten(), lg_0.flatten()[:c.n_half]])

    Bp_pyr = [sm, lg]
    B_feat = [np.array([[0, 1, 2, 3, 4],
                        [5, 6, 7, 8, 9]]),
              np.array([[10, 11, 12, 13, 14],
                        [15, 16, 17, 18, 19]])]

    q, level, row, col = [1, 1, 0, 0]

    correct_BBp_feat = np.hstack([np.array([15, 16, 17, 18, 19]),
                                  correct_feat_0])

    BBp_feat = extract_pixel_feature(Bp_pyr, B_feat, q, level, row, col, c)

    assert(np.allclose(correct_BBp_feat, BBp_feat))


def test_extract_coherence_neighborhood():
    img = np.arange(5 * 4).reshape((5, 4))
    imh, imw = img.shape

    c.num_ch, c.padding_sm, c.padding_lg, c.weights = c.setup_vars(img)

    tests = [(0, 1),
             (0, imw - 1),
             (imh - 1, 0),
             (imh - 1, imw - 1),
             (np.floor(imh/2.), np.floor(imw/2.))]

    answers = [np.array([0]),
               np.array([1, 2]),
               np.array([8, 9, 10, 12, 13, 14]),
               np.array([9, 10, 11, 13, 14, 15, 17, 18]),
               np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])]

    for (row, col), answer in zip(tests, answers):
        neighborhood = extract_coherence_neighborhood(img, row, col, c)

        assert np.allclose(neighborhood, answer)





