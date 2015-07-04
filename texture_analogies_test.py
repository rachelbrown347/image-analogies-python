import numpy as np
from texture_analogies import compute_features

import config as c

def test_compute_features():
    # Make and test 2D image

    sm = 0.5 * np.ones((4, 5))
    sm[0, 0] = 0
    lg = 0.3 * np.ones((7, 10))
    lg[0, 0] = 1

    # First test a full feature

    c.num_ch, c.padding_sm, c.padding_lg, c.weights = c.setup_vars(lg)
    feat = compute_features([sm, lg], c, full_feat=True)

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

    feat = compute_features([sm, lg], c, full_feat=False)

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
    feat = compute_features([sm, lg], c, full_feat=True)

    sm_3d_0 = np.dstack([sm_0, sm_0, sm_0])
    lg_3d_0 = np.dstack([lg_0, lg_0, lg_0])
    correct_feat_0 = np.hstack([sm_3d_0.flatten(), lg_3d_0.flatten()])

    assert(len(feat) == 2)
    assert(feat[0] == [])
    assert(feat[1].shape == (7 * 10, (9 + 25) * 3))
    assert(np.allclose(feat[1][0], correct_feat_0))

    # Next test a half feature

    feat = compute_features([sm, lg], c, full_feat=False)

    correct_feat_0 = np.hstack([sm_3d_0.flatten(), lg_3d_0.flatten()[:c.n_half * 3]])

    assert(len(feat) == 2)
    assert(feat[0] == [])
    assert(feat[1].shape == (7 * 10, (9 + c.n_half) * 3))
    assert(np.allclose(feat[1][0], correct_feat_0))


