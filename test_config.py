import numpy as np

from config import compute_weights, matlab_style_gauss2D

def test_compute_weights():
    n_sm, n_lg, n_half, num_ch = 3, 5, 12, 1

    gauss_sm = matlab_style_gauss2D((n_sm, n_sm), 0.5)
    gauss_lg = matlab_style_gauss2D((n_lg, n_lg), 1)

    w_sm = ((1./(n_sm**2)) * gauss_sm).flatten()
    w_lg = ((1./(n_lg**2)) * gauss_lg).flatten()
    w_half = ((1./n_half) * gauss_lg.flatten()[:n_half])

    assert(len(w_sm) == 9)
    assert(len(w_lg) == 25)
    assert(len(w_half) == 12)

    assert(np.isclose(np.sum(w_sm), 1./9))
    assert(np.isclose(np.sum(w_lg), 1./25))
    assert(np.sum(w_half) < 0.5 * (1./n_half))

    correct_weights = np.hstack([w_sm, w_lg, w_sm, w_half])
    weights = compute_weights(n_sm, n_lg, n_half, num_ch)

    print('weights', weights)
    print('correct weights', correct_weights)

    assert(np.allclose(weights, correct_weights))