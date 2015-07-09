from itertools import chain, repeat
import numpy as np


# Set Parameters and Variables
convert = True  # Convert to YIQ
k       = 25     # 0.5 <= k <= 5 for texture synthesis
n_sm    = 3     # coarse scale neighborhood size
n_lg    = 5     # fine scale neighborhood size
n_half  = np.floor((n_lg * n_lg)/2.) # half feature for fine scale
pad_sm  = np.floor(n_sm/2.)
pad_lg  = np.floor(n_lg/2.)

num_ch = None
padding_sm = None
padding_lg = None
weights = None


def setup_vars(img):
    assert 1 < len(img.shape) <= 3
    num_ch = 1 if len(img.shape) == 2 else img.shape[2]

    if num_ch == 1:
        padding_sm = int(pad_sm)
        padding_lg = int(pad_lg)
    else:
        padding_sm = ((pad_sm, pad_sm), (pad_sm, pad_sm), (0, 0))
        padding_lg = ((pad_lg, pad_lg), (pad_lg, pad_lg), (0, 0))

    weights = compute_weights(n_sm, n_lg, n_half, num_ch)

    return num_ch, padding_sm, padding_lg, weights


def compute_weights(n_sm, n_lg, n_half, num_ch):
    def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        m, n = [(ss-1.)/2. for ss in shape]
        y, x = np.ogrid[-m:m+1,-n:n+1]
        h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
        h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    gauss_sm = matlab_style_gauss2D((n_sm, n_sm), 0.5)
    gauss_lg = matlab_style_gauss2D((n_lg, n_lg),   1)

    w_sm = (1./(n_sm * n_sm)) * np.dstack(list(chain.from_iterable(repeat(e, num_ch) for e in [gauss_sm]))).flatten()
    w_lg = (1./(n_lg * n_lg)) * np.dstack(list(chain.from_iterable(repeat(e, num_ch) for e in [gauss_lg]))).flatten()
    w_half = (1./n_half) * np.dstack(list(chain.from_iterable(repeat(e, num_ch) for e in [gauss_lg]))).flatten()[: n_half * num_ch]

    return np.hstack([w_sm, w_lg, w_sm, w_half])

