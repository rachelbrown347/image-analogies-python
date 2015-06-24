import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import feature_extraction
from skimage.transform import pyramid_gaussian
import sys
import time


def compute_gaussian_pyramid(A, n_sm):
    A_h, A_w = A.shape[0:2]
    min_size = np.min([A_h, A_w])

    levels = 0
    while min_size > n_sm:
        min_size = np.floor(min_size/2.)
        levels += 1

    A_pyr = list(pyramid_gaussian(A, max_layer=levels))

    A_pyr.reverse() # smallest to largest
    return A_pyr


def compute_features(A_pyr, n_sm, n_lg, n_half, full_feat):
    # features will be organized like this:
    # sm_imA, lg_imA (channels shuffled C-style)

    # create a list of features for each pyramid level
    A_features = []
    A_features.append([])

    # pad each pyramid level to avoid edge problems
    pd_lg = np.floor(n_lg/2.)
    pd_sm = np.floor(n_sm/2.)

    for level in range(1, len(A_pyr)):
        padded_sm = np.pad(A_pyr[level - 1], ((pd_sm, pd_sm), (pd_sm, pd_sm), (0, 0)), mode='reflect')
        patches_sm = feature_extraction.image.extract_patches_2d(padded_sm, (n_sm, n_sm))
        patches_sm = patches_sm.reshape(patches_sm.shape[0], \
                                        patches_sm.shape[1] * patches_sm.shape[2], \
                                        patches_sm.shape[3]) # flatten 2D

        padded_lg = np.pad(A_pyr[level], ((pd_lg, pd_lg), (pd_lg, pd_lg), (0, 0)), mode='reflect')
        patches_lg = feature_extraction.image.extract_patches_2d(padded_lg, (n_lg, n_lg))
        patches_lg = patches_lg.reshape(patches_lg.shape[0], \
                                        patches_lg.shape[1] * patches_lg.shape[2], \
                                        patches_lg.shape[3]) # flatten 2D

        if not full_feat:
            # discard second half of larger feature vector
            patches_lg = patches_lg[:, : n_half, :]

        # concatenate small and large patches
        level_features = []
        imh, imw = A_pyr[level].shape[0:2]
        for r in range(imh):
            for c in range(imw):
                # pull out corresponding pixel features, flatten, and stack
                level_features.append( np.hstack( [patches_sm[np.ceil(r/2) * np.ceil(imw/2) + \
                                                              np.ceil(c/2)].flatten(), \
                                                   patches_lg[r * imw + c].flatten()] ) )
        # final feature array is n_pixels by f_length
        A_features.append(np.vstack(level_features))
    return A_features


def best_approximate_match(flnn, BBp_feat):
    indices, dist = flnn.knnSearch(BBp_feat.astype(np.float32), 1, params={})
    return indices[0][0]


def best_coherence_match(AAp_feat, BBp_feat, s, q, n_half):
    assert(len(s) >= 1)

    # search through all synthesized pixels within most recent half-neighborhood
    num_r = min(len(s), n_half)
    set_r = np.arange(q - num_r, q, dtype=int)
    set_a = (s[set_r] + q - set_r).astype(int)

    min_r = np.argmin(np.sum(np.abs(AAp_feat[set_a, :] - BBp_feat)**2, axis=1))

    return s[min_r] + q - min_r


def compute_weights(n_sm, n_lg, n_half, num_channels):
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

    gauss_sm   = matlab_style_gauss2D((n_sm, n_sm), 0.5)
    gauss_lg   = matlab_style_gauss2D((n_lg, n_lg),   1)

    w_sm = (1./(n_sm * n_sm)) * np.dstack([gauss_sm, gauss_sm, gauss_sm]).flatten()
    w_lg = (1./(n_lg * n_lg)) * np.dstack([gauss_lg, gauss_lg, gauss_lg]).flatten()
    w_half = (1./n_half) * np.dstack([gauss_lg, gauss_lg, gauss_lg]).flatten()[: n_half * num_channels]

    return np.hstack([w_sm, w_lg, w_sm, w_half])


def compute_distance(AAp_p, BBp_q, weights):
    diff = (AAp_p - BBp_q) * weights
    return np.sum((np.abs(diff))**2)


if __name__ == '__main__':

    argv = sys.argv

    if len(argv) != 5:
        print "Usage: python", argv[0], "[imageA] [imageA'] [imageB] [output_file]"
        exit()

    # Read image files
    # setup A, A', B, B', s
    A_fname  = (argv[1])
    Ap_fname = (argv[2])
    B_fname  = (argv[3])
    Bp_fname = argv[4]

    # # This is all the setup code

    # Load Images

    begin_time = time.time()

    A_orig = plt.imread(A_fname)
    Ap_orig = plt.imread(Ap_fname)
    B_orig = plt.imread(B_fname)

    A  = A_orig/255.
    Ap = Ap_orig/255.
    B  = B_orig/255.

    # Set Parameters and Variables

    n_sm = 3    # coarse scale neighborhood size
    n_lg = 5    # fine scale neighborhood size
    k    = 1.5  # 0.5 <= k <= 5 for texture synthesis

    n_half = np.floor((n_lg * n_lg)/2.) # half feature for fine scale
    num_channels = Ap.shape[2]

    weights = compute_weights(n_sm, n_lg, n_half, num_channels)

    # Create Pyramids

    start_time = time.time()

    A_pyr  = compute_gaussian_pyramid(A,  n_sm)
    Ap_pyr = compute_gaussian_pyramid(Ap, n_sm)
    B_pyr  = compute_gaussian_pyramid(B,  n_sm)

    # Create Random Initialization of Bp

    Bp_flat = []
    for level in range(len(A_pyr)):
        Bp_flat.append(np.random.rand(B_pyr[level].shape[0] * B_pyr[level].shape[1], B_pyr[level].shape[2]))

    stop_time = time.time()
    print 'Pyramid Creation: %f' % (stop_time - start_time)

    # Compute Feature Vectors

    start_time = time.time()

    A_feat  = compute_features(A_pyr,  n_sm, n_lg, n_half, full_feat=True)
    Ap_feat = compute_features(Ap_pyr, n_sm, n_lg, n_half, full_feat=False)
    B_feat  = compute_features(B_pyr,  n_sm, n_lg, n_half, full_feat=True)

    # Build Structures for ANN

    params = dict(algorithm=1, trees=4)
    flnn = [list([]) for _ in xrange(len(A_pyr))]
    As = [list([]) for _ in xrange(len(A_pyr))]
    for level in range(1, len(A_feat)):
        As[level] = np.hstack([A_feat[level], Ap_feat[level]])
        flnn[level] = cv2.flann_Index(As[level].astype(np.float32), params)

    stop_time = time.time()
    print 'Feature Creation: %f' % (stop_time - start_time)

    # # This is the Algorithm Code

    # now we iterate per pixel in each level

    num_levels = len(A_pyr)
    for level in range(1, num_levels):
        start_time = time.time()
        ann_time = 0
        print('Computing level %d of %d' % (level, num_levels - 1))

        imh, imw = A_pyr[level].shape[0:2]
        s = np.array([])

        for r in range(imh):
            for c in range(imw):
                q = r * imw + c
                q_sm = np.ceil(r/2.) * np.ceil(imw/2.) + np.ceil(c/2.)

                # THIS NEEDS TO BE FIXED TO USE NEIGHBORHOOD SIZE VARIABLES

                # Create B/Bp Feature Vector

                Bp_feat = np.hstack([Bp_flat[level - 1][  q_sm - 4:   q_sm + 4 + 1, :].flatten(), \
                                     Bp_flat[level    ][q - n_half: q + n_half + 1, :].flatten()])
                BBp_feat = np.hstack([B_feat[level][q, :], Bp_feat])

                # Find Approx Nearest Neighbor
                ann_start_time = time.time()

                p_app = best_approximate_match(flnn[level], BBp_feat)

                ann_stop_time = time.time()
                ann_time = ann_time + (ann_stop_time - ann_start_time)

                # is this the first iteration for this level?
                # then skip coherence step
                if len(s) <= 1:
                    p = p_app

                # Find Coherence Match and Compare Distances

                else:
                    p_coh = best_coherence_match(As[level], BBp_feat, s, q, n_half)

                    d_app = compute_distance(As[level][p_app, :], BBp_feat, weights)
                    d_coh = compute_distance(As[level][p_coh, :], BBp_feat, weights)

                    if d_coh <= d_app * (1 + 2**(level - num_levels) * k):
                        p = p_coh
                    else:
                        p = p_app

                # Get Pixel Value from Ap

                p_col = p % imw
                p_row = (p - p_col) / imw
                p_val = Ap_pyr[level][p_row, p_col, :]

                # Set Bp and Update s

                Bp_flat[level][q, :] = p_val
                np.append(s, p)

        stop_time = time.time()
        print 'Level %d time: %f' % (level, stop_time - start_time)

    # Output Image

    im_out = Bp_flat[-1].reshape(Ap.shape)

    end_time = time.time()
    print 'Total time: %f' % (end_time - begin_time)
    print('ANN time: %f' % ann_time)

    #plt.imshow(im_out)
    #plt.show()

    plt.imsave(Bp_fname, im_out)