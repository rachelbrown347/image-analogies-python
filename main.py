import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

from img_preprocess import convert_to_YIQ, convert_to_RGB, compute_gaussian_pyramid, initialize_Bp, remap_luminance
from texture_analogies import compute_features, best_approximate_match, best_coherence_match, compute_distance
import config as c


if __name__ == '__main__':

    # argv = sys.argv
    #
    # if len(argv) != 5:
    #     print "Usage: python", argv[0], "[imageA] [imageA'] [imageB] [output_file]"
    #     exit()

    # Read image files

    # A_fname  = (argv[1])
    # Ap_fname = (argv[2])
    # B_fname  = (argv[3])
    # Bp_fname = argv[4]
    #
    # A_orig = plt.imread(A_fname)
    # Ap_orig = plt.imread(Ap_fname)
    # B_orig = plt.imread(B_fname)

    # Files for Testing
    # A_orig = plt.imread('./images/freud-src.jpg')
    # Ap_orig = plt.imread('./images/freud-filt.jpg')
    # B_orig = plt.imread('./images/shore-src.jpg')
    # Bp_fname = './output/shore-filt-freud-25-3.jpg'

    A_orig = plt.imread('./../sample-images/analogies/wood_orig_sm.jpg')
    Ap_orig = plt.imread('./../sample-images/analogies/real_wood_orig_sm.jpg')
    B_orig = plt.imread('./../sample-images/analogies/wood_relit_sm_2p5_2.jpg')
    Bp_fname = './../sample-images/analogies/output/real_wood_relit_sm_2p5_2_k25.jpg'

    artistic_filter = False

    assert(A_orig.shape == Ap_orig.shape == B_orig.shape)

    # This is all the setup code

    begin_time = time.time()
    start_time = time.time()

    # Do conversions

    if c.convert:
        A_yiq  = convert_to_YIQ( A_orig/255.)
        Ap_yiq = convert_to_YIQ(Ap_orig/255.)
        B_yiq  = convert_to_YIQ( B_orig/255.)
        A  =  A_yiq[:, :, 0]
        Ap = Ap_yiq[:, :, 0]
        B  =  B_yiq[:, :, 0]
    else:
        A  =  A_orig/255.
        Ap = Ap_orig/255.
        B  =  B_orig/255.

    c.num_ch, c.padding_sm, c.padding_lg, c.weights = c.setup_vars(A)

    # Remap Luminance

    #A, Ap = remap_luminance(A, Ap, B)

    # Create Pyramids

    A_pyr  = compute_gaussian_pyramid( A, c.n_sm)
    Ap_pyr = compute_gaussian_pyramid(Ap, c.n_sm)
    B_pyr  = compute_gaussian_pyramid( B, c.n_sm)

    if not artistic_filter:
        Ap_color_pyr = compute_gaussian_pyramid(Ap_orig, c.n_sm)
        Bp_color_pyr = compute_gaussian_pyramid(np.nan * np.ones(Ap_orig.shape), c.n_sm)

    # Create Random Initialization of Bp

    Bp_pyr = initialize_Bp(Ap_pyr, init_rand=True)

    stop_time = time.time()
    print 'Environment Setup: %f' % (stop_time - start_time)

    # Compute Feature Vectors

    start_time = time.time()

    A_feat  = compute_features(A_pyr,  c, full_feat=True)
    Ap_feat = compute_features(Ap_pyr, c, full_feat=False)
    B_feat  = compute_features(B_pyr,  c, full_feat=True)

    stop_time = time.time()
    print 'Feature Extraction: %f' % (stop_time - start_time)

    # Build Structures for ANN

    start_time = time.time()

    FLANN_INDEX_LSH = 4
    params = dict(algorithm = FLANN_INDEX_LSH,
                  table_number = 20,
                  key_size = 20,
                  multi_probe_level = 2)

    # FLANN_INDEX_KDTREE = 1
    # params = dict(algorithm = FLANN_INDEX_KDTREE,
    #                           trees=20)

    # FLANN_INDEX_AUTOTUNED = 5
    # params = dict(algorithm = FLANN_INDEX_AUTOTUNED,
    #               target_precision = 1,
    #               build_weight = 1,
    #               memory_weight = 1,
    #               sample_fraction = 1)

    flnn = [list([]) for _ in xrange(len(A_pyr))]
    As = [list([]) for _ in xrange(len(A_pyr))]
    for level in range(1, len(A_feat)):
        As[level] = np.hstack([A_feat[level], Ap_feat[level]])
        flnn[level] = cv2.flann_Index(As[level].astype(np.float32), params)

    stop_time = time.time()
    print 'ANN Index Creation: %f' % (stop_time - start_time)

    # # This is the Algorithm Code

    # now we iterate per pixel in each level

    num_levels = len(A_pyr)
    for level in range(1, num_levels):
        start_time = time.time()
        ann_time = 0
        print('Computing level %d of %d' % (level, num_levels - 1))

        imh, imw = A_pyr[level].shape[0:2]
        s = np.array([])

        for row in range(imh):
            for col in range(imw):
                q = row * imw + col

                # Create B/Bp Feature Vector
                # pad each pyramid level to avoid edge problems

                Bp_sm = np.pad(Bp_pyr[level - 1], c.padding_sm, mode='reflect')
                Bp_lg = np.pad(Bp_pyr[level    ], c.padding_lg, mode='reflect')

                Bp_feat = np.hstack([Bp_sm[np.floor(row/2.) : np.floor(row/2.) + 2 * c.pad_sm + 1, \
                                           np.floor(col/2.) : np.floor(col/2.) + 2 * c.pad_sm + 1].flatten(),
                                     Bp_lg[row : row + c.pad_lg + 1, col : col + 2 * c.pad_lg + 1].flatten()])
                # we pull the first pad + 1 rows and discard the last pad + 1 columns of the last row
                BBp_feat = np.hstack([B_feat[level][q, :], Bp_feat[:-(c.num_ch * (c.pad_lg + 1))]])

                assert(BBp_feat.shape[0] == A_feat[level].shape[1] + Ap_feat[level].shape[1])

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
                    p_coh = best_coherence_match(As[level], BBp_feat, s, q, c.n_half)

                    d_app = compute_distance(As[level][p_app, :], BBp_feat, c.weights)
                    d_coh = compute_distance(As[level][p_coh, :], BBp_feat, c.weights)

                    if d_coh <= d_app * (1 + 2**(level - num_levels) * c.k):
                        p = p_coh
                    else:
                        p = p_app

                # Get Pixel Value from Ap
                p_col = p % imw
                p_row = (p - p_col) // imw

                p_val = Ap_pyr[level][p_row, p_col]

                # Set Bp and Update s

                Bp_pyr[level][row, col] = p_val

                if not artistic_filter:
                    Bp_color_pyr[level][row, col] = Ap_color_pyr[level][p_row, p_col]

                np.append(s, p)

        plt.imsave(Bp_fname[:-4] + '_bw.jpg', Bp_pyr[level], cmap='gray')

        stop_time = time.time()
        print 'Level %d time: %f' % (level, stop_time - start_time)

    # Output Image

    if artistic_filter:
        im_out = convert_to_RGB(np.dstack([Bp_pyr[-1], B_yiq[:, :, 1:]]))
        im_out = np.clip(im_out, 0, 1)
    else:
        im_out = Bp_color_pyr[-1]

    # if c.convert:
    #     im_out = convert_to_RGB(Bp_color_pyr[-1])
    # else:
    #     im_out = Bp_pyr[-1]

    end_time = time.time()
    print 'Total time: %f' % (end_time - begin_time)
    print('ANN time: %f' % ann_time)

    #plt.imshow(im_out)
    #plt.show()

    plt.imsave(Bp_fname, im_out)