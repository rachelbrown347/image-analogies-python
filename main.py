import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings

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
    A_orig = plt.imread('./images/lf_originals/half_size/lights-src.jpg')
    Ap_orig = plt.imread('./images/lf_originals/half_size/lights-filt.jpg')
    B_orig = plt.imread('./images/lf_originals/half_size/boat-src.jpg')
    out_path = './images/lf_originals/output/boat/'

    # A_orig = plt.imread('./images/crosshatch/crosshatch_blurred.jpg')
    # Ap_orig = plt.imread('./images/crosshatch/crosshatch.jpg')
    # B_orig = plt.imread('./images/crosshatch/piano_gradient.jpg')
    # Bp_fname = './images/crosshatch/output/piano_gradient_crosshatch.jpg'
    # out_path = './images/crosshatch/output/'

    # A_orig = plt.imread('./../sample-images/analogies/wood_orig_sm.jpg')
    # Ap_orig = plt.imread('./../sample-images/analogies/real_wood_orig_sm.jpg')
    # B_orig = plt.imread('./../sample-images/analogies/wood_relit_sm_2p5_2.jpg')
    # Bp_fname = './../sample-images/analogies/output/real_wood_relit_sm_2p5_2_k25.jpg'

    artistic_filter = True

    assert(A_orig.shape == Ap_orig.shape)
    assert(len(A_orig.shape) == len(B_orig.shape)) # same number of channels

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

    if len(A_pyr) != len(B_pyr):
        max_levels = min(len(A_pyr), len(B_pyr))
        warnings.warn('Warning: input images are very different sizes! The minimum number of levels will be used.')
    else:
        max_levels = len(B_pyr)

    if not artistic_filter:
        Ap_color_pyr = compute_gaussian_pyramid(Ap_orig, c.n_sm)
        Bp_color_pyr = compute_gaussian_pyramid(np.nan * np.ones(B_orig.shape), c.n_sm)

    # Create Random Initialization of Bp

    Bp_pyr = initialize_Bp(B_pyr, init_rand=False)

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
                  table_number = 16,
                  key_size = 30,
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
    ann_time_total = 0

    for level in range(1, max_levels):
        start_time = time.time()
        ann_time_level = 0
        print('Computing level %d of %d' % (level, max_levels - 1))

        imh, imw = B_pyr[level].shape[0:2]
        s = np.array([])

        # debugging structures
        p_src = np.nan * np.ones((imh, imw))
        coh_dist = np.nan * np.ones((imh, imw))
        app_dist = np.nan * np.ones((imh, imw))

        for row in range(imh):
            for col in range(imw):
                q = row * imw + col

                # Extract B/Bp Feature Vector
                # pad each pyramid level to avoid edge problems

                Bp_sm = np.pad(Bp_pyr[level - 1], c.padding_sm, mode='reflect')
                Bp_lg = np.pad(Bp_pyr[level    ], c.padding_lg, mode='reflect')

                Bp_feat = np.hstack([Bp_sm[np.floor(row/2.) : np.floor(row/2.) + 2 * c.pad_sm + 1, \
                                           np.floor(col/2.) : np.floor(col/2.) + 2 * c.pad_sm + 1].flatten(),
                                     Bp_lg[row : row + c.pad_lg + 1, col : col + 2 * c.pad_lg + 1].flatten()])
                # we pull the first pad + 1 rows and discard the last pad + 1 columns of the last row
                BBp_feat = np.hstack([B_feat[level][q, :], Bp_feat[:-(c.num_ch * (c.pad_lg + 1))]])

                assert(BBp_feat.shape[0] == As[level].shape[1])

                # Find Approx Nearest Neighbor
                ann_start_time = time.time()

                p_app = best_approximate_match(flnn[level], BBp_feat)

                ann_stop_time = time.time()
                ann_time_level = ann_time_level + ann_stop_time - ann_start_time
                ann_time_total = ann_time_total + ann_time_level

                # is this the first iteration for this level?
                # then skip coherence step

                if len(s) < 1:
                    p = p_app

                # Find Coherence Match and Compare Distances

                else:
                    p_coh = best_coherence_match(As[level], BBp_feat, s, q, c.n_half)

                    if np.isnan(p_coh):
                        p = p_app
                        p_src[row, col] = 0.5
                    else:
                        d_app = compute_distance(As[level][p_app, :], BBp_feat, c.weights)
                        d_coh = compute_distance(As[level][p_coh, :], BBp_feat, c.weights)

                        app_dist[row, col] = d_app
                        coh_dist[row, col] = d_coh

                        if d_coh <= d_app * (1 + 2**(level - max_levels) * c.k):
                            p = p_coh
                            p_src[row, col] = 1.0
                        else:
                            p = p_app
                            p_src[row, col] = 0.0

                # Get Pixel Value from Ap
                p_col = p % Ap_pyr[level].shape[1]
                p_row = (p - p_col) // Ap_pyr[level].shape[1]

                p_val = Ap_pyr[level][p_row, p_col]

                # Set Bp and Update s

                Bp_pyr[level][row, col] = p_val

                if not artistic_filter:
                    Bp_color_pyr[level][row, col] = Ap_color_pyr[level][p_row, p_col]

                s = np.append(s, p)

        #print(np.isnan(p_src))
        #print(np.isnan(app_dist))
        #print(np.isnan(coh_dist))
        # print(np.max(p_src), np.min(p_src))
        # print(np.max(app_dist), np.min(app_dist))
        # print(np.max(coh_dist), np.min(coh_dist))

        plt.imsave(out_path + '%d_psrc.jpg' % (level), p_src, cmap='gray')
        plt.imsave(out_path + '%d_appdist.jpg' % (level), app_dist, cmap='gray')
        plt.imsave(out_path + '%d_cohdist.jpg' % (level), coh_dist, cmap='gray')
        plt.imsave(out_path + '%d_output.jpg' % (level), Bp_pyr[level], cmap='gray')
        #plt.imsave(Bp_fname[:-4] + '_bw.jpg', Bp_pyr[level], cmap='gray')

        stop_time = time.time()
        print 'Level %d time: %f' % (level, stop_time - start_time)
        print('Level %d ANN time: %f' % (level, ann_time_level))

    # Output Image

    if artistic_filter:
        #im_out = convert_to_RGB(np.dstack([Bp_pyr[-1], B_yiq[:, :, 1:]]))
        #im_out = np.clip(im_out, 0, 1)
        im_out = Bp_pyr[-1]
    else:
        im_out = Bp_color_pyr[-1]

    # if c.convert:
    #     im_out = convert_to_RGB(Bp_color_pyr[-1])
    # else:
    #     im_out = Bp_pyr[-1]

    end_time = time.time()
    print 'Total time: %f' % (end_time - begin_time)
    print('ANN time: %f' % ann_time_total)

    #plt.imshow(im_out)
    #plt.show()

    #plt.imsave(Bp_fname, im_out, cmap='gray')