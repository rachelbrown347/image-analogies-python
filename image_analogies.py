import os
import pickle
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np

from algorithms import create_index, compute_feature_array, extract_pixel_feature, best_approximate_match,  \
                       best_coherence_match, compute_distance
from config import setup_vars, save_metadata
from img_preprocess import convert_to_YIQ, convert_to_RGB, compute_gaussian_pyramid, initialize_Bp, remap_luminance, \
                           compress_values, ix2px, px2ix, Ap_ix2px, Ap_px2ix, savefig_noborder, pad_img_pair



def img_setup(A_fname, Ap_fname_list, B_fname, out_path, c):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    A_orig = plt.imread(A_fname)
    B_orig = plt.imread(B_fname)
    assert(len(A_orig.shape) == len(B_orig.shape))  # same number of channels (for now)

    Ap_orig_list = []
    for Ap_fname in Ap_fname_list:
        Ap_orig = plt.imread(Ap_fname)
        assert(A_orig.shape == Ap_orig.shape) # src alignment
        Ap_orig_list.append(Ap_orig)

    # Make sure all images are floats on 0 to 1 scale
    scales = []
    for img in [A_orig, B_orig, Ap_orig[0]]:
        if np.max(img) > 1.0:
            scales.append(255.)
        else:
            scales.append(1.0)

    # Do conversions
    if c.convert:
        A_yiq = convert_to_YIQ( A_orig/scales[0])
        B_yiq = convert_to_YIQ( B_orig/scales[1])
        A = A_yiq[:, :, 0]
        B = B_yiq[:, :, 0]

        Ap_yiq_list = []
        Ap_list = []
        for Ap_orig in Ap_orig_list:
            Ap_yiq_list.append(convert_to_YIQ(Ap_orig/scales[2]))
            Ap_list.append(Ap_yiq_list[-1][:, :, 0])
    else:
        A = A_orig/scales[0]
        B = B_orig/scales[1]
        Ap_list = []
        for Ap_orig in Ap_orig_list:
            Ap_list.append(Ap_orig/scales[2])

    # Process input images
    if c.remap_lum:
        A, Ap_list = remap_luminance(A, Ap_list, B)

    if not c.init_rand:
        B_orig_pyr = compute_gaussian_pyramid(B, c.n_sm)

    A, B = compress_values(A, B, c.AB_weight)

    c.num_ch, c.padding_sm, c.padding_lg, c.weights = setup_vars(A)

    # Create Pyramids
    A_pyr = compute_gaussian_pyramid(A, c.n_sm)
    B_pyr = compute_gaussian_pyramid(B, c.n_sm)

    Ap_pyr_list = []
    for Ap in Ap_list:
        Ap_pyr_list.append(compute_gaussian_pyramid(Ap, c.n_sm))

    if c.convert:
        color_pyr_list = [compute_gaussian_pyramid(B_yiq, c.n_sm)]
    else:
        color_pyr_list = [compute_gaussian_pyramid(Ap_orig, c.n_sm) for Ap_orig in Ap_list]

    if len(A_pyr) != len(B_pyr):
        c.max_levels = min(len(A_pyr), len(B_pyr))
        warnings.warn('Warning: input images are very different sizes! The minimum number of levels will be used.')
    else:
        c.max_levels = len(B_pyr)

    # Create Random Initialization of Bp
    if c.init_rand:
        Bp_pyr = initialize_Bp(B_pyr, init_rand=True)
    else:
        Bp_pyr = initialize_Bp(B_orig_pyr, init_rand=False)

    return A_pyr, Ap_pyr_list, B_pyr, Bp_pyr, color_pyr_list, c


def image_analogies_main(A_fname, Ap_fname_list, B_fname, out_path, c, debug=False):
    # # This is the setup code
    begin_time = time.time()
    start_time = time.time()

    # Load images
    A_pyr, Ap_pyr_list, B_pyr, Bp_pyr, color_pyr_list, c = img_setup(A_fname, Ap_fname_list, B_fname, out_path, c)

    # Save parameters for reference
    names = ['A_fname', 'Ap_fname_list', 'B_fname', 'c.convert', 'c.remap_lum', 'c.init_rand', 'c.AB_weight', 'c.k']
    vars  = [ A_fname,   Ap_fname_list,   B_fname,   c.convert,   c.remap_lum,   c.init_rand,   c.AB_weight,   c.k ]
    save_metadata(out_path, names, vars)

    # Pull features from B
    B_features = compute_feature_array(B_pyr, c, full_feat=True)

    stop_time = time.time()
    print 'Environment Setup: %f' % (stop_time - start_time)

    # Build Structures for ANN
    start_time = time.time()

    flann, flann_params, As, As_size = create_index(A_pyr, Ap_pyr_list, c)

    stop_time = time.time()
    ann_time_total = stop_time - start_time
    print 'ANN Index Creation: %f' % (ann_time_total)

    # ##########################################################################################

    # # This is the Algorithm Code

    # now we iterate per pixel in each level
    for level in range(1, c.max_levels):
        start_time = time.time()
        ann_time_level = 0
        print('Computing level %d of %d' % (level, c.max_levels - 1))

        imh, imw = Bp_pyr[level].shape[:2]
        color_im_out = np.nan * np.ones((imh, imw, 3))

        s = []
        im = []

        if debug:
            # make debugging structures
            sa = []
            sc = []
            rstars = []
            p_src     = np.nan * np.ones((imh, imw, 3))
            img_src   = np.zeros((imh, imw))
            app_dist  = np.zeros((imh, imw))
            coh_dist  = np.zeros((imh, imw))
            app_color = np.array([1, 0, 0])
            coh_color = np.array([1, 1, 0])
            err_color = np.array([0, 0, 0])

            paths = ['%d_psrc.eps'    % (level),
                     '%d_appdist.eps' % (level),
                     '%d_cohdist.eps' % (level),
                     '%d_output.eps'  % (level),
                     '%d_imgsrc.eps'  % (level)]
            vars = [p_src, app_dist, coh_dist, Bp_pyr[level], img_src]

        for row in range(imh):
            for col in range(imw):
                px = np.array([row, col])

                # we pad on each iteration so Bp features will be more accurate
                Bp_pd = pad_img_pair(Bp_pyr[level - 1], Bp_pyr[level], c)
                BBp_feat = np.hstack([B_features[level][px2ix(px, imw), :],
                                      extract_pixel_feature(Bp_pd, px, c, full_feat=False)])

                assert(BBp_feat.shape == (As_size[level][1],))

                # Find Approx Nearest Neighbor
                ann_start_time = time.time()

                p_app_ix = best_approximate_match(flann[level], flann_params[level], BBp_feat)
                assert(p_app_ix < As_size[level][0])

                ann_stop_time = time.time()
                ann_time_level = ann_time_level + ann_stop_time - ann_start_time

                # translate p_app_ix back to row, col
                Ap_imh, Ap_imw = Ap_pyr_list[0][level].shape[:2]
                p_app, i_app = Ap_ix2px(p_app_ix, Ap_imh, Ap_imw)

                # is this the first iteration for this level?
                # then skip coherence step
                if len(s) < 1:
                    p = p_app
                    i = i_app

                # Find Coherence Match and Compare Distances
                else:
                    p_coh, i_coh, r_star = best_coherence_match(As[level], (Ap_imh, Ap_imw), BBp_feat, s, im, px, imw, c)

                    if np.allclose(p_coh, np.array([-1, -1])):
                        p = p_app
                        i = i_app

                    else:
                        AAp_feat_app = As[level][p_app_ix]
                        AAp_feat_coh = As[level][Ap_px2ix(p_coh, i_coh, Ap_imh, Ap_imw)]

                        d_app = compute_distance(AAp_feat_app, BBp_feat, c.weights)
                        d_coh = compute_distance(AAp_feat_coh, BBp_feat, c.weights)

                        if d_coh <= d_app * (1 + (2**(level - c.max_levels)) * c.k):
                            p = p_coh
                            i = i_coh
                        else:
                            p = p_app
                            i = i_app

                # Update Bp and s
                Bp_pyr[level][row, col] = Ap_pyr_list[i][level][tuple(p)]

                if not c.convert:
                    color_im_out[row, col, :] = color_pyr_list[i][level][tuple(p)]

                s.append(p)
                im.append(i)

                if debug:
                    sa.append(p_app)
                    if len(s) > 1 and not np.allclose(p_coh, np.array([-1, -1])):
                        sc.append(p_coh)
                        rstars.append(r_star)
                        app_dist[row, col] = d_app
                        coh_dist[row, col] = d_coh
                        if np.allclose(p, p_coh):
                            p_src[row, col] = coh_color
                        elif np.allclose(p, p_app):
                            p_src[row, col] = app_color
                        else:
                            print('Look, a bug! Squash it!')
                            raise
                    else:
                        sc.append((0, 0))
                        rstars.append((0, 0))
                        p_src[row, col] = err_color

        ann_time_total = ann_time_total + ann_time_level

        if debug:
            assert(len(im) == np.product(img_src.shape))
            img_src[:, :] = (np.array(im).astype(np.float64)/np.max(im)).reshape(img_src.shape)
            # Save debugging structures
            for path, var in zip(paths, vars):
                fig = plt.imshow(var, interpolation='nearest', cmap='gray')
                savefig_noborder(out_path + path, fig)
                plt.close()

            with open(out_path + '%d_srcs.pickle' % level, 'w') as f:
                pickle.dump([sa, sc, rstars, s, im], f)

        # Save color output images
        if c.convert:
            color_im_out = convert_to_RGB(np.dstack([Bp_pyr[level], color_pyr_list[i][level][:, :, 1:]]))
            color_im_out = np.clip(color_im_out, 0, 1)
        plt.imsave(out_path + 'level_%d_color.jpg' % level, color_im_out)
        plt.imsave(out_path + out_path.split('/')[-2] + '.jpg', color_im_out)

        stop_time = time.time()
        print 'Level %d time: %f' % (level, stop_time - start_time)
        print('Level %d ANN time: %f' % (level, ann_time_level))

    end_time = time.time()
    print 'Total time: %f' % (end_time - begin_time)
    print('ANN time: %f' % ann_time_total)
