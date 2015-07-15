import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pickle

from img_preprocess import compute_gaussian_pyramid


def show_pair(src_img, out_img, s_a, s_c, s_r, s):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.imshow(out_img, interpolation='nearest')

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.imshow(src_img, interpolation='nearest')

    plt.show(block=False)

    while True:
        c_out = fig1.ginput(n=1, timeout=0)[0]
        px = tuple(np.rint(c_out[::-1]).astype(int))

        ax1.clear()
        ax2.clear()
        ax1.imshow(out_img, interpolation='nearest')
        ax2.imshow(src_img, interpolation='nearest')

        imh, imw = out_img.shape[:2]
        ix = px[0] * imw + px[1]
        r_star = s_r[ix]
        sr_star = s[r_star[0] * imw + r_star[1]]

        print('sr', s_r[:10])
        print('rx', r_star)
        print('s[rx]', sr_star)

        print(imw, ix, px)

        if s_a[ix] == s_c[ix]:
            ec = 'black'
        elif s[ix] == s_a[ix]:
            ec = 'red'
        elif s[ix] == s_c[ix]:
            ec = 'yellow'
        else:
            print('You made a mistake, try again!')
            print('s[ix], s_a[ix], s_c[ix]:', s[ix], s_a[ix], s_c[ix])
            raise

        #print('px, s[ix], s_a[ix], s_c[ix]:', px, s[ix], s_a[ix], s_c[ix])
        ax1.add_patch(Rectangle(np.array(       px)[::-1] - 0.5, 1, 1, fill=None, alpha=1, linewidth=2, edgecolor=ec))
        ax1.add_patch(Rectangle(np.array(   r_star)[::-1] - 0.5, 1, 1, fill=None, alpha=1, linewidth=2, edgecolor='blue'))
        ax2.add_patch(Rectangle(np.array(s_a[ix])[::-1] - 0.5, 1, 1, fill=None, alpha=1, linewidth=2, edgecolor='red'))
        ax2.add_patch(Rectangle(np.array(s_c[ix])[::-1] - 0.5, 1, 1, fill=None, alpha=1, linewidth=2, edgecolor='yellow'))
        ax2.add_patch(Rectangle(np.array(  sr_star)[::-1] - 0.5, 1, 1, fill=None, alpha=1, linewidth=2, edgecolor='blue'))
        fig1.canvas.draw()
        fig2.canvas.draw()


def load_imgs(src_path, out_path):
    src_img = plt.imread(src_path)
    src_pyr = compute_gaussian_pyramid(src_img, min_size = 3)

    out_pyr = [[]]
    sas = [[]]
    scs = [[]]
    srs = [[]]
    ss = [[]]

    for level in range(1, len(src_pyr)):
        with open(out_path + '%d_srcs.pickle' % level) as f:
            s_a, s_c, s_r, s = pickle.load(f)

        assert(len(s_a) == len(s_c) == len(s_r) == len(s))

        sas.append(s_a)
        scs.append(s_c)
        srs.append(s_r)
        ss.append(s)
        out_img = plt.imread(out_path + 'im_out_color_%d.jpg' % level)

        out_pyr.append(out_img)

    return src_pyr, out_pyr, sas, scs, srs, ss

# src_path = './images/lf_originals/half_size/fruit-src.jpg'
# out_path = './images/lf_originals/output/boat/matching_test/'

src_path = './images/lf_originals/half_size/fruit-filt.jpg'
out_path = './images/lf_originals/output/boat/full_alg_kmeans_kappa25/'

src_pyr, out_pyr, sas, scs, srs, ss = load_imgs(src_path, out_path)

level = 6
show_pair(src_pyr[level], out_pyr[level], sas[level], scs[level], srs[level], ss[level])



