import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pickle

from img_preprocess import compute_gaussian_pyramid


def show_pair(src_img, out_img, s_a, s_c, s):
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
        s_ix = px[0] * imw + px[1]

        print(imw, s_ix, px)

        if s_a[s_ix] == s_c[s_ix]:
            ec = 'blue'
        elif s[s_ix] == s_a[s_ix]:
            ec = 'red'
        elif s[s_ix] == s_c[s_ix]:
            ec = 'yellow'
        else:
            print('You made a mistake, try again!')
            print('s[s_ix], s_a[s_ix], s_c[s_ix]:', s[s_ix], s_a[s_ix], s_c[s_ix])
            raise

        print('px, s[s_ix], s_a[s_ix], s_c[s_ix]:', px, s[s_ix], s_a[s_ix], s_c[s_ix])
        ax1.add_patch(Rectangle(np.array(       px)[::-1] - 0.5, 1, 1, fill=None, alpha=1, edgecolor=ec))
        ax2.add_patch(Rectangle(np.array(s_a[s_ix])[::-1] - 0.5, 1, 1, fill=None, alpha=1, edgecolor='red'))
        ax2.add_patch(Rectangle(np.array(s_c[s_ix])[::-1] - 0.5, 1, 1, fill=None, alpha=1, edgecolor='yellow'))
        fig1.canvas.draw()
        fig2.canvas.draw()


def load_imgs(src_path, out_path):
    src_img = plt.imread(src_path)
    src_pyr = compute_gaussian_pyramid(src_img, min_size = 3)

    out_pyr = [[]]
    sas = [[]]
    scs = [[]]
    ss = [[]]

    for level in range(1, len(src_pyr)):
        with open(out_path + '%d_srcs.pickle' % level) as f:
            s_a, s_c, s = pickle.load(f)

        print(np.product(src_pyr[level].shape[:2]))

        if level == 1:
            print('s', s)
            print('sa', s_a)
            print('sc', s_c)

        sas.append(s_a)
        scs.append(s_c)
        ss.append(s)
        out_img = plt.imread(out_path + 'im_out_color_%d.jpg' % level)

        print(len(s_a), len(s_c), len(s), np.product(out_img.shape[:2]))
        assert(len(s_a) == len(s_c) == len(s) == np.product(out_img.shape[:2]))

        out_pyr.append(out_img)

    return src_pyr, out_pyr, sas, scs, ss


# src_path = './images/lf_originals/half_size/fruit-src.jpg'
# out_path = './images/lf_originals/output/boat/matching_test/'

src_path = './images/lf_originals/half_size/fruit-filt.jpg'
out_path = './images/lf_originals/output/boat/full_alg_kmeans_2_test/'

src_pyr, out_pyr, sas, scs, ss = load_imgs(src_path, out_path)

level = 5
show_pair(src_pyr[level], out_pyr[level], sas[level], scs[level], ss[level])


