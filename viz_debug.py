import pickle

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from img_preprocess import compute_gaussian_pyramid


def show_pair(src_img, out_img, sa, sc, rs, s):
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
        rstar = rs[ix]
        s_rstar = s[rstar[0] * imw + rstar[1]]

        print('source row/col', s[ix])
        print('output row/col', px)

        if sa[ix] == sc[ix]:
            ec = 'black'
        elif s[ix] == sa[ix]:
            ec = 'red'
        elif s[ix] == sc[ix]:
            ec = 'yellow'
        else:
            print('You made a mistake, try again!')
            print('s[ix], sa[ix], sc[ix]:', s[ix], sa[ix], sc[ix])
            raise

        #print('px, s[ix], sa[ix], sc[ix]:', px, s[ix], sa[ix], sc[ix])
        ax1.add_patch(Rectangle(np.array(     px)[::-1] - 0.5, 1, 1, fill=None, alpha=1, linewidth=2, edgecolor=ec))
        ax1.add_patch(Rectangle(np.array(  rstar)[::-1] - 0.5, 1, 1, fill=None, alpha=1, linewidth=2, edgecolor='blue'))
        ax2.add_patch(Rectangle(np.array( sa[ix])[::-1] - 0.5, 1, 1, fill=None, alpha=1, linewidth=2, edgecolor='red'))
        ax2.add_patch(Rectangle(np.array( sc[ix])[::-1] - 0.5, 1, 1, fill=None, alpha=1, linewidth=2, edgecolor='yellow'))
        ax2.add_patch(Rectangle(np.array(s_rstar)[::-1] - 0.5, 1, 1, fill=None, alpha=1, linewidth=2, edgecolor='blue'))
        fig1.canvas.draw()
        fig2.canvas.draw()


def load_imgs(src_path, out_path):
    src_img = plt.imread(src_path)
    src_pyr = compute_gaussian_pyramid(src_img, min_size = 3)

    out_pyr = [[]]
    sas = [[]]
    scs = [[]]
    rss = [[]]
    ss = [[]]
    ims = [[]]

    for level in range(1, len(src_pyr)):
        with open(out_path + '%d_srcs.pickle' % level) as f:
            sa, sc, rstars, s, im = pickle.load(f)

        assert(len(sa) == len(sc) == len(rstars) == len(s) == len(im))

        sas.append(sa)
        scs.append(sc)
        rss.append(rstars)
        ss.append(s)
        ims.append(im)
        out_img = plt.imread(out_path + 'im_out_color_%d.jpg' % level)

        out_pyr.append(out_img)

    return src_pyr, out_pyr, sas, scs, rss, ss, ims


src_path = './images/lf_originals/half_size/fruit-filt.jpg'
out_path = './images/lf_originals/output/boat/working_test_2/'
level = 6

src_pyr, out_pyr, sas, scs, rss, ss, ims = load_imgs(src_path, out_path)
print('Images Loaded! Level = %d' % level)

show_pair(src_pyr[level], out_pyr[level], sas[level], scs[level], rss[level], ss[level], ims[level])



