import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d

def compute_features(A_pyr, c, full_feat):
    # features will be organized like this:
    # sm_imA, lg_imA (channels shuffled C-style)

    # create a list of features for each pyramid level
    # level 0 is empty
    A_features = [[]]

    # pad each pyramid level to avoid edge problems
    for level in range(1, len(A_pyr)):
        padded_sm = np.pad(A_pyr[level - 1], c.padding_sm, mode='reflect')
        padded_lg = np.pad(A_pyr[level    ], c.padding_lg, mode='reflect')

        patches_sm = extract_patches_2d(padded_sm, (c.n_sm, c.n_sm))
        patches_lg = extract_patches_2d(padded_lg, (c.n_lg, c.n_lg))

        # discard second half of larger feature vector
        if not full_feat:
            patches_lg = patches_lg.reshape(patches_lg.shape[0], -1)[:, : c.num_ch * c.n_half]

        # concatenate small and large patches
        level_features = []
        imh, imw = A_pyr[level].shape[:2]
        for row in range(imh):
            for col in range(imw):
                level_features.append(np.hstack([
                    patches_sm[np.floor(row/2.) * np.ceil(imw/2.) + np.floor(col/2.)].flatten(),
                    patches_lg[row * imw + col].flatten()
                ]))

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


def compute_distance(AAp_p, BBp_q, weights):
    diff = (AAp_p - BBp_q) * weights
    return np.sum((np.abs(diff))**2)