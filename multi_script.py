from glob import glob
from image_analogies import image_analogies_main
import config as c

c.convert, c.remap_lum, c.init_rand, c.AB_weight = [False, False, True, 1]
if c.remap_lum: assert c.convert

# materials = ['desk', 'granite', 'paint', 'wood', 'woodfloor']
# angles = ['5_4', '5_10', '5_4_p20', '20_10_p20']
# weights = [0.1, 0.3, 0.5]
# w_paths = ['0p1', '0p3', '0p5']

materials = ['paint', 'granite', 'woodfloor', 'wood', 'desk']
angles = ['5_4', '5_10', '5_4_p20', '20_10_p20']
kappa = [0.5, 5]
kappa_str = ['k0p5', 'k5']
path_tag = 'w1_random'

for mat in materials:
    for angle in angles:
        for k, ks in zip(kappa, kappa_str):
            print('Computing material %s, angle %s' % (mat, angle))
            A_fname = './images/texture_angles/angle_orig_sm.jpg'
            B_fname = './images/texture_angles/angle_relit_sm_' + angle + '.jpg'

            out_path = './images/texture_output/single_frame/' + '_'.join([mat, angle, ks, path_tag]) + '/'

            #Ap_fname_list = glob('./images/texture_originals/frames/%s-*' % mat)
            Ap_fname_list = ['./images/texture_originals/frames/%s-001.jpg' % mat]

            c.k = k
            image_analogies_main(A_fname, Ap_fname_list, B_fname, out_path, c, debug=True)
