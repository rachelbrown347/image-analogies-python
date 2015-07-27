# from image_analogies import image_analogies_main
# import config as c
#
# c.convert, c.remap_lum, c.init_rand, c.k = [False, False, True, 0.5]
# if c.remap_lum: assert c.convert
#
#
# A_fname  = './images/texture_fabric/fabric_orig_sm_2_gb.jpg'
# B_fname  = './images/texture_fabric/model_orig_sm_1_gb.jpg'
#
# Ap_fname = './images/texture_fabric/fabric_orig_sm_2.jpg'
# out_path = './images/texture_fabric/output/src2_gbfabric_tgt1_gbmodel/kappa2/'
#
# weights = [0.1, 0.3, 0.5]
# w_str = ['0p1_nonrand', '0p3_nonrand', '0p5_nonrand']
# inits = [False, False, False]
#
# for w, weight, init in zip(w_str, weights, inits):
#     c.AB_weight = weight
#     c.init_rand = init
#     image_analogies_main(A_fname, [Ap_fname], B_fname, out_path + w + '/', c, debug=True)


from glob import glob
from image_analogies import image_analogies_main
import config as c

c.convert, c.remap_lum, c.init_rand, c.AB_weight, c.k = [False, False, True, 1, 0.5]
if c.remap_lum: assert c.convert

for mat in ['granite', 'desk']:
    A_fname = './images/texture_angles/angle_orig_sm.jpg'
    B_fname = './images/texture_angles/angle_relit_color.jpg'

    out_path = './images/texture_output/color_test/%s/' % mat

    Ap_fname_list = glob('./images/texture_originals/frames/%s-*' % mat)
    #Ap_fname_list = ['./images/texture_originals/frames/%s-001.jpg' % mat]

    image_analogies_main(A_fname, Ap_fname_list, B_fname, out_path, c, debug=True)
