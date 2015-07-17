from main import image_analogies
import config as c


#angles = ['5_4', '5_10', '5_4_p20', '20_10_p20']

A_fname  = './images/texture_angles/angle_orig_sm.jpg'
Ap_fname = './images/texture_originals/desk_sm.jpg'
B_fname  = './images/texture_angles/angle_relit_sm_20_10_p20.jpg'
out_path = './images/texture_output/test/sm_20_10_p20/'

c.convert, c.remap_lum, c.init_rand, c.k = [False, False, True, 0.5]
if c.remap_lum: assert c.convert

image_analogies(A_fname, Ap_fname, B_fname, out_path, c, debug=False)

