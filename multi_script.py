from image_analogies import image_analogies_main
import config as c

c.convert, c.remap_lum, c.init_rand, c.k = [False, False, True, 0.5]
if c.remap_lum: assert c.convert

materials = ['desk', 'granite', 'paint', 'wood', 'woodfloor']
angles = ['5_4', '5_10', '5_4_p20', '20_10_p20']
weights = [0.1, 0.3, 0.5]
w_paths = ['0p1', '0p3', '0p5']

# materials = ['wood']
# angles = ['5_4']
# weights = [0.1, 0.3]
# w_paths = ['0p1', '0p3']

for mat in materials:
    for angle in angles:
        for weight, w_path in zip(weights, w_paths):
            A_fname  = './images/texture_angles/angle_orig_sm.jpg'
            Ap_fname = './images/texture_originals/' + mat + '_sm.jpg'
            B_fname  = './images/texture_angles/angle_relit_sm_' + angle + '.jpg'
            out_path = './images/texture_output/' + '/'.join([mat, angle, w_path, ''])

            c.AB_weight = weight
            image_analogies_main(A_fname, Ap_fname, B_fname, out_path, c, debug=False)
