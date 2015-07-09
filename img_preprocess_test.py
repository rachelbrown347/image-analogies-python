import numpy as np

from img_preprocess import convert_to_YIQ, convert_to_RGB, compute_gaussian_pyramid, initialize_Bp


def test_converts():
    np.random.seed(0xba5eba11)
    img = np.random.rand(25, 25, 3)
    assert np.allclose(convert_to_RGB(convert_to_YIQ(img)), img, atol=0.05)
    img = np.ones((25, 25, 3))
    assert np.allclose(convert_to_RGB(convert_to_YIQ(img)), img)
    img = np.zeros((25, 25, 3))
    assert np.allclose(convert_to_RGB(convert_to_YIQ(img)), img)


def test_initialize_Bp():
    np.random.seed(0xba5eba11)
    img = np.random.rand(25, 40)
    img_pyr = compute_gaussian_pyramid(img, min_size=3)

    init_pyr = initialize_Bp(img_pyr, init_rand=False)
    for layer in [0, 1]:
        assert(np.allclose(img_pyr[layer], init_pyr[layer]))
    for layer in range(2, len(img_pyr)):
        assert(not np.allclose(img_pyr[layer], init_pyr[layer]))

    init_pyr = initialize_Bp(img_pyr, init_rand=True)

    for layer in range(len(img_pyr)):
        assert(not np.allclose(img_pyr[layer], init_pyr[layer]))
