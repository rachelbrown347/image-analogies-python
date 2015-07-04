import numpy as np

from img_preprocess import convert_to_YIQ, convert_to_RGB


def test_converts():
    np.random.seed(0xba5eba11)
    img = np.random.rand(25, 25, 3)
    assert np.allclose(convert_to_RGB(convert_to_YIQ(img)), img, atol=0.05)
    img = np.ones((25, 25, 3))
    assert np.allclose(convert_to_RGB(convert_to_YIQ(img)), img)
    img = np.zeros((25, 25, 3))
    assert np.allclose(convert_to_RGB(convert_to_YIQ(img)), img)