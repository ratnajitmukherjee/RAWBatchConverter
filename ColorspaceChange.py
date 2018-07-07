"""
Class to convert images from RGB to LAB color space and convert lin2srgb and vice versa
"""
import numpy as np

class ColorspaceChange:
    def __init__(self):
        pass

    def srgb2lin(self, srgb):
        # normalize the image
        rgb_norm = srgb.astype(np.float32) / 255.0
        linRGB = np.zeros(rgb_norm.shape, dtype=np.float32)

        # constants
        t = 0.04045
        a = 0.055

        # perform inverse gamma correction
        linRGB[rgb_norm <= t] = rgb_norm[rgb_norm <= t] / 12.92
        linRGB[rgb_norm > t] = np.power(((rgb_norm[rgb_norm > t] + a) / (1 + a)), 2.4)

        return linRGB

    def lin2srgb(self, linrgb):
        sRGB = np.zeros(linrgb.shape, dtype=np.float32)
        # constants
        t = 0.0031308
        a = 0.055

        # perform forward gamma correction
        sRGB[linrgb <= t] = linrgb[linrgb <= t] * 12.92
        sRGB[linrgb > t] = np.power(((1 + a) * linrgb[linrgb > t]), (1 / 2.4)) - a

        return sRGB

    def rgb2lab(self, linrgb):
        # perform linRGB2LAB color space operation
        RGB_s = np.array(linrgb) * 100

        X = RGB_s[:, :, 0] * 0.4124 + RGB_s[:, :, 1] * 0.3576 + RGB_s[:, :, 2] * 0.1805
        Y = RGB_s[:, :, 0] * 0.2126 + RGB_s[:, :, 1] * 0.7152 + RGB_s[:, :, 2] * 0.0722
        Z = RGB_s[:, :, 0] * 0.0193 + RGB_s[:, :, 1] * 0.1192 + RGB_s[:, :, 2] * 0.9505
        XYZ = np.dstack((X, Y, Z))

        XYZ[:, :, 0] /= 95.047  # ref_X =  95.047   Observer= 2°, Illuminant= D65
        XYZ[:, :, 1] /= 100.0  # ref_Y = 100.000
        XYZ[:, :, 2] /= 108.883  # ref_Z = 108.883

        XYZ[XYZ > 0.008856] = np.power(XYZ[XYZ > 0.008856], (1/3))
        XYZ[XYZ <= 0.008856] = 7.787 * XYZ[XYZ <= 0.008856] + (16 / 116)

        L = 116 * XYZ[:, :, 1] - 16
        a = 500 * (XYZ[:, :, 0] - XYZ[:, :, 1])
        b = 200 * (XYZ[:, :, 1] - XYZ[:, :, 2])

        Lab = np.dstack((L, a, b))

        return Lab