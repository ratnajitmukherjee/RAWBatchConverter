"""
Class to convert images from RGB to LAB color space and convert lin2srgb and vice versa
"""
import numpy as np

class ColorspaceChange:
    def __init__(self, sRGB):
        print("Calling color space transformations")

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
        # PART 1: linRGB to XYZ conversion
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

        # PART 2: XYZ to LAB conversion
        L = 116 * XYZ[:, :, 1] - 16
        a = 500 * (XYZ[:, :, 0] - XYZ[:, :, 1])
        b = 200 * (XYZ[:, :, 1] - XYZ[:, :, 2])

        Lab = np.dstack((L, a, b))

        return Lab

    def lab2rgb(self, lab):
        # PART 1: LAB to XYZ conversion
        Xn = 95.047  # Observer = 2°, Illuminant = D65
        Yn = 100.000
        Zn = 108.883

        xyz = np.zeros(shape=lab.shape, dtype=np.float32)
        xyz[:,:,1] = (lab[:,:,0] + 16)/116
        xyz[:,:,0] = (lab[:,:,1] / 500) + xyz[:,:,1]
        xyz[:,:,2] = xyz[:,:,1] - lab[:,:, 2] / 200

        xyz[np.power(xyz, 3)> 0.008856] = np.power(xyz[np.power(xyz, 3)> 0.008856], 3)
        xyz[np.power(xyz, 3) <= 0.008856] = (xyz[np.power(xyz, 3) <= 0.008856] - 16/116) / 7.787

        xyz[:, :, 0] *= Xn  # multiplication by the reference numbers
        xyz[:, :, 1] *= Yn
        xyz[:, :, 2] *= Zn

        # PART 2: XYZ to linRGB conversion
        xyz_norm = np.divide(xyz, 100) # this normalizes XYZ (which is still in LAB scale)
        linrgb = np.zeros(xyz_norm.shape, dtype=np.float32)

        linrgb[:, :, 0] = xyz_norm[:, :, 0] * 3.2406 - 1.5372 * xyz_norm[:, :, 1] - 0.4986 * xyz_norm[:, :, 2]
        linrgb[:, :, 1] = -0.9689 * xyz_norm[:, :, 0] + 1.8758 * xyz_norm[:, :, 1] + 0.0415 * xyz_norm[:, :, 2]
        linrgb[:, :, 2] = 0.0557 * xyz_norm[:, :, 0] - 0.2040* xyz_norm[:, :, 1] + 1.0570 * xyz_norm[:, :, 2]

        return linrgb