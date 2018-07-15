"""
RAW Batch Conversion and processing to color neutralization using gray world algorithm
"""

# various imports
import numpy as np
import rawpy
import os
import ColorspaceChange as cspace

class RAWBatchConversion:
    def __init__(self, input_folder_name, output_folder_name):
        print("\n Input Folder Name: {0}\n Output Folder Name {1}".format(input_folder_name, output_folder_name))
        return

    def convertRAWFiles(self, raw_img):
        with rawpy.imread(raw_img) as raw:
            raw_rgb = raw.postprocess()

        # get lab color space image
        cs = cspace.ColorspaceChange()
        linrgb = cs.srgb2lin(raw_rgb)
        lab = cs.rgb2lab(linrgb)

        # call color correction with gray-world assumption
        corrected_lab = self.colorCorrection(lab)

        # convert back to sRGB from color corrected lab
        linrgb = cs.lab2rgb(corrected_lab)
        corrected_sRGB = cs.lin2srgb(linrgb)

        return corrected_sRGB

    def colorCorrection(self, lab_img):
        # get average a and b from lab
        mean_a = np.mean(lab_img[:, :, 1].ravel())
        mean_b = np.mean(lab_img[:, :, 2].ravel())

        # compute a_delta and b_delta
        a_delta = mean_a * (lab_img[:, :, 0] / 100) * 1.1
        b_delta = mean_b * (lab_img[:, :, 0] / 100) * 1.1

        # shift the colors in a and b channels
        lab_img[:, :, 1] = lab_img[:, :, 1] - a_delta
        lab_img[:, :, 2] = lab_img[:, :, 2] - b_delta

        return lab_img
