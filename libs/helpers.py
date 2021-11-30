import numpy as np
import cv2
import matplotlib.pyplot as plt


def scrimg_postprocess(scr, dilation=7, nocare_area=21, blur = False, blursize=(5, 5), var = 6.0, custom_blur = None):

    # Compute foreground
    if scr.max() == 1:
        kernel_fg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation))
        fg = cv2.dilate(scr.astype(np.uint8), kernel=kernel_fg).astype(scr.dtype)
    else:
        fg = scr

    # Compute nocare area
    if nocare_area is None:
        nocare = None
    else:
        kernel_nc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (nocare_area, nocare_area))
        nocare = cv2.dilate(fg, kernel=kernel_nc) - fg
    if blur:
        fg = cv2.GaussianBlur(fg,ksize=blursize,sigmaX=var)
    elif custom_blur:
        c_kernel = np.array([[1,2,3,2,1],[2,4,9,4,2],[3,9,64,9,3],[2,4,9,4,2],[1,2,3,2,1]])
        c_kernel = c_kernel/np.sum(c_kernel)
        fg = cv2.filter2D(fg,ddepth=-1,kernel = c_kernel)

    return fg, nocare