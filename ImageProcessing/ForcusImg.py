import numpy as np
import math
import matplotlib.pyplot as plt
import pyautogui
import cv2
"""
GPU mode is not recommended.

Since it is not yet optimized, the CPU mode is faster at computing.
"""
try:
    import cupy as cp
    GPU = True
except ImportError:
    GPU = False

class ForcusImg:
    def __init__(self, gpu=False):
        if gpu:
            print("[Using GPU]")
        else:
            print("[Not using GPU]")
        self.gpu = gpu
        self.alpha = None
        self.img_shape = None
        self.bias = None
        
    def bias_func(self, img_shape):
        xlim = np.zeros(img_shape[0]*2)
        ylim = np.zeros(img_shape[1]*2)
        
        beta = self.alpha*10 - 2
        for i, x in enumerate(range(-int(img_shape[0]), int(img_shape[0]), 1)):
            xlim[i] = math.exp(-(x**2/(img_shape[0]*math.exp(beta))))
        for i, x in enumerate(range(-int(img_shape[1]), int(img_shape[1]), 1)):
            ylim[i] = math.exp(-(x**2/(img_shape[0]*math.exp(beta))))
        xlim = np.round(xlim, 2)
        ylim = np.round(ylim, 2)
        
        return xlim, ylim

    def forcus_img(self, img, point, alpha):
        if (img.shape[0]%2 != 0):
            img = img[:img.shape[0]-1, :]
        if (img.shape[1]%2 != 0):
            img = img[:, :img.shape[1]-1]
        if self.gpu:
            return self.forcus_img_gpu(img, point, alpha)
        if (img.shape == self.img_shape and self.alpha == alpha):
                bias = self.bias
        else:
            self.img_shape = img.shape
            self.alpha = alpha
            bias = np.zeros((img.shape[0]*2, img.shape[1]*2))
            xlim, ylim = self.bias_func(img.shape)
            for i in range(img.shape[0]*2):
                for j in range(img.shape[1]*2):
                    bias[i, j] = (xlim[i] * ylim[j])
            self.bias = bias
        
        flt = bias[img.shape[0]-point[0]:2*img.shape[0]-point[0],
                img.shape[1]-point[1]:2*img.shape[1]-point[1]]
        out_img = img * flt
        
        return out_img

    def forcus_img_gpu(self, img, point, alpha):
        if (img.shape == self.img_shape and self.alpha == alpha):
                bias_cp = self.bias
        else:
            self.img_shape = img.shape
            self.alpha = alpha
            bias = np.zeros((img.shape[0]*2, img.shape[1]*2))
            xlim, ylim = self.bias_func(img.shape)
            img_cp = cp.asarray(img)
            bias_cp = cp.asarray(bias)
            xlim_cp = cp.asarray(xlim)
            ylim_cp = cp.asarray(ylim)
            for i in range(img.shape[0]*2):
                for j in range(img.shape[1]*2):
                    bias_cp[i, j] = (xlim_cp[i] * ylim_cp[j])
            self.bias = cp.asnumpy(bias_cp)
        
        flt_cp = bias_cp[img_cp.shape[0]-point[0]:2*img_cp.shape[0]-point[0],
                         img_cp.shape[1]-point[1]:2*img_cp.shape[1]-point[1]]
        out_img_cp = img_cp * flt_cp
        out_img = cp.asnumpy(out_img_cp)
        return out_img


if __name__ == "__main__":
    alpha = 0.5
    #fi = ForcusImg(GPU)
    fi = ForcusImg()
    scale = 0.5
    for i in range(50):
        img = pyautogui.screenshot()
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
        point = pyautogui.position()
        point = (int(point.y*scale), int(point.x*scale))
        out_img = fi.forcus_img(img, point, alpha)
        plt.imshow(out_img, cmap="gray")
        plt.pause(0.001)
