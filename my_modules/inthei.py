"""Return interfacial height of a multiphaseflow image."""

#from numpy import zeros, ones, floor_divide, array, min
import numpy as np
from scipy import ndimage as ndi
from skimage import color
from skimage.io import imread, ImageCollection
from skimage import morphology
from skimage.filters import threshold_otsu


def imread_x(f,path,box=(0,792,0,85)):
    """Return a collection of grayscale cropped images."""
    return color.rgb2gray(imread(
            r'D:\Mestrado\Imagens\Antigas'
            + path + '(%d).jpg'%f)[box[0]:box[1],
                                   box[2]:box[3]])

def im_start(pic,path,box):
    """Return an array of images."""
    im_matrix = ImageCollection(
            pic,path=path,box=box,
            load_func=imread_x)
    return(im_matrix.concatenate())

def bg_im(im):
    """Return image background."""
    bg = np.zeros(im.shape,dtype=int)
    bg[:,0:np.floor_divide(im.shape[1],2)] = 255
    return(bg)

def bg_removal(im_mat):
    """Remove the background of an image."""
    bg = bg_im(im_mat[0])
    im_int = im_mat * 255
    im_int = im_int.astype(int)
    no_bg = im_int - bg
    no_bg[no_bg < 0] = 0
    return(no_bg)

def im_proc(im):
    """Apply series of morphological procedures on image."""
    th = threshold_otsu(im)
    im_bin = im > th
    return(ndi.binary_fill_holes(
                morphology.closing(
                im_bin,np.ones((3,3)))))

def interface_height(im,scale):
    """Return a dictionary of interfacial heights."""
    D = {n:np.array([i for i, j in enumerate(k) if j]) for n, k in enumerate(im)}
    height = []
    for v in D.values():
        if len(v) == 0:
            height.append(height[-1])
        else:
            height.append(np.min(v))
    return(D,height)

"""Fourier analysis bit lacking"""