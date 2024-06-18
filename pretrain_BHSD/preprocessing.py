import cv2
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

def load_image3d(data, n_slice, img_size): # return ndarray (1,22,256,256)

    def get_nifti(dat):
        dat = nib.load(dat)
        dat = dat.get_fdata()
        dat = np.transpose(dat, axes=(2, 0, 1))
        return dat

    img, mask = data
    img, mask = get_nifti(img), get_nifti(mask)

    mask[mask == 2] = 1 
    mask[mask == 3] = 1 
    mask[mask != 1] = 0 

    img, mask = np.stack([np.rot90(windowing(cv2.resize(slice, (img_size, img_size)))) for slice in img]), np.stack([np.rot90(cv2.resize(slice, (img_size, img_size), interpolation=cv2.INTER_NEAREST)) for slice in mask])
    
    img, mask = change_depth_siz(img, n_slice), change_depth_siz(mask, n_slice)
    # Fixed slice number
    # middle = img.shape[0] //2
    # num_imgs2 = n_slice//2
    # p1 = max(0, middle - num_imgs2)
    # p2 = min(img.shape[0], middle + num_imgs2)
    # img, mask = img[p1:p2], mask[p1:p2]
    return np.expand_dims(img, 0), np.expand_dims(mask, 0)

def windowing(img, lower=0, upper=120):
    X = np.clip(img.copy(), lower, upper)

    # min-max method
    if np.min(X) < np.max(X): # Xがすべて同じ値を持つとき、Falseとなる
        X = X - np.min(X)
        X = X / np.max(X)
    return X

def change_depth_siz(img, n_slice):
    desired_depth = n_slice
    current_depth = img.shape[0]
    depth = current_depth / desired_depth
    depth_factor = 1 / depth
    img_new = zoom(img, (depth_factor, 1, 1), mode='nearest')
    return img_new
