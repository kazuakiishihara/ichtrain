import numpy as np
import cv2
import pydicom
from scipy.ndimage import zoom
from torchvision import transforms as transforms

def load_dicom_image3d(filelist_str, n_slice, img_size):

    img_size = int(img_size * 0.875)
    
    dicom_files = filelist_str.split(';')[:-1]
    dicoms_all = []
    for d in dicom_files: # ImagePositionPatient属性を持っていない時の例外処理
        try:
            dicom_data = pydicom.dcmread(d)
            if hasattr(dicom_data, 'pixel_array') and hasattr(dicom_data, 'ImagePositionPatient'):
                dicoms_all.append(dicom_data)
        except Exception as e:
            print(f"Error reading DICOM file {d}: {str(e)}")

    first_date = None
    first_time = None

    for dcm in dicoms_all:
        study_date = dcm.StudyDate
        study_time = dcm.StudyTime

        # 最初の日付と時間を設定
        if first_date is None or (study_date < first_date) or (study_date == first_date and study_time < first_time):
            first_date = study_date
            first_time = study_time

    # 最初の日付と時間に一致するDICOMデータを抽出
    dicoms = []
    for dcm in dicoms_all:
        study_date = dcm.StudyDate
        study_time = dcm.StudyTime

        if study_date == first_date and study_time == first_time:
            dicoms.append(dcm)
    
    z_pos = [float(d.ImagePositionPatient[-1]) for d in dicoms] # 実際の頭部CTはImagePositionPatient[-1]
    img_list = [cv2.resize(d.pixel_array, (img_size, img_size)) for d in dicoms]
    img_shape = img_list[0].shape
    img_list = [cv2.resize(img, (img_size, img_size)) for img in img_list if img.shape == img_shape]
    img_list = [img for _, img in sorted(zip(z_pos, img_list), key=lambda x: x[0])]
    img = np.stack(img_list)
    
    # SSS
    # middle = len(dicoms)//2
    # num_imgs2 = n_slice//2
    # p1 = max(0, middle - num_imgs2)
    # p2 = min(len(dicoms), middle + num_imgs2)
    # img = img[p1:p2]
    # # スライス数が足りない場合、スライス軸に対してゼロで埋める
    # if len(img) < n_slice:
    #     img = np.pad(img, ((0, n_slice - len(img)), (0, 0), (0, 0)), mode='constant', constant_values=0)

    # SIZ
    img = change_depth_siz(img, n_slice)
    
    # convert to HU
    M = float(dicoms[0].RescaleSlope)
    B = float(dicoms[0].RescaleIntercept)
    img = img * M + B
    
    # Windowing
    img = windowing(img)
    # cropping and padding
    img = np.stack([add_pad(crop_image(_)) for _ in img])
    
    return np.expand_dims(img, 0)

def windowing(img):
    X = np.clip(img.copy(), 15, 100)
    # min-max normalization
    if np.min(X) < np.max(X):
        X = X - np.min(X)
        X = X / np.max(X)
    return X


def crop_image(image, display=False):
    # Create a mask with the background pixels
    mask = image == 0
    # Find the brain area
    coords = np.array(np.nonzero(~mask))
    
    # Check if coords is not empty before finding the minimum and maximum
    if coords.size > 0:
        # Find the top-left and bottom-right coordinates of the non-background pixels
        top_left = np.min(coords, axis=1)
        bottom_right = np.max(coords, axis=1)
        
        # Remove the background by cropping the image using the top-left and bottom-right coordinates
        cropped_image = image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
        return cropped_image
    else:
        # Handle the case when coords is empty by returning the original image
        return image

def add_pad(image, new_height=256, new_width=256):

    height, width = image.shape
    add_pad_image = np.zeros((new_height, new_width))
    pad_left = int((new_width - width) / 2)
    pad_top = int((new_height - height) / 2)
    # Replace the pixels with the image's pixels
    add_pad_image[pad_top:pad_top + height, pad_left:pad_left + width] = image
    return add_pad_image

def change_depth_siz(img, n_slice):
    desired_depth = n_slice
    current_depth = img.shape[0]
    depth = current_depth / desired_depth
    depth_factor = 1 / depth
    img_new = zoom(img, (depth_factor, 1, 1), mode='nearest')
    return img_new
