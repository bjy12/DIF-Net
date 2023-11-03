import os
import json
import scipy
import numpy as np
import glob
import os
from glob import glob
from tqdm import tqdm
import SimpleITK as sitk


def read_nifti(path):
    itk_img = sitk.ReadImage(path)
    image = sitk.GetArrayFromImage(itk_img)
    spacing = np.array(itk_img.GetSpacing())
    print(" spacing : " , spacing )
    return image ,spacing


def save_nifti(image, path , new_spcaing):
    out = sitk.GetImageFromArray(image)
    if new_spcaing is not None:
        out.SetSpacing(new_spcaing)
    sitk.WriteImage(out, path)


def load_names():
    with open('info_copy.json', 'r') as f:
        info = json.load(f)
        names = []
        for s in ['train', 'test', 'eval']:
            names += info[s]
        return names


def resample_testdemo(path):
    os.makedirs('./resampled', exist_ok=True)
    ref_spacing = np.array([0.8, 0.8, 0.8])
    #* 想把原始数据的spacing都设定为0.8 

    itk_img = sitk.ReadImage(path)
    spacing = np.array(itk_img.GetSpacing())
    print( " spacing : " , spacing)
    image = sitk.GetArrayFromImage(itk_img)
    image = image.transpose(2, 1, 0)

    image = np.clip(image, a_min=-400, a_max=500)
    image = (image - image.min()) / (image.max() - image.min())
    image = image.astype(np.float32)
    #* 这里使用图像缩放的方式来更改原始spacing 
    #* 假如原始图像的spcaing为1，想将其转化为0.5
    #* 就是把图像放大了两倍，这样在同一个像素的本身的距离为1 ，经过放大后就是0.5 ， 这样scale 就是 2 
    #* scaling 的计算公式  应该是 缩放因子等于 当前图像spacing 比 设定图像的spacing
    scaling = spacing / ref_spacing
    print(" scaling : " , scaling )
    image = scipy.ndimage.zoom(
        image, 
        scaling, 
        order=3, 
        prefilter=False
    )

    image = (image * 255).astype(np.uint8)
    print( " image resample shape " , image.shape)
    save_path = f'./resampled/resampled_testdata.nii.gz'
    save_nifti(image, save_path , ref_spacing)



def resample():
    os.makedirs('./resampled', exist_ok=True)
    ref_spacing = np.array([0.8, 0.8, 0.8])

    for name in tqdm(load_names(), ncols=50):
        a, b = name.split('-')
        path = f'F:\Project\DIF-NET\dataexample\S-Knee-WithoutMetalRod\{a}-{b}.mhd'
        itk_img = sitk.ReadImage(path)
        #print( " itk_img " , itk_img.shape)
        spacing = np.array(itk_img.GetSpacing())
        image = sitk.GetArrayFromImage(itk_img)
        image = image.transpose(2, 1, 0)

        image = np.clip(image, a_min=-400, a_max=500)
        image = (image - image.min()) / (image.max() - image.min())
        image = image.astype(np.float32)

        scaling = spacing / ref_spacing
        image = scipy.ndimage.zoom(
            image, 
            scaling, 
            order=3, 
            prefilter=False
        )

        image = (image * 255).astype(np.uint8)
        save_path = f'./resampled/{name}.nii.gz'
        save_nifti(image, save_path , ref_spacing)

def crop_pad_test(path):
    os.makedirs('./processed', exist_ok=True)
    files = glob(path + "*.nii.gz")

    image , spacing = read_nifti(files)
    print(" image shape " , image.shape)
    image = image[0]
        # w, h
    if image.shape[0] > 256: # crop
        p = image.shape[0] // 2 - 128
        image = image[p:p+256, p:p+256, :]
    elif image.shape[0] < 256: # padding
        image_tmp = np.full([256, 256, image.shape[-1]], fill_value=0, dtype=np.uint8)
        p = 128 - image.shape[0] // 2
        print( " p " , p)
        l = image.shape[0]
        print(" l " , l)
        image_tmp[p:p+l, p:p+l, :] = image
        image = image_tmp

        # d
    if image.shape[-1] > 256: # crop
        p = image.shape[-1] // 2 - 128
        image = image[..., p:p+256]
    elif image.shape[-1] < 256: # padding
        image_tmp = np.full(list(image.shape[:2]) + [256], fill_value=0, dtype=np.uint8)
        p = 128 - image.shape[-1] // 2
        l = image.shape[-1]
        image_tmp[..., p:p+l] = image
        image = image_tmp
    print(" save image : " , image.shape)

    save_path = f'./processed/test_processed.nii.gz'
    save_nifti(image, save_path ,spacing)



def crop_pad():
    os.makedirs('./processed', exist_ok=True)
    files = glob('./resampled/*.nii.gz')
    for file in tqdm(files, ncols=50):
        print(" file : " , file)
        name = file.split('\\')[-1].split('.')[0]
        image , spacing = read_nifti(file)  
        # w, h
        if image.shape[0] > 256: # crop
            p = image.shape[0] // 2 - 128
            image = image[p:p+256, p:p+256, :]
        elif image.shape[0] < 256: # padding
            image_tmp = np.full([256, 256, image.shape[-1]], fill_value=0, dtype=np.uint8)
            p = 128 - image.shape[0] // 2
            l = image.shape[0]
            image_tmp[p:p+l, p:p+l, :] = image
            image = image_tmp

        # d
        if image.shape[-1] > 256: # crop
            p = image.shape[-1] // 2 - 128
            image = image[..., p:p+256]
        elif image.shape[-1] < 256: # padding
            image_tmp = np.full(list(image.shape[:2]) + [256], fill_value=0, dtype=np.uint8)
            p = 128 - image.shape[-1] // 2
            l = image.shape[-1]
            image_tmp[..., p:p+l] = image
            image = image_tmp

        save_path = f'./processed/{name}.nii.gz'
        save_nifti(image, save_path , spacing)

            
if __name__ == '__main__':
    #* test  preprocess.PY
    #* read  ct image  basic information 
    #* 512 512 408 
    #* spacing 0.408203125 0.408203125 0.60000017199017197
    #raw_file_path = "./knee/test_data.mhd"
    #resample_testdemo(raw_file_path)
    #resample_path = "./resampled/"
    #crop_pad_test(resample_path)
    #resample()
    crop_pad()
        