import numpy as np
import cv2
import nibabel
import copy
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

def pad_image_training(filename_tomo, filename_mask, output_dir):
    loc = filename_tomo.rfind('/')
    loc2 = filename_tomo[:loc].rfind('/')
    new_name_mat = output_dir + '/' + filename_tomo[loc+1:-7]   + '_' + "mat.nii.gz"
    new_name_mask = output_dir + '/' + filename_tomo[loc+1:-7]   + '_' + "mask.nii.gz"
    
    if os.path.isfile(new_name_mat) and os.path.isfile(new_name_mask):
        return new_name_mat, new_name_mask

    full_image_nifti = nibabel.load(filename_tomo)
    full_image = full_image_nifti.get_fdata()
    full_image_mask_nifti = nibabel.load(filename_mask)
    full_image_mask = full_image_mask_nifti.get_fdata()
    full_image_mask[full_image_mask == 128] = 1
    full_image_mask[full_image_mask == 255] = 2

    new_header = copy.deepcopy(full_image_nifti.header)
    new_header.default_x_flip = False
    new_header_mask = copy.deepcopy(full_image_mask_nifti.header)
    new_header_mask.default_x_flip = False

    if 'LMLO' in new_name_mat or 'LCC' in new_name_mat:
        full_image = np.rot90(full_image, k=2)
        full_image_mask = np.rot90(full_image_mask, k=2)

    right = max(0, 1024-full_image.shape[1])
    down = max(0, 2048-full_image.shape[0])

    new_image_padded = np.zeros((full_image.shape[0] + down, full_image.shape[1] + right, full_image.shape[2]))
    new_mask_padded = np.zeros((full_image.shape[0] + down, full_image.shape[1] + right, full_image.shape[2]))

    if 'RCC' in new_name_mat or 'RMLO' in new_name_mat:
        new_mask_padded[0:full_image.shape[0], right:new_mask_padded.shape[1], :] = full_image_mask
        new_image_padded[0:full_image.shape[0], right:new_mask_padded.shape[1], :] = full_image
    else:
        new_mask_padded[0:full_image.shape[0], 0:full_image.shape[1], :] = full_image_mask
        new_image_padded[0:full_image.shape[0], 0:full_image.shape[1], :] = full_image

    final_nib_image = nibabel.Nifti1Image(new_image_padded, np.eye(4), new_header)
    final_nib_mask = nibabel.Nifti1Image(new_mask_padded, np.eye(4), new_header_mask)
    nibabel.save(final_nib_image, new_name_mat)
    nibabel.save(final_nib_mask, new_name_mask)

    del full_image
    del final_nib_image
    del final_nib_mask
    del full_image_mask
    del new_image_padded
    del new_mask_padded

    return new_name_mat, new_name_mask

def pad_image_inference(filename_tomo, output_dir):
    loc = filename_tomo.rfind('/')
    loc2 = filename_tomo[:loc].rfind('/')
    new_name_mat = output_dir + '/' + filename_tomo[loc+1:-7]   + '_' + "mat.nii.gz"
    
    if os.path.isfile(new_name_mat):
        return new_name_mat

    full_image_nifti = nibabel.load(filename_tomo)
    full_image = full_image_nifti.get_fdata()

    new_header = copy.deepcopy(full_image_nifti.header)
    new_header.default_x_flip = False


    if 'LMLO' in new_name_mat or 'LCC' in new_name_mat:
        full_image = np.rot90(full_image, k=2)

    right = max(0, 1024-full_image.shape[1])
    down = max(0, 2048-full_image.shape[0])

    new_image_padded = np.zeros((full_image.shape[0] + down, full_image.shape[1] + right, full_image.shape[2]))

    if 'RCC' in new_name_mat or 'RMLO' in new_name_mat:
        new_image_padded[0:full_image.shape[0], right:new_image_padded.shape[1], :] = full_image
    else:
        new_image_padded[0:full_image.shape[0], 0:full_image.shape[1], :] = full_image

    final_nib_image = nibabel.Nifti1Image(new_image_padded, np.eye(4), new_header)
    nibabel.save(final_nib_image, new_name_mat)

    del full_image
    del final_nib_image
    del new_image_padded

    return new_name_mat


# Preprocess images so that they can be properly used by the DL algorithm
# This ensures that image dimensions are at least 2048 voxels in the x- and y- directions
# Usage:
# python pad_tomo_images.py 'input_dir' 'output_dir' 'mode'
# input_dir -- Location of reconstructed 3D DBT images before preprocessing
# output_dir -- Desired path where preprocessed images will be stored
# mode -- 'train' for training; 'inference' for inference
if __name__ == '__main__':
    mode = sys.argv[3]
    
    # Preprocess images for training
    if mode == "train":
        input_dir, output_dir = sys.argv[1], sys.argv[2]
        
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        
        final_list = []
        count = 0
        for subdir, dirs, files, in os.walk(input_dir, followlinks=True):
            for file in files:
                print(file)
                if 'mask' in file:
                    continue
                new_name_tomo, new_name_mask = pad_image_training(subdir + '/' + file, subdir + '/' + file[:-10]+"mask.nii.gz", output_dir)
                final_list.append([new_name_tomo, new_name_mask])

        final_df = pd.DataFrame(final_list, columns = [' Channel_0', ' Label'])
        final_df.index.names = ['SubjectID']
        final_df.to_csv('padded_data_training.csv')
         
    # Preprocess images for inference
    elif mode == "inference":
        input_dir, output_dir = sys.argv[1], sys.argv[2]
        
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
            
        final_list = []
        count = 0
        for subdir, dirs, files, in os.walk(input_dir, followlinks=True):
            for file in files:
                if 'mask' in file:
                    continue
                new_tomoname = pad_image_inference(subdir + '/' + file, output_dir)
                final_list.append(new_tomoname)
                print(new_tomoname)

        final_df = pd.DataFrame(final_list, columns = [' Channel_0'])
        final_df.index.names = ['SubjectID']
        final_df.to_csv('padded_data_inference.csv')
                
    else:
        exit(1)
