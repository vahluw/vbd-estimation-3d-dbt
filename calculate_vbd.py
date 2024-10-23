import numpy as np
import cv2
import nibabel
import copy
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

if __name__ == '__main__':
    #csv_file --- CSV file output by preprocess_images.py that contains paths to reconstructed tomo images
    #output_file --- Name of new output file to which the results of this script will be written
    #orig_path --- Directory where GaNDLF inference predictions will be stored
    
    csv_file, output_file, orig_path = sys.argv[1], sys.argv[2], sys.argv[3]
    df_train = pd.read_csv(csv_file)
    df_train = df_train.reset_index()
    out_path = open(output_file, "w")
    out_path.write("SubjectID, Channel_0, Pred_VBD\n")

    counter = 0
    for index, row in df_train.iterrows():
        tomopath, subID = row[' Channel_0'], row['SubjectID']
        
        # Calculate VBD from inference
        finalpath = orig_path + "/testing/" + str(subID)+ "/" + str(subID) + "_seg.nii.gz"
        nifti_pred = nibabel.load(finalpath)
        full_pred = np.array(nifti_pred.get_fdata())
        flat_pred = full_pred.flatten()
        total_breast_pred = np.sum(flat_pred>=1) * 1.0
        total_dense_pred = np.sum(flat_pred==2) * 1.0
        total_vbd_pred = total_dense_pred/total_breast_pred
        
        out_path.write(str(subID) + "," + tomopath + "," + str(total_vbd_pred) + "\n")
        print(str(subID) + "," + tomopath + "," + str(total_vbd_pred) + "\n")

        del nifti_pred
        del full_pred



