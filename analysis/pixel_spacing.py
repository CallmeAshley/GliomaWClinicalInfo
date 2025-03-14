
import os
import natsort
import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk
from copy import deepcopy
from multiprocessing.pool import Pool
from turtle import down
from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import *
from monai.transforms import *
import shutil
from scipy.ndimage import binary_fill_holes
import pandas as pd
from operator import itemgetter

def create_nonzero_mask(data):
    assert len(data.shape) == 4 or len(data.shape) == 3, "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = np.zeros(data.shape[1:], dtype=bool)
    for c in range(data.shape[0]):
        this_mask = data[c] != data.min()
        nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    return nonzero_mask

def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    if len(mask.shape) == 3:
        minzidx = int(np.min(mask_voxel_coords[0]))
        maxzidx = int(np.max(mask_voxel_coords[0])) + 1
        minyidx = int(np.min(mask_voxel_coords[1]))
        maxyidx = int(np.max(mask_voxel_coords[1])) + 1
        minxidx = int(np.min(mask_voxel_coords[2]))
        maxxidx = int(np.max(mask_voxel_coords[2])) + 1
        return [[minzidx, maxzidx], [minyidx, maxyidx], [minxidx, maxxidx]]
    elif len(mask.shape) == 2:
        minyidx = int(np.min(mask_voxel_coords[0]))
        maxyidx = int(np.max(mask_voxel_coords[0])) + 1
        minxidx = int(np.min(mask_voxel_coords[1]))
        maxxidx = int(np.max(mask_voxel_coords[1])) + 1
        return [[minyidx, maxyidx], [minxidx, maxxidx]]


def train_dataset_conversion(patient_name, data_dir, target_base, target_spacing):
    print(patient_name)

    # T1_itk =  sitk.ReadImage(os.path.join(data_dir, patient_name, patient_name+'_T1_regi_bet.nii.gz'))
    T1_itk =  sitk.ReadImage(os.path.join(data_dir, patient_name+'_0000.nii.gz'))
    T1 =  sitk.GetArrayFromImage(T1_itk)
        
    subject_spacing = T1_itk.GetSpacing()[::-1] # (z, y, x)
    d, h, w = T1.shape
        
    return_dict = {'name' : patient_name, 'spacing':subject_spacing, 'size': T1.shape}
    # return_dict = subject_spacing

    return return_dict


if __name__ == "__main__":
    # You should change here ---------------------------------------------------------------------------
    task_name = "Task001_SEV"
    # data_dir = "/mai_nas/BYS/brain_metastasis/data/SEV/Co_Regi_Data/"
    data_dir = "/mai_nas/BYS/brain_metastasis/pre_processed/Task001_SEV/crop_images/"
    out_path = "/mai_nas/BYS/brain_metastasis/pre_processed/"
    target_spacing = [1.0, 1.0, 1.0]
    num_workers = 20  # multiprocessing workers
    # --------------------------------------------------------------------------------------
    target_base = join(out_path, task_name, 'analysis')
    os.makedirs(target_base, exist_ok=True)
    
    names = os.listdir("/mai_nas/BYS/brain_metastasis/data/SEV/Co_Regi_Data/")
    names = natsort.natsorted(names)
    
    df_from_excel = pd.read_excel('/mai_nas/BYS/brain_metastasis/data/SEV/SEV_label.xlsx')
    excluded_label = np.array(df_from_excel['exclusion'])
    chosen_label = 1 - excluded_label
    names = list(itemgetter(*chosen_label.nonzero()[0])(names))
    
    print('total = ' , len(names))
    mean_spacing = np.zeros((len(names), 3)) # z y x
    header_dict = {}
    
    p = Pool(num_workers)
    args = [[patient_name, data_dir, target_base, target_spacing] for patient_name in names]
    results = p.starmap_async(train_dataset_conversion, args)
    results = results.get()
    p.close()
    p.join()
    
    all_spacing = []
    all_names = []
    all_sizes = []
    
    for pat_dict in results:
        pat_name = pat_dict['name']
        spacing = pat_dict['spacing']
        arr_size = pat_dict['size']
        
        all_names.append(pat_name)
        all_spacing.append(spacing)
        all_sizes.append(arr_size)
    
        
    all_spacing = np.array(all_spacing)
    all_sizes = np.array(all_sizes)
    
    
    
    df = pd.DataFrame({'case' : all_names, 'slice_thickness':all_spacing[:, 0], 'y_spacing' : all_spacing[:, 1], 'x_spacing': all_spacing[:, 2],
                       'z_slice' : all_sizes[:, 0], 'y_slice' : all_sizes[:, 1], 'x_slice' : all_sizes[:, 2]})
    df.to_excel(os.path.join(target_base, 'size_and_spacing' + '.xlsx'), sheet_name = 'Sheet1', float_format = "%.3f",header = True,
                index = True)