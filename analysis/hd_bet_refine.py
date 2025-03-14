import subprocess
import numpy as np
import os
import natsort
import pandas as pd
from operator import itemgetter
import SimpleITK as sitk
from multiprocessing.pool import Pool
import shutil

def create_nonzero_mask(data):
    from scipy.ndimage import binary_fill_holes
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
    

def SEV_refine_ver1(patient_name, data_dir):
    # Read Bet Mask
    T1C_itkm =  sitk.ReadImage(os.path.join(data_dir, patient_name, patient_name+'_T1C_regi_bet_mask.nii.gz'))
    T1Cm =  sitk.GetArrayFromImage(T1C_itkm)
    
    # Read Image
    T1_itk =  sitk.ReadImage(os.path.join(data_dir, patient_name, patient_name+'_T1_regi.nii.gz'))
    T1C_itk =  sitk.ReadImage(os.path.join(data_dir, patient_name, patient_name+'_T1C_regi.nii.gz'))
    T2_itk =  sitk.ReadImage(os.path.join(data_dir, patient_name, patient_name+'_T2_regi.nii.gz'))
    FLAIR_itk =  sitk.ReadImage(os.path.join(data_dir, patient_name, patient_name+'_FLAIR_regi.nii.gz'))
    
    T1 =  sitk.GetArrayFromImage(T1_itk)
    T1C =  sitk.GetArrayFromImage(T1C_itk)
    T2 =  sitk.GetArrayFromImage(T2_itk)
    FLAIR =  sitk.GetArrayFromImage(FLAIR_itk)
    
    T1 = T1 * T1Cm
    T1C = T1C * T1Cm
    T2 = T2 * T1Cm
    FLAIR = FLAIR * T1Cm
    
    os.makedirs(os.path.join(data_dir, patient_name), exist_ok=True)
    
    new_t1_itk = sitk.GetImageFromArray(T1)
    new_t1_itk.CopyInformation(T1_itk)
    sitk.WriteImage(new_t1_itk, os.path.join(data_dir, patient_name, patient_name + "_T1_bet.nii.gz"))
    
    new_t1c_itk = sitk.GetImageFromArray(T1C)
    new_t1c_itk.CopyInformation(T1C_itk)
    sitk.WriteImage(new_t1c_itk, os.path.join(data_dir, patient_name, patient_name + "_T1C_bet.nii.gz"))
    
    new_t2_itk = sitk.GetImageFromArray(T2)
    new_t2_itk.CopyInformation(T2_itk)
    sitk.WriteImage(new_t2_itk, os.path.join(data_dir, patient_name, patient_name + "_T2_bet.nii.gz"))

    new_fl_itk = sitk.GetImageFromArray(FLAIR)
    new_fl_itk.CopyInformation(FLAIR_itk)
    sitk.WriteImage(new_fl_itk, os.path.join(data_dir, patient_name, patient_name + "_FLAIR_bet.nii.gz"))
    
    # os.rename(os.path.join(data_dir, patient_name, patient_name+'_T1C_regi_bet.nii.gz'), os.path.join(data_dir, patient_name, patient_name+'_T1C_bet.nii.gz'))        
    
    if os.path.isfile(os.path.join(data_dir, patient_name, patient_name+'_ADC_regi.nii.gz')):
        ADC_itk =  sitk.ReadImage(os.path.join(data_dir, patient_name, patient_name+'_ADC_regi.nii.gz'))
        ADC =  sitk.GetArrayFromImage(ADC_itk)
        ADC = ADC * T1Cm
        new_adc_itk = sitk.GetImageFromArray(ADC)
        new_adc_itk.CopyInformation(ADC_itk)
        sitk.WriteImage(new_adc_itk, os.path.join(data_dir, patient_name, patient_name + "_ADC_bet.nii.gz"))
    print(patient_name + '  done')
    

    
    
    
    

def SEV_refine_ver2(patient_name, data_dir):
    # Read Bet Mask
    T1C_itkm =  sitk.ReadImage(os.path.join(data_dir, patient_name, patient_name+'_T1C_Co-Regi_bet_mask.nii.gz'))
    T1Cm =  sitk.GetArrayFromImage(T1C_itkm)

    # Read Image
    T1_itk =  sitk.ReadImage(os.path.join(data_dir, patient_name, patient_name+'_T1_Co-Regi.nii.gz'))
    T1C_itk =  sitk.ReadImage(os.path.join(data_dir, patient_name, patient_name+'_T1C_Co-Regi.nii.gz'))
    T2_itk =  sitk.ReadImage(os.path.join(data_dir, patient_name, patient_name+'_T2_Co-Regi.nii.gz'))
    FLAIR_itk =  sitk.ReadImage(os.path.join(data_dir, patient_name, patient_name+'_FLAIR_Co-Regi.nii.gz'))
    
    T1 =  sitk.GetArrayFromImage(T1_itk)
    T1C =  sitk.GetArrayFromImage(T1C_itk)
    T2 =  sitk.GetArrayFromImage(T2_itk)
    FLAIR =  sitk.GetArrayFromImage(FLAIR_itk)
    
    T1 = T1 * T1Cm
    T1C = T1C * T1Cm
    T2 = T2 * T1Cm
    FLAIR = FLAIR * T1Cm
    
    os.makedirs(os.path.join(data_dir, patient_name), exist_ok=True)
    
    new_t1_itk = sitk.GetImageFromArray(T1)
    new_t1_itk.CopyInformation(T1_itk)
    sitk.WriteImage(new_t1_itk, os.path.join(data_dir, patient_name, patient_name + "_T1_bet.nii.gz"))
    
    new_t1c_itk = sitk.GetImageFromArray(T1C)
    new_t1c_itk.CopyInformation(T1C_itk)
    sitk.WriteImage(new_t1c_itk, os.path.join(data_dir, patient_name, patient_name + "_T1C_bet.nii.gz"))
    
    new_t2_itk = sitk.GetImageFromArray(T2)
    new_t2_itk.CopyInformation(T2_itk)
    sitk.WriteImage(new_t2_itk, os.path.join(data_dir, patient_name, patient_name + "_T2_bet.nii.gz"))

    new_fl_itk = sitk.GetImageFromArray(FLAIR)
    new_fl_itk.CopyInformation(FLAIR_itk)
    sitk.WriteImage(new_fl_itk, os.path.join(data_dir, patient_name, patient_name + "_FLAIR_bet.nii.gz"))
    
    # os.rename(os.path.join(data_dir, patient_name, patient_name+'_T1C_Co-Regi_bet.nii.gz'), os.path.join(data_dir, patient_name, patient_name+'_T1C_bet.nii.gz'))        
    
    if os.path.isfile(os.path.join(data_dir, patient_name, patient_name+'_ADC_Co-Regi.nii.gz')):
        ADC_itk =  sitk.ReadImage(os.path.join(data_dir, patient_name, patient_name+'_ADC_Co-Regi.nii.gz'))
        ADC =  sitk.GetArrayFromImage(ADC_itk)
        ADC = ADC * T1Cm
        new_adc_itk = sitk.GetImageFromArray(ADC)
        new_adc_itk.CopyInformation(ADC_itk)
        sitk.WriteImage(new_adc_itk, os.path.join(data_dir, patient_name, patient_name + "_ADC_bet.nii.gz"))
    
    print(patient_name + '  done')
    
    
def TCGA_refine(patient_name, data_dir):
    # Read Bet Mask
    T1C_itkm =  sitk.ReadImage(os.path.join(data_dir, patient_name, patient_name+'_T1C_regi_bet_mask.nii.gz'))
    T1Cm =  sitk.GetArrayFromImage(T1C_itkm)
    
    # Read Image
    T1_itk =  sitk.ReadImage(os.path.join(data_dir, patient_name, patient_name+'_T1_regi.nii.gz'))
    T1C_itk =  sitk.ReadImage(os.path.join(data_dir, patient_name, patient_name+'_T1C_regi.nii.gz'))
    T2_itk =  sitk.ReadImage(os.path.join(data_dir, patient_name, patient_name+'_T2_regi.nii.gz'))
    FLAIR_itk =  sitk.ReadImage(os.path.join(data_dir, patient_name, patient_name+'_FLAIR_regi.nii.gz'))
    
    T1 =  sitk.GetArrayFromImage(T1_itk)
    T1C =  sitk.GetArrayFromImage(T1C_itk)
    T2 =  sitk.GetArrayFromImage(T2_itk)
    FLAIR =  sitk.GetArrayFromImage(FLAIR_itk)
    
    T1 = T1 * T1Cm
    T1C = T1C * T1Cm
    T2 = T2 * T1Cm
    FLAIR = FLAIR * T1Cm
    
    os.makedirs(os.path.join(data_dir, patient_name), exist_ok=True)
    
    new_t1_itk = sitk.GetImageFromArray(T1)
    new_t1_itk.CopyInformation(T1_itk)
    sitk.WriteImage(new_t1_itk, os.path.join(data_dir, patient_name, patient_name + "_T1_bet.nii.gz"))
    
    new_t1c_itk = sitk.GetImageFromArray(T1C)
    new_t1c_itk.CopyInformation(T1C_itk)
    sitk.WriteImage(new_t1c_itk, os.path.join(data_dir, patient_name, patient_name + "_T1C_bet.nii.gz"))
    
    new_t2_itk = sitk.GetImageFromArray(T2)
    new_t2_itk.CopyInformation(T2_itk)
    sitk.WriteImage(new_t2_itk, os.path.join(data_dir, patient_name, patient_name + "_T2_bet.nii.gz"))

    new_fl_itk = sitk.GetImageFromArray(FLAIR)
    new_fl_itk.CopyInformation(FLAIR_itk)
    sitk.WriteImage(new_fl_itk, os.path.join(data_dir, patient_name, patient_name + "_FLAIR_bet.nii.gz"))
    
    os.rename(os.path.join(data_dir, patient_name, patient_name+'_T1C_regi_bet.nii.gz'), os.path.join(data_dir, patient_name, patient_name+'_T1C_bet.nii.gz'))        
    
    if os.path.isfile(os.path.join(data_dir, patient_name, patient_name+'_ADC_regi.nii.gz')):
        ADC_itk =  sitk.ReadImage(os.path.join(data_dir, patient_name, patient_name+'_ADC_regi.nii.gz'))
        ADC =  sitk.GetArrayFromImage(ADC_itk)
        ADC = ADC * T1Cm
        new_adc_itk = sitk.GetImageFromArray(ADC)
        new_adc_itk.CopyInformation(ADC_itk)
        sitk.WriteImage(new_adc_itk, os.path.join(data_dir, patient_name, patient_name + "_ADC_bet.nii.gz"))
    print(patient_name + '  done')
    

    

    
data_dir = "/mai_nas/BYS/brain_metastasis/data/SEV/sev_ver12/"
data_dir = "/mai_nas/BYS/brain_metastasis/data/TCGA/Co-Regi_Data/"

ver = 3
# --------------------------------------------------------------------------------------

names = os.listdir(data_dir)
names = natsort.natsorted(names)


p = Pool(12)

if ver == 1:
    names = [name for name in names if name.startswith('SEV')]
    args = [[patient_name, data_dir] for i, patient_name in enumerate(names)]
    p.starmap_async(SEV_refine_ver1, args)
    p.close()
    p.join() 

elif ver == 2:
    names = [name for name in names if name.startswith('AA')]
    args = [[patient_name, data_dir] for i, patient_name in enumerate(names)]
    p.starmap_async(SEV_refine_ver2, args)
    p.close()
    p.join() 

elif ver == 3:
    names = [name for name in names if name.startswith('TCGA')]
    names = names[150:]
    args = [[patient_name, data_dir] for i, patient_name in enumerate(names)]
    p.starmap_async(TCGA_refine, args)
    p.close()
    p.join() 