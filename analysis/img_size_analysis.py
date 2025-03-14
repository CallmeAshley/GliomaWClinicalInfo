import subprocess
import numpy as np
import os
import natsort
import pandas as pd
from operator import itemgetter
import SimpleITK as sitk
from multiprocessing.pool import Pool
import shutil


def read_size(patient_name, data_dir):
    
    # Read Image
    T1_itk =  sitk.ReadImage(os.path.join(data_dir, patient_name, patient_name+'_T1.nii.gz'))
    T1 =  sitk.GetArrayFromImage(T1_itk)
    
    seg_itk = sitk.ReadImage(os.path.join(data_dir, patient_name, patient_name+'_seg.nii.gz'))
    seg = sitk.GetArrayFromImage(seg_itk)
    
    seg[seg !=0] = 1    
    
    z_seg = seg.sum(-1).sum(-1)
    z_roi_idx = z_seg.nonzero()[0].tolist()

    y_seg = seg.sum(0).sum(-1)
    y_roi_idx = y_seg.nonzero()[0].tolist()
        
    x_seg = seg.sum(0).sum(0)
    x_roi_idx = x_seg.nonzero()[0].tolist()
    
    num_of_tumor_slice = (len(z_roi_idx), len(y_roi_idx), len(x_roi_idx))
        
    
    # Retrun Dictionary
    return_dict = {'name' : patient_name, 'size': T1.shape, 'tumor_slice': num_of_tumor_slice}


    print(patient_name + '  done')
    
    return return_dict
    

data_dir = "/mai_nas/BYS/brain_metastasis/preprocessed/TCGA/"
out_dir = "/mai_nas/BYS/brain_metastasis/preprocessed/tcga_analysis/"
# --------------------------------------------------------------------------------------

names = os.listdir(data_dir)
names = natsort.natsorted(names)

p = Pool(6)
args = [[patient_name, data_dir] for i, patient_name in enumerate(names)]

results = p.starmap_async(read_size, args)
results = results.get()
p.close()
p.join()


subtask_sizes = []
subtask_names = []
subtask_tumor_slice = []

for pat_dict in results:
    pat_name = pat_dict['name']
    arr_size = pat_dict['size']
    tumor_slice = pat_dict['tumor_slice']

    subtask_sizes.append(arr_size)
    subtask_names.append(pat_name)
    subtask_tumor_slice.append(tumor_slice)


all_sizes = np.array(subtask_sizes)
all_names = np.array(subtask_names)
all_tumor_slice = np.array(subtask_tumor_slice)
    
df = pd.DataFrame({'case' : all_names,
                   'z_slice' : all_sizes[:, 0], 'y_slice' : all_sizes[:, 1], 'x_slice' : all_sizes[:, 2],
                   'z_tumor_slice' : all_tumor_slice[:, 0], 'y_tumor_slice' : all_tumor_slice[:, 1], 'x_tumor_slice' : all_tumor_slice[:, 2]})
# df.to_excel(os.path.join('/mai_nas/BYS/brain_metastasis/data/SEV/', 'fingerprint' + '.xlsx'), sheet_name = 'Sheet1', float_format = "%.3f",header = True,
#             index = True)
df.to_excel(os.path.join(out_dir, 'fingerprint' + '.xlsx'), sheet_name = 'Sheet1', float_format = "%.3f",header = True,
            index = True)