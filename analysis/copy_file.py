import os
import shutil
import natsort
from tqdm import tqdm

root_path = "/mai_nas/BYS/brain_metastasis/data/SEV/Co_Regi_Data/"
out_path = "/mai_nas/BYS/medical_segmentation/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task082_BraTS2020/imagesTs3/"
names = os.listdir(root_path)
names = natsort.natsorted(names)

names = names[3::4]

for name in tqdm(names):
    shutil.copy(os.path.join(root_path, name, name+'_T1_new.nii.gz'), os.path.join(out_path, name+'_0000.nii.gz'))
    shutil.copy(os.path.join(root_path, name, name+'_T1C_new.nii.gz'), os.path.join(out_path, name+'_0001.nii.gz'))
    shutil.copy(os.path.join(root_path, name, name+'_T2_new.nii.gz'), os.path.join(out_path, name+'_0002.nii.gz'))
    shutil.copy(os.path.join(root_path, name, name+'_FLAIR_new.nii.gz'), os.path.join(out_path, name+'_0003.nii.gz'))
