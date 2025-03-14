import os
import shutil
import natsort
from tqdm import tqdm

root_path = "/mai_nas/BYS/brain_metastasis/data/SEV/Co_Regi_ver1/"
names = os.listdir(root_path)
names = natsort.natsorted(names)
      
      
for name in tqdm(names):
    file0 = os.path.join(root_path, name, name+'_FLAIR_regi_bet_mask.nii.gz')
    
    if os.path.isfile(file0):
        os.remove(file0)
        
        

# import numpy as np
# import pandas as pd
# import shutil

# root_path = "/mai_nas/BYS/brain_metastasis/data/SEV/Co_Regi_Data/"
# names = os.listdir(root_path)
# names = natsort.natsorted(names)

# df_from_excel = pd.read_excel('/mai_nas/BYS/brain_metastasis/data/SEV/SEV_label.xlsx')
# excluded_label = np.array(df_from_excel['exclusion'])


# print(excluded_label.sum())
# for i, name in tqdm(enumerate(names)):
#     if excluded_label[i] == 1:
#         shutil.rmtree(os.path.join(root_path, name))