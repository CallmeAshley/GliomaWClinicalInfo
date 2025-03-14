import subprocess
import numpy as np
import os

root_path = '/mai_nas/BYS/brain_metastasis/data/UCSF/images'
patient_list = os.listdir(root_path)
patient_list = [patient for patient in patient_list if patient.startswith('UCSF')]
patient_list.sort()


gpu_num = '0'
patient_list = patient_list[0:2
                            ]
# patient_list = patient_list[1::3]
# patient_list = patient_list[2::3]


for i, patient in enumerate(patient_list):
    # sub_num = '%04d' %i
    sub_num = '%03d' %i
    target_path = os.path.join(root_path, patient, patient.split('_')[0] + '_T1c_bias.nii.gz')
    subprocess.run(['hd-bet', '-i', target_path, '-device', gpu_num, '--overwrite_existing', '1'], shell=False)
