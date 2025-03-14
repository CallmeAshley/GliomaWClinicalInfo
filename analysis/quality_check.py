import os
import natsort
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing.pool import Pool
import pandas as pd
from operator import itemgetter
from tqdm import tqdm

def plot_images(data_path, label_path, out_path, name, grade, idh_mutant):

    # Read Image
    T1_itk =  sitk.ReadImage(os.path.join(data_path, name, name+'_T1_bet.nii.gz'))
    T1C_itk =  sitk.ReadImage(os.path.join(data_path, name, name+'_T1C_bet.nii.gz'))
    T2_itk =  sitk.ReadImage(os.path.join(data_path, name, name+'_T2_bet.nii.gz'))
    FLAIR_itk =  sitk.ReadImage(os.path.join(data_path, name, name+'_FLAIR_bet.nii.gz'))

    T1 =  sitk.GetArrayFromImage(T1_itk)
    T1C =  sitk.GetArrayFromImage(T1C_itk)
    T2 =  sitk.GetArrayFromImage(T2_itk)
    FLAIR =  sitk.GetArrayFromImage(FLAIR_itk)

    
    seg_itk = sitk.ReadImage(os.path.join(label_path, name + ".nii.gz"))
    seg = sitk.GetArrayFromImage(seg_itk)
    
    binary_seg = seg.copy()
    binary_seg[binary_seg !=0] = 1
    max_idx = np.argmax(binary_seg.sum(-1).sum(-1))
    
    img = np.stack([T1, T1C, T2, FLAIR], 0)
    plot_img = img[:, max_idx]
    
    _, D, H, W = img.shape
    
    x_size = W*4
    y_size = H*2
    
    new_x_size = 40 * x_size / (x_size + y_size)
    new_y_size = 40 * y_size / (x_size + y_size) + 1
    
    fig = plt.figure(figsize=(new_x_size, new_y_size))
    rows = 2
    cols = 4
    
    mask = seg[max_idx]
    masked = np.ma.masked_where(mask == 0, mask)

    for i in range(cols):
        ax = fig.add_subplot(rows, cols, i+1)
        ax.imshow(plot_img[i], cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    
    for i in range(cols):
        ax = fig.add_subplot(rows, cols, i+5)
        ax.imshow(plot_img[i], cmap='gray')
        ax.imshow(masked, cmap='jet', alpha=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('G'+ str(grade) + '_' + idh_mutant, fontsize=20)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        

    plt.tight_layout()
    plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
    # plt.title(idh_mutant)
    plt.savefig(os.path.join(out_path, name + '.png'), bbox_inches='tight', pad_inches=0)
    plt.close()
    plt.clf()
    
    print(name)

# data_path = "/mai_nas/BYS/brain_metastasis/data/SEV/sev_ver12/"
# label_path = "/mai_nas/BYS/brain_metastasis/nnunet/nnunet_results/"

data_path = "/mai_nas/BYS/brain_metastasis/data/TCGA/Co-Regi_Data/"
label_path = "/mai_nas/BYS/brain_metastasis/nnunet/nnunet_results/TCGA/"

out_path = "/mai_nas/BYS/brain_metastasis/analysis/QC/TCGA"
os.makedirs(out_path, exist_ok=True)

names = os.listdir(data_path)
names = [name.split('.')[0] for name in names]
names = natsort.natsorted(names)

# df_from_excel = pd.read_excel('/mai_nas/BYS/brain_metastasis/data/SEV/SEV_whole_label.xlsx')
df_from_excel = pd.read_excel('/mai_nas/BYS/brain_metastasis/data/TCGA/TCGA_whole_label.xlsx')
grade_list = list(df_from_excel['WHO'])
class_list = list(df_from_excel['IDH/codel subtype'])

p = Pool(4)

args = [[data_path, label_path, out_path, name, grade_list[i], class_list[i]] for i, name in enumerate(names)]
p.starmap_async(plot_images, args)
p.close()
p.join()
