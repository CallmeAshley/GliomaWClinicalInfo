import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from torch.utils.data import Dataset

import os
import time
import random
import warnings
import logging
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from monai import transforms
from monai.transforms import Resized, ToTensord

from src.utils import is_main_process
from src.total_utils import normalize_images, sequence_combination

import hydra
from omegaconf import DictConfig
from typing import Any, Dict, List, Optional, Tuple


# A logger for this file
logger = logging.getLogger(__name__)
warnings.filterwarnings(action='ignore')


class Brain_Dataset(Dataset):
    def __init__(self, train_val_test, cfg, transform=None):
        self.train_val_test = train_val_test
        self.cfg = cfg
        
        self.data_root = cfg.paths.data_root
        self.label_root = cfg.paths.label_root
        self.slice_percentile = cfg.data.slice_percentile

        self.cls_mode = cfg.cls_mode
        self.seq_comb = cfg.data.seq_comb
        self.k_fold = cfg.data.k_fold  
        self.fold_num = cfg.data.fold_num  
        
        self.transform = transform      
        
        img_name_list, cls_list = self.patient_and_class_selection()
        
        assert train_val_test not in ['train', 'val', 'test']
        
        self.img_name_list = img_name_list
        self.cls_list = cls_list

        print('{} data, length of dataset: {}'.format(self.train_val_test, len(self.img_name_list)))
            
    def __len__(self):
        return len(self.img_name_list)

    def cls_bal_selection(self):
        rand_prob = random.uniform(0, 1)
        cls_array = np.array(self.cls_list)
        cls_unique = np.unique(cls_array)
        
        prob_step = 1 / len(cls_unique)
        
        for i in range(len(cls_unique)):
            if rand_prob < (i+1) * prob_step:
                idx_list = np.where(cls_array == i)[0].tolist()
                break
    
        idx = idx_list[random.randint(0, len(idx_list)-1)]
        return idx

    def sequence_combination(self, t1, t1c, t2, flair, adc):
        if self.seq_comb == '4seq':
            img_npy = np.stack([t1, t1c, t2, flair], 0)
        elif self.seq_comb == '4seq+adc':
            img_npy = np.stack([t1, t1c, t2, flair, adc], 0)
        elif self.seq_comb == 't2':
            img_npy = np.stack([t2], 0)
        elif self.seq_comb == 't2+adc':
            img_npy = np.stack([t2, adc], 0)
        
        return img_npy

    def patient_and_class_selection(self):
        
        df_from_excel = pd.read_excel(self.label_root)
        df_from_excel.replace(['M','male'], -1, inplace=True)
        df_from_excel.replace(['F','female'], 1, inplace=True)
        
        # ---------------------------------------------------------------------------------------
        if self.cls_mode == 'idh':
            name_list = np.array(df_from_excel['Anony_ID'])
            cls_list = np.array(df_from_excel['IDH_mutation'])
            mask = None
        elif self.cls_mode == '1p_19q':
            mutation = np.array(df_from_excel['IDH_mutation']).astype(np.bool_)
            name_list = np.array(df_from_excel['Anony_ID'])
            cls_list = np.array(df_from_excel['1p/19q codeletion'])
            mask = mutation
        elif self.cls_mode == 'subtype':
            name_list = np.array(df_from_excel['Anony_ID'])
            cls_list = np.array(df_from_excel['Mole_Group_no'])
            cls_list = cls_list - 1 # 1, 2, 3 -> 0, 1, 2
            mask = None
        elif self.cls_mode == 'lgg_hgg':
            name_list = np.array(df_from_excel['Anony_ID'])
            cls_list = np.array(df_from_excel['WHO_23_4']) # LGG : 0,  HGG : 1
            mask = None
        elif self.cls_mode == 'grade':
            name_list = np.array(df_from_excel['Anony_ID'])
            cls_list = np.array(df_from_excel['WHO']) # LGG : 0,  HGG : 1
            cls_list = cls_list - 2 # 2, 3, 4 -> 0, 1, 2
            mask = None
        
        name_list = np.squeeze(name_list[mask])
        cls_list = np.squeeze(cls_list[mask])
        
        return name_list, cls_list

        
    def split_class_balanced_data(self, img_name_array, cls_array, k_fold, k):   
        '''
        k_fold : total num of fold
        k : fold num
        '''
        train_name_list = []
        train_cls_list = []

        test_name_list = []
        test_cls_list = []
        
        unique = np.unique(cls_array)
        
        for cls in unique:
            mask = (cls_array == cls)
            temp_name_list = img_name_array[mask]
            
            len_train = 0
            for i in range(k_fold):
                if i == k:
                    test_name_temp = temp_name_list[k::k_fold]
                    test_name_list += test_name_temp.tolist()
                else:
                    train_name_temp = temp_name_list[i::k_fold]
                    train_name_list += train_name_temp.tolist()
                    len_train += len(train_name_temp.tolist())
            
            test_cls_list += [cls] * len(test_name_temp)
            train_cls_list += [cls] * len_train
        
        return train_name_list, train_cls_list, test_name_list, test_cls_list  
        
    def __getitem__(self, idx):        
        name = self.img_name_list[idx]
                
        t1 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.data_root, name, name +'_T1.nii.gz')))        
        t1c = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.data_root, name, name +'_T1C.nii.gz')))
        t2 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.data_root, name, name +'_T2.nii.gz')))
        flair = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.data_root, name, name +'_FLAIR.nii.gz')))
        
        if 'adc' in self.seq_comb:
            adc = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.data_root, name, name +'_ADC.nii.gz')))
        else:
            adc = None
        
        seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.data_root, name, name + '_seg.nii.gz')))
        seg[seg !=0] = 1
        
        t1, t1c, t2, flair, adc = normalize_images(t1, t1c, t2, flair, adc)
        img_npy = sequence_combination(t1, t1c, t2, flair, adc, self.seq_comb)
        
        z_seg = seg.sum(-1).sum(-1)

        # 상위 %
        glioma_vol_lower_bound = np.percentile(z_seg[z_seg.nonzero()[0]], 100-self.slice_percentile, axis=0)
        roi_mask = z_seg > glioma_vol_lower_bound
        roi_idx_list = np.where(roi_mask==True)[0].tolist()
        # ---------------------------------------------------------------------------------------------------------------------------------------
        label = np.array(self.cls_list[idx])
        label = torch.from_numpy(label).long()
        
        img_npy_2d = img_npy[:, roi_idx_list] # C x S x H x W
        S, A, H, W = img_npy_2d.shape
        img_npy_2d = img_npy_2d.reshape(S*A, H, W)
        seg_2d = seg[roi_idx_list]
        data_dict = {'image' : img_npy_2d, 'label' : label, 'name' : name, 'seg' : seg_2d}
        
        
        if self.transform is not None:
            data_dict = self.transform(data_dict)
                
        return data_dict


def dataloader(cfg, workers_per_gpu):    
    if isinstance(cfg.data.training_size, int):
        size = (cfg.data.training_size, cfg.data.training_size)
    elif isinstance(cfg.data.training_size, tuple):
        size = cfg.data.training_size
        
    test_transform = transforms.Compose([
                                        Resized(keys=["image", "seg"], spatial_size=size, mode=["bicubic", "nearest"]),
                                        ToTensord(keys=["image", "seg"]),
                                        ])
    
    testset = Brain_Dataset(cfg.task_name, cfg, transform=test_transform)
    

    print("[!] Data Loading Done")
    
    
    test_loader = torch.utils.data.DataLoader(testset, pin_memory=cfg.data.pin_memory, batch_size=cfg.data.batch_size,
                                                sampler=None, shuffle=False, num_workers=workers_per_gpu
                                                )

    
    return test_loader

@hydra.main(version_base="1.3", config_path="./configs", config_name="external_eval.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    
    print("[!] Glioma Subtyping and Grading Clasification According to WHO2021")
    print("[!] Created by MAI-LAB, Yunsu Byeon")
    
    assert cfg.data.batch_size == 1, "Batch size should be 1 for slice ensembling"
    

    # # gpu setting
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpus)
    # print("[Info] Finding Empty GPU {}".format(cfg.gpus))
    
    # main
    print("[!] Single-GPU Evaluation")
    test_worker(cfg)
    print('[Info] Save dir:', cfg.paths.output_dir)



def test_worker(cfg):
    distributed = False
    
    # Define network
    net = hydra.utils.instantiate(cfg.model)
    net = net.cuda()
    
    # Load Dataset
    test_loader = dataloader(cfg, cfg.data.workers)

    
    if cfg.ckpt:
        checkpoint = torch.load(cfg.ckpt)
        
        if distributed:
            net.module.load_state_dict(checkpoint['net'])
        else:
            net.load_state_dict(checkpoint['net'])
        
        print("[!] Model loaded from ", cfg.ckpt)

        del checkpoint
    
    best_thres = evaluate(net, test_loader, cfg)
    print("RETURN: " + str(best_thres)) # Do not remove this print line, this is for capturing the path of model

def evaluate(net, data_loader, cfg):

    metric_dict, val_pred_and_label = test(net, data_loader, cfg)

    outputs, targets, names = val_pred_and_label
    
    if outputs.shape[1] == 1:
        prob = F.sigmoid(outputs)
    else:
        prob = F.softmax(outputs, 1)
    
    prob = prob.detach().cpu().numpy()
    prob = np.around(prob, 3)
    
    excel_dict = {}
    excel_dict['names'] = names
    excel_dict['target'] = targets.detach().cpu().numpy().tolist()
    for i in range(prob.shape[1]):
        cls_name = 'pred_'+str(i)
        excel_dict[cls_name] = prob[:, i].tolist()
        
    df = pd.DataFrame(excel_dict)
    df.to_excel(os.path.join(cfg.paths.output_dir, 'prediction.xlsx'), sheet_name = 'Sheet1', float_format = "%.3f",header = True, index = True)
    
    import json
    with open(os.path.join(cfg.paths.output_dir, 'metric.json'),'w') as f:
        json.dump(metric_dict, f, indent=4)
    
    if metric_dict.get("Best_thres") == None:
        return 'no thres'
    else:
        return metric_dict['Best_thres']




def slice_ensemble(inputs, seg, targets, net, attribution_generator, name, cfg):
    B, SA, H, W = inputs.shape
    assert B == 1, "Batch size must be 1 in inference mode"
    S, A = cfg.model.in_chans, SA // cfg.model.in_chans
    inputs = inputs.view(B, S, A, H, W).contiguous()
    
    max_prob = 0
    max_prob_slice = 0
    max_roi_slice = seg.sum(-1).sum(-1)[0].argmax().item()
    
    for i in range(A):
        slice_input = inputs[:, :, i]
        
        outputs = net(slice_input)
        
        class_idx = targets[0]
        
        if cfg.model.num_classes == 1:
            prob = F.sigmoid(outputs)
            class_prob = prob.item()
        else:
            prob = F.softmax(outputs, 1)
            class_prob = prob[0][class_idx].item()
            
        if max_prob < class_prob:
            max_prob = class_prob
            max_prob_slice = i

        if i ==0 :
            final_out = outputs.detach().cpu()
        else:
            final_out += outputs.detach().cpu()
    
    if cfg.model.num_classes == 1:
        attribution_generator.plot_and_save(net, inputs[:, :, max_prob_slice], seg[:, max_prob_slice], name+'_max_prob', cfg, 0)
        attribution_generator.plot_and_save(net, inputs[:, :, max_roi_slice], seg[:, max_roi_slice], name+'_max_roi', cfg, 0)
    else:
        attribution_generator.plot_and_save(net, inputs[:, :, max_prob_slice], seg[:, max_prob_slice], name+'_max_prob', cfg, class_idx)
        attribution_generator.plot_and_save(net, inputs[:, :, max_roi_slice], seg[:, max_roi_slice], name+'_max_roi', cfg, class_idx)
        
    final_out /= A
    
    return final_out




# Patch Validation
def test(net, val_loader, cfg): 
    test_metrics = hydra.utils.instantiate(cfg.metric)
    attribution_generator = hydra.utils.instantiate(cfg.xai)
    
    net.eval()
    total_outputs = []
    total_targets = []
    total_names = []
    
    for batch_idx, data_dict in enumerate(val_loader):
        inputs = data_dict['image'].cuda(non_blocking=True).float()
        targets = data_dict['label'].cuda(non_blocking=True).long()

        name = data_dict['name'][0]
        seg = data_dict['seg'].cuda(non_blocking=True).long()
        
        final_out = slice_ensemble(inputs, seg, targets, net, attribution_generator, name, cfg)
        test_metrics.update(final_out, targets)
        
        if isinstance(total_outputs, list):
            total_outputs = final_out
            total_targets = targets
        else:
            total_outputs = torch.cat((total_outputs, final_out), 0)
            total_targets = torch.cat((total_targets, targets), 0)
        
        total_names += data_dict['name']
        

    print('Compute evaluation metrics')
    metrics_dict = test_metrics.on_epoch_end_compute(cfg.best_thres)
    test_metrics.plot_graphs(cfg.paths.output_dir)
    
    sorteddict = list(metrics_dict.keys())
    sorteddict.sort()
    metrics_dict = {i: metrics_dict[i] for i in sorteddict}
    
    if is_main_process():
        logger = logging.getLogger(cfg.task_name)
        print('[!] Evaluating the data in ' + str(cfg.paths.data_root))
        for key, value in metrics_dict.items():
            if isinstance(value, float):
                if key in ['Accuracy', 'Sensitivity', 'Specificity']:
                    logger.info(str(key)+ "  :  " + str(round(value*100, 1)))
                else:
                    logger.info(str(key)+ "  :  " + str(round(value, 3)))
            else:
                logger.info(str(key)+ "  :  {}".format(value))
    
    return metrics_dict, [total_outputs, total_targets, total_names]
        
        
        
if __name__ == '__main__':   
    import matplotlib  
    matplotlib.use('Agg')  

    main()