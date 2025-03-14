from timm.models import create_model
import torch.nn.functional as F
from transformers import AutoTokenizer, BertModel
import torch

# text = ["female", "male"]
# text = ["a MR image of an old patient", "a MR image of a young patient"]
text = ["an image of an old patient", "an image of a young patient"]
# text = ["a photo of a male patient", "a photo of a female patient"]
# text = ["a photo of a male patient", "a photo of an old patient"]
# text = ["old", "young"]

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

last_hidden_state= outputs.last_hidden_state
last_hidden_state = last_hidden_state[:, 0]

print(last_hidden_state)

sim = F.cosine_similarity(last_hidden_state[0:1], last_hidden_state[1:])
print(sim)

# print(outputs[0].shape, outputs[1].shape)
# from torchsummary import summary
# # vit_base_patch16_224
# # swin_base_patch4_window7_224
# model = create_model('vit_base_patch16_224', in_chans=4, num_classes=1).cuda()

# print(model)
# pytorch_total_params = sum(p.numel() for p in model.parameters())
# print(pytorch_total_params/1000000,'M')
# summary(model, (4, 224, 224), batch_size=1)

# ResNet18:  11
# Resent50:  23
# densenet121:   7
# densenet201:   18
# efficentnet_b0:   4 
# efficentnet_b2:   7
# efficentnet_b3:   10
# efficentnet_b4:   17
# efficentnet_b5:   28

# vit_tiny_patch16_224:   5
# vit_small_patch16_224:   21
# vit_base_patch16_224:   86
# vit_large_patch16_224:   303


# import torch

# a = torch.load('/mai_nas/BYS/Correlational-Image-Modeling/exps/cim_job/checkpoint-220.pth')
# a = torch.load('/mai_nas/BYS/mae/output_dir/mae_pretrain_vit_base.pth')

# b = torch.load('/mai_nas/BYS/mae/output_dir/mae_ch4_25percent_ACS/checkpoint-299.pth')
# b=1
# new_dict = {}
# for k, v in a['model'].items():
#     if "_c." in k:
#         new_dict[k.replace('_c', '')] = v

# new_dict['cls_token'] = a['model']['cls_token']
# new_dict['pos_embed'] = a['model']['pos_embed']








# from monai import transforms
# from monai.transforms import Resized, RandAffined, RandFlipd, RandGaussianNoised, RandGaussianSmoothd, RandGaussianSmoothd, RandScaleIntensityd, RandAdjustContrastd, ToTensord
# import numpy as np
# import SimpleITK as sitk
# import os 
# from src.total_utils import normalize_images, sequence_combination, load_custom_pretrined, convert_to_distributed_network
# import random
# import matplotlib.pyplot as plt

# transform =transforms.Compose([   
#                                         Resized(keys=["image"], spatial_size=(224,224), mode="bicubic"),
#                                         RandAffined(keys=["image"], mode="bilinear", prob=1.0, 
#                                                     rotate_range=(np.pi/12, np.pi/12), scale_range=(0.15, 0.15), shear_range=(0.15, 0.15), padding_mode="border"),
#                                         RandGaussianNoised(keys=['image'], mean=0, std=0.2, prob=1.0),
#                                         RandGaussianSmoothd(keys=['image'], sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), prob=1.0),
#                                         RandScaleIntensityd(keys=["image"], factors=0.25, prob=1.0),
#                                         RandAdjustContrastd(keys=['image'], gamma=(0.75, 1.25), prob=1.0),
#                                         ToTensord(keys=["image"])
#                                         ])


# data_root="/mai_nas/BYS/brain_metastasis/preprocessed/SEV/"
# name="AA0001"
# adc=None
# seq_comb="4seq"
# slice_percentile=50


# t1 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_root, name, name +'_T1.nii.gz')))        
# t1c = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_root, name, name +'_T1C.nii.gz')))
# t2 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_root, name, name +'_T2.nii.gz')))
# flair = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_root, name, name +'_FLAIR.nii.gz')))

# seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_root, name, name + '_seg.nii.gz')))
# seg[seg !=0] = 1



# t1, t1c, t2, flair, adc = normalize_images(t1, t1c, t2, flair, adc)
# img_npy = sequence_combination(t1, t1c, t2, flair, adc, seq_comb)

# z_seg = seg.sum(-1).sum(-1)

# # 상위 %
# glioma_vol_lower_bound = np.percentile(z_seg[z_seg.nonzero()[0]], 100-slice_percentile, axis=0)
# roi_mask = z_seg > glioma_vol_lower_bound
# roi_idx_list = np.where(roi_mask==True)[0].tolist()
# # ---------------------------------------------------------------------------------------------------------------------------------------

# roi_random_idx = random.choice(roi_idx_list)
# img_npy_2d = img_npy[:, roi_random_idx] #CHW
# data_dict = {'image' : img_npy_2d, 'name' : name}

# data_dict = transform(data_dict)

# print(img_npy_2d.min(), img_npy_2d.max(), img_npy_2d.mean(), img_npy_2d.std())
# print(data_dict['image'].min(), data_dict['image'].max(), data_dict['image'].mean(), data_dict['image'].std())

# fig = plt.figure(frameon=False, dpi=600)
# for j in range(4):
#     ax = fig.add_subplot(2,4,j+1)
#     ax.imshow(img_npy_2d[j], cmap=plt.cm.gray, interpolation='nearest')
#     ax.axis('off')
    
#     ax = fig.add_subplot(2,4,j+5)
#     ax.imshow(data_dict['image'][j].numpy(), cmap=plt.cm.gray, interpolation='nearest')
#     ax.axis('off')

# plt.tight_layout()
# plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)


# plt.savefig('a.png')
# plt.close()
# plt.clf()

