#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from timm.models import create_model
from torchsummary import summary

model=["resnet18","resnet50","resnet101","resnet152","resnet200","densenet121","densenet201",
       "efficientnet_b0", "efficientnet_b2","efficientnet_b3","efficientnet_b4","efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
       "efficientnetv2_rw_m","tf_efficientnetv2_l","tf_efficientnetv2_xl",
       "vit_tiny_patch16_224", "vit_small_patch16_224","vit_base_patch16_224", "vit_large_patch16_224",
       "deit_tiny_patch16_224","deit_small_patch16_224","deit_base_patch16_224",
       "swin_tiny_patch4_window7_224","swin_small_patch4_window7_224", "swin_base_patch4_window7_224", "swin_large_patch4_window7_224"]

params = []

for i, m in enumerate(model):
    net = create_model(m, in_chans=4, num_classes=1).cuda()
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print(model[i], round(pytorch_total_params/1000000, 2),'M')
    params.append(round(pytorch_total_params/1000000,2))

excel_dict = {'model': model, '# of parms': params}
df = pd.DataFrame(excel_dict)
df.to_excel('model_num_of_params.xlsx', sheet_name = 'Sheet1', float_format = "%.3f",header = True, index = True)
    
# df = pd.read_excel('/mai_nas/BYS/brain_metastasis/results_mae.xlsx')

# maskratio = df['Masking Ratio'].tolist()
# val_auc = df['Val_AUC'].tolist()
# test_auc = df['Test_AUC'].tolist()


# fig, ax = plt.subplots()
# ax.plot(maskratio, val_auc, "o-g")
# ax.plot(maskratio, test_auc, "s-m")
# ax.set_title("AUC")
# ax.set_xlabel("Days of the week")
# ax.set_ylabel("Steps walked")
# ax.grid(True)
# ax.legend(["Internal Validation", "Internal Test"])
# plt.grid(False)
# plt.xlim(0, 80)
# plt.show()
# plt.savefig('a.png', dpi=1200)
# plt.close()
# %%
