#%%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_excel('/mai_nas/BYS/brain_metastasis/results_mae2.xlsx')

# maskratio = df['Pretraining Method'].tolist()
# val_auc = df['Val_AUC'].tolist()
# test_auc = df['Test_AUC'].tolist()


# seaborn 스타일 설정
# sns.set_style({'font.family':'serif', 'font.serif':['Times New Roman']})
sns.set_theme(style="whitegrid")
sns.set_context('paper', font_scale=1.25)

# ax = sns.barplot(x='InVal', y='task', hue='model', data= df)
ax = sns.barplot(x='auc', y='val_test', hue='method', data= df)
ax.legend_.remove()

# 제목 추가
ax.set_title('AUC')
ax.set_xlim(0.5, 1)
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles=handles, labels=labels)
# plt.legend(loc='center right', title='')

# 그래프 보여주기
plt.show()
plt.savefig('a.png', dpi=1200)
plt.close()

# %%
