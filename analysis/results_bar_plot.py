#%%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_excel('/mai_nas/BYS/brain_metastasis/results.xlsx')

task = df['task'].tolist()[:3]
inval = np.array(df['InVal'].tolist()).reshape(2,3)
exval = np.array(df['ExVal'].tolist()).reshape(2,3)


# 데이터 예시
categories = task

# seaborn 스타일 설정
# sns.set_style({'font.family':'serif', 'font.serif':['Times New Roman']})
sns.set_theme(style="whitegrid")
sns.set_context('paper', font_scale=1.25)

# ax = sns.barplot(x='InVal', y='task', hue='model', data= df)
ax = sns.barplot(x='InVal', y='task', hue='model', data= df, legend = False)

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

# 데이터 예시
categories = task

# seaborn 스타일 설정
# sns.set_style({'font.family':'serif', 'font.serif':['Times New Roman']})

sns.set_theme(style="whitegrid")
# ax = sns.barplot(x='ExVal', y='task', hue='model', data= df)
ax = sns.barplot(x='ExVal', y='task', hue='model', data= df, legend = False)

# 제목 추가
ax.set_title('AUC')
ax.set_xlim(0.5, 1)
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles=handles, labels=labels)
# plt.legend(loc='center right', title='')

# 그래프 보여주기
plt.show()
plt.savefig('b.png', dpi=1200)

# %%
