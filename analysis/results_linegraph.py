#%%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_excel('/mai_nas/BYS/brain_metastasis/results_mae.xlsx')

maskratio = df['Masking Ratio'].tolist()
val_auc = df['Val_AUC'].tolist()
test_auc = df['Test_AUC'].tolist()


fig, ax = plt.subplots()
ax.plot(maskratio, val_auc, "o-g")
ax.plot(maskratio, test_auc, "s-m")
ax.set_title("AUC")
ax.set_xlabel("Days of the week")
ax.set_ylabel("Steps walked")
ax.grid(True)
ax.legend(["Internal Validation", "Internal Test"])
plt.grid(False)
plt.xlim(0, 80)
plt.show()
plt.savefig('a.png', dpi=1200)
plt.close()
# %%
