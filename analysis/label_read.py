
import json
import pandas as pd
import os

grade = 'oligodendroglioma, nos'

with open('/mai_nas/BYS/brain_metastasis/data/TCGA/' + grade + '.json', "r") as st_json:
    abc = json.load(st_json)


a = 1


all_names = []
all_sex = []
all_grades=[]

for sub in abc:
    name = sub['submitter_id']
    sex = sub['demographic']['gender']
    
    all_names.append(name)
    all_sex.append(sex)
    all_grades.append(grade)

df = pd.DataFrame({'name' : all_names, 'sex':all_sex, 'Primary Diagnosis': all_grades})
df.to_excel(os.path.join('/mai_nas/BYS/brain_metastasis/data/TCGA/', grade + '.xlsx'), sheet_name = 'Sheet1', float_format = "%.3f",header = True,
            index = True)