#Correlation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df_2 = dataset.copy()
df_2['Attrition'] = df_2['Attrition'].apply(lambda x: 0 if x=='No' else 1)
df_2['Gender'] = df_2['Gender'].apply(lambda x: 0 if x=='Male' else 1)
df_2['OverTime'] = df_2['OverTime'].apply(lambda x: 0 if x=='No' else 1)

BusinessTravel_map = {'Non-Travel':0, 'Travel_Rarely':1, 'Travel_Frequently':2}
df_2['BusinessTravel'] = df_2['BusinessTravel'].map(BusinessTravel_map)

MaritalStatus_map = {'Single':0, 'Divorced':1, 'Married':2}
df_2['MaritalStatus'] = df_2['MaritalStatus'].map(MaritalStatus_map)

EducationField_map = {'Other':0, 'Life Sciences':1, 'Medical':2, 'Marketing':3, 'Technical Degree':4,
                     'Human Resources':5}
df_2['EducationField'] = df_2['EducationField'].map(EducationField_map)

JobRole_map = {'Sales Executive':0,'Sales Representative':1,'Laboratory Technician':2,'Manufacturing Director':3,
              'Healthcare Representative':4,'Manager':5,'Research Scientist':6,'Research Director':7,'Human Resources':8}
df_2['JobRole'] = df_2['JobRole'].map(JobRole_map)

Department_map = {'Sales':0, 'Research & Development':1, 'Human Resources':2}
df_2['Department'] = df_2['Department'].map(Department_map)
df_2.info()

plt.figure(figsize=(30,30))
correlation = sns.heatmap(df_2.corr(), vmin=-1, vmax=1, annot=True, linewidths=1)
correlation.set_title('Correlation Graph', fontdict={'fontsize': 24})

rgb = (255/255, 255/255, 255/255)
plt.gcf().patch.set_facecolor(rgb)

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=20)

plt.show()
