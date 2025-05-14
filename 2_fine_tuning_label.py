import os
import pandas as pd
import numpy as np

labeled_data_file = 'data/processed/labeled_output.csv'
df = pd.read_csv(labeled_data_file)
model_list = ['claude', 'gpt4']

df["agreement_score"] = df[[f'rating_{m}' for m in model_list]].nunique(axis=1)
df['target'] = np.nan

df.loc[df['agreement_score']==1, 'target'] = df.loc[df['agreement_score']==1, 'rating_claude']
unagreement_index = df[df['agreement_score']==2].index

batch_size = 10
batches = [unagreement_index[i:i + batch_size] for i in range(0, len(unagreement_index), batch_size)]
df.loc[batches[0], 'target'] = [-1, 0, 0, 0, -1, 0, 1, -1, 0, 1]
df.loc[batches[1], 'target'] = [0, 0, -1, 0, -1, -1, 1, 1, -1, 0]
df.loc[batches[2], 'target'] = [0, 1, -1, 0, 0, 1, 1, 0, -1, 1]
df.loc[batches[3], 'target'] = [0, -1, 0, -1, 0, 0, -1, 0, -1, 0]
df.loc[batches[4], 'target'] = [0, -1, -1, 1, 0]

df_final = df.loc[df['target']!=0, ['text', 'target']]
df_final['target'] = df_final['target'].replace(-1, 0)
df_final.to_csv('data/processed/final_label.csv', index=False)