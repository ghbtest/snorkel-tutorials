# 2022-03-21 15:38:54
# checking dataset
import os
import pandas as pd
import pickle
#%%
dtfp="/Users/robinlu/NLP/data_programming/snorkel-tutorials/spam/data"
dfs=[]
for f in os.listdir(dtfp):
    df=pd.read_csv(dtfp+"/"+f)
    dfs.append(df)
sum([dfs[x].shape[0] for x in range(5)]) #1956
dfs[0].columns
dfs[0].iloc[0,:]

#%%data from spam_tutorial.py
dfall=df_train.append(df_test)
dfall.shape
dfall.columns  #['author', 'date', 'text', 'label', 'video']     
dfall[:5]
pickle.dump(dfall,open(dtfp+"/dfall.pickle","wb"))
#%% google search
concatenate dataframe vertically