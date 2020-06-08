# -*- coding: utf-8 -*-
"""
Esse arquivo converte os arquivos .csv para .h5.
Além disso também divide nos conjuntos de treino, validação e teste para o
treinamento do autoencoder.
"""

import pandas as pd
import h5py
import numpy as np
import os 
import json

directory = "Data"

cat_list = os.listdir(directory+"/cat")
cat = {}
for i in range(len(cat_list)):
    temp = cat_list[i][:-4]
    cat[temp] = pd.read_csv(directory+"/cat/"+cat_list[i], index_col=0)
    cat_list[i] = temp
NaN_cat  = pd.read_csv(directory+"/NaN_cat.gz", index_col=0)
num_bool = pd.read_csv(directory+"/num_bool.gz", index_col=0)
NaN_num  = pd.read_csv(directory+"/NaN_num.gz", index_col=0)

#%%
NaN_num_columns = list(NaN_num.columns)
not_NaN_num_columns = [x for x in num_bool.columns if x not in NaN_num_columns]

#%%
def create_h5_datasets(group, n_samples):
    dataset_data = {}
    for c in cat_list:
        dataset_data[c] = group.create_dataset(c,(n_samples, len(cat[c].columns)), 
                                               dtype='i1', compression = "lzf")
    dataset_data["data"] = group.create_dataset("data",(n_samples, 
                                                        len(not_NaN_num_columns)), 
                                                compression = "lzf")
    dataset_data["data_na"] = group.create_dataset("data_na",(n_samples, 
                                                              len(NaN_num_columns)),
                                                   compression = "lzf")
    dataset_na_cat = group.create_dataset("NaN_cat", (n_samples, 
                                                      len(NaN_cat.columns)), 
                                          dtype='i1', compression = "lzf")
    dataset_na_num = group.create_dataset("NaN_num", (n_samples, 
                                                      len(NaN_num.columns)), 
                                          dtype='i1', compression = "lzf")
    group.attrs["NaN_cat"] = json.dumps(list(NaN_cat.columns))
    group.attrs["NaN_num"] = json.dumps(list(NaN_num_columns))
    return dataset_data, dataset_na_cat, dataset_na_num

#%%
NaN_sum = NaN_cat.sum(axis = 1) + NaN_num.sum(axis = 1)
index = np.random.permutation(NaN_sum[NaN_sum == 0].index)

#%%
length = {"All"  : len(num_bool),
          "train": len(index)//2,
          "val"  : len(index)//4, 
          "test" : len(index)-len(index)//2-len(index)//4}
slices = {"All"  : num_bool.index,
          "train": index[:length["train"]],
          "val"  : index[length["train"]: length["train"]+length["val"]], 
          "test" : index[length["train"]+length["val"]: len(index)]}

with open(directory+"/All_index.json", "w") as File:
    json.dump(dict(zip(num_bool.index, range(len(num_bool.index)))), File)
    
for part in ["All", "train", "val", "test"]:
    print(part)
    File_h5 = h5py.File(directory+'/'+part+'.h5', 'w')

    ds, ds_NaN_cat, ds_NaN_num = create_h5_datasets(File_h5, length[part])
    
    print(" - cat_list")
    for c in cat_list:
        ds[c][:] = cat[c].loc[slices[part]].to_numpy(dtype=int)
    
    print(" - data")
    ds["data"][:]    = num_bool[not_NaN_num_columns] \
                           .loc[slices[part]].to_numpy(dtype=float)
    print(" - data_na")
    ds["data_na"][:] = num_bool[NaN_num_columns]\
                           .loc[slices[part]].to_numpy(dtype=float)
    print(" - ds_NaN_cat")
    ds_NaN_cat[:] = NaN_cat.loc[slices[part]].to_numpy(dtype=int)
    print(" - ds_NaN_num")
    ds_NaN_num[:] = NaN_num.loc[slices[part]].to_numpy(dtype=int)
    
    File_h5.close()