# -*- coding: utf-8 -*-
"""
A partir do arquivo h5 cria as funções geradoras para a entrada de dados
do autoencoder.
"""

import json
import tensorflow as tf
import numpy as np
from numpy.random import binomial

def GetH5Generator(h5_file, preload=False, batch=1, 
                   prob=0, out=False, losses=False):
    NaN_cat_columns = json.loads(h5_file.attrs['NaN_cat'])
    datasets = ["NaN_cat", "NaN_num", "data", "data_na"] + NaN_cat_columns
    cat_columns = [x for x in h5_file.keys() if x not in datasets]
    
    if preload:
        Data         = h5_file["data"][:]
        Data_NaN     = h5_file["data_na"][:]
        Data_cat     = {x: h5_file[x][:] for x in cat_columns}
        Data_cat_NaN = {x: h5_file[x][:] for x in NaN_cat_columns}
    else:
        Data         = h5_file["data"]
        Data_NaN     = h5_file["data_na"]
        Data_cat     = {x: h5_file[x] for x in cat_columns}
        Data_cat_NaN = {x: h5_file[x] for x in NaN_cat_columns}
    
    num_samples = Data.shape[0]
    num_batches = int(np.ceil(num_samples/batch))
    
    def get_gen(data):
        def generator(i):
            inic = i*batch
            return (data[inic:inic+batch,:])
        return {"generator": generator, 
                "output_types": data[0].dtype, 
                "output_shapes": tf.TensorShape([batch] + [data.shape[1]]),
                "num_batches": num_batches}
    
    def get_gen_NaN(data, row=False):
        if row:
            row = 1
        else:
            row = data.shape[1]
        def generator(i):
            inic = i*batch
            temp = data[inic:inic+batch,:]
            mask = binomial(1, prob, size=data[inic:inic+batch,:].shape[0]*row)
            mask = mask.reshape(data[inic:inic+batch,:].shape[0], row).astype("bool")
            temp = np.multiply(temp, np.logical_not(mask))
            return (np.hstack((temp, mask)))
        return {"generator": generator, 
                "output_types": data[0].dtype, 
                "output_shapes": tf.TensorShape([batch] + [data.shape[1]+row]),
                "num_batches": num_batches}
    
    Inputs = {}
    Inputs["Data"]      = get_gen(Data)
    Inputs["Data_NaN"]  = get_gen_NaN(Data_NaN, row=False)
    for x in cat_columns:
        Inputs[x]  = get_gen(Data_cat[x])
    for x in NaN_cat_columns:
        Inputs[x]  = get_gen_NaN(Data_cat_NaN[x], row=True)
        
    R = (Inputs,)
    
    if out:
        Outputs = {}
        Outputs["Data_out"]     = get_gen(Data)
        Outputs["Data_NaN_out"] = get_gen(Data_NaN)
        for x in cat_columns:
            Outputs[x+"_out"] = get_gen(Data_cat[x])
        for x in NaN_cat_columns:
            Outputs[x+"_out"] = get_gen(Data_cat_NaN[x])
        R += (Outputs,)
    
    if losses:
        loss = ["mse"]*2
        loss+= ['categorical_crossentropy']*(len(cat_columns)+
                                             len(NaN_cat_columns))
        R += (loss,)
        
    return R
    
def fuse_generators(gen_list, tar=None):
    generators     = {}
    outputs_types  = {}
    outputs_shapes = {}
    for gen in gen_list:
        generators[gen]     = gen_list[gen]["generator"]
        outputs_types[gen]  = gen_list[gen]["output_types"]
        outputs_shapes[gen] = gen_list[gen]["output_shapes"]
    n_batches = gen_list[gen]["num_batches"]
    if tar:
        tar_generators     = {}
        tar_outputs_types   = {}
        tar_outputs_shapes = {}
        for gen in tar:
            tar_generators[gen]     = tar[gen]["generator"]
            tar_outputs_types[gen]  = tar[gen]["output_types"]
            tar_outputs_shapes[gen] = tar[gen]["output_shapes"]
        def fused_gen():
            for i in range(n_batches):
                out = {}
                for gen in generators:
                    out[gen] = generators[gen](i)
                tar = {}
                for gen in tar_generators:
                    tar[gen] = tar_generators[gen](i)
                yield (out, tar) 
        return {"generator": fused_gen, 
                "output_types": (outputs_types, tar_outputs_types), 
                "output_shapes": (outputs_shapes, tar_outputs_shapes),}
    else:
        def fused_gen():
            for i in range(n_batches):
                out = {}
                for gen in generators:
                    out[gen] = generators[gen](i)
                yield out
                
        return {"generator": fused_gen, 
                "output_types": outputs_types, 
                "output_shapes": outputs_shapes}
