# -*- coding: utf-8 -*-
"""
Created on Sun May 31 18:08:16 2020

@author: lucas
"""
import h5py
from tensorflow.keras.models import load_model
from tensorflow import data
import tensorflow as tf

import numpy as np
from functions.generator import GetH5Generator, fuse_generators
from functions.get_encoder import get_encoder
from functions.evaluate import predict

#%%
Data_All = h5py.File("Data/All.h5")

model_name = "16_4"

batch = 13597*2
Input_gen = GetH5Generator(Data_All, preload=False, batch=batch)[0]
Input = data.Dataset.from_generator(**fuse_generators(Input_gen))

autoencoder = load_model("Autoencoder/"+model_name+"/model.h5")
encoder = get_encoder(autoencoder)

num_batches = Data_All["data"].shape[0]//batch
print(num_batches, "batches")

#%%
shape = (Data_All["data"].shape[0], encoder.output.shape[1])
File_h5   = h5py.File('Data/'+model_name+'encoded.h5', 'w')
File_mean = File_h5.create_dataset("mean", shape)
File_std  = File_h5.create_dataset("std", shape)

#%%
count = 0
for x in Input:
    for key in x:
        x[key] = tf.dtypes.cast(x[key], tf.float32)
    print("%i of %i" %(count+1, num_batches))
    
    predictions = predict(encoder, x, 200, 42)
    File_mean[count*batch:(count+1)*batch,:] = np.mean(predictions, axis=1)
    File_std[count*batch:(count+1)*batch,:]  = np.std(predictions, axis=1)
    count+=1
    
#%%
File_h5.close()