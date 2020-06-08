# -*- coding: utf-8 -*-
"""
Created on Sat May 30 10:28:30 2020

@author: lucas
"""

from sys import exit
import numpy as np
import json
import os
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

def predict(model, x, n_predictions=100, gen_num=None):
    predictions = []
    for i in range(n_predictions):
        if gen_num is None:
            data = x
        else:
            data = tf.dtypes.cast(x(gen_num), tf.float32)
        predictions.append(model(data, training=True))
    predictions = np.stack(predictions, axis=1)
    return predictions

#%%
def evaluate(paths, Test, seed=42, n_predictions=100, clear_session=True):
    np.random.seed(seed)
    
    if isinstance(paths, str):
        paths = [paths]
        
    with open("Autoencoder/loss_weights.json", "r") as File:
        loss_weights = json.load(File)
        
    num_batches = 0
    for x in Test:
        num_batches+=1

    mean_losses = {}
    for dir_path in paths:
        try:
            autoencoder = load_model(dir_path+"/model.h5")
        except:
            print("Load model %s failed\n" % (dir_path+"/model.h5"))
            input("Press enter to exit")
            exit()   
                
        losses = []
        for l in autoencoder.loss:
            losses.append(tf.keras.losses.get(l))
            
        loss = []
        count = 0 
        for x, y in Test:
            for key in x:
                x[key] = tf.dtypes.cast(x[key], tf.float32)
            y = list(y.values())
            print("%i of %i" %(count+1, num_batches))
            count+=1
            
            predictions = predict(autoencoder, x, n_predictions)
            
            temp_loss = []
            mean = []
            for i in range(len(predictions)):
                mean.append(np.mean(predictions[i], axis=0))
                temp = tf.convert_to_tensor(list(mean[i]))
                temp_loss.append(np.mean(losses[i](y[i], temp)))    
            temp_loss = dict(zip(list(x.keys()), temp_loss))
            temp_loss["loss"] = sum([(temp_loss[key]*loss_weights[key]) \
                                     for key in loss_weights])
            loss.append(temp_loss)
            #print(temp_loss)
    
        mean_loss = np.array([list(i.values()) for i in loss]).mean(axis=0)
        mean_loss = dict(zip(loss[0].keys(), list(mean_loss)))
        mean_losses[dir_path] = mean_loss
        
        with open(dir_path+"/test_loss.json", "w") as File:
            json.dump(mean_loss, File)
        print(mean_loss)
        
        if clear_session:
            K.clear_session()
    return mean_losses

#%%
def evaluate_Deep_One(paths, gen, center, num_batches, 
                      seed=42, n_predictions=100, clear_session=True):
    np.random.seed(seed)
    
    if isinstance(paths, str):
        paths = [paths]

    mean_losses = {}
    for dir_path in paths:
        try:
            model = load_model(dir_path+"/model.h5")
        except:
            print("Load model %s failed\n" % (dir_path+"/model.h5"))
            input("Press enter to exit")
            exit()   
                
        losses = tf.keras.losses.get(model.loss)
            
        loss = []
        for n in range(num_batches):
            print("%i of %i" %(n+1, num_batches))
             
            predictions = predict(model, gen, gen_num=n)
            
            temp_loss = []
            mean = []
            for i in range(len(predictions)):
                mean.append(np.mean(predictions[i], axis=0))
                temp = tf.convert_to_tensor(list(mean[i]))
                temp_loss.append(np.mean(losses(center, temp)))
            loss.append(temp_loss)
    
        mean_loss = {"loss": float(np.mean(loss))}
        mean_losses[dir_path] = mean_loss
        print(mean_loss)
        
        if clear_session:
            K.clear_session()
    return mean_losses


#%%
def get_all_losses():
    with open("Autoencoder/loss_weights.json", "r") as File:
        loss_weights = json.load(File)
        
    all_losses = {}
    for x in os.listdir("Autoencoder"):
        if os.path.isdir("Autoencoder/"+x):
            with open("Autoencoder/"+x+"/test_loss.json", "r") as File:
                loss = json.load(File)
                
            if int(x.split("_")[1]) <= 3:
                loss["loss"] = sum([(loss[key]*loss_weights[key]) for key in loss_weights])
            print(x, loss["loss"])
            all_losses[x] = loss
    df = pd.DataFrame(all_losses)
    df.to_csv("Autoencoder/all_losses.csv")
    return all_losses
