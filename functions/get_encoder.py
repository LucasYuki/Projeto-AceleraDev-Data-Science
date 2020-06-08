# -*- coding: utf-8 -*-
"""
Created on Sat May 30 19:14:55 2020

@author: lucas
"""

from tensorflow.keras import  Model

def get_encoder(autoencoder):
    out = None
    flag = False
    for layer in autoencoder.layers:
        if layer.__class__.__name__ == "Dense":
            if flag:
                break
            else:
                flag = True
            out = layer
        else:
            flag = False
            
    encoder = Model(inputs=autoencoder.input,
                    outputs=autoencoder.get_layer(out.name).output)
    return encoder