# -*- coding: utf-8 -*-
"""
Este arquivo treina o autencoder

@author: lucas
"""

# Setup

import os
from sys import exit
import h5py

from tensorflow import data
from tensorflow.keras import Input, Model
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import concatenate, Dense, Dropout
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, \
                                       ModelCheckpoint, History

from functions.generator import GetH5Generator, fuse_generators
from functions.plot_train_hist import plot_train_hist
from functions.evaluate import evaluate, get_all_losses
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('expand_frame_repr', False)

dropout_prob = 0.2
nan_prob = 0.2
preload = True
batch = 59569

patience = 100
epochs = 2000
save_train_history =True

test_batch = 839 # o tamanho do batch de teste deve ser menor
n_predictions = 100
seed = 42

#%% carrega os dados e cria o modelo
Train_File = h5py.File("Data/train.h5", 'r')
Val_File = h5py.File("Data/val.h5", 'r')
Test_File = h5py.File("Data/test.h5", 'r')

Train_Input_gen, Train_Output_gen, losses = GetH5Generator(Train_File, 
                                                           preload=preload, 
                                                           batch=batch, 
                                                           out=True, 
                                                           prob=nan_prob,
                                                           losses = True)
Val_Input_gen, Val_Output_gen = GetH5Generator(Val_File, preload=preload, 
                                       batch=batch, out=True, prob=nan_prob)

Test_Input_gen, Test_Output_gen = GetH5Generator(Test_File, preload=preload, 
                                       batch=test_batch, out=True, prob=nan_prob)

Train = data.Dataset.from_generator(**fuse_generators(Train_Input_gen,
                                                      Train_Output_gen))
Val   = data.Dataset.from_generator(**fuse_generators(Val_Input_gen,
                                                      Val_Output_gen))
Test  = data.Dataset.from_generator(**fuse_generators(Test_Input_gen,
                                                      Test_Output_gen))

#%%
for num in [18, 20]:
    # Verifica se o diretorio onde será salvo já existe
    dir_path = "Autoencoder/"+str(num)+"_4"
    
    if os.path.isdir(dir_path):
        print(dir_path)
        print("directory already exist, please chose other name")
        input("Press enter to exit")
        Train_File.close()  
        Val_File.close()  
        exit()
    else:
        try:
            os.makedirs(dir_path)
        except:
            print("Creation of the directory %s failed\n" % (dir_path))
            input("Press enter to exit")
            Train_File.close()  
            Val_File.close()  
            exit()
        
    Model_Inputs   = []
    Model_Outputs  = []
    Losses_weights = {}
    for inp in Train_Input_gen.keys():
        Model_Inputs.append(Input(shape=(Train_Input_gen[inp]["output_shapes"][1],),
                                  name=inp))
        
        units = Train_Output_gen[inp+"_out"]["output_shapes"][1]
        if losses[list(Train_Input_gen.keys()).index(inp)]=="mse":
            activation = "linear"
            Losses_weights[inp+"_out"] = units
        else:
            activation = "softmax"
            Losses_weights[inp+"_out"] = 1
        
        Model_Outputs.append(Dense(units,
                                   name=inp+"_out", activation=activation))
        
    total_weights = sum(Losses_weights.values())
    Losses_weights = {key: Losses_weights[key]/total_weights for key in Losses_weights}
    
    Input_layer = concatenate(Model_Inputs)
    hidden_encoder = Dense(256, activation="tanh")(Input_layer)
    hidden_encoder = Dropout(dropout_prob)(hidden_encoder, training=True)
    hidden_encoder = Dense(256, activation="tanh")(hidden_encoder)
    hidden_encoder = Dropout(dropout_prob)(hidden_encoder, training=True)
    encoder_out = Dense(num, activation="tanh", name="encoder_out")(hidden_encoder)
    
    decoder_inp = Dense(256, activation="tanh")(encoder_out)
    hidden_decoder = Dropout(dropout_prob)(decoder_inp, training=True)
    hidden_decoder = Dense(256, activation="tanh")(hidden_decoder)
    hidden_decoder = Dropout(dropout_prob)(hidden_decoder, training=True)
    
    for i in range(len(Model_Outputs)):
        Model_Outputs[i] = Model_Outputs[i](hidden_decoder)

    autoencoder = Model(Model_Inputs, Model_Outputs)
    
    autoencoder.compile(optimizer="Adam", loss=losses, loss_weights=Losses_weights)
    plot_model(autoencoder, dir_path + '/plot_model.png', show_shapes=True)
    
    # treina o modelo
    print(autoencoder.summary())
    with open(dir_path + '\\model_description.txt', 'w') as txt:
        autoencoder.summary(print_fn=lambda x: txt.write(x + '\n'))
    
    print("\n------------------------------------------------------------\n")
    print('The model will be save in the file ' + dir_path +"\\model.h5")
    
    """trains the neural network"""
    callbacks=[]
    cl = CSVLogger(dir_path+'\\train_history.csv', separator=',', append=False)
    mc = ModelCheckpoint(dir_path+"\\model.h5", monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    callbacks.append(cl)
    callbacks.append(mc)
    
    if patience>=1:
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
        callbacks.append(es)
    
    if save_train_history:
        history = History()
        callbacks.append(history)
    
    autoencoder.fit(Train, epochs=epochs, validation_data= Val, 
                    verbose=2, callbacks=callbacks)
    
    plot_train_hist(dir_path)
    
    # faz a avaliação do modelo com o conjunto de teste
    loss = evaluate(dir_path, seed=seed, n_predictions=n_predictions,
                    Test=Test, clear_session=False)
    df = pd.DataFrame(loss)
    
    K.clear_session()
    
df = pd.DataFrame(get_all_losses())
Test_File.close()  
Train_File.close()  
Val_File.close()  
