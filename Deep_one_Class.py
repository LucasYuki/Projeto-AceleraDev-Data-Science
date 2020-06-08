# -*- coding: utf-8 -*-
"""
O Deep one-Class é uma rede utilizada para fazer classificação quando só
tem disponível uma classe. 

O objetivo desta rede é fazer com que os dados de treinamento tenham como saída
um ponto o mais próximo possível do centro de uma hiperesfera pré-determinada,
desta forma esta rede fará com que os dados que se assemelham com os dados de 
treino estejam próximas ao centro da hiperesfera, desta forma classificando 
os dados.

Para mais informações veja o artigo:

Ruff, L., Vandermeulen, R., Goernitz, N., Deecke, L., Siddiqui, S.A., 
Binder, A., Müller, E. & Kloft, M.. (2018). Deep One-Class Classification. 
Proceedings of the 35th International Conference on Machine Learning, 
in PMLR 80:4393-4402

@author: lucas
"""

# Setup
import tensorflow as tf
from tensorflow import data, random
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model, regularizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, \
                                       ModelCheckpoint, History
                                       
import pandas as pd
import numpy as np
import json
import h5py
import os

from functions.plot_train_hist import plot_train_hist
from functions.evaluate import evaluate_Deep_One, predict

def generator(mean, std, batch):
    num_samples = mean.shape[0]
    num_batches = int(np.ceil(num_samples/batch))
    
    def gen(i):
        inic = i*batch
        temp = mean[inic:inic+batch,:]
        temp+= np.random.normal(scale=std[inic:inic+batch,:], 
                                size=temp.shape)
        return temp
    return {"generator": gen, 
            "output_types": mean[0].dtype, 
            "output_shapes": tf.TensorShape([batch] + [mean.shape[1]]),
            "num_batches": num_batches}

def gen_build(gen, center=None):
    if center is None:
        outputs_types  = gen["output_types"]
        outputs_shapes = gen["output_shapes"]
        def generator():
            for i in range(gen["num_batches"]):
                yield gen["generator"](i)
    else:
        outputs_types  = (gen["output_types"], center.dtype)
        outputs_shapes = (gen["output_shapes"], 
                          tf.TensorShape([gen["output_shapes"][0], center.shape[0]]))
        def generator():
            for i in range(gen["num_batches"]):
                yield (gen["generator"](i), np.tile(center, (gen["output_shapes"][0], 1)))
    return {"generator": generator, 
            "output_types": outputs_types, 
            "output_shapes": outputs_shapes}
    
def Deep_One(data_path, save_dir, portifolio_num):
    dropout_prob = 0.2
    patience = 10
    epochs = 100
    save_train_history = True
    
    n_predictions = 100
    seed = 42
    n_models = 10
    
    n_out = 1
    
    h5py_file = h5py.File(data_path, "r")
    
    index = pd.read_csv("Data/Portifolios/estaticos_portfolio" + \
                        str(portifolio_num) + ".csv")["id"].to_list()
    with open("Data/All_index.json", "r") as File:
        all_index = json.load(File)
    for i in range(len(index)):
        index[i] = all_index[index[i]]
    
    np.random.seed(seed)
    random.set_seed(seed)
    
    index = np.random.permutation(index)
        
    train_size = int(len(index)*0.8)
    val_size  = (len(index) - train_size)//2
    test_size = len(index) - val_size - train_size
    
    train_index = np.sort(index[:train_size])
    val_index   = np.sort(index[train_size:train_size+val_size])
    test_index  = np.sort(index[train_size+val_size:])
    
    Train_gen = generator(h5py_file["mean"][:][train_index], 
                          h5py_file["std"][:][train_index],
                          batch=train_size)
    Val_gen   = generator(h5py_file["mean"][:][val_index],
                          h5py_file["std"][:][val_index],
                          batch=val_size)
    Test_gen  = generator(h5py_file["mean"][:][test_index],
                          h5py_file["std"][:][test_index],
                          batch=test_size)
    
    loss = {"Train":[], "Val":[], "Test":[], "error val":[], "error test":[]}
    for n_model in range(n_models):
        dir_path = "Deep_One-Class/"+save_dir+"/"+str(n_model)
        
        if os.path.isdir(dir_path):
            print(dir_path)
            print("directory already exist, please chose other name")
            input("Press enter to exit")
            h5py_file.close()  
            exit()
        else:
            try:
                os.makedirs(dir_path)
            except:
                print("Creation of the directory %s failed\n" % (dir_path))
                input("Press enter to exit")
                h5py_file.close()  
                exit()
        
        Model_Input  = Input(shape=h5py_file["mean"].shape[1], name="input")
        hidden_layer = Dense(256, activation="relu", use_bias=False,
                             kernel_regularizer=regularizers.l2())(Model_Input)
        hidden_layer = Dropout(dropout_prob)(hidden_layer, training=True)
        hidden_layer = Dense(256, activation="relu", use_bias=False,
                             kernel_regularizer=regularizers.l2())(hidden_layer)
        hidden_layer = Dropout(dropout_prob)(hidden_layer, training=True)
        Model_Output = Dense(n_out, use_bias=False,
                             kernel_regularizer=regularizers.l2())(hidden_layer)
    
        model = Model(Model_Input, Model_Output)
        
        model.compile(optimizer="adam", loss="mse")
        plot_model(model, dir_path + '/plot_model.png', show_shapes=True)
        
        # treina o modelo
        print(model.summary())
        with open(dir_path + '\\model_description.txt', 'w') as txt:
            model.summary(print_fn=lambda x: txt.write(x + '\n'))
        
        print("\n------------------------------------------------------------\n")
        print('The model will be save in the file ' + dir_path +"\\model.h5")
        
        """trains the neural network"""
        callbacks=[]
        cl = CSVLogger(dir_path+'\\train_history.csv', separator=',', append=False)
        mc = ModelCheckpoint(dir_path+"\\model.h5", monitor='val_loss', 
                             mode='min', verbose=1, save_best_only=True)
        callbacks.append(cl)
        callbacks.append(mc)
        
        if patience>=1:
            es = EarlyStopping(monitor='val_loss', mode='min', 
                               verbose=1, patience=patience)
            callbacks.append(es)
        
        if save_train_history:
            history = History()
            callbacks.append(history)
            
        center = []
        for k in range(10):
            center.append(np.sum(model(Train_gen["generator"](0)), axis=0))
        center = np.sum(center, axis=0)
            
        with open(dir_path + "/center.json", "w") as File:
            json.dump(center.tolist(), File)
        
        Train = data.Dataset.from_generator(**gen_build(Train_gen, center=center))
        Val   = data.Dataset.from_generator(**gen_build(Val_gen, center=center))
    
        model.fit(Train, epochs=epochs, validation_data= Val, 
                        verbose=2, callbacks=callbacks)
        
        # envia os resutados para a conversa do telegram configurada
        plot_train_hist(dir_path)
        
        # faz a avaliação do modelo com o conjunto de teste
        datasets = {"Train":Train_gen, "Val":Val_gen, "Test":Test_gen}
        for d in datasets:
            print(d)
            loss[d].append(evaluate_Deep_One(dir_path, gen=datasets[d]["generator"], 
                                             center=center, seed=seed,
                                             num_batches=datasets[d]["num_batches"],
                                             n_predictions=n_predictions,
                                             clear_session=False)[dir_path]["loss"])
        loss["error val"].append(abs(loss["Val"][-1]/loss["Train"][-1]-1))
        loss["error test"].append(abs(loss["Test"][-1]/loss["Train"][-1]-1))
        
        K.clear_session()
        
    df = pd.DataFrame(loss)
    df.to_csv("Deep_One-Class/"+save_dir+"/losses.csv")
    h5py_file.close() 
    
def Aply_Deep_One(data_path, models_dir, portifolio_num, model_num):
    dir_path = models_dir+"/"+str(model_num)
    
    seed = 42
    
    port_index = pd.read_csv("Data/Portifolios/estaticos_portfolio" + \
                             str(portifolio_num) + ".csv")["id"].to_list()
    with open("Data/All_index.json", "r") as File:
        all_index = json.load(File)
    index = []
    index_name = []
    for i in all_index:
        if i not in port_index:
            index.append(all_index[i])
            index_name.append(i)
    
    h5py_file = h5py.File(data_path, "r")
    with open(dir_path + "/center.json", "r") as File:
        center = json.load(File)
        
    Data = generator(h5py_file["mean"][index], 
                     h5py_file["std"][index],
                     batch=len(index))["generator"]
    np.random.seed(seed)
    random.set_seed(seed)
    
    model = load_model(dir_path + "\model.h5")
    
    Results = pd.DataFrame(index_name, columns=["index"])
    
    predictions = predict(model, Data, gen_num=0)-center
    mean = pd.DataFrame(np.square(np.mean(predictions, axis=1)), 
                        columns=["mean"])
    std  = pd.DataFrame(np.std(predictions, axis=1) , columns=["std"])
    
    Results = pd.concat([Results, mean, std], axis=1).set_index("index")

    Results.to_csv(dir_path+"/results.csv")

    K.clear_session()
    h5py_file.close() 