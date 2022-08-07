#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   main.py
@Time    :   2022/06/21 22:27:16
@Author  :   QuYue 
@Email   :   quyue1541@gmail.com
@Desc    :   main
'''

# %% Import Package
# Basic
import os
import random
import torch
import numpy as np
import time
import datetime

# Add path
os.chdir(os.path.dirname(__file__))

# Self-defined
import utils
import dataprocessor as dp
import modeler as ml
import recorder as rd

# %% Set Super-parameters
class PARAM():
    def __init__(self) -> None:
        # Dataset
        self.dataset_name = "Synthetic"     # Dataset name
        self.train_valid_test = [4, 1, 1]   # The ratio of training, validation and test data
        self.cv = 5                         # Fold number for cross-validation
        self.dataset_setting()              # Dataset setting 

        # Random seed
        self.seed = 1           # Random seed
        self.random_setting()   # Random seed setting

        # Device
        self.gpu = True             # Used GPU, when bool (if use GPU), when int (the ID of GPU) 
        self.device_setting(True)   # Device setting

        # Training
        self.epochs = 10           # Epochs
        self.batch_size = 1000      # Batch size
        self.learn_rate = 0.01      # Learning rate
        self.test_epoch = 1         # Test once every few epochs

        # Model
        self.model_name_list = ["s_learner", "t_learner"]   # Model name list
        self.model_setting()

        # Records
        self.ifrecord = True                # If record
        self.now = datetime.datetime.now()  # Current time
    
    @property
    def train_ratio(self):
        return self.train_valid_test[0] / (self.train_valid_test[0]+self.train_valid_test[2])
    
    def model_param_setting(self, model_name):
        # Setting of parameters of models
        model_param = utils.tools.MyStruct('model_param', [model_name])
        name = model_name.lower().strip()
        if name == "s_learner":
            model_param.name = "S_Learner"
            model_param.input_size = self.dataset.data_dimensions
            model_param.output_size = 1
            model_param.hidden_size = 15
            model_param.layer_number = 3
        elif name == "t_learner":
            model_param.name = "T_Learner"
            model_param.input_size = self.dataset.data_dimensions
            model_param.output_size = 1
            model_param.hidden_size = 15
            model_param.layer_number = 3
        else:
            raise ValueError("Model name is not defined.")
        return model_param
    
    def dataset_setting(self):
        # Setting of dataset
        self.dataset = utils.tools.MyStruct('dataset', [self.dataset_name])
        self.dataset.cv = self.cv
        self.dataset.train_ratio = self.train_ratio
        if self.dataset_name.lower().strip() == "synthetic":
            self.dataset.data_number = 10000
            self.dataset.data_dimensions = 10
            self.dataset.ifprint = False
            self.dataset.stratify = 't'
            self.dataset.keylist = ['x', 't', 'y', 'potential_y']
            self.dataset.typelist = ['float', 'long', 'float', 'float']

    def random_setting(self):
        # Setting of random seed
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)   # If you are using multi-GPU.
    
    def device_setting(self, ifprint=True):
        # Setting of device
        if isinstance(self.gpu, bool):
            if self.gpu:
                try:
                    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "mps:0")
                except:
                    self.device = torch.device("cpu")
            else:
                self.device = torch.device("cpu")
        elif isinstance(self.gpu, int):
            device_count = torch.cuda.device_count()
            self.gpu = min(self.gpu, device_count-1)
            if self.gpu >= 0:
                self.device = torch.device("cuda:" + str(self.gpu) if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device("cpu")
        else:
            raise ValueError("The type of gpu must be bool or int.")
        device = 'CPU' if self.device == torch.device("cpu") else f"GPU {self.device.index} ({torch.cuda.get_device_name(self.device.index)})"
        if ifprint:
            print(f"The experimental environment is set to {device}.")
    
    def model_setting(self):
        # Setting of models
        self.model_param_list = [self.model_param_setting(name) for name in self.model_name_list]
        self.model_list = [ml.get_model(param.name, param.dict) for param in self.model_param_list]


Parm = PARAM()

# %% Main Function
if __name__ == "__main__":
    print("Loading dataset ...")
    dataset = dp.datasets.load_dataset(Parm.dataset_name, seed=Parm.seed, **Parm.dataset.dict)
    print("Start training ...")
    recorder = rd.Recorder_nones([dataset.cv, Parm.epochs])
    for cv in range(dataset.cv):
        print(f"Cross Validation {cv}: {datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
        train_loader, test_loader = dp.process.dataloader(dataset[cv], batch_size=Parm.batch_size, **Parm.dataset.dict)
        for epoch in range(Parm.epochs):
            print(f"Epoch {epoch}: {datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
            # Training
            record = rd.Record(index=epoch)
            for batch_idx, data in enumerate(train_loader):
                data = [data.to(Parm.device) for data in data]
                data = dict(zip(Parm.dataset.keylist, data))
                batchrecord = rd.BatchRecord(size=data['x'].shape[0], index=batch_idx)  
                batchrecord['new'] = [1, 2, 3]
                batchrecord['newnew'] = [1, 2, 3]
                record.add_batch(batchrecord)
            record.aggregate({'new': 'sum', 'newnew': 'mean'})
            record['time'] = time.time()
            str = record.print_all_str()

            # # Testing
            # for batch_idx, data in enumerate(test_loader):
            #     data = [data.to(Parm.device) for data in data]
            #     data = dict(zip(Parm.dataset.keylist, data))

            recorder[cv, epoch] = record
            
# %% 1
recorder.save("save.json")
del recorder

recorder1 = rd.read_json("save.json")

