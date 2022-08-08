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
class MyParam(utils.parameter.PARAM):
    def __init__(self) -> None:
        # Random seed
        self.seed = 1       # Random seed
        # Device
        self.gpu = 0        # Used GPU, when bool (if use GPU), when int (the ID of GPU) 
        # Dataset
        self.dataset_name = "Synthetic"     # Dataset name
        self.train_valid_test = [4, 1, 1]   # The ratio of training, validation and test data
        self.cv = 5                         # Fold number for cross-validation
        # Dataset Parameters
        self.dataset_set = {"synthetic":
                                {"name": "Synthetic",
                                 "data_number": 10000,
                                 "data_dimensions": 10,
                                 "ifprint": False,
                                 "stratify": 't',
                                 "keylist": ['x', 't', 'y', 'potential_y'],
                                 "type_list": ['float', 'long', 'float', 'float']}}
        # Model
        self.model_name_list = ["s_learner", "t_learner"]   # Model name list
        # Model Parameters
        self.model_param_set = {"s_learner":
                                    {"name": "S_Learner", 
                                     "input_size": self.dataset_set[self.dataset_name.lower().strip()]["data_dimensions"],
                                     "output_size": 1,
                                     "hidden_size": 15,
                                     "layer_number": 3},
                                "t_learner":
                                    {"name": "T_Learner",
                                     "input_size":self.dataset_set[self.dataset_name.lower().strip()]["data_dimensions"],
                                     "output_size":1,
                                     "hidden_size":15,
                                     "layer_number":3}}
        # Training
        self.epochs = 10            # Epochs
        self.batch_size = 1000      # Batch size
        self.learn_rate = 0.01      # Learning rate
        self.test_epoch = 1         # Test once every few epochs
        # Records
        self.ifrecord = True                # If record
        self.now = datetime.datetime.now()  # Current time
        self.recorder = None
        self.save_path = f"../../Results/Experiments_1/{self.now.strftime('%Y-%m-%d_%H-%M-%S')}"

        # Setting
        self.setting()

Parm = MyParam()

# %% Main Function
if __name__ == "__main__":
    print("Loading dataset ...")
    dataset = dp.datasets.load_dataset(Parm.dataset_name, seed=Parm.seed, **Parm.dataset.dict)
    print("Start training ...")
    Parm.recorder = rd.Recorder_nones([dataset.cv, Parm.epochs])
    for cv in range(dataset.cv):
        print(f"Cross Validation {cv}: {datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
        train_loader, test_loader = dp.process.dataloader(dataset[cv], batch_size=Parm.batch_size, **Parm.dataset.dict)
        for epoch in range(Parm.epochs):
            print(f"Epoch {epoch}: {datetime.datetime .now().strftime('%Y-%m-%d_%H:%M:%S')}")
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

            # Testing
            for batch_idx, data in enumerate(test_loader):
                data = [data.to(Parm.device) for data in data]
                data = dict(zip(Parm.dataset.keylist, data))
            Parm.recorder[cv, epoch] = record
        Parm.save(f"cv{cv}.json")
    Parm.save("final.json")
# %% 1
# recorder.save("save.json")
# del recorder

# recorder1 = rd.read_json("save.json")

# quick sort
