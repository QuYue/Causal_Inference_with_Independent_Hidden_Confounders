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

# Add path
os.chdir(os.path.dirname(__file__))

# Self-defined
import dataprocess as dp
import modelbase as mb

# %% Set Super-parameters
class PARAM():
    def __init__(self) -> None:
        # Dataset
        self.dataset_name = "Synthetic"     # Dataset name

        # Random seed
        self.seed = 1           # Random seed
        self.random_setting()   # Random seed setting

    def random_setting(self):
        # Setting of random seed
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)   # If you are using multi-GPU.
        
Parm = PARAM()

# %% Main Function
if __name__ == "__main__":
    dp.datasets.load(Parm.dataset_name)



