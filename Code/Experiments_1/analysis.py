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
import json
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

# %% Get Parameters
path = "../../Results/Experiments_1/2022-08-09_01-16-35/final.json"
Parm = utils.parameter.read_json(path)

# %% Main Function
    
            
