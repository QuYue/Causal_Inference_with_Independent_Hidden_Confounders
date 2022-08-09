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

# Add path
os.chdir(os.path.dirname(__file__))

# Self-defined
import utils

# %% Get Parameters
path = "../../Results/Experiments_1/2022-08-09_01-16-35/final.json"
Parm = utils.parameter.read_json(path)
recorder = Parm.recorder

# %% Main Function
def main():
    pass

# %%
if __name__ == "__main__":
    main()
            
