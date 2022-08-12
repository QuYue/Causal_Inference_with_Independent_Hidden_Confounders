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

# %% Functions
def get_path(file_name_list):
    path_list = []
    for file_name in file_name_list:
        path_list.append([file_name, f"../../Results/Experiments_1/{file_name}/final.json"])
    return path_list


# %% Main Function
if __name__ == "__main__":
    file_name = ["2022-08-12_13-46-58"]
    path_list = get_path(file_name)

    for file_name, path in path_list:
        print(file_name)
        Parm = utils.parameter.read_json(path)
        new = Parm.recorder['test'].query("s_learner_test_loss")
        break
            
# %%
