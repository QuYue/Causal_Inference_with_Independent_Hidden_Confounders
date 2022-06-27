#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   datasets.py
@Time    :   2022/06/21 22:28:40
@Author  :   QuYue 
@Email   :   quyue1541@gmail.com
@Desc    :   datasets
'''

# %% Import Packages
# Basic

# %% Classes


# %% Functions
def load(dataset_name):
    """
    The function of loading datasets which can: 
        - Load the dataset.
        - Preprocess the data.
        - Return train/test split.

    Args:
        dataset_name (_type_): _description_
    """
    print(dataset_name)

    # if dataset_name is string
    if not isinstance(dataset_name, str):
        raise ValueError("dataset_name must be a string.")

    if dataset_name.lower() == "":
        pass
    elif dataset_name == "":
        pass
    else:
        raise ValueError(f"There is no dataset called '{dataset_name}'.")

# %% Main Function
if __name__ == "__main__":
    print('datasets')
