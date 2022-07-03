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
import torch
import torch.utils.data as Data
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
    # If dataset_name is string
    if not isinstance(dataset_name, str):
        raise ValueError("The type of dataset_name must be a string.")

    # Find dataset
    dataset_name_lower = dataset_name.lower().strip()
    if dataset_name_lower == "synthetic":
        from .dataset_Synthetic import load as load_synthetic
        data_orginal = load_synthetic(random_seed=seed, **kwargs)
        dataset = data_orginal
    elif dataset_name_lower == "":
        pass
    else:
        raise ValueError(f"There is no dataset called '{dataset_name}'.")

    return dataset

# %% Main Function
if __name__ == "__main__":
    print('datasets')
