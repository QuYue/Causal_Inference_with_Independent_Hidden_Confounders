#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   s_learner.py
@Time    :   2022/07/05 07:35:54
@Author  :   QuYue 
@Email   :   quyue1541@gmail.com
@Desc    :   s_learner
'''

# %% Import Packages
# Basic
import torch
import torch.nn as nn

# Modules
if __package__ is None:
    import layers
else:
    from . import layers

# %% T_Learner (Classes)
class T_Learner(nn.Module):
    """
    T_Learner.
    """
    def __init__(self, input_size, output_size=1, hidden_size=10, layer_number=3):
        """
        Initialize S_Learner model.
        """
        super(T_Learner, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer_number = layer_number

        self.net0 = nn.Sequential()
        self.net1 = nn.Sequential()

        self.net0.add_module('fc0', nn.Linear(self.input_size, self.hidden_size))
        self.net1.add_module('fc0', nn.Linear(self.input_size, self.hidden_size))
        for i in range(self.layer_number-1):
            self.net0.add_module(f'fc{i+1}', layers.FullyConnected(hidden_size, hidden_size, 0.5, [nn.ReLU()]))
            self.net1.add_module(f'fc{i+1}', layers.FullyConnected(hidden_size, hidden_size, 0.5, [nn.ReLU()]))
        self.net0.add_module(f'fc_{layer_number}', layers.FullyConnected(hidden_size, output_size))
        self.net1.add_module(f'fc_{layer_number}', layers.FullyConnected(hidden_size, output_size))

    def forward(self, x):
        """
        Forward propagation.
        """
        pred_y0 = self.net0(x)
        pred_y1 = self.net1(x)
        return pred_y0, pred_y1


# %% Main Function
if __name__ == '__main__':
    x = torch.ones([10, 5])
    t = torch.ones([10, 1])

    model = T_Learner(5, 1, 10, 3)
    pred_y0, pred_y1 = model(x)
    print(f"x: {x.shape}")
    print(f"t: {t.shape}")
    print(f"pred_y0: {pred_y0.shape}")
    print(f"pred_y1: {pred_y1.shape}")