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

# %% S_Learner (Classes)
class S_Learner(nn.Module):
    """
    S_Learner.
    """
    def __init__(self, input_size, output_size=1, hidden_size=10, layer_number=3, **kwargs):
        """
        Initialize S_Learner model.
        """
        super(S_Learner, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer_number = layer_number

        self.net = nn.Sequential()
        self.net.add_module('fc0', nn.Linear(self.input_size+1, self.hidden_size))
        for i in range(self.layer_number-1):
            self.net.add_module(f'fc{i+1}', layers.FullyConnected(hidden_size, hidden_size, 0.5, [nn.ReLU()]))
        self.net.add_module(f'fc_{layer_number}', layers.FullyConnected(hidden_size, output_size))

    def forward(self, x):
        """
        Forward propagation.
        """
        sample_number = x.shape[0]
        pred_y0 = self.net(torch.cat([x, torch.zeros([sample_number, 1]).to(x.device)], dim=1))
        pred_y1 = self.net(torch.cat([x, torch.ones([sample_number, 1]).to(x.device)], dim=1))
        return pred_y0, pred_y1
    
    def predict(self, data):
        """
        Predict.
        """
        if isinstance(data, dict):
            x = data["x"]
        else:
            x = data
        pred = self.forward(x)
        pred = [i.unsqueeze(1) for i in pred]
        pred = torch.cat(pred, dim=1)
        return {"y_pred": pred}

    # def fit(self, data, target):
    #     """
    #     Fit.
    #     """
    #     pred_y0, pred_y1 = self.forward(data)
    #     loss = torch.mean(torch.abs(pred_y1-pred_y0-target))
    #     return loss

    

# %% Main Function
if __name__ == '__main__':
    x = torch.ones([10, 5])
    t = torch.ones([10, 1])

    model = S_Learner(5, 1, 10, 3)
    pred_y0, pred_y1 = model(x)
    print(f"x: {x.shape}")
    print(f"t: {t.shape}")
    print(f"pred_y0: {pred_y0.shape}")
    print(f"pred_y1: {pred_y1.shape}")