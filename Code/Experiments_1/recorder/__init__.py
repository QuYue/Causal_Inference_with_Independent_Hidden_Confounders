#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   __init__.py
@Time    :   2022/07/15 05:07:34
@Author  :   QuYue 
@Email   :   quyue1541@gmail.com
@Desc    :   __init__ of recorder
'''

# %% Import Packages
# Basic
import os
import sys
import datetime

# Add path
if __package__ is None:
    os.chdir(os.path.dirname(__file__))
    sys.path.append("..")

# Self-defined
import utils

# %% Classes
class Recorder():
    def __init__(self):
        pass

class BatchRecord():
    """
    A Record for one batch.
    
    example:
    --------
    a = BatchRecord(2)
    a['key1'] = 1
    a['key2'] = [0,1,2,3]

    a['key2'].append(3)
    a['key2'].extend([4,5,6])
    """
    def __init__(self, size=0, index=0):
        self.info = dict()
        self.size = size
        self.index = index
    
    def __getitem__(self, key):
        return self.info[key]

    def __setitem__(self, key, value):
        self.info[key] = value
    
    def __len__(self):
        return (len(self.info), self.size)
    
    def keys(self):
        return list(self.info.keys())

    @property
    def shape(self):
        # (number of keys, batch size)
        return (len(self.info), self.size)

    def list_keys(self):
        list_keys = []
        keys = self.keys()
        for k in keys:
            if isinstance(self.info[k], list):
                list_keys.append(k)
        return list_keys

    def num_keys(self):
        num_keys = []
        keys = self.keys()
        for k in keys:
            if not isinstance(self.info[k], list):
                num_keys.append(k)
        return num_keys
    
    def __repr__(self) -> str:
        s = self.shape
        return f"BatchRecord()"


class Record(BatchRecord):
    def __init__(self, size=0, index=0):
        super().__init__(size, index)
        self.time = datetime.datetime.now()

    def add_batch(self, batch_record: BatchRecord):
        self.size += batch_record.size
        for k in batch_record.list_keys():
            if k in self.info:
                self.info[k].extend(batch_record[k])
            else:
                self.info[k] = batch_record[k]
        for k in batch_record.num_keys():
            if k in self.info:
                self.info[k].append(batch_record[k])
            else:
                self.info[k] = [batch_record[k]]
    
    def aggregate(self, key_dict):
        if not isinstance(key_dict, dict):
            raise ValueError("The type of argument 'key_dict' should be dict.")
        for k, v in key_dict.items():
            if k in self.info:
                if isinstance(v, str):
                    if v == "mean":
                        self.info[k] = self.info[k] / len(self.info[k])
                    elif v == "mean_size":
                        self.info[k] /= self.size
                    elif v == "sum":
                        self.info[k] = sum(self.info[k])
                    else:
                        raise ValueError("The value of dict should be 'mean', 'mean_size', 'sum'.")
                if hasattr(v, "__call__"):
                    self.info[k] = v(self.info[k])
            else:
                raise ValueError(f"The key '{k}' is not in this Record.")
                
        


    
# %% Functions
def main():
    pass

# %% Main Function
if __name__ == '__main__':
    main()