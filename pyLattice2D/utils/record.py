from collections import defaultdict
from torch import Tensor
from numpy import ndarray
import pickle
import os

class Record:
    def __init__(self):
        self.data = defaultdict(list)

    def __getitem__(self, item):
        return self.data[item]
    
    @property
    def keys(self):
        return self.data.keys()
    
    def add(self, variables):
        for key in variables:
            if isinstance(variables[key], Tensor):
                value = variables[key].detach().numpy().tolist()
            elif isinstance(variables[key], ndarray):
                value = variables[key].tolist()
            else:
                value = variables[key]
            self.data[key].append(value)
            
class RecordAndSave(Record):
    def __init__(self, path, filename):
        super().__init__()
        self.path = path
        self.filename = filename
        self.load()
    
    def save(self):
        with open('{}/{}.pickle'.format(self.path, self.filename), 'wb') as handle:
            pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load(self):
        if os.path.exists('{}/{}.pickle'.format(self.path, self.filename)):
            with open('{}/{}.pickle'.format(self.path, self.filename), 'rb') as handle:
                self.data = pickle.load(handle)
