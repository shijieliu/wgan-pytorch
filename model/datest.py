#!/usr/bin/env python
#encoding=utf-8

import torch
from torch.utils.data import Dataset
from torch import LongTensor
from vocab import vocab
import json
import sys
import numpy as np

class datest(Dataset):
    def __init__(self,FLAGS):
        super(datest,self).__init__()
        self.load(FLAGS)


    def __getitem__(self,index):
        return self._poem_dict[index]

    def __len__(self):
        return len(self._poem_dict)

    def load(self,FLAGS):
        self._poem_dict = []
        print FLAGS.input
        with open(FLAGS.input,'r') as f:
            for i,lines in enumerate(f):
                try:
                    lines = lines.strip().decode('utf-8')
                    jsonobj = json.loads(lines)
                except:
                    print >> sys.stderr,lines,i
                    continue
                poem = jsonobj["poem"]
                if len(poem) > 50:
                    continue
                if not poem:
                    continue
                tensor = torch.Tensor(poem)
                self._poem_dict.append(tensor)

    def changeonehot(self,poem,vocab_size):
        list_ = []
        for i in poem:
            new_tensor = np.zeros(vocab_size)
            new_tensor[i] = 1
            list_.append(torch.from_numpy(new_tensor))
        return torch.stack(list_)
