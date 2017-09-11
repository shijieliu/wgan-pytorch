#!/usr/bin/env python
#encoding=utf-8

import json
import os

class vocab(object):
    def __init__(self,filename):
        self.num_to_token = {}
        self.token_to_num = {}
        self.filename = filename
        if not os.path.isfile(filename):
            raise
        self.__load()

    def __load(self):
        with open(self.filename,'r') as f:
            for i,lines in enumerate(f):
                try:
                    lines = lines.strip().decode('utf-8')
                except:
                    print lines
                    continue
                token,idx = lines.split('\t')
                idx = int(idx)
                self.num_to_token[idx] = token
                if token in self.token_to_num:
                    print >> sys.stderr,token.encode('utf-8'),i
                self.token_to_num[token] = idx
        
    def __getitem__(self,index):
        #print index.encode('utf-8'),type(index)
        if isinstance(index,int):
            return self.num_to_token[index]
        if isinstance(index,unicode):
            return self.token_to_num[index]

    def __len__(self):
        assert len(self.num_to_token) == len(self.token_to_num)
        return  len(self.num_to_token)
