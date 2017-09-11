#!/usr/bin/env python
#encoding=utf-8

from torch.nn import *
from torch.nn.functional import *
import torch
from torch.autograd import Variable

class networkG(Module):
    def __init__(self,FLAGS):
        super(networkG,self).__init__()
        self.convlayer = Sequential(
                        Conv2d(3,16,3,stride = 2),
                        ReLU(),
                        MaxPool2d(3)
                        )
        self.lstm = LSTM(400,FLAGS.vocab_size,1)
        self.linear = Linear(FLAGS.vocab_size,1)

    def forward(self,input,step):
        convout = self.convlayer(input)
        convout = convout.view(1,-1)
        lstm_input = [ convout for i in range(step)]
        stacklstm_input = torch.stack(lstm_input,0)
        lstmoutput,_ = self.lstm(stacklstm_input)
        lstmoutput = lstmoutput.permute(1,0,2)
        return lstmoutput


class networkD(Module):
    def __init__(self,FLAGS):
        super(networkD,self).__init__()
        self.embedding = Linear(FLAGS.vocab_size,400)
        self.lstm = LSTM(400,32,3)

    def forward(self,input):
        net = self.embedding(input)
        net = net.permute(1,0,2)
        lstmoutput,_ = self.lstm(net)
        return lstmoutput
