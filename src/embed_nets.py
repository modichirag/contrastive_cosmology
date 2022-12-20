import numpy as np
import torch
from torch import nn
import sys, os

act_dict = {'relu': nn.ReLU, 
        'identity': nn.Identity, 
        'none' : nn.Identity,
        'sigmoid' : nn.Sigmoid
        }

#####
class Arch():
    
    def __init__(self, input_size, output_size, nlayers=2, activation='relu', out_activation='identity', nhidden=[32]):

        self.input_size = input_size
        self.output_size = output_size
        self.nlayers = nlayers  # nlayers = inlayer + hidden + outlayer
        self.activation = activation
        self.out_activation = out_activation
        self.nhidden = nhidden
        self._build()

    def _build(self):
 
        try:
            iterator = iter(self.nhidden)
        except TypeError:
            self.nhidden = [self.nhidden for _ in range(self.nlayers-1)]            
        print(self.nhidden)
        assert len(self.nhidden) == self.nlayers-1

    def parse_act(self, act):
        return act_dcit[act]


#####
class Simple_MLP(nn.Module):

    def __init__(self, arch):
        super(Simple_MLP, self).__init__()
        self.arch = arch
        self._build()

    def _build(self):
        
        arch = self.arch
        act = act_dict[arch.activation]
        
        self.f0 = nn.Linear(arch.input_size, arch.nhidden[0])
        self.a0 = act(inplace=False)
        self.layers = [self.f0, self.a0]

        for i in range(1, arch.nlayers-1):
            print(i)
            attr = "f%d"%i
            value = nn.Linear(arch.nhidden[i-1], arch.nhidden[i])
            setattr(self,attr,value)
            self.layers.append(value)
            attr = "a%d"%i
            value = act(inplace=False)
            setattr(self,attr,value)
            self.layers.append(value)

        out_act = act_dict[arch.out_activation]
        attr = "f%d"%(i+1)
        value = nn.Linear(arch.nhidden[-1], arch.output_size)
        setattr(self,attr,value)
        self.layers.append(value)
        attr = "a%d"%(i+1)
        value = out_act(inplace=False)
        setattr(self,attr,value)
        self.layers.append(value)

    def forward(self, x):

        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x



