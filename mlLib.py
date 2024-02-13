import torch
#import numpy
#import random
import math

def libTranspose(x):
    return torch.transpose(x, 0, -1)

def libActRelu(x):
    return torch.nn.functional.relu(x)

def libActReluLeaky(x):
    return torch.nn.functional.leaky_relu(x, negative_slope=0.01) 

def libActElephant(x, d=2.5):
    # from paper "Elephant networks"
    # looks to be great for lifelong learning!

    z0 = torch.abs(x)
    z1 = torch.pow(z0, d)
    return 1.0 / (1.0 + z1)



# weight initialization
def xavierInitializer(in_features, out_features):
    std = math.sqrt(2.0 / (in_features + out_features))
    weight = torch.randn(in_features, out_features) * std
    return weight
