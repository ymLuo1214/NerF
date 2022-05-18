import torch

a=torch.arange(6).view([2,3])
b=torch.arange(2).view([2])
print(a/b[...,None])
