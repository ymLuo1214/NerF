from re import X
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from cmath import pi


def posEmbed(x:torch.tensor, L:int)->torch.tensor:  # x:N*L*3
    '''
    func = [torch.sin, torch.cos]
    freq = 2**torch.linspace(0, L-1, L)*pi
    B, N, S = x.size()
    freq_pos = torch.zeros(B, N, S*2*L)
    for i in range(B):
        for j in range(N):
            for k in range(S):
                freq_pos[i, j, k*2*L:(k+1)*2*L] = torch.tensor([f(p)
                                                                for p in x[i, j, k]*freq for f in func])
    return freq_pos
    '''
    result=[]
    B,N=x.shape[0],x.shape[1]
    freq=2.**torch.linspace(0,L-1,L)*pi
    func=[torch.cos,torch.sin]
    for fre in freq:
        for f in func:
            result.append(f(fre*x.unsqueeze(-1)))
    return torch.cat(result,dim=-1).view(B,N,-1)


def mlpMaker(in_cha, out_cha, act_fun=nn.ReLU(), norm=None):
    modules = [nn.Linear(in_cha, out_cha)]
    if norm is not None:
        modules.append(norm(out_cha))
    if act_fun is not None:
        modules.append(act_fun)
    return modules


class Nerf(nn.Module):
    def __init__(self, xins=60, dins=24, W=256, mlp_num=8):
        super().__init__()
        self.xins = xins  # dimensions of gama(x)
        self.dins = dins  # dimensions of gama(d)
        self.W = W  # dimessions of MLP
        self.mlp_num = mlp_num

        assert self.mlp_num % 2 == 0
        self.linear1 = nn.ModuleList([nn.Linear(
            self.xins, self.W)]+[nn.Linear(self.W, self.W) for i in range(self.mlp_num % 2-1)])
        self.linear2 = nn.Linear(self.W, self.W)
        self.linear3 =nn.ModuleList([nn.Linear(
            self.W+self.xins, self.W)]+[nn.Linear(self.W, self.W) for i in range(self.mlp_num % 2-1)])
        self.linear_sigma=nn.Linear(self.W,1)
        self.linear_rgb=nn.ModuleList([nn.Linear(self.W+self.dins,self.W//2),nn.Linear(self.W//2,3)])

    def forward(self, x, d):
        x = posEmbed(x, self.xins//6)
        d = posEmbed(d, self.dins//4)
        gamax=x
        for l in self.linear1:
            x=l(x)
            x=nn.ReLU(x)
        x=self.linear2(x)
        x=torch.cat((gamax,x),dim=-1)
        for l in self.linear3:
            x=l(x)
            x=nn.ReLU(x)
        sigma=self.linear_sigma(x)
        x=nn.ReLU(torch.cat([d,self.linear_rgb[0]],dim=-1))
        rgb=self.linear_rgb(x)
        output=[sigma,rgb]
        return output



model = Nerf()
x = torch.arange(10*3).view(10, 3)
d = torch.arange(10*3).view(10, 3)
model(x, d)
