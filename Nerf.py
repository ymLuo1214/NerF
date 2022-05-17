from re import X
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from cmath import pi
import matplotlib.pyplot as plt
from matplotlib import projections
import numpy as np

def posEmbed(x: torch.tensor, L: int) -> torch.tensor: 
    result = []
    B, N = x.shape[0], x.shape[1]
    freq = 2.**torch.linspace(0, L-1, L)
    func = [torch.cos, torch.sin]
    for fre in freq:
        for f in func:
            result.append(f(fre*x))
    return torch.cat(result, dim=-1)

class Nerf(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.xins = args.xins  
        self.dins = args.dins 
        self.W = args.W 
        self.mlps = args.mlps

        assert self.mlps % 2 == 0
        self.linear1 = nn.ModuleList([nn.Linear(
            self.xins, self.W)]+[nn.Linear(self.W, self.W) for i in range(self.mlps // 2)])
        self.linear2 = nn.ModuleList([nn.Linear(
            self.W+self.xins, self.W)]+[nn.Linear(self.W, self.W) for i in range(self.mlps //2)])
        self.linear_sigma = nn.Linear(self.W, 1)
        self.linear_rgb = nn.ModuleList(
            [nn.Linear(self.W+self.dins, self.W//2), nn.Linear(self.W//2, 3)])

    def forward(self, x, d): #x:B*H*W*pts*3
        x = posEmbed(x, self.xins//6)   
        d = posEmbed(d, self.dins//6)
        gamax = x
        for l in self.linear1:
            x = l(x)
            x = F.relu(x, inplace=True)
        x = torch.cat((gamax, x), dim=-1)
        for l in self.linear2:
            x = l(x)
            x = F.relu(x, inplace=True)
        sigma = self.linear_sigma(x)
        sigma=F.relu(sigma)
        x = torch.cat([d, x], dim=-1)
        x = self.linear_rgb[0](x)
        x = F.relu(x, inplace=True)
        rgb = self.linear_rgb[1](x)
        rgb=F.relu(rgb)
        output = [sigma, rgb]
        return output


def raysGet(K,c2w):
    H=int(K[0][2])
    W=int(K[1][2])
    focal=float(K[0][0])
    x, y = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    x, y = x.t(), y.t()
    car_dir = torch.stack(((x-0.5*W)/focal, -(y-0.5*H) /focal, -torch.ones_like(x)), dim=-1)
    rays_dir = torch.matmul(c2w[:,None,None,:3, :3], car_dir.unsqueeze(dim=-1)).squeeze(-1)
    rays_o = c2w[:,None,None,:3, -1].expand(rays_dir.size())
    return rays_o, rays_dir


def randomraysSample(rays_o, rays_dir, pts_num, d_near, d_far):
    B,H, W, _ = rays_o.size()
    t_bound = torch.linspace(d_near, d_far, pts_num)
    t_val = torch.rand(B,H, W, pts_num)
    t = t_bound+t_val
    sample = rays_o[:,:,:,None,:]+t.unsqueeze(-1)*rays_dir.unsqueeze(-2)
    return sample

def view(sample, rays_o, rays_dir):
    scale = torch.linspace(0, 3.5, 2)
    rays = rays_o[:,:,:,None,:]+scale.unsqueeze(-1)*rays_dir.unsqueeze(-2)
    rays = rays.numpy()
    origin = rays_o.numpy()
    points = sample.numpy()

    B, H,W, pts, _ = points.shape
    print(B)
    assert (B%2==0 or B==1)
    fig = plt.figure('graph')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('3D viewer')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    for i in range(0,B,2):
        for j in range(0,H,80):
            for k in range(0,W,80):
                ax.plot(rays[i, j, k,:, 0], rays[i, j,k, :, 1], rays[i, j, k,:, 2], linewidth=1)
                ax.scatter(points[i, j,k, :, 0], points[i, j,k, :, 1], points[i, j, k,:, 2],s=1)
        
        ax.scatter(origin[i,0,0, 0], origin[i,0,0, 1], origin[i,0,0,2],s=2)
       
    plt.show()
