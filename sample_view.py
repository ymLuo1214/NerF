from Nerf import raysGet
from cmath import pi
import torch
import math
import matplotlib.pyplot as plt
import numpy as np


def raysGet(H, W, focal, c2w):
    x, y = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    x, y = x.t(), y.t()
    car_dir = torch.stack(((x-0.5*W)/focal, -(y-0.5*H) /
                          focal, -torch.ones_like(x)), dim=-1)
    rays_dir = torch.matmul(c2w[:3, :3], car_dir.unsqueeze(dim=-1)).squeeze(-1)

    rays_o = c2w[:3, -1].expand(rays_dir.size())
    return rays_o, rays_dir


def randomraysSample(rays_o, rays_dir, pts_num, d_near,d_far):
    H, W, _ = rays_o.size()
    t_bound = torch.linspace(d_near, d_far, pts_num)
    t_val = torch.rand(H, W, pts_num)
    t = t_bound+t_val
    sample = rays_o.unsqueeze(-2)+t.unsqueeze(-1)*rays_dir.unsqueeze(-2)
    return sample


def view(sample, rays_o, rays_dir):
    scale = torch.linspace(0, 3.5, 2)
    rays = rays_o.unsqueeze(-2)+scale.unsqueeze(-1)*rays_dir.unsqueeze(-2)
    rays = rays.numpy()
    origin = rays_o.numpy()
    points = sample.numpy()

    B, N, pts, _ = points.shape
    fig = plt.figure('graph')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('3D viewer')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    print(B,N)
    for i in range(0,B,80):
        for j in range(0,N,80):
            ax.plot(rays[i, j, :, 0], rays[i, j, :, 1],
                    rays[i, j, :, 2], linewidth=1)
            ax.scatter(points[i, j, :, 0], points[i, j, :, 1], points[i, j, :, 2],s=2)
    ax.scatter(origin[0, 0, 0], origin[0, 0, 1], origin[0, 0, 2])
    plt.show()


if __name__ == '__main__':
    camera_config = torch.tensor([
        [-0.9999021887779236, 0.004192245192825794, -0.013345719315111637, -0.05379832163453102],
        [-0.013988681137561798,-0.2996590733528137,0.95394366979599,3.845470428466797],
        [-4.656612873077393e-10,0.9540371894836426, 0.29968830943107605,1.2080823183059692],
        [0.0,0.0,0.0,1.0]
    ])
    H,W=800,800
    camera_angle=torch.tensor([0.6911112070083618])
    focal=W/(2*torch.tan(camera_angle))
    rays_o,rays_dir=raysGet(800,800,focal,camera_config)
    sample = randomraysSample(rays_o, rays_dir, 10, 2, 6)
    view(sample, rays_o, rays_dir)
