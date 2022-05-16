from sklearn.utils import shuffle
from lego_loder import MyDataset
from torch.utils.data import DataLoader
import torch
from Nerf import *
import configargparse

parser = configargparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=150*1e3, help='total epochs')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='initial learning rate')
parser.add_argument('--cpts_num', type=int, default=64,
                    help='numbers of coarse sample points')
parser.add_argument('--fpts_num', type=int, default=128,
                    help='addtional numbers of fine sample points')
parser.add_argument('--near', type=float, default=2., help='z of near plane')
parser.add_argument('--far', type=float, default=6., help='z of far plane')
args = parser.parse_args()


def train():
    train_data = MyDataset(root_dir='./lego', half_res=True, is_train=True)
    train_loder = DataLoader(train_data, batch_size=8,
                             shuffle=True, num_workers=4)
    optimizer = torch.optim.Adam(
        [model_coarse.parameters(), model_fine.parammeters()], lr=args.lr)
    for i in range(args.epoch):
        for i, (img, K, tfs) in enumerate(train_loder):
            optimizer.zero_grad()
            rays_o, rays_dir = raysGet(K, tfs)
            coarse_sample = randomraysSample(
                rays_o, rays_dir, args.cpts_num, args.near, args.far)
            coarse_sigma, coarse_RGB = model_coarse(coarse_sample)
            coarse_render = colRender(coarse_sample, coarse_sigma, coarse_RGB)
            fine_sample = invSample(coarse_sample, coarse_sigma, args.fpts_num)
            fine_sigma, fine_RGB = model_fine(fine_sample)
            # TODO:concat concat fine_render and coarse_render
            fine_render = cloRender(fine_sample, fine_sigma, fine_RGB)
            loss = img2mse(fine_render, img)+img2mse(coarse_render, img)
            loss.backward()
            optimizer.step()
            new_lr = args.lr*(0.1**(i/args.epoch))
            for param in optimizer.param_groups:
                param['lr'] = new_lr
