from sklearn.utils import shuffle
from lego_loder import MyDataset
from torch.utils.data import DataLoader
import torch

def train():
    train_data=MyDataset(root_dir='./lego',half_res=True,is_train=True)
    train_loder=DataLoader(train_data,batch_size=8,shuffle=True,num_workers=4)
    optimizer=torch.optim.Adam([model_coarse.parameters(),model_fine.parammeters()],lr=args.lr)
    for i in range(args.epoch):
        for i,(img,K,tfs) in enumerate(train_loder):
            optimizer.zero_grad()
            rays_o,rays_dir=raysGet(K,tfs)
            coarse_sample=randomraysSample(rays_o,rays_dir,args.cpts_num,args.near,args.far)
            coarse_sigma,coarse_RGB=model_coarse(coarse_sample)
            coarse_render=colRender(coarse_sample,coarse_sigma,coarse_RGB)
            fine_sample=invSample(coarse_sample,coarse_sigma)
            fine_sigma,fine_RGB=model_fine(fine_sample)
            #TODO:concat concat fine_render and coarse_render
            fine_render=cloRender(fine_sample,fine_sigma,fine_RGB)
            loss=img2mse(fine_render,img)+img2mse(coarse_render,img)
            loss.backward()
            optimizer.step()
            new_lr=args.lr*(0.1**(i/args.epoch))
            for param in optimizer.param_groups:
                param['lr']=new_lr



