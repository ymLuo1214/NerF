from lego_loder import MyDataset
from torch.utils.data import DataLoader
import torch
from Nerf import *
import configargparse
import time
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import torch.nn as nn

parser = configargparse.ArgumentParser()
parser.add_argument('--half_res', type=bool, default=True,
                    help='resolution of 400')
parser.add_argument('--epoch', type=int, default=150000, help='total epochs')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='initial learning rate')
parser.add_argument('--cpts_num', type=int, default=64,
                    help='numbers of coarse sample points')
parser.add_argument('--fpts_num', type=int, default=128,
                    help='addtional numbers of fine sample points')
parser.add_argument('--rays_batch', type=int, default=2048,
                    help='batch_size of rays')
parser.add_argument('--near', type=float, default=2., help='z of near plane')
parser.add_argument('--far', type=float, default=6., help='z of far plane')
parser.add_argument('--xins', type=int, default=60,
                    help='dimensions of gama(x)')
parser.add_argument('--dins', type=int, default=36,
                    help='dimensions of gama(d)')
parser.add_argument('--W', type=int, default=256,
                    help='dimensions of each mlp')
parser.add_argument('--mlps', type=int, default=8, help='layers of mlp')
args = parser.parse_args()

if not torch.cuda.is_available():
    print("CUDA not available.")
    exit(-1)

def getSummaryWriter(epochs:int):
    logdir = './logs/'
    time_stamp = "{0:%Y-%m-%d/%H-%M-%S}-epoch{1}/".format(datetime.now(), epochs)
    return SummaryWriter(log_dir = logdir + time_stamp)

MSE=nn.MSELoss()
writer=getSummaryWriter(args.epoch)
torch.cuda.manual_seed(1)
def main():
    train_data = MyDataset(root_dir='./lego/', half_res=args.half_res, is_train=True)
    train_loder = DataLoader(train_data, batch_size=1,shuffle=True, num_workers=4)
    test_data=MyDataset(root_dir='./lego/', half_res=args.half_res, is_train=False)
    test_loader=DataLoader(test_data, batch_size=1,shuffle=False, num_workers=4)
    if args.half_res:
        H, W = 400, 400
    else:
        H,W = 800,800   
    focal = W/(2*torch.tan(0.5*train_data.cam_fov))
    K = torch.tensor([[focal, 0, W], [0, focal, H], [0, 0, 1]])
    model_coarse = Nerf(args)
    model_coarse=model_coarse.cuda()
    model_fine = Nerf(args)
    model_fine=model_fine.cuda()
    grad_vars = list(model_coarse.parameters())
    grad_vars += list(model_fine.parameters())
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lr)
    num_train=0
    num_test=0
    for i in range(args.epoch):
        for j, (img, tfs) in enumerate(train_loder):
            start_time=time.time()
            """
            img:B*3*H*W
            tfs:B*4*4
            """
            train_loss=0.
            model_coarse.train()
            model_fine.train()
            img=img.cuda()
            tfs=tfs.cuda()
            rays_ori, rays_dirs, rays_dists = raysGet(K, tfs)
            coarse_sample, coarse_sample_dist = randomraysSample(rays_ori, rays_dirs, rays_dists, args.cpts_num, args.near, args.far)
            # view(coarse_sample,rays_ori,rays_dirs,pt_fine=False)
            rays_dirs = rays_dirs[:, :, :, None,:].expand(coarse_sample.size())
            coarse_sample_loader, rays_ori_loader, rays_dir_loader, rays_dist_loader, coarse_d_loader,pixel_loader = raysBatchify(coarse_sample, rays_ori, rays_dirs, rays_dists, coarse_sample_dist, img,args.rays_batch)
            I=0
            for coarse_s, rays_o, rays_dir, rays_dist, coarse_dist ,pixel in zip(coarse_sample_loader, rays_ori_loader, rays_dir_loader, rays_dist_loader, coarse_d_loader,pixel_loader):
                I+=1
                num_train+=1
                optimizer.zero_grad()
                coarse_sigma, coarse_RGB = model_coarse(coarse_s, rays_dir)
                coarse_cr, weight = colRender(coarse_dist, rays_dist, coarse_sigma, coarse_RGB)
                fine_sample, fine_sample_dist, rays_dir_fine = invSample( weight, args.fpts_num, rays_o, rays_dir, rays_dist, args.near, args.far, coarse_dist)
                # view(fine_sample,rays_o,rays_dir,pt_fine=True)
                fine_sigma, fine_RGB = model_fine(fine_sample, rays_dir_fine)
                fine_cr, _ = colRender(fine_sample_dist, rays_dist, fine_sigma, fine_RGB)
                pixel=pixel.permute((0,2,1))
                loss = MSE(fine_cr, pixel)+MSE(coarse_cr,pixel)
                loss.backward()
                optimizer.step()
                loss_float=float(loss.detach().cpu().numpy())
                print("%d-th epoch,%d-th picture,%d-th rays_batch,loss%f:"%(i,j,I,loss_float))
                train_loss=train_loss+loss_float
                writer.add_scalar('Loss/train/rays',loss_float,num_train)
            end_time=time.time()
            print("%dth epoch,%d-th picture,loss:%f,time:%f"%(i,j,train_loss,end_time-start_time))
            writer.add_scalar('Loss/train/pic',train_loss,num_train)
            model_coarse_save_path='./model/'+str((i*100+j))+'model.tar'
            model_fine_save_path='./model/'+str((i*100+j))+'model.tar'
            if j%20==0:
                torch.save({'state_dict':model_coarse.state_dict(),'train_loss':train_loss},model_coarse_save_path)
                torch.save({'state_dict':model_fine.state_dict(),'train_loss':train_loss},model_fine_save_path)
                test_pic=[]
                for k,(img,tfs) in enumerate(test_loader):
                    model_coarse.eval()
                    model_fine.eval()
                    img=img.cuda()
                    tfs=tfs.cuda()
                    rays_ori, rays_dirs, rays_dists = raysGet(K, tfs)
                    coarse_sample, coarse_sample_dist = randomraysSample(rays_ori, rays_dirs, rays_dists, args.cpts_num, args.near, args.far)
                    rays_dirs = rays_dirs[:, :, :, None,:].expand(coarse_sample.size())
                    coarse_sample_loader, rays_ori_loader, rays_dir_loader, rays_dist_loader, coarse_d_loader,pixel_loader = raysBatchify(coarse_sample, rays_ori, rays_dirs, rays_dists, coarse_sample_dist, img,args.rays_batch)
                    for coarse_s, rays_o, rays_dir, rays_dist, coarse_dist ,pixel in zip(coarse_sample_loader, rays_ori_loader, rays_dir_loader, rays_dist_loader, coarse_d_loader,pixel_loader):
                        with torch.no_grad():
                            num_test+=1
                            coarse_sigma, coarse_RGB = model_coarse(coarse_s, rays_dir)
                            coarse_r, weight = colRender(coarse_dist, rays_dist, coarse_sigma, coarse_RGB)
                            fine_sample, fine_sample_dist, rays_dir_fine = invSample( weight, args.fpts_num, rays_o, rays_dir, rays_dist, args.near, args.far, coarse_dist)
                            fine_sigma, fine_RGB = model_fine(fine_sample, rays_dir_fine)
                            fine_cr, _ = colRender(fine_sample_dist, rays_dist, fine_sigma, fine_RGB)
                            pixel=pixel.permute((0,2,1))
                            loss = MSE(fine_cr, pixel)+MSE(coarse_r,pixel)
                            loss_float=float(loss.detach().cpu().numpy())
                            loss_float/=args.rays_batch
                            writer.add_scalar('Loss/test',loss_float,num_test)
                            test_pic.append(fine_cr.cpu())
                    break
                pic=torch.cat(test_pic,dim=-2).squeeze(0)
                pic=pic.contiguous().view(H,W,3).permute(2,0,1).numpy()
                pic=255*pic
                pic_name='picture/'+str(i*100+j)+'.png'
                plt.savefig(pic_name)

        new_lr = args.lr*(0.1**(i/args.epoch))
        for param in optimizer.param_groups:
            param['lr'] = new_lr        
    writer.close
        


if __name__ == '__main__':
    main()
