from lego_loder import MyDataset
from torch.utils.data import DataLoader
import torch
from Nerf import *
import configargparse
import time
from torch.utils.tensorboard import SummaryWriter
import os,shutil
from datetime import datetime
import torch.nn as nn
from torchvision.utils import save_image
from torch.cuda import amp
from timm.utils import NativeScaler
from timm.scheduler import CosineLRScheduler

parser = configargparse.ArgumentParser()
parser.add_argument('--half_res', type=bool, default=True,
                    help='resolution of 400')
parser.add_argument('--epoch', type=int, default=10*100, help='total epochs')
parser.add_argument('--lr', type=float, default=8e-4,
                    help='initial learning rate')
parser.add_argument('--cpts_num', type=int, default=64,
                    help='numbers of coarse sample points')
parser.add_argument('--fpts_num', type=int, default=128,
                    help='addtional numbers of fine sample points')
parser.add_argument('--rays_batch', type=int, default=3200,
                    help='batch_size of rays')
parser.add_argument('--near', type=float, default=2., help='z of near plane')
parser.add_argument('--far', type=float, default=6., help='z of far plane')
parser.add_argument('--xins', type=int, default=63,
                    help='dimensions of gama(x)')
parser.add_argument('--dins', type=int, default=39,
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
    # if os.path.exists(logdir):
    #     shutil.rmtree(logdir)
    time_stamp = "{0:%Y-%m-%d/%H-%M-%S}-epoch{1}/".format(datetime.now(), epochs)
    return SummaryWriter(log_dir = logdir + time_stamp)



MSE=nn.MSELoss()
writer=getSummaryWriter(args.epoch)
torch.cuda.manual_seed(1)
train_data = MyDataset(root_dir='./lego/', half_res=args.half_res, is_train=True)
train_loder = DataLoader(train_data, batch_size=1,shuffle=True, num_workers=4)
test_data=MyDataset(root_dir='./lego/', half_res=args.half_res, is_train=False)
test_loader=DataLoader(test_data, batch_size=1,shuffle=False, num_workers=4)
model_coarse = Nerf(args)
model_coarse=model_coarse.cuda()
model_fine = Nerf(args)
model_fine=model_fine.cuda()
grad_vars = list(model_coarse.parameters())
grad_vars += list(model_fine.parameters())
optimizer = torch.optim.Adam(params=grad_vars, lr=args.lr)
amp_scaler = NativeScaler()
if args.half_res:
    H, W = 400, 400
else:
    H,W = 800,800  
focal = W/(2*torch.tan(0.5*train_data.cam_fov))
K = torch.tensor([[focal, 0, W], [0, focal, H], [0, 0, 1]])
total_ite=args.epoch*100
pre_num=0#3250
# scheduler = CosineLRScheduler(optimizer, t_initial=num_epoch, decay_rate=1., lr_min=1e-5)

def raysGet_rays(x,y,K,c2w):
    """
    x:B*R
    y:B*R
    """
    
    H=int(K[0][2])
    W=int(K[1][2])
    focal=float(K[0][0])

    cam_dir = torch.stack(((x-0.5*W)/focal, -(y-0.5*H) /focal, -torch.ones_like(x)), dim=-1).cuda()
    rays_dir = torch.matmul(c2w[:,None,:3, :3], cam_dir.unsqueeze(dim=-1)).squeeze(-1)
    rays_dist=torch.sqrt(torch.matmul(cam_dir[:,:,None,:],cam_dir[:,:,:,None])).squeeze(-1)
    rays_o = c2w[:,None,:3, -1].expand(rays_dir.size())
    """
    rays_o : B*R*3
    rays_dir : B*R*3
    rays_dist : B*R*1 
    """
    return rays_o, rays_dir,rays_dist.view(args.rays_batch,1)

def trainSample(img,N,S):
    assert img.shape[0]==1
    index=torch.tensor(random.sample(range(N),S)).cuda()
    x=index%W
    y=index//H
    img=img.view(-1,3,N)
    pixel=torch.index_select(img,-1,index)
    return x.view(1,S),y.view(1,S),pixel

def train_one_ray(epoch,num_pic,img,tfs):
    model_coarse.train()
    model_fine.train()
    img=img.cuda()
    tfs=tfs.cuda()
    x,y,pixel=trainSample(img,H*W,args.rays_batch)
    rays_ori, rays_dirs, rays_dists = raysGet_rays(x,y,K, tfs)
    coarse_sample, coarse_sample_dist = randomraysSample(rays_ori, rays_dirs, rays_dists, args.cpts_num, args.near, args.far)
    # view(coarse_sample,rays_ori,rays_dirs,pt_fine=False)
    rays_dirs = rays_dirs[:, :, None,:].expand(coarse_sample.size())
    rays_dists=rays_dists.view(1,args.rays_batch)
    rays_dir_nor = rays_dirs / rays_dirs.norm(dim=-1, keepdim=True)
    optimizer.zero_grad()
    with amp.autocast():
        coarse_sigma, coarse_RGB = model_coarse(coarse_sample, rays_dir_nor)
        coarse_cr, weight = colRender(coarse_sample_dist, rays_dists, coarse_sigma, coarse_RGB)
        fine_sample, fine_sample_dist, rays_dir_fine = invSample(weight, args.fpts_num, rays_ori, rays_dir_nor, rays_dists,args.near, args.far, coarse_sample_dist)
        #view(fine_sample,rays_o,rays_dir,pt_fine=True)
        rays_dir_fine_nor = rays_dir_fine / rays_dir_fine.norm(dim=-1, keepdim=True)
        fine_sigma, fine_RGB = model_fine(fine_sample, rays_dir_fine_nor)
        fine_cr, _ = colRender(fine_sample_dist, rays_dists, fine_sigma, fine_RGB)
        pixel = pixel.permute((0, 2, 1))
        loss = MSE(fine_cr, pixel) + MSE(coarse_cr, pixel)
    if not amp_scaler is None:
        amp_scaler(loss, optimizer, clip_grad=None, parameters = grad_vars, create_graph = False)
    else:
        loss.backward()
        optimizer.step()
    loss_float = float(loss.detach().cpu().numpy())
    num_train=epoch*100+num_pic
    writer.add_scalar('Loss/train/rays', loss_float, num_train)
    return loss_float

def train_ray_batch(epoch,num_pic,img,tfs,rays_epoch):
    model_coarse.train()
    model_fine.train()
    img=img.cuda()
    tfs=tfs.cuda()
    rays_ori, rays_dirs, rays_dists = raysGet(K, tfs)
    coarse_sample, coarse_sample_dist = randomraysSample(rays_ori, rays_dirs, rays_dists, args.cpts_num, args.near, args.far)
    #view(coarse_sample,rays_ori,rays_dirs,pt_fine=False)
    rays_dirs = rays_dirs[:, :, None,:].expand(coarse_sample.size())
    sum_loss=0
    if epoch<2:
        for i in range(rays_epoch):
            coarse_s, rays_o, rays_dir, rays_dist, coarse_dist, pixel=randomBatch(coarse_sample, rays_ori, rays_dirs, rays_dists, coarse_sample_dist, img,H*W,args.rays_batch)
            rays_dir_nor = rays_dir / rays_dir.norm(dim=-1, keepdim=True)
            optimizer.zero_grad()
            with amp.autocast():
                coarse_sigma, coarse_RGB = model_coarse(coarse_s, rays_dir_nor)
                coarse_cr, weight = colRender(coarse_dist, rays_dist, coarse_sigma, coarse_RGB)
                fine_sample, fine_sample_dist, rays_dir_fine = invSample(weight, args.fpts_num, rays_o, rays_dir_nor, rays_dist,args.near, args.far, coarse_dist)
                #view(fine_sample,rays_o,rays_dir,pt_fine=True)
                rays_dir_fine_nor = rays_dir_fine / rays_dir_fine.norm(dim=-1, keepdim=True)
                fine_sigma, fine_RGB = model_fine(fine_sample, rays_dir_fine_nor)
                fine_cr, _ = colRender(fine_sample_dist, rays_dist, fine_sigma, fine_RGB)
                pixel = pixel.permute((0, 2, 1))
                loss = MSE(fine_cr, pixel) + MSE(coarse_cr, pixel)
            if not amp_scaler is None:
                amp_scaler(loss, optimizer, clip_grad=None, parameters = grad_vars, create_graph = False)
            else:
                loss.backward()
                optimizer.step()
            loss_float = float(loss.detach().cpu().numpy())
            print("%d-th epoch,%d-th picture,%d-th rays_batch,loss%f:" % (epoch, num_pic, i, loss_float))
            num_train=epoch*100*(H*W//args.rays_batch)+(pre_num+num_pic)*(H*W//args.rays_batch)+i
            writer.add_scalar('Loss/train/rays', loss_float, num_train)
            sum_loss+=loss_float
            
    else:
        coarse_sample_loader, rays_ori_loader, rays_dir_loader, rays_dist_loader, coarse_d_loader,pixel_loader = raysBatchify(coarse_sample, rays_ori, rays_dirs, rays_dists, coarse_sample_dist, img,args.rays_batch)
        i=0
        for coarse_s, rays_o, rays_dir, rays_dist, coarse_dist ,pixel in zip(coarse_sample_loader, rays_ori_loader, rays_dir_loader, rays_dist_loader, coarse_d_loader,pixel_loader):
            i+=1
            optimizer.zero_grad()
            rays_dir_nor=rays_dir/rays_dir.norm(dim=-1,keepdim=True)
            with amp.autocast():
                coarse_sigma, coarse_RGB = model_coarse(coarse_s, rays_dir_nor)
                coarse_cr, weight = colRender(coarse_dist, rays_dist, coarse_sigma, coarse_RGB)
                fine_sample, fine_sample_dist, rays_dir_fine = invSample( weight, args.fpts_num, rays_o, rays_dir_nor, rays_dist, args.near, args.far, coarse_dist)
                # view(fine_sample,rays_o,rays_dir,pt_fine=True)
                rays_dir_fine_nor=rays_dir_fine/rays_dir_fine.norm(dim=-1,keepdim=True)    
                fine_sigma, fine_RGB = model_fine(fine_sample, rays_dir_fine_nor)
                fine_cr, _ = colRender(fine_sample_dist, rays_dist, fine_sigma, fine_RGB)
                pixel=pixel.permute((0,2,1))
                loss = MSE(fine_cr, pixel)+MSE(coarse_cr,pixel)
            if not amp_scaler is None:
                amp_scaler(loss, optimizer, clip_grad=None, parameters = grad_vars, create_graph = False)
            else:
                loss.backward()
                optimizer.step()
            loss_float = float(loss.detach().cpu().numpy())
            print("%d-th epoch,%d-th picture,%d-th rays_batch,loss%f:" % (epoch, num_pic, i, loss_float))
            num_train=epoch*100*(H*W//args.rays_batch)+(pre_num+num_pic)*(H*W//args.rays_batch)+i
            writer.add_scalar('Loss/train/rays', loss_float, num_train)
            sum_loss+=loss_float
    return sum_loss,num_train

def test(img,tfs,num_test):
    model_coarse.eval()
    model_fine.eval()
    img = img.cuda()
    tfs = tfs.cuda()
    test_pic = []
    rays_ori, rays_dirs, rays_dists = raysGet(K, tfs)
    coarse_sample, coarse_sample_dist = randomraysSample(rays_ori, rays_dirs, rays_dists, args.cpts_num, args.near,args.far)
    rays_dirs = rays_dirs[:, :,  None, :].expand(coarse_sample.size())
    I=0
    coarse_sample_loader, rays_ori_loader, rays_dir_loader, rays_dist_loader, coarse_d_loader, pixel_loader = raysBatchify(coarse_sample, rays_ori, rays_dirs, rays_dists, coarse_sample_dist, img, args.rays_batch)
    for coarse_s, rays_o, rays_dir, rays_dist, coarse_dist, pixel in zip(coarse_sample_loader, rays_ori_loader,rays_dir_loader, rays_dist_loader,coarse_d_loader, pixel_loader):
        I+=1
        with torch.no_grad():
            rays_dir_nor = rays_dir / rays_dir.norm(dim=-1, keepdim=True)
            coarse_sigma, coarse_RGB = model_coarse(coarse_s, rays_dir_nor)
            coarse_r, weight = colRender(coarse_dist, rays_dist, coarse_sigma, coarse_RGB)
            fine_sample, fine_sample_dist, rays_dir_fine = invSample(weight, args.fpts_num, rays_o, rays_dir_nor,rays_dist, args.near, args.far, coarse_dist)
            rays_dir_fine_nor = rays_dir_fine / rays_dir_fine.norm(dim=-1, keepdim=True)
            fine_sigma, fine_RGB = model_fine(fine_sample, rays_dir_fine_nor)
            fine_cr, _ = colRender(fine_sample_dist, rays_dist, fine_sigma, fine_RGB)
            pixel = pixel.permute((0, 2, 1))
            loss = MSE(fine_cr, pixel) + MSE(coarse_r, pixel)
            loss_float = float(loss.detach().cpu().numpy())
            loss_float /= args.rays_batch
            writer.add_scalar('Loss/test', loss_float, num_test*(H*W//args.rays_batch)+I)
            test_pic.append(fine_cr.cpu())
    pic = torch.cat(test_pic, dim=-2).squeeze(0)
    pic = pic.contiguous().view(H, W, 3).permute(2, 0, 1)
    pic_name = 'picture/' + str(num_test+1+pre_num//10) + '.png'
    save_image(pic, pic_name)


def main_batch():
    num_train=0
    num_test=0
    for i in range(args.epoch):
        for j, (img, tfs) in enumerate(train_loder):
            start_time=time.time()
            rays_epoch=(H*W)//args.rays_batch
            """
            img:B*3*H*W
            tfs:B*4*4
            """
            train_loss,num=train_ray_batch(i,j,img,tfs,rays_epoch)
            end_time = time.time()
            print("%dth epoch,%d-th picture,loss:%f,time:%f"%(i,j,train_loss,end_time-start_time))
            writer.add_scalar('Loss/train/pic', train_loss, (i * 100 + j))
            model_coarse_save_path = './model/' + str((i * 100 + j)) + 'model_coarse.tar'
            model_fine_save_path = './model/' + str((i * 100 + j)) + 'model_fine.tar'
            if j % 10 == 0:
                torch.save({'state_dict': model_coarse.state_dict(), 'train_loss': train_loss}, model_coarse_save_path)
                torch.save({'state_dict': model_fine.state_dict(), 'train_loss': train_loss}, model_fine_save_path)
                num_test+=1
                new_lr = args.lr * (0.1 ** (num / total_ite))
                for param in optimizer.param_groups:
                    param['lr'] = new_lr
                for k, (img, tfs) in enumerate(test_loader):
                    test(img,tfs,num_test)
                    break
    writer.close()

def main_one():
    num_train=0
    num_test=0
    for i in range(args.epoch):
        start_time=time.time()
        sum_loss=0.
        for j, (img, tfs) in enumerate(train_loder):
            """
            img:B*3*H*W
            tfs:B*4*4
            """

            train_loss=train_one_ray(i,j,img,tfs)
            sum_loss+=train_loss
            print("%dth epoch,%d-th picture,loss:%f"%(i,j,train_loss))
        num_train+=1
        end_time = time.time()
        writer.add_scalar('Loss/train/pic', sum_loss, (i))
        print("%dth epoch,loss:%f,time:%f"%(i,sum_loss,end_time-start_time))
        model_coarse_save_path = './model/' + str((i//10)) + 'model_coarse.tar'
        model_fine_save_path = './model/' + str((i//10)) + 'model_fine.tar'
        if i % 20 == 0:
            torch.save({'state_dict': model_coarse.state_dict(), 'train_loss': train_loss}, model_coarse_save_path)
            torch.save({'state_dict': model_fine.state_dict(), 'train_loss': train_loss}, model_fine_save_path)
            num_test+=1
            new_lr = args.lr * (0.1 ** (num_train / args.epoch))
            for param in optimizer.param_groups:
                param['lr'] = new_lr
            for k, (img, tfs) in enumerate(test_loader):
                test(img,tfs,num_test)
                break
        
        
    writer.close()



if __name__ == '__main__':
    main_one()
