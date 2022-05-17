from torch.utils.data import Dataset,DataLoader
import os
import natsort
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import  matplotlib.pyplot as plt

class MyDataset(Dataset):   #return img,K,tfs
    def __init__(self,root_dir,half_res=True,is_train=True):
        super().__init__()
        self.root_dir=root_dir
        self.half_res=True
        self.is_train=is_train
        self.main_dir=self.root_dir+('train/' if is_train else 'test/')
        img_names=list(filter(lambda x:x.endswith('png'),os.listdir(self.main_dir)))
        self.imgs=natsort.natsorted(img_names)
        self.cam_fov,self.tfs=self._camparamGet()
        if self.half_res:
            self.transform=transforms.Compose([transforms.Resize((400,400)),transforms.ToTensor()])
        else:
            self.transform=transforms.Compose([transforms.ToTensor()])

    @staticmethod
    def jsonRead(path:str):
        with open(path,'r') as file:
            items=json.load(file)
        cam_fov=torch.tensor(items["camera_angle_x"])
        print('Camera fov: %lf'%(cam_fov))
        tf_np=np.stack([frame["transform_matrix"] for frame in items["frames"]],axis=0)
        tfs=torch.from_numpy(tf_np).float()
        return cam_fov,tfs

    def _camparamGet(self):
        json_file="%stransforms_%s.json"%(self.root_dir,"train" if self.is_train else "test")
        cam_fov,tfs=MyDataset.jsonRead(json_file)
        return cam_fov,tfs

    def cameraGet(self):
        return self.cam_fov,self.tfs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_file=os.path.join(self.main_dir,self.imgs[index])
        image=Image.open(img_file).convert('RGB')
        img=self.transform(image)
        return img,self.tfs[index]

    def datasetGet(self):
        result=[]
        for im in self.imgs:
            img_name=os.path.join(self.main_dir,im)
            result.append(self.transform(Image.open(img_name).convert('RGB')))
        all_images=torch.stack(result,dim=0)
        return self.cam_fov,self.tfs,all_images

if __name__=="__main__":
    dataset=MyDataset('./lego/',half_res=True,is_train=True)
    print(type(dataset))
    trainloader=DataLoader(dataset,batch_size=8,shuffle=True,num_workers=4)
    for i,(img,tfs) in enumerate(trainloader):
        for i in range(3):
            plt.subplot(1,3,i+1)
            plt.imshow(img[i].permute(1,2,0))
        print(img.size())
        print(tfs.size())
        break
    plt.show()


