import torch
import torch.nn as nn
class Unet(nn.Module):
    '''U-Net Architecture'''
    def __init__(self,inp,out):
        super(Unet,self).__init__()
        self.c1=self.contracting_block(inp,64)
        self.c2=self.contracting_block(64,128)
        self.c3=self.contracting_block(128,256)
        self.c4=self.contracting_block(256,512)
        self.c5=self.contracting_block(512,1024)
        self.maxpool=nn.MaxPool2d(2)
        self.upsample=nn.Upsample(scale_factor=2,mode="bilinear",align_corners=True)
        self.c6=self.contracting_block(512+1024,512)
        self.c7=self.contracting_block(512+256,256)
        self.c8=self.contracting_block(256+128,128)
        self.c9=self.contracting_block(128+64,64)
        self.c10=nn.Conv2d(64,1,1)
        

    def contracting_block(self,inp,out,k=3):
        block =nn.Sequential(
            nn.Conv2d(inp, out, padding=1,kernel_size=3),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True),
            nn.Conv2d(out, out,padding=1,kernel_size=3),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True)
        )
        return block


    def forward(self,x):
        conv1=self.c1(x) #256x256x64
        conv1=self.maxpool(conv1) #128x128x64
        conv2=self.c2(conv1) #128x128x128
        conv2=self.maxpool(conv2) #64x64x128
        conv3=self.c3(conv2) #64x64x256
        conv3=self.maxpool(conv3) #32x32x256
        conv4=self.c4(conv3) #32x32x512
        conv4=self.maxpool(conv4) #16x16x512
        conv5=self.c5(conv4) #8x8x1024
        conv5=self.maxpool(conv5)
        x=self.upsample(conv5) ##16x16x1024
        #print(x.shape)
        x=torch.cat([x,conv4],axis=1) #16x16x1536
        x=self.c6(x) #16x16x512
        x=self.upsample(x) #32x32x512
        x=torch.cat([x,conv3],axis=1) 
        x=self.c7(x) #32x32x256
        x=self.upsample(x) #64x64x256
        x=torch.cat([x,conv2],axis=1)
        x=self.c8(x) #64x64x128
        x=self.upsample(x) #128x128x128
        x=torch.cat([x,conv1],axis=1) 
        x=self.c9(x) #128x128x64
        x=self.upsample(x)#256x256x64
        x=self.c10(x)
        return x


if __name__=="__main__":
    x=torch.ones(1,3,256,512)
    net=Unet(3,1)
    print(net(x).shape)
