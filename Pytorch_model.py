#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from torch import nn, cat
from torch.autograd import Variable


# In[ ]:


class nnUnet(nn.Module):
    def __init__(self, trainer):
       super(nnUnet,self).__init__()
       self.conv_blocks_context = nn.Sequential(*list(trainer.conv_blocks_context.children()))
       self.conv_blocks_localization = nn.Sequential(*list(trainer.conv_blocks_localization.children())) 
       self.td = nn.Sequential(*list(trainer.td.children()))
       self.tu = nn.Sequential(*list(trainer.tu.children()))
       self.seg_outputs = nn.Sequential(*list(trainer.seg_outputs.children()))
       
    def forward(self,x):

        #ENCODER
        skips = []
        #seg_outputs = []
        
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)        
            skips.append(x)            
        
        x = self.conv_blocks_context[-1](x)  
        x_encode = x      
        
        #SEGMENTATION
        for u in range(len(self.tu)):            
            x = self.tu[u](x)
            x = cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
        #seg_outputs = self.seg_outputs[-1](x)
        #seg_outputs = torch.nn.functional.softmax(seg_outputs,1)    

        return x_encode, x

        #return x


# In[ ]:


class Encodeur_nnUnet(nn.Module):
    def __init__(self,trainer):
       super(Encodeur_nnUnet,self).__init__()
       self.conv_blocks_context = nn.Sequential(*list(trainer.conv_blocks_context.children()))
       '''self.conv_blocks_localization = nn.Sequential(*list(trainer.conv_blocks_localization.children())) 
       self.td = nn.Sequential(*list(trainer.td.children()))
       self.tu = nn.Sequential(*list(trainer.tu.children()))
       self.seg_outputs = nn.Sequential(*list(trainer.seg_outputs.children()))'''
       
    def forward(self,x):
        
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)                
        
        x = self.conv_blocks_context[-1](x)  

        return x


# In[ ]:


class Decoder3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(Decoder3D, self).__init__()

        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(out_channels, out_channels, kernel_size=stride, stride=stride)
        ]

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


# In[ ]:


class Encoder3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(Encoder3D, self).__init__()

        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride,padding=1),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)            
        ]

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


# In[ ]:


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_block, self).__init__()

        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)            
        ]

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)


# In[ ]:


class Multitask_nnUNet(nn.Module):
    def __init__(self, model,drop_encode=0.5, n_classes = 3):
        super(Multitask_nnUNet,self).__init__()

        self.nnunet = model
        self.seg = nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1)    

        channel = (320, 320, 256, 128, 64, 32)
        self.upsample_1 = Decoder3D(channel[0], channel[1], stride=(2, 1, 2))
        self.upsample_2 = Decoder3D(channel[1], channel[2], stride=2)
        self.upsample_3 = Decoder3D(channel[2], channel[3], stride=2)
        self.upsample_4 = Decoder3D(channel[3], channel[4], stride=2)
        self.upsample_5 = Decoder3D(channel[4], channel[5], stride=2)
        self.reconst = nn.Conv3d(channel[5], 1, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(320, n_classes)
        self.dropout = nn.Dropout(p=drop_encode)    
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))       

    def forward(self, image):
        
        encode, seg = self.nnunet(image) 
        self.features_conv = encode
        
        seg = self.seg(seg)
        seg = nn.Sigmoid()(seg)

        #RECONSTRUCTION
        reconst = self.upsample_1(encode)
        reconst = self.upsample_2(reconst)
        reconst = self.upsample_3(reconst)
        reconst = self.upsample_4(reconst)
        reconst = self.upsample_5(reconst)
        reconst = self.reconst(reconst)

        #CLASSIFICATION
        
        classe = self.avgpool(encode)
        classe = classe.view(classe.size()[0], -1)
        classe = nn.ReLU()(classe)

        classe = self.dropout(classe)
        classe = self.fc1(classe)
        classe = nn.Softmax(dim=1)(classe) 

        return classe, reconst, seg


# In[ ]:


class Model_Encodeur(nn.Module):
    def __init__(self, model,drop_encode=0.5, n_classes = 3):
        super(Model_Encodeur,self).__init__()
        self.nnunet = model
        self.fc1 = nn.Linear(320, n_classes)
        self.dropout = nn.Dropout(p=drop_encode)    
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))       

    def forward(self, image):
        
        encode = self.nnunet(image) 

        #CLASSIFICATION
        
        classe = self.avgpool(encode)
        classe = classe.view(classe.size()[0], -1)
        classe = nn.ReLU()(classe)

        classe = self.dropout(classe)
        classe = self.fc1(classe)
        classe = nn.Softmax(dim=1)(classe) 

        return classe


# In[ ]:


class Multitask_nnUNet_diffuse(nn.Module):
    def __init__(self, model, drop_encode=0.5, n_classes = 3):
        super(Multitask_nnUNet_diffuse,self).__init__()

        self.nnunet = model
        self.seg = nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1)    

        channel = (320, 320, 256, 128, 64, 32)
        self.upsample_1 = Decoder3D(channel[0], channel[1], stride=(2, 1, 2))
        self.upsample_2 = Decoder3D(channel[1], channel[2], stride=2)
        self.upsample_3 = Decoder3D(channel[2], channel[3], stride=2)
        self.upsample_4 = Decoder3D(channel[3], channel[4], stride=2)
        self.upsample_5 = Decoder3D(channel[4], channel[5], stride=2)
        self.reconst = nn.Conv3d(channel[5], 1, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(320, n_classes)
        self.fc1_df = nn.Linear(46080, 4)
        self.dropout = nn.Dropout(p=drop_encode)    
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))       

    def forward(self, image):
        
        encode, seg = self.nnunet(image) 
        self.features_conv = encode
        
        seg = self.seg(seg)
        seg = nn.Sigmoid()(seg)

        #RECONSTRUCTION
        reconst = self.upsample_1(encode)
        reconst = self.upsample_2(reconst)
        reconst = self.upsample_3(reconst)
        reconst = self.upsample_4(reconst)
        reconst = self.upsample_5(reconst)
        reconst = self.reconst(reconst)

        #CLASSIFICATION
        
        classe = self.avgpool(encode)
        classe = classe.view(classe.size()[0], -1)
        classe = nn.ReLU()(classe)

        classe = self.dropout(classe)
        classe = self.fc1(classe)
        classe = nn.Softmax(dim=1)(classe)

        #CLASSIFICATION DIFFUSE
        #classe_df = self.avgpool(encode)
        classe_df = encode.view(encode.size()[0], -1)
        classe_df = nn.ReLU()(classe_df)

        classe_df = self.dropout(classe_df)
        classe_df = self.fc1_df(classe_df)
        classe_df = nn.Softmax(dim=1)(classe_df) 

        return classe, classe_df, reconst, seg


# In[ ]:


class Multitask_nnUNet_diffuse_3(nn.Module):
    def __init__(self, model, drop_encode=0.5, n_classes = 3):
        super(Multitask_nnUNet_diffuse_3,self).__init__()

        self.nnunet = model
        self.seg = nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1)    

        channel = (320, 320, 256, 128, 64, 32)
        self.upsample_1 = Decoder3D(channel[0], channel[1], stride=(2, 1, 2))
        self.upsample_2 = Decoder3D(channel[1], channel[2], stride=2)
        self.upsample_3 = Decoder3D(channel[2], channel[3], stride=2)
        self.upsample_4 = Decoder3D(channel[3], channel[4], stride=2)
        self.upsample_5 = Decoder3D(channel[4], channel[5], stride=2)
        self.reconst = nn.Conv3d(channel[5], 1, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(46080, n_classes)
        self.fc1_df = nn.Linear(46080, 4)
        self.dropout = nn.Dropout(p=drop_encode)    
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))       

    def forward(self, image):
        
        encode, seg = self.nnunet(image) 
        self.features_conv = encode
        
        seg = self.seg(seg)
        seg = nn.Sigmoid()(seg)

        #RECONSTRUCTION
        reconst = self.upsample_1(encode)
        reconst = self.upsample_2(reconst)
        reconst = self.upsample_3(reconst)
        reconst = self.upsample_4(reconst)
        reconst = self.upsample_5(reconst)
        reconst = self.reconst(reconst)

        #CLASSIFICATION
        flatten = encode.view(encode.size()[0], -1)
        classe = nn.ReLU()(flatten)

        classe = self.dropout(classe)
        classe = self.fc1(classe)
        classe = nn.Softmax(dim=1)(classe)

        #CLASSIFICATION DIFFUSE
        classe_df = nn.ReLU()(flatten)

        classe_df = self.dropout(classe_df)
        classe_df = self.fc1_df(classe_df)
        classe_df = nn.Softmax(dim=1)(classe_df) 

        return classe, classe_df, reconst, seg


# In[ ]:


class Multitask_nnUNet_nodiffuse(nn.Module):
    def __init__(self, model, drop_encode=0.5, n_classes = 3):
        super(Multitask_nnUNet_nodiffuse,self).__init__()

        self.nnunet = model
        self.seg = nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1)    

        channel = (320, 320, 256, 128, 64, 32)
        self.upsample_1 = Decoder3D(channel[0], channel[1], stride=(2, 1, 2))
        self.upsample_2 = Decoder3D(channel[1], channel[2], stride=2)
        self.upsample_3 = Decoder3D(channel[2], channel[3], stride=2)
        self.upsample_4 = Decoder3D(channel[3], channel[4], stride=2)
        self.upsample_5 = Decoder3D(channel[4], channel[5], stride=2)
        self.reconst = nn.Conv3d(channel[5], 1, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(320, n_classes)
        self.dropout = nn.Dropout(p=drop_encode)    
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))       

    def forward(self, image):
        
        encode, seg = self.nnunet(image) 
        self.features_conv = encode
        
        seg = self.seg(seg)
        seg = nn.Sigmoid()(seg)

        #RECONSTRUCTION
        reconst = self.upsample_1(encode)
        reconst = self.upsample_2(reconst)
        reconst = self.upsample_3(reconst)
        reconst = self.upsample_4(reconst)
        reconst = self.upsample_5(reconst)
        reconst = self.reconst(reconst)

        #CLASSIFICATION
        
        classe = self.avgpool(encode)
        classe = classe.view(classe.size()[0], -1)
        classe = nn.ReLU()(classe)

        classe = self.dropout(classe)
        classe = self.fc1(classe)
        classe = nn.Softmax(dim=1)(classe)

        return classe, reconst, seg


# In[ ]:


class Multitask_nnUNet_nodiffuse_noSeg(nn.Module):
    def __init__(self, model, drop_encode=0.5, n_classes = 3):
        super(Multitask_nnUNet_nodiffuse_noSeg,self).__init__()

        self.nnunet = model
    
        channel = (320, 320, 256, 128, 64, 32)
        self.upsample_1 = Decoder3D(channel[0], channel[1], stride=(2, 1, 2))
        self.upsample_2 = Decoder3D(channel[1], channel[2], stride=2)
        self.upsample_3 = Decoder3D(channel[2], channel[3], stride=2)
        self.upsample_4 = Decoder3D(channel[3], channel[4], stride=2)
        self.upsample_5 = Decoder3D(channel[4], channel[5], stride=2)
        self.reconst = nn.Conv3d(channel[5], 1, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(320, n_classes)
        self.dropout = nn.Dropout(p=drop_encode)    
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))       

    def forward(self, image):
        
        encode = self.nnunet(image) 
        
        #RECONSTRUCTION
        reconst = self.upsample_1(encode)
        reconst = self.upsample_2(reconst)
        reconst = self.upsample_3(reconst)
        reconst = self.upsample_4(reconst)
        reconst = self.upsample_5(reconst)
        reconst = self.reconst(reconst)

        #CLASSIFICATION
        
        classe = self.avgpool(encode)
        classe = classe.view(classe.size()[0], -1)
        classe = nn.ReLU()(classe)

        classe = self.dropout(classe)
        classe = self.fc1(classe)
        classe = nn.Softmax(dim=1)(classe)

        return classe, reconst


# In[ ]:


class Multitask_nnUNet_diffuse_onlySeg(nn.Module):
    def __init__(self, model):
        super(Multitask_nnUNet_diffuse_onlySeg,self).__init__()

        self.nnunet = model
        self.seg = nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1)    

    def forward(self, image):
        
        _, seg = self.nnunet(image) 
        
        seg = self.seg(seg)
        seg = nn.Sigmoid()(seg)

        return seg


# In[ ]:


class Multitask_nnUNet_diffuse_noSeg2(nn.Module):
    def __init__(self, model, drop_encode=0.5, n_classes = 3):
        super(Multitask_nnUNet_diffuse_noSeg2,self).__init__()

        self.nnunet = model
        self.seg = nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1)    

        channel = (320, 320, 256, 128, 64, 32)
        self.upsample_1 = Decoder3D(channel[0], channel[1], stride=(2, 1, 2))
        self.upsample_2 = Decoder3D(channel[1], channel[2], stride=2)
        self.upsample_3 = Decoder3D(channel[2], channel[3], stride=2)
        self.upsample_4 = Decoder3D(channel[3], channel[4], stride=2)
        self.upsample_5 = Decoder3D(channel[4], channel[5], stride=2)
        self.reconst = nn.Conv3d(channel[5], 1, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(320, n_classes)
        self.fc1_df = nn.Linear(46080, 4)
        self.dropout = nn.Dropout(p=drop_encode) 
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))       

    def forward(self, image):
        
        encode = self.nnunet(image) 
        #self.features_conv = encode
        
        #RECONSTRUCTION
        reconst = self.upsample_1(encode)
        reconst = self.upsample_2(reconst)
        reconst = self.upsample_3(reconst)
        reconst = self.upsample_4(reconst)
        reconst = self.upsample_5(reconst)
        reconst = self.reconst(reconst)

        #CLASSIFICATION
        
        classe = self.avgpool(encode)
        classe = classe.view(classe.size()[0], -1)
        classe = nn.ReLU()(classe)

        classe = self.dropout(classe)
        classe = self.fc1(classe)
        classe = nn.Softmax(dim=1)(classe)

        #CLASSIFICATION DIFFUSE
        classe_df = encode.view(encode.size()[0], -1)
        classe_df = nn.ReLU()(classe_df)

        classe_df = self.dropout(classe_df)
        classe_df = self.fc1_df(classe_df)
        classe_df = nn.Softmax(dim=1)(classe_df) 

        return classe, classe_df, reconst


# In[ ]:


class Multitask_nnUNet_diffuse_noSeg3(nn.Module):
    def __init__(self, model, drop_encode=0.5, n_classes = 3):
        super(Multitask_nnUNet_diffuse_noSeg3,self).__init__()

        self.nnunet = model
        self.seg = nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1)    

        channel = (320, 320, 256, 128, 64, 32)
        self.upsample_1 = Decoder3D(channel[0], channel[1], stride=(2, 1, 2))
        self.upsample_2 = Decoder3D(channel[1], channel[2], stride=2)
        self.upsample_3 = Decoder3D(channel[2], channel[3], stride=2)
        self.upsample_4 = Decoder3D(channel[3], channel[4], stride=2)
        self.upsample_5 = Decoder3D(channel[4], channel[5], stride=2)
        self.reconst = nn.Conv3d(channel[5], 1, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(46080, n_classes)
        self.fc1_df = nn.Linear(46080, 4)
        self.dropout = nn.Dropout(p=drop_encode)    
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))       

    def forward(self, image):
        
        encode = self.nnunet(image) 
        #self.features_conv = encode
        
        #RECONSTRUCTION
        reconst = self.upsample_1(encode)
        reconst = self.upsample_2(reconst)
        reconst = self.upsample_3(reconst)
        reconst = self.upsample_4(reconst)
        reconst = self.upsample_5(reconst)
        reconst = self.reconst(reconst)

        #CLASSIFICATION
        flatten = encode.view(encode.size()[0], -1)
        classe = nn.ReLU()(flatten)

        classe = self.dropout(classe)
        classe = self.fc1(classe)
        classe = nn.Softmax(dim=1)(classe)

        #CLASSIFICATION DIFFUSE
        classe_df = nn.ReLU()(flatten)

        classe_df = self.dropout(classe_df)
        classe_df = self.fc1_df(classe_df)
        classe_df = nn.Softmax(dim=1)(classe_df) 

        return classe, classe_df, reconst


# In[ ]:


class Multitask_nnUNet_D100(nn.Module):
    def __init__(self, model, drop_encode=0.5, n_classes = 3):
        super(Multitask_nnUNet_D100,self).__init__()

        self.nnunet = Encodeur_nnUnet(model)
        self.seg = nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1)    

        channel = (320, 320, 256, 128, 64, 32)
        self.upsample_1 = Decoder3D(channel[0], channel[1], stride=(2, 1, 2))
        self.upsample_2 = Decoder3D(channel[1], channel[2], stride=2)
        self.upsample_3 = Decoder3D(channel[2], channel[3], stride=2)
        self.upsample_4 = Decoder3D(channel[3], channel[4], stride=2)
        self.upsample_5 = Decoder3D(channel[4], channel[5], stride=2)
        self.reconst = nn.Conv3d(channel[5], 1, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(320, n_classes)
        self.dropout = nn.Dropout(p=drop_encode)    
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))       

    def forward(self, image):
        
        encode = self.nnunet(image) 
        self.features_conv = encode

        #RECONSTRUCTION
        reconst = self.upsample_1(encode)
        reconst = self.upsample_2(reconst)
        reconst = self.upsample_3(reconst)
        reconst = self.upsample_4(reconst)
        reconst = self.upsample_5(reconst)
        reconst = self.reconst(reconst)

        #CLASSIFICATION
        
        classe = self.avgpool(encode)
        classe = classe.view(classe.size()[0], -1)
        classe = nn.ReLU()(classe)

        classe = self.dropout(classe)
        classe = self.fc1(classe)
        classe = nn.Softmax(dim=1)(classe) 

        return classe, reconst


# In[ ]:


class Encodeur_Net(nn.Module):
    def __init__(self, model, drop_encode=0.5):
        super(Encodeur_Net,self).__init__()

        self.nnunet = model  

        self.fc1 = nn.Linear(320, 3)
        self.dropout = nn.Dropout(p=drop_encode)    
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1)) 

    def forward(self, image):
        
        encode = self.nnunet(image)

        #CLASSIFICATION
        
        classe = self.avgpool(encode)
        classe = classe.view(classe.size()[0], -1)
        classe = nn.ReLU()(classe)

        classe = self.dropout(classe)
        classe = self.fc1(classe)
        classe = nn.Softmax(dim=1)(classe) 

        return classe

    


# In[ ]:


class UNet(nn.Module):
    def __init__(self,in_channels=1,base_channel=32,stride=2,nb_blocks=5):
        super(UNet,self).__init__()
        self.conv_block = conv_block(in_channels=in_channels,out_channels=base_channel)

        #Encoder
        blocks = []
        for i in range(nb_blocks):
            temp = Encoder3D(in_channels=base_channel*(2**i),out_channels=base_channel*(2**(i+1)),stride=stride)
            blocks.append(temp)

        self.encode = nn.Sequential(*blocks)

        # decode
        
        blocks = []
        for i in range(nb_blocks-1):
            temp = Decoder3D(in_channels=base_channel*(2**(i+1))*2,out_channels=base_channel*(2**i),stride=2)
            blocks.append(temp)
            
        temp = Decoder3D(in_channels=base_channel*(2**nb_blocks),out_channels=base_channel*(2**(nb_blocks-1)),stride=2)
        blocks.append(temp)
        self.decode = nn.Sequential(*blocks)

        #segmentation
        self.seg1 = conv_block(in_channels=base_channel*2,out_channels=base_channel)
        self.seg = nn.Conv3d(in_channels=base_channel, out_channels=1, kernel_size=3, padding=1)
                                 
    
    def forward(self, x):
               
        # ENCODER
        x = self.conv_block(x)

        skips = [x]        
        for down in range(len(self.encode)-1):
            x = self.encode[down](x)
            skips.append(x)

        x = self.encode[-1](x)  
        x_encode = x      
        
        # DECODER
        for up in range(len(self.decode)): 
            x = self.decode[-(up+1)](x)   
            x = cat((x, skips[-(up + 1)]), dim=1)

        x = self.seg1(x)
        seg_outputs = self.seg(x)
        #seg_outputs = nn.functional.softmax(seg_outputs,1)  
        seg_outputs = nn.Sigmoid()(seg_outputs)
         
        return x_encode, seg_outputs
            


# In[ ]:


class Multitask_Net(nn.Module):
    def __init__(self,drop_encode=0.5,in_channels=1,base_channel=32,stride=2,nb_blocks=4):
        super(Multitask_Net,self).__init__()

        self.nnunet = UNet(in_channels=in_channels,base_channel=base_channel,stride=stride,nb_blocks=nb_blocks)
        
        blocks = []
        for i in range(nb_blocks):
            temp = Decoder3D(in_channels=base_channel*(2**(nb_blocks-i)),out_channels=base_channel*(2**(nb_blocks-i-1)),stride=2)
            blocks.append(temp)
            
        self.decode = nn.Sequential(*blocks)
        self.reconst = nn.Conv3d(base_channel, 1, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(int(base_channel*(2**nb_blocks)), 3)
        self.dropout = nn.Dropout(p=drop_encode)    
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1)) 



    def forward(self, image):
        
        encode, seg = self.nnunet(image)
        
        #RECONSTRUCTION
        reconst = self.decode(encode)
        reconst = self.reconst(reconst)

        #CLASSIFICATION
        
        classe = self.avgpool(encode)
        classe = classe.view(classe.size()[0], -1)
        classe = nn.ReLU()(classe)

        classe = self.dropout(classe)
        classe = self.fc1(classe)
        classe = nn.Softmax(dim=1)(classe) 

        return classe, reconst, seg

   


# In[ ]:


class nnUnet_modif(nn.Module):
    def __init__(self, trainer):
       super(nnUnet_modif,self).__init__()
       self.conv_blocks_context = nn.Sequential(*list(trainer.conv_blocks_context.children()))
       self.conv_blocks_localization = nn.Sequential(*list(trainer.conv_blocks_localization.children())) 
       self.td = nn.Sequential(*list(trainer.td.children()))
       self.tu = nn.Sequential(*list(trainer.tu.children()))
       self.seg_outputs = nn.Sequential(*list(trainer.seg_outputs.children()))
       #self.encode = Encoder3D(in_channels=320,out_channels=512,stride=2)
       
    def forward(self,x):

        #ENCODER
        skips = []
        #seg_outputs = []
        
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)        
            skips.append(x)            
        
        x = self.conv_blocks_context[-1](x)  
        x_encode_1 = x   
        x_encode_2 = Encoder3D(in_channels=320,out_channels=1024,stride=2).cuda()(x)
        
        #SEGMENTATION
        for u in range(len(self.tu)):            
            x = self.tu[u](x)
            x = cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
        #seg_outputs = self.seg_outputs[-1](x)
        #seg_outputs = torch.nn.functional.softmax(seg_outputs,1)    

        return x_encode_1, x_encode_2, x

        #return x


# In[ ]:


class Multitask_nnUNet_diffuse_2(nn.Module):
    def __init__(self, model, drop_encode=0.5, n_classes = 3):
        super(Multitask_nnUNet_diffuse_2,self).__init__()

        self.nnunet = model
        self.seg = nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1)    

        channel = (320, 320, 256, 128, 64, 32)
        self.upsample_1 = Decoder3D(channel[0], channel[1], stride=(2, 1, 2))
        self.upsample_2 = Decoder3D(channel[1], channel[2], stride=2)
        self.upsample_3 = Decoder3D(channel[2], channel[3], stride=2)
        self.upsample_4 = Decoder3D(channel[3], channel[4], stride=2)
        self.upsample_5 = Decoder3D(channel[4], channel[5], stride=2)
        self.reconst = nn.Conv3d(channel[5], 1, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(320, n_classes)
        self.fc1_df = nn.Linear(1024, 256)
        #self.fc2_df = nn.Linear(256, 64)
        #self.fc3_df = nn.Linear(64, 4)
        self.dropout = nn.Dropout(p=drop_encode)    
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))       

    def forward(self, image):
        
        encode_1, encode_2, seg = self.nnunet(image) 
        #self.features_conv = encode
        
        seg = self.seg(seg)
        seg = nn.Sigmoid()(seg)

        #RECONSTRUCTION
        reconst = self.upsample_1(encode_1)
        reconst = self.upsample_2(reconst)
        reconst = self.upsample_3(reconst)
        reconst = self.upsample_4(reconst)
        reconst = self.upsample_5(reconst)
        reconst = self.reconst(reconst)

        #CLASSIFICATION
        
        classe_gap = self.avgpool(encode_1)
        classe_gap = classe_gap.view(classe_gap.size()[0], -1)
        classe = nn.ReLU()(classe_gap)

        classe = self.dropout(classe)
        classe = self.fc1(classe)
        classe = nn.Softmax(dim=1)(classe)

        #CLASSIFICATION DIFFUSE
        classe_df_gap = self.avgpool(encode_2) 
        #classe_df_gap = classe_df_gap.view(classe_df_gap.size()[0], -1)
        #classe_df_gap = cat((classe_gap,classe_df_gap),-1)
        classe_df = nn.ReLU()(classe_df_gap)

        #classe_df = self.dropout(classe_df)
        classe_df = self.fc1_df(classe_df)
        classe_df = nn.ReLU()(classe_df)
        classe_df = self.fc2_df(classe_df)
        classe_df = nn.ReLU()(classe_df)
        classe_df = self.fc3_df(classe_df)
        classe_df = nn.ReLU()(classe_df)
        classe_df = nn.Softmax(dim=1)(classe_df) 

        return classe, classe_df, reconst, seg


# In[ ]:


class Multitask_Net_diffuse(nn.Module):
    def __init__(self,drop_encode=0.5,in_channels=1,base_channel=32,stride=2,nb_blocks=4):
        super(Multitask_Net_diffuse,self).__init__()

        self.nnunet = UNet(in_channels=in_channels,base_channel=base_channel,stride=stride,nb_blocks=nb_blocks)
        
        blocks = []
        for i in range(nb_blocks):
            temp = Decoder3D(in_channels=base_channel*(2**(nb_blocks-i)),out_channels=base_channel*(2**(nb_blocks-i-1)),stride=2)
            blocks.append(temp)
            
        self.decode = nn.Sequential(*blocks)
        self.reconst = nn.Conv3d(base_channel, 1, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(int(base_channel*(2**nb_blocks)), 3)
        self.fc1_df = nn.Linear(int(base_channel*(2**nb_blocks)), 4)
        self.dropout = nn.Dropout(p=drop_encode)    
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1)) 



    def forward(self, image):
        
        encode, seg = self.nnunet(image)
        
        #RECONSTRUCTION
        reconst = self.decode(encode)
        reconst = self.reconst(reconst)

        #CLASSIFICATION
        
        classe = self.avgpool(encode)
        classe = classe.view(classe.size()[0], -1)
        classe = nn.ReLU()(classe)

        classe = self.dropout(classe)
        classe = self.fc1(classe)
        classe = nn.Softmax(dim=1)(classe) 

        #CLASSIFICATION DIFFUSE
        classe_df = self.avgpool(encode)
        classe_df = classe_df.view(classe_df.size()[0], -1) 
        classe_df = nn.ReLU()(classe_df)

        #classe_df = self.dropout(classe_df)
        classe_df = self.fc1_df(classe_df)
        classe_df = nn.Softmax(dim=1)(classe_df) 

        return classe, classe_df, reconst, seg


# In[ ]:


if __name__ == "__main__":
    from torchsummary import summary
    import os
    from nnunet.inference.predict import load_model_and_checkpoint_files

    base = '/media/nguyen-k/nnUNet_trained_models/nnUNet/3d_fullres'
    task = '001'
    list_task = os.listdir(base)    
    for t in list_task :
        if task in t :
            folders = os.path.join(base, t, 'nnUNetTrainerV2__nnUNetPlansv2.1')
    trainer, params_tr = load_model_and_checkpoint_files(folders, folds=None, mixed_precision=None, checkpoint_name="model_best")
    nn_Unet = nnUnet(trainer.network)
    #multitask = Multitask_nnUNet(nn_Unet).cuda()
    #summary(multitask,(1,32*4, 32*3, 32*6))

    state_dict =trainer.network.state_dict()
    keys = list(state_dict.keys())
    for e in keys:
        if 'conv_blocks_context' not in e :
            del state_dict[e]
            keys = state_dict.keys()
        

