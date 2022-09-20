#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from skimage.transform import resize
from torch.autograd import Variable
from sklearn.preprocessing import OneHotEncoder
import math


# In[ ]:


def Grad_Cam(prob,pred,model,dim):

    # get the gradient of the output with respect to the parameters of the model

    prob[:,pred].backward()

    # pull the gradients out of the model
    gradients = model.get_activations_gradient()

    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3, 4])

    # get the activations of the last convolutional layer
    activations = model.get_activations().detach()

    # weight the channels by corresponding gradients
    for i in range(activations.shape[0]):
        activations[:, i, :, :, :] *= pooled_gradients[i]
        
    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()

    # relu on top of the heatmap
    heatmap = np.maximum(heatmap.cpu(), 0)

    # normalize the heatmap
    heatmap /= torch.max(heatmap)

    # resize heatmap
    heatmap_resized = resize(heatmap, (dim[0], dim[1],dim[2]))

    return heatmap_resized    


# In[ ]:


def cam(model,img,dim):
    model.eval()
    features = model.nnunet.conv_blocks_context
    maps = features(Variable(img.cuda()))
    
    maps_avg = nn.AdaptiveAvgPool3d((1,1,1))(maps)
    maps_avg = maps_avg.view(maps_avg.size()[0], -1)

    out = model.fc1(maps_avg)

    index = np.argmax(out.cpu().data.numpy(), axis=1).reshape(-1, 1)


    weights = model.fc1.weight
    
    b, k, d, h, w = maps.size()
    saliency_map = torch.zeros(1,1,d,h,w).cuda()
    for i in range(k):
        saliency_map[0,0,:,:,:] += weights[0,i]*maps[0,i,:,:,:]
    saliency_map = F.interpolate(saliency_map, size=(dim[0], dim[1], dim[2]), mode='trilinear', align_corners=False).data.cpu().numpy()
    return np.squeeze(saliency_map)




# In[ ]:


def gradcam(model,img,dim):
    model.eval()
    features = model.nnunet.conv_blocks_context
    maps = features(Variable(img.cuda()))

    grads_ = []
    maps.register_hook(lambda x: grads_.append(x))
    
    maps_avg = nn.AdaptiveAvgPool3d((1,1,1))(maps)
    maps_avg = maps_avg.view(maps_avg.size()[0], -1)

    out = model.fc1(maps_avg)

    index = np.argmax(out.cpu().data.numpy(), axis=1).reshape(-1, 1)

    ohe_index = out.cpu().data.numpy()*0
    ohe_index[0][index] = 1

    out.backward(torch.from_numpy(ohe_index).float().cuda())
  

    gradients = grads_[0]

    b, k, d, h, w = gradients.size()

    alpha = gradients.view(b, k, -1).mean(-1)
    weights = alpha.view(b, k, 1, 1, 1)

    saliency_map = (weights*maps).sum(1, keepdim=True)
    saliency_map = F.relu(saliency_map)
    saliency_map = F.interpolate(saliency_map, size=(dim[0], dim[1], dim[2]), mode='trilinear', align_corners=False).data.cpu().numpy()



    '''weights = grads_[0].squeeze().view(320,-1).mean(-1,keepdim=True) 
    #heatmap = resize(F.relu(weigh_maps(weights, maps)).data.cpu().numpy(), (dim[0], dim[1],dim[2]))
    heatmap = F.relu(weigh_maps(weights, maps))
  
    heatmap = F.upsample(heatmap[None,None,:], size=(dim[0], dim[1], dim[2]), mode='trilinear', align_corners=False).data.cpu().numpy()'''
       
    return np.squeeze(saliency_map)


# In[ ]:


def gradcam_MRD(model,img,dim):
    model.eval()
    features = model.nnunet.conv_blocks_context
    maps = features(Variable(img.cuda()))

    grads_ = []
    maps.register_hook(lambda x: grads_.append(x))
    
    #maps_avg = nn.AdaptiveAvgPool3d((1,1,1))(maps)
    maps_avg = maps.view(maps.size()[0], -1)

    out = model.fc1(maps_avg)

    index = np.argmax(out.cpu().data.numpy(), axis=1).reshape(-1, 1)

    ohe_index = out.cpu().data.numpy()*0
    ohe_index[0][index] = 1

    out.backward(torch.from_numpy(ohe_index).float().cuda())
  

    gradients = grads_[0]

    b, k, d, h, w = gradients.size()

    alpha = gradients.view(b, k, -1).mean(-1)
    weights = alpha.view(b, k, 1, 1, 1)

    saliency_map = (weights*maps).sum(1, keepdim=True)
    saliency_map = F.relu(saliency_map)
    saliency_map = F.interpolate(saliency_map, size=(dim[0], dim[1], dim[2]), mode='trilinear', align_corners=False).data.cpu().numpy()



    '''weights = grads_[0].squeeze().view(320,-1).mean(-1,keepdim=True) 
    #heatmap = resize(F.relu(weigh_maps(weights, maps)).data.cpu().numpy(), (dim[0], dim[1],dim[2]))
    heatmap = F.relu(weigh_maps(weights, maps))
  
    heatmap = F.upsample(heatmap[None,None,:], size=(dim[0], dim[1], dim[2]), mode='trilinear', align_corners=False).data.cpu().numpy()'''
       
    return np.squeeze(saliency_map)


# In[ ]:


def gradcam_pp(model,img,dim):
    
    features = model.nnunet.conv_blocks_context
    maps = features(Variable(img.cuda()))

    grads_ = []
    maps.register_hook(lambda x: grads_.append(x))
    
    maps_avg = nn.AdaptiveAvgPool3d((1,1,1))(maps)
    maps_avg = maps_avg.view(maps_avg.size()[0], -1)

    out = model.fc1(maps_avg)

    index = np.argmax(out.cpu().data.numpy(), axis=1).reshape(-1, 1)

    ohe_index = out.cpu().data.numpy()*0
    ohe_index[0][index] = 1

    out.backward(torch.from_numpy(ohe_index).float().cuda())

    gradients = grads_[0]

    b, k, d, h, w = gradients.size()

    alpha_num = gradients.pow(2)
    alpha_denom = gradients.pow(2).mul(2) + \
            maps.mul(gradients.pow(3)).view(b, k, d*h*w).sum(-1, keepdim=True).view(b, k, 1, 1, 1)
    alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

    alpha = alpha_num.div(alpha_denom+1e-7)
    score = out.cpu().detach().numpy()[0]
   
    positive_gradients = F.relu(math.exp(score[index][0][0])*gradients) # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
    weights = (alpha*positive_gradients).view(b, k, d*h*w).sum(-1).view(b, k, 1, 1, 1)

    saliency_map = (weights*maps).sum(1, keepdim=True)
    saliency_map = F.relu(saliency_map)
    saliency_map = F.interpolate(saliency_map, size=(dim[0], dim[1], dim[2]), mode='trilinear', align_corners=False).data.cpu().numpy()
    
    return np.squeeze(saliency_map)

