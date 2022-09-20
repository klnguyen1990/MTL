#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import matplotlib.pyplot as plt
import nrrd
from scipy.ndimage.interpolation import rotate
from preprocessing_Pytorch import load_image_and_label, preprocessing_image
from scipy.ndimage import label
from scipy.ndimage.measurements import center_of_mass
from tqdm import tqdm


# In[2]:


base_nrrd = '/home/nguyen-k/Bureau/segCassiopet/dcm_test'
max_SUV = 0.4


# In[ ]:


def nb_lesion_focale(p) :
    nb = 0
    if os.path.isfile(os.path.join(base_nrrd, p, 'majorityLabel1.nrrd')) : 
        list = os.listdir(os.path.join(base_nrrd, p))
        for i in list :
            if 'nrrd' in i :
                nb += 1
    return nb


# In[ ]:


def count_nb_lesion(patient, image_seg, dim, spacing) : 
    smooth = 1e-6
    nb_lesion = nb_lesion_focale(patient)
    nb_lesion_pred = 0
    _, _, positions, s_patient = load_image_and_label(patient)         
        
    for l in range(1, nb_lesion+1) :
        img_nrrd, _ = nrrd.read(os.path.join(base_nrrd, patient,'majorityLabel'+str(l)+'.nrrd')) 
        img_nrrd, _ = preprocessing_image(img=img_nrrd, positions=positions, s_patient=s_patient,flip=False,blur=False, 
                                            scale=(1, 1, 1), sigma=(1, 1, 1), dim=dim, spacing=spacing)
        intersection_ct = image_seg * img_nrrd
        if np.sum(intersection_ct)/(np.sum(img_nrrd)+smooth) > 0.5 :
            nb_lesion_pred +=1
            
    return nb_lesion, nb_lesion_pred


# In[3]:


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# In[4]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# In[5]:


def create_mip_from_3D(img_data, angle=0):

    img_data = np.array(img_data, dtype=np.float32)
    img_data+=1e-5    
    vol_angle= rotate(img_data,angle)

    mip=np.amax(vol_angle,axis=1)
    mip-=1e-5
    mip[mip<1e-5]=0
    mip=np.flipud(mip.T)

    return mip


# In[6]:


def plot_image_mip(img, angle, text, x, y, z,color='jet',alpha=1) : 
    mip = create_mip_from_3D(img, angle=angle)  
    ax = plt.subplot(x, y, z)
    plt.imshow(mip, cmap=color,alpha=alpha)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.title(text)
    
    return mip


# In[7]:


def plot_graphic(fold, nb_train, train_c1, train_c2, train_c3, nb_val, val_c1, val_c2, val_c3, dim, spacing, scale, sigma, nb_parametres, score_weight, 
                drop_encode, weight_decay, l1_fc1, l2_fc1, learning_rate, lr, patience, epoch, train_micro_f1_score, val_micro_f1_score, val_macro_f1_score, 
                val_weighted_f1_score, train_loss, val_loss, train_loss_class, val_loss_class, train_dice, val_dice, train_loss_seg, val_loss_seg, seed) :
    plt.subplot(2, 3, 1)
    plt.text(0, 0.1, ' DATA - FOLD ' + str(fold) + \
                    '\n Train data = ' + str(nb_train) + ' - C1 - C2 - C3 = ' + str(train_c1) + ' - ' + str(train_c2) + ' - ' + str(train_c3) + \
                    '\n Validation data = ' + str(nb_val) + ' - C1 - C2 - C3 = ' + str(val_c1) + ' - ' + str(val_c2) + ' - ' + str(val_c3) + \
                    '\n Image dimensions = ' + str(dim[0]) + ' x '+ str(dim[1]) + ' x ' + str(dim[2]) + ' - Spacing = ' + str(spacing) + \
                    '\n ' + \
                    '\n IMAGE GENERATOR' + \
                    '\n Scale = ' + str(scale[0]) + ' - ' + str(scale[1]) + \
                    '\n Gaussian blur = Random - Sigma = '  + str(sigma[0]) + ' - ' + str(sigma[1]) + \
                    '\n ' + \
                    '\n MODELE CLASSIFICATION - SEGMENTATION - RECONSTRUCTION' + \
                    '\n Nombre de paramètres trainables = ' + str(round(nb_parametres/1e6, 2)) + ' M' + \
                    '\n Score : Segmentation : Reconstruction = ' + str(score_weight) + ' : 1 : 1' + \
                    '\n Drop encode = ' + str(drop_encode) + \
                    '\n Optimiseur Adam - Weight decay = ' + str(weight_decay) + \
                    '\n Fc1 Regulateur : L1 = ' + str(l1_fc1) + ' - L2 = ' + str(l2_fc1) + \
                    '\n Learning rate = ' + str(learning_rate) + ' - Learning rate scheduler = ' + str(lr) + ' - Patience = ' + str(patience) + \
                    '\n Seed = ' + str(seed))

    plt.subplot(2, 3, 2)
    plt.plot(range(epoch),train_micro_f1_score[0:epoch], linestyle='solid', color='green', label='Train F1 score (Accuracy)')
    plt.plot(range(epoch), val_micro_f1_score[0:epoch], linestyle='solid', color='black', label='Val micro F1 score (Accuracy)')
    plt.plot(range(epoch), val_macro_f1_score[0:epoch], linestyle='dashed', color='black', label='Val macro F1 score')
    plt.plot(range(epoch), val_weighted_f1_score[0:epoch], linestyle='dotted', color='black', label='Val weighted F1 score')
    plt.legend(loc='upper left')
    plt.xlabel("Epoch")          
                                
    plt.subplot(2, 3, 4)
    plt.plot(range(epoch),train_loss[0:epoch]/nb_train,linestyle='solid', color='blue', label='Train Loss Total')
    plt.plot(range(epoch),val_loss[0:epoch]/nb_val, linestyle='solid',color='red', label='Val Loss Total')
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")  

    plt.subplot(2, 3, 5)    
    plt.plot(range(epoch),train_loss_class[0:epoch]/nb_train,linestyle='dotted', color='blue', label='Train Loss Class')
    plt.plot(range(epoch),val_loss_class[0:epoch]/nb_val, linestyle='dotted',color='red', label='Val Loss Class')  
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")  

    plt.subplot(2, 3, 3)    
    plt.plot(range(epoch),train_dice[0:epoch], linestyle='solid',color='blue', label='Train Dice')  
    plt.plot(range(epoch),val_dice[0:epoch], linestyle='solid',color='red', label='Val Dice')  
    plt.legend(loc='lower right')
    plt.xlabel("Epoch")  

    plt.subplot(2, 3, 6)    
    plt.plot(range(epoch),train_loss_seg[0:epoch]/nb_train,linestyle='dashed', color='blue', label='Train Loss Seg')
    plt.plot(range(epoch),val_loss_seg[0:epoch]/nb_val, linestyle='dashed',color='red', label='Val Loss Seg')  
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")  


# In[ ]:


def plot_graphic(fold, nb_train, train_c1, train_c2, train_c3, nb_val, val_c1, val_c2, val_c3, dim, spacing, scale, sigma, nb_parametres, score_weight, 
                drop_encode, weight_decay, l1_fc1, l2_fc1, learning_rate, lr, patience, epoch, train_micro_f1_score, val_micro_f1_score, val_macro_f1_score, 
                val_weighted_f1_score, train_loss, val_loss, train_loss_class, val_loss_class, train_dice, val_dice, train_loss_seg, val_loss_seg, seed) :
    plt.subplot(2, 3, 1)
    plt.text(0, 0.1, ' DATA - FOLD ' + str(fold) + \
                    '\n Train data = ' + str(nb_train) + ' - C1 - C2 - C3 = ' + str(train_c1) + ' - ' + str(train_c2) + ' - ' + str(train_c3) + \
                    '\n Validation data = ' + str(nb_val) + ' - C1 - C2 - C3 = ' + str(val_c1) + ' - ' + str(val_c2) + ' - ' + str(val_c3) + \
                    '\n Image dimensions = ' + str(dim[0]) + ' x '+ str(dim[1]) + ' x ' + str(dim[2]) + ' - Spacing = ' + str(spacing) + \
                    '\n ' + \
                    '\n IMAGE GENERATOR' + \
                    '\n Scale = ' + str(scale[0]) + ' - ' + str(scale[1]) + \
                    '\n Gaussian blur = Random - Sigma = '  + str(sigma[0]) + ' - ' + str(sigma[1]) + \
                    '\n ' + \
                    '\n MODELE CLASSIFICATION - SEGMENTATION - RECONSTRUCTION' + \
                    '\n Nombre de paramètres trainables = ' + str(round(nb_parametres/1e6, 2)) + ' M' + \
                    '\n Score : Segmentation : Reconstruction = ' + str(score_weight) + ' : 1 : 1' + \
                    '\n Drop encode = ' + str(drop_encode) + \
                    '\n Optimiseur Adam - Weight decay = ' + str(weight_decay) + \
                    '\n Fc1 Regulateur : L1 = ' + str(l1_fc1) + ' - L2 = ' + str(l2_fc1) + \
                    '\n Learning rate = ' + str(learning_rate) + ' - Learning rate scheduler = ' + str(lr) + ' - Patience = ' + str(patience) + \
                    '\n Seed = ' + str(seed))

    plt.subplot(2, 3, 2)
    plt.plot(range(epoch),train_micro_f1_score[0:epoch], linestyle='solid', color='green', label='Train F1 score (Accuracy)')
    plt.plot(range(epoch), val_micro_f1_score[0:epoch], linestyle='solid', color='black', label='Val micro F1 score (Accuracy)')
    plt.plot(range(epoch), val_macro_f1_score[0:epoch], linestyle='dashed', color='black', label='Val macro F1 score')
    plt.plot(range(epoch), val_weighted_f1_score[0:epoch], linestyle='dotted', color='black', label='Val weighted F1 score')
    plt.legend(loc='upper left')
    plt.xlabel("Epoch")          
                                
    plt.subplot(2, 3, 4)
    plt.plot(range(epoch),train_loss[0:epoch]/nb_train,linestyle='solid', color='blue', label='Train Loss Total')
    plt.plot(range(epoch),val_loss[0:epoch]/nb_val, linestyle='solid',color='red', label='Val Loss Total')
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")  

    plt.subplot(2, 3, 5)    
    plt.plot(range(epoch),train_loss_class[0:epoch]/nb_train,linestyle='dotted', color='blue', label='Train Loss Class')
    plt.plot(range(epoch),val_loss_class[0:epoch]/nb_val, linestyle='dotted',color='red', label='Val Loss Class')  
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")  

    plt.subplot(2, 3, 3)    
    plt.plot(range(epoch),train_dice[0:epoch], linestyle='solid',color='blue', label='Train Dice')  
    plt.plot(range(epoch),val_dice[0:epoch], linestyle='solid',color='red', label='Val Dice')  
    plt.legend(loc='lower right')
    plt.xlabel("Epoch")  

    plt.subplot(2, 3, 6)    
    plt.plot(range(epoch),train_loss_seg[0:epoch]/nb_train,linestyle='dashed', color='blue', label='Train Loss Seg')
    plt.plot(range(epoch),val_loss_seg[0:epoch]/nb_val, linestyle='dashed',color='red', label='Val Loss Seg')  
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")  


# In[ ]:


def plot_graphic_diffuse(fold, nb_train, train_c1, train_c2,train_c3, nb_val, val_c1, val_c2, val_c3, dim, spacing, scale, sigma, nb_parametres, score_weight, 
                drop_encode, weight_decay, l1_fc1, l2_fc1,l1_fc1_df, l2_fc1_df, learning_rate, lr, patience, epoch, train_micro_f1_score, val_micro_f1_score, 
                val_macro_f1_score, val_weighted_f1_score,train_micro_f1_score_df,val_micro_f1_score_df, val_macro_f1_score_df, val_weighted_f1_score_df,  
                train_loss, val_loss, train_loss_class, val_loss_class, train_loss_class_diffuse, val_loss_class_diffuse,
                train_dice, val_dice, train_loss_seg, val_loss_seg, seed) :
    ax = plt.subplot(3, 3, 1)
    plt.text(-0.05, 0.1, ' DATA - FOLD ' + str(fold) + \
                    '\n Train data = ' + str(nb_train) + ' - C1 - C2 - C3 = ' + str(train_c1) + ' - ' + str(train_c2) + ' - ' + str(train_c3) + \
                    '\n Validation data = ' + str(nb_val) + ' - C1 - C2 - C3 = ' + str(val_c1) + ' - ' + str(val_c2) + ' - ' + str(val_c3) + \
                    '\n Image dimensions = ' + str(dim[0]) + ' x '+ str(dim[1]) + ' x ' + str(dim[2]) + ' - Spacing = ' + str(spacing) + \
                    '\n ' + \
                    '\n IMAGE GENERATOR' + \
                    '\n Scale = ' + str(scale[0]) + ' - ' + str(scale[1]) + \
                    '\n Gaussian blur = Random - Sigma = '  + str(sigma[0]) + ' - ' + str(sigma[1]) + \
                    '\n ' + \
                    '\n MODELE CLASSIFICATION DV score - CLASSIFICATION Diffusion - SEGMENTATION - RECONSTRUCTION' + \
                    '\n Nombre de paramètres trainables = ' + str(round(nb_parametres/1e6, 2)) + ' M' + \
                    '\n Deauville : Diffision : Segmentation : Reconstruction = ' + str(score_weight) + ' : 1 : 1 : 1' + \
                    '\n Drop encode = ' + str(drop_encode) + \
                    '\n Optimiseur Adam - Weight decay = ' + str(weight_decay) + \
                    '\n Fc1 Regulateur : L1 = ' + str(l1_fc1) + ' - L2 = ' + str(l2_fc1) + \
                    '\n Fc1 Regulateur df : L1 = ' + str(l1_fc1_df) + ' - L2 = ' + str(l2_fc1_df) + \
                    '\n Learning rate = ' + str(learning_rate) + ' - Learning rate scheduler = ' + str(lr) + ' - Patience = ' + str(patience) + \
                    '\n Seed = ' + str(seed))
    ax.spines['right'].set_visible(False) 
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False, right=False,
        labelbottom=False,
        labeltop=False, labelleft=False, labelright=False) # labels along the bottom edge are off

    ax.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False, right=False,
        labelbottom=False,
        labeltop=False, labelleft=False, labelright=False) # labels along the bottom edge are off

    plt.subplot(3, 3, 2)
    plt.title('Deauville')
    plt.plot(range(epoch),train_micro_f1_score[0:epoch], linestyle='solid', color='green', label='Train F1 score (Accuracy)')
    plt.plot(range(epoch), val_micro_f1_score[0:epoch], linestyle='solid', color='black', label='Val micro F1 score (Accuracy)')
    plt.plot(range(epoch), val_macro_f1_score[0:epoch], linestyle='dashed', color='black', label='Val macro F1 score')
    plt.plot(range(epoch), val_weighted_f1_score[0:epoch], linestyle='dotted', color='black', label='Val weighted F1 score')
    plt.legend(loc='upper left')
    plt.xlabel("Epoch")  

    plt.subplot(3, 3, 3)
    plt.title('Diffuse sans LF')
    plt.plot(range(epoch),train_micro_f1_score_df[0:epoch], linestyle='solid', color='green', label='Train F1 score (Accuracy)')
    plt.plot(range(epoch), val_micro_f1_score_df[0:epoch], linestyle='solid', color='black', label='Val micro F1 score (Accuracy)')
    plt.plot(range(epoch), val_macro_f1_score_df[0:epoch], linestyle='dashed', color='black', label='Val macro F1 score')
    plt.plot(range(epoch), val_weighted_f1_score_df[0:epoch], linestyle='dotted', color='black', label='Val weighted F1 score')
    plt.legend(loc='upper left')
    plt.xlabel("Epoch")           
                                
    plt.subplot(3, 3, 4)
    plt.plot(range(epoch),train_loss[0:epoch]/nb_train,linestyle='solid', color='blue', label='Train Loss Total')
    plt.plot(range(epoch),val_loss[0:epoch]/nb_val, linestyle='solid',color='red', label='Val Loss Total')
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")  

    plt.subplot(3, 3, 5)    
    plt.plot(range(epoch),train_loss_class[0:epoch]/nb_train,linestyle='dotted', color='blue', label='Train Loss Class')
    plt.plot(range(epoch),val_loss_class[0:epoch]/nb_val, linestyle='dotted',color='red', label='Val Loss Class')  
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")  

    plt.subplot(3, 3, 6)    
    plt.plot(range(epoch),train_dice[0:epoch], linestyle='solid',color='blue', label='Train Dice')  
    plt.plot(range(epoch),val_dice[0:epoch], linestyle='solid',color='red', label='Val Dice')  
    plt.legend(loc='lower right')
    plt.xlabel("Epoch")  

    plt.subplot(3, 3, 7)    
    plt.plot(range(epoch),train_loss_seg[0:epoch]/nb_train,linestyle='dashed', color='blue', label='Train Loss Seg')
    plt.plot(range(epoch),val_loss_seg[0:epoch]/nb_val, linestyle='dashed',color='red', label='Val Loss Seg')  
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")  

    plt.subplot(3, 3, 8)    
    plt.plot(range(epoch),train_loss_class_diffuse[0:epoch]/nb_train,linestyle='dashed', color='blue', label='Train Loss diffuse')
    plt.plot(range(epoch),val_loss_class_diffuse[0:epoch]/nb_val, linestyle='dashed',color='red', label='Val Loss diffuse')  
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")  

    ax = plt.subplot(3, 3, 9)  
    ax.spines['right'].set_visible(False) 
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False, right=False,
        labelbottom=False,
        labeltop=False, labelleft=False, labelright=False) # labels along the bottom edge are off

    ax.tick_params(
        axis='y',          # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False, right=False,
        labelbottom=False,
        labeltop=False, labelleft=False, labelright=False) # labels along the bottom edge are off


# In[ ]:


def plot_graphic_MRD_diffuse(fold, nb_train, train_c1, train_c2, nb_val, val_c1, val_c2, dim, spacing, scale, sigma, nb_parametres, score_weight, 
                drop_encode, weight_decay, l1_fc1, l2_fc1,l1_fc1_df, l2_fc1_df, learning_rate, lr, patience, epoch, train_micro_f1_score, val_micro_f1_score, 
                val_macro_f1_score, val_weighted_f1_score,train_micro_f1_score_df,val_micro_f1_score_df, val_macro_f1_score_df, val_weighted_f1_score_df,  
                train_loss, val_loss, train_loss_class, val_loss_class, train_loss_class_diffuse, val_loss_class_diffuse,
                train_dice, val_dice, train_loss_seg, val_loss_seg, seed) :
    ax = plt.subplot(3, 3, 1)
    plt.text(-0.05, 0.1, ' DATA - FOLD ' + str(fold) + \
                    '\n Train data = ' + str(nb_train) + ' - C1 - C2 = ' + str(train_c1) + ' - ' + str(train_c2) + \
                    '\n Validation data = ' + str(nb_val) + ' - C1 - C2 = ' + str(val_c1) + ' - ' + str(val_c2) + \
                    '\n Image dimensions = ' + str(dim[0]) + ' x '+ str(dim[1]) + ' x ' + str(dim[2]) + ' - Spacing = ' + str(spacing) + \
                    '\n ' + \
                    '\n IMAGE GENERATOR' + \
                    '\n Scale = ' + str(scale[0]) + ' - ' + str(scale[1]) + \
                    '\n Gaussian blur = Random - Sigma = '  + str(sigma[0]) + ' - ' + str(sigma[1]) + \
                    '\n ' + \
                    '\n MODELE CLASSIFICATION MRD score - CLASSIFICATION Diffusion - SEGMENTATION - RECONSTRUCTION' + \
                    '\n Nombre de paramètres trainables = ' + str(round(nb_parametres/1e6, 2)) + ' M' + \
                    '\n Deauville : Diffision : Segmentation : Reconstruction = ' + str(score_weight) + ' : 1 : 1 : 1' + \
                    '\n Drop encode = ' + str(drop_encode) + \
                    '\n Optimiseur Adam - Weight decay = ' + str(weight_decay) + \
                    '\n Fc1 Regulateur : L1 = ' + str(l1_fc1) + ' - L2 = ' + str(l2_fc1) + \
                    '\n Fc1 Regulateur df : L1 = ' + str(l1_fc1_df) + ' - L2 = ' + str(l2_fc1_df) + \
                    '\n Learning rate = ' + str(learning_rate) + ' - Learning rate scheduler = ' + str(lr) + ' - Patience = ' + str(patience) + \
                    '\n Seed = ' + str(seed))
    ax.spines['right'].set_visible(False) 
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False, right=False,
        labelbottom=False,
        labeltop=False, labelleft=False, labelright=False) # labels along the bottom edge are off

    ax.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False, right=False,
        labelbottom=False,
        labeltop=False, labelleft=False, labelright=False) # labels along the bottom edge are off

    plt.subplot(3, 3, 2)
    plt.title('MRD')
    plt.plot(range(epoch),train_micro_f1_score[0:epoch], linestyle='solid', color='green', label='Train F1 score (Accuracy)')
    plt.plot(range(epoch), val_micro_f1_score[0:epoch], linestyle='solid', color='black', label='Val micro F1 score (Accuracy)')
    plt.plot(range(epoch), val_macro_f1_score[0:epoch], linestyle='dashed', color='black', label='Val macro F1 score')
    plt.plot(range(epoch), val_weighted_f1_score[0:epoch], linestyle='dotted', color='black', label='Val weighted F1 score')
    plt.legend(loc='upper left')
    plt.xlabel("Epoch")  

    plt.subplot(3, 3, 3)
    plt.title('Diffuse')
    plt.plot(range(epoch),train_micro_f1_score_df[0:epoch], linestyle='solid', color='green', label='Train F1 score (Accuracy)')
    plt.plot(range(epoch), val_micro_f1_score_df[0:epoch], linestyle='solid', color='black', label='Val micro F1 score (Accuracy)')
    plt.plot(range(epoch), val_macro_f1_score_df[0:epoch], linestyle='dashed', color='black', label='Val macro F1 score')
    plt.plot(range(epoch), val_weighted_f1_score_df[0:epoch], linestyle='dotted', color='black', label='Val weighted F1 score')
    plt.legend(loc='upper left')
    plt.xlabel("Epoch")           
                                
    plt.subplot(3, 3, 4)
    plt.plot(range(epoch),train_loss[0:epoch]/nb_train,linestyle='solid', color='blue', label='Train Loss Total')
    plt.plot(range(epoch),val_loss[0:epoch]/nb_val, linestyle='solid',color='red', label='Val Loss Total')
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")  

    plt.subplot(3, 3, 5)    
    plt.plot(range(epoch),train_loss_class[0:epoch]/nb_train,linestyle='dotted', color='blue', label='Train Loss Class')
    plt.plot(range(epoch),val_loss_class[0:epoch]/nb_val, linestyle='dotted',color='red', label='Val Loss Class')  
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")  

    plt.subplot(3, 3, 6)    
    plt.plot(range(epoch),train_dice[0:epoch], linestyle='solid',color='blue', label='Train Dice')  
    plt.plot(range(epoch),val_dice[0:epoch], linestyle='solid',color='red', label='Val Dice')  
    plt.legend(loc='lower right')
    plt.xlabel("Epoch")  

    plt.subplot(3, 3, 7)    
    plt.plot(range(epoch),train_loss_seg[0:epoch]/nb_train,linestyle='dashed', color='blue', label='Train Loss Seg')
    plt.plot(range(epoch),val_loss_seg[0:epoch]/nb_val, linestyle='dashed',color='red', label='Val Loss Seg')  
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")  

    plt.subplot(3, 3, 8)    
    plt.plot(range(epoch),train_loss_class_diffuse[0:epoch]/nb_train,linestyle='dashed', color='blue', label='Train Loss diffuse')
    plt.plot(range(epoch),val_loss_class_diffuse[0:epoch]/nb_val, linestyle='dashed',color='red', label='Val Loss diffuse')  
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")  

    ax = plt.subplot(3, 3, 9)  
    ax.spines['right'].set_visible(False) 
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False, right=False,
        labelbottom=False,
        labeltop=False, labelleft=False, labelright=False) # labels along the bottom edge are off

    ax.tick_params(
        axis='y',          # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False, right=False,
        labelbottom=False,
        labeltop=False, labelleft=False, labelright=False) # labels along the bottom edge are off


# In[ ]:


def plot_graphic_diffuse_noSeg(fold, nb_train, train_c1, train_c2,train_c3, nb_val, val_c1, val_c2, val_c3, dim, spacing, scale, sigma, nb_parametres, score_weight, 
                drop_encode, weight_decay, l1_fc1, l2_fc1,l1_fc1_df, l2_fc1_df, learning_rate, lr, patience, epoch, train_micro_f1_score, val_micro_f1_score, 
                val_macro_f1_score, val_weighted_f1_score,train_micro_f1_score_df,val_micro_f1_score_df, val_macro_f1_score_df, val_weighted_f1_score_df,  
                train_loss, val_loss, train_loss_class, val_loss_class, train_loss_class_diffuse, val_loss_class_diffuse,seed) :
    ax = plt.subplot(2, 3, 1)
    plt.text(-0.05, 0.1, ' DATA - FOLD ' + str(fold) + \
                    '\n Train data = ' + str(nb_train) + ' - C1 - C2 - C3 = ' + str(train_c1) + ' - ' + str(train_c2) + ' - ' + str(train_c3) + \
                    '\n Validation data = ' + str(nb_val) + ' - C1 - C2 - C3 = ' + str(val_c1) + ' - ' + str(val_c2) + ' - ' + str(val_c3) + \
                    '\n Image dimensions = ' + str(dim[0]) + ' x '+ str(dim[1]) + ' x ' + str(dim[2]) + ' - Spacing = ' + str(spacing) + \
                    '\n ' + \
                    '\n IMAGE GENERATOR' + \
                    '\n Scale = ' + str(scale[0]) + ' - ' + str(scale[1]) + \
                    '\n Gaussian blur = Random - Sigma = '  + str(sigma[0]) + ' - ' + str(sigma[1]) + \
                    '\n ' + \
                    '\n MODELE CLASSIFICATION DV score - CLASSIFICATION Diffusion - SEGMENTATION - RECONSTRUCTION' + \
                    '\n Nombre de paramètres trainables = ' + str(round(nb_parametres/1e6, 2)) + ' M' + \
                    '\n Deauville : Diffision : Reconstruction = ' + str(score_weight) + ' : 1 : 1' + \
                    '\n Drop encode = ' + str(drop_encode) + \
                    '\n Optimiseur Adam - Weight decay = ' + str(weight_decay) + \
                    '\n Fc1 Regulateur : L1 = ' + str(l1_fc1) + ' - L2 = ' + str(l2_fc1) + \
                    '\n Fc1 Regulateur df : L1 = ' + str(l1_fc1_df) + ' - L2 = ' + str(l2_fc1_df) + \
                    '\n Learning rate = ' + str(learning_rate) + ' - Learning rate scheduler = ' + str(lr) + ' - Patience = ' + str(patience) + \
                    '\n Seed = ' + str(seed))
    ax.spines['right'].set_visible(False) 
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False, right=False,
        labelbottom=False,
        labeltop=False, labelleft=False, labelright=False) # labels along the bottom edge are off

    ax.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False, right=False,
        labelbottom=False,
        labeltop=False, labelleft=False, labelright=False) # labels along the bottom edge are off

    plt.subplot(2, 3, 2)
    plt.title('Deauville')
    plt.plot(range(epoch),train_micro_f1_score[0:epoch], linestyle='solid', color='green', label='Train F1 score (Accuracy)')
    plt.plot(range(epoch), val_micro_f1_score[0:epoch], linestyle='solid', color='black', label='Val micro F1 score (Accuracy)')
    plt.plot(range(epoch), val_macro_f1_score[0:epoch], linestyle='dashed', color='black', label='Val macro F1 score')
    plt.plot(range(epoch), val_weighted_f1_score[0:epoch], linestyle='dotted', color='black', label='Val weighted F1 score')
    plt.legend(loc='upper left')
    plt.xlabel("Epoch")  

    plt.subplot(2, 3, 3)
    plt.title('Diffuse sans LF')
    plt.plot(range(epoch),train_micro_f1_score_df[0:epoch], linestyle='solid', color='green', label='Train F1 score (Accuracy)')
    plt.plot(range(epoch), val_micro_f1_score_df[0:epoch], linestyle='solid', color='black', label='Val micro F1 score (Accuracy)')
    plt.plot(range(epoch), val_macro_f1_score_df[0:epoch], linestyle='dashed', color='black', label='Val macro F1 score')
    plt.plot(range(epoch), val_weighted_f1_score_df[0:epoch], linestyle='dotted', color='black', label='Val weighted F1 score')
    plt.legend(loc='upper left')
    plt.xlabel("Epoch")           
                                
    plt.subplot(2, 3, 4)
    plt.plot(range(epoch),train_loss[0:epoch]/nb_train,linestyle='solid', color='blue', label='Train Loss Total')
    plt.plot(range(epoch),val_loss[0:epoch]/nb_val, linestyle='solid',color='red', label='Val Loss Total')
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")  

    plt.subplot(2, 3, 5)    
    plt.plot(range(epoch),train_loss_class[0:epoch]/nb_train,linestyle='dotted', color='blue', label='Train Loss Class')
    plt.plot(range(epoch),val_loss_class[0:epoch]/nb_val, linestyle='dotted',color='red', label='Val Loss Class')  
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")  

    plt.subplot(2, 3, 6)    
    plt.plot(range(epoch),train_loss_class_diffuse[0:epoch]/nb_train,linestyle='dashed', color='blue', label='Train Loss diffuse')
    plt.plot(range(epoch),val_loss_class_diffuse[0:epoch]/nb_val, linestyle='dashed',color='red', label='Val Loss diffuse')  
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")  


# In[ ]:


def plot_graphic_MRD_diffuse_noSeg(fold, nb_train, train_c1, train_c2, nb_val, val_c1, val_c2, dim, spacing, scale, sigma, nb_parametres, score_weight, 
                drop_encode, weight_decay, l1_fc1, l2_fc1,l1_fc1_df, l2_fc1_df, learning_rate, lr, patience, epoch, train_micro_f1_score, val_micro_f1_score, 
                val_macro_f1_score, val_weighted_f1_score,train_micro_f1_score_df,val_micro_f1_score_df, val_macro_f1_score_df, val_weighted_f1_score_df,  
                train_loss, val_loss, train_loss_class, val_loss_class, train_loss_class_diffuse, val_loss_class_diffuse,seed) :
    ax = plt.subplot(2, 3, 1)
    plt.text(-0.05, 0.1, ' DATA - FOLD ' + str(fold) + \
                    '\n Train data = ' + str(nb_train) + ' - C1 - C2 = ' + str(train_c1) + ' - ' + str(train_c2) + \
                    '\n Validation data = ' + str(nb_val) + ' - C1 - C2 = ' + str(val_c1) + ' - ' + str(val_c2) + \
                    '\n Image dimensions = ' + str(dim[0]) + ' x '+ str(dim[1]) + ' x ' + str(dim[2]) + ' - Spacing = ' + str(spacing) + \
                    '\n ' + \
                    '\n IMAGE GENERATOR' + \
                    '\n Scale = ' + str(scale[0]) + ' - ' + str(scale[1]) + \
                    '\n Gaussian blur = Random - Sigma = '  + str(sigma[0]) + ' - ' + str(sigma[1]) + \
                    '\n ' + \
                    '\n MODELE CLASSIFICATION DV score - CLASSIFICATION Diffusion - RECONSTRUCTION' + \
                    '\n Nombre de paramètres trainables = ' + str(round(nb_parametres/1e6, 2)) + ' M' + \
                    '\n Deauville : Diffision : Reconstruction = ' + str(score_weight) + ' : 1 : 1' + \
                    '\n Drop encode = ' + str(drop_encode) + \
                    '\n Optimiseur Adam - Weight decay = ' + str(weight_decay) + \
                    '\n Fc1 Regulateur : L1 = ' + str(l1_fc1) + ' - L2 = ' + str(l2_fc1) + \
                    '\n Fc1 Regulateur df : L1 = ' + str(l1_fc1_df) + ' - L2 = ' + str(l2_fc1_df) + \
                    '\n Learning rate = ' + str(learning_rate) + ' - Learning rate scheduler = ' + str(lr) + ' - Patience = ' + str(patience) + \
                    '\n Seed = ' + str(seed))
    ax.spines['right'].set_visible(False) 
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False, right=False,
        labelbottom=False,
        labeltop=False, labelleft=False, labelright=False) # labels along the bottom edge are off

    ax.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False, right=False,
        labelbottom=False,
        labeltop=False, labelleft=False, labelright=False) # labels along the bottom edge are off

    plt.subplot(2, 3, 2)
    plt.title('Deauville')
    plt.plot(range(epoch),train_micro_f1_score[0:epoch], linestyle='solid', color='green', label='Train F1 score (Accuracy)')
    plt.plot(range(epoch), val_micro_f1_score[0:epoch], linestyle='solid', color='black', label='Val micro F1 score (Accuracy)')
    plt.plot(range(epoch), val_macro_f1_score[0:epoch], linestyle='dashed', color='black', label='Val macro F1 score')
    plt.plot(range(epoch), val_weighted_f1_score[0:epoch], linestyle='dotted', color='black', label='Val weighted F1 score')
    plt.legend(loc='upper left')
    plt.xlabel("Epoch")  

    plt.subplot(2, 3, 3)
    plt.title('Diffuse sans LF')
    plt.plot(range(epoch),train_micro_f1_score_df[0:epoch], linestyle='solid', color='green', label='Train F1 score (Accuracy)')
    plt.plot(range(epoch), val_micro_f1_score_df[0:epoch], linestyle='solid', color='black', label='Val micro F1 score (Accuracy)')
    plt.plot(range(epoch), val_macro_f1_score_df[0:epoch], linestyle='dashed', color='black', label='Val macro F1 score')
    plt.plot(range(epoch), val_weighted_f1_score_df[0:epoch], linestyle='dotted', color='black', label='Val weighted F1 score')
    plt.legend(loc='upper left')
    plt.xlabel("Epoch")           
                                
    plt.subplot(2, 3, 4)
    plt.plot(range(epoch),train_loss[0:epoch]/nb_train,linestyle='solid', color='blue', label='Train Loss Total')
    plt.plot(range(epoch),val_loss[0:epoch]/nb_val, linestyle='solid',color='red', label='Val Loss Total')
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")  

    plt.subplot(2, 3, 5)    
    plt.plot(range(epoch),train_loss_class[0:epoch]/nb_train,linestyle='dotted', color='blue', label='Train Loss Class')
    plt.plot(range(epoch),val_loss_class[0:epoch]/nb_val, linestyle='dotted',color='red', label='Val Loss Class')  
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")  

    plt.subplot(2, 3, 6)    
    plt.plot(range(epoch),train_loss_class_diffuse[0:epoch]/nb_train,linestyle='dashed', color='blue', label='Train Loss diffuse')
    plt.plot(range(epoch),val_loss_class_diffuse[0:epoch]/nb_val, linestyle='dashed',color='red', label='Val Loss diffuse')  
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")  


# In[ ]:


def plot_graphic_NoDF_noSeg(fold, nb_train, train_c1, train_c2,train_c3, nb_val, val_c1, val_c2, val_c3, dim, spacing, scale, sigma, nb_parametres, score_weight, 
                drop_encode, weight_decay, l1_fc1, l2_fc1, learning_rate, lr, patience, epoch, train_micro_f1_score, val_micro_f1_score, 
                val_macro_f1_score, val_weighted_f1_score, train_loss, val_loss, train_loss_class, val_loss_class,seed) :
    ax = plt.subplot(2, 3, 1)
    plt.text(-0.05, 0.1, ' DATA - FOLD ' + str(fold) + \
                    '\n Train data = ' + str(nb_train) + ' - C1 - C2 - C3 = ' + str(train_c1) + ' - ' + str(train_c2) + ' - ' + str(train_c3) + \
                    '\n Validation data = ' + str(nb_val) + ' - C1 - C2 - C3 = ' + str(val_c1) + ' - ' + str(val_c2) + ' - ' + str(val_c3) + \
                    '\n Image dimensions = ' + str(dim[0]) + ' x '+ str(dim[1]) + ' x ' + str(dim[2]) + ' - Spacing = ' + str(spacing) + \
                    '\n ' + \
                    '\n IMAGE GENERATOR' + \
                    '\n Scale = ' + str(scale[0]) + ' - ' + str(scale[1]) + \
                    '\n Gaussian blur = Random - Sigma = '  + str(sigma[0]) + ' - ' + str(sigma[1]) + \
                    '\n ' + \
                    '\n MODELE CLASSIFICATION DV score - SEGMENTATION - RECONSTRUCTION' + \
                    '\n Nombre de paramètres trainables = ' + str(round(nb_parametres/1e6, 2)) + ' M' + \
                    '\n Deauville : Reconstruction = ' + str(score_weight) + ' : 1' + \
                    '\n Drop encode = ' + str(drop_encode) + \
                    '\n Optimiseur Adam - Weight decay = ' + str(weight_decay) + \
                    '\n Fc1 Regulateur : L1 = ' + str(l1_fc1) + ' - L2 = ' + str(l2_fc1) + \
                    '\n Learning rate = ' + str(learning_rate) + ' - Learning rate scheduler = ' + str(lr) + ' - Patience = ' + str(patience) + \
                    '\n Seed = ' + str(seed))
    ax.spines['right'].set_visible(False) 
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False, right=False,
        labelbottom=False,
        labeltop=False, labelleft=False, labelright=False) # labels along the bottom edge are off

    ax.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False, right=False,
        labelbottom=False,
        labeltop=False, labelleft=False, labelright=False) # labels along the bottom edge are off

    plt.subplot(2, 3, 2)
    plt.title('Deauville')
    plt.plot(range(epoch),train_micro_f1_score[0:epoch], linestyle='solid', color='green', label='Train F1 score (Accuracy)')
    plt.plot(range(epoch), val_micro_f1_score[0:epoch], linestyle='solid', color='black', label='Val micro F1 score (Accuracy)')
    plt.plot(range(epoch), val_macro_f1_score[0:epoch], linestyle='dashed', color='black', label='Val macro F1 score')
    plt.plot(range(epoch), val_weighted_f1_score[0:epoch], linestyle='dotted', color='black', label='Val weighted F1 score')
    plt.legend(loc='upper left')
    plt.xlabel("Epoch")     
                                
    plt.subplot(2, 3, 3)
    plt.plot(range(epoch),train_loss[0:epoch]/nb_train,linestyle='solid', color='blue', label='Train Loss Total')
    plt.plot(range(epoch),val_loss[0:epoch]/nb_val, linestyle='solid',color='red', label='Val Loss Total')
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")  

    plt.subplot(2, 3, 5)    
    plt.plot(range(epoch),train_loss_class[0:epoch]/nb_train,linestyle='dotted', color='blue', label='Train Loss Class')
    plt.plot(range(epoch),val_loss_class[0:epoch]/nb_val, linestyle='dotted',color='red', label='Val Loss Class')  
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")  

    ax = plt.subplot(2, 3, 4)
    ax.spines['right'].set_visible(False) 
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False, right=False,
        labelbottom=False,
        labeltop=False, labelleft=False, labelright=False) # labels along the bottom edge are off

    ax.tick_params(
        axis='y',          # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False, right=False,
        labelbottom=False,
        labeltop=False, labelleft=False, labelright=False) # labels along the bottom edge are off

    ax = plt.subplot(2, 3, 6)
    ax.spines['right'].set_visible(False) 
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False, right=False,
        labelbottom=False,
        labeltop=False, labelleft=False, labelright=False) # labels along the bottom edge are off

    ax.tick_params(
        axis='y',          # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False, right=False,
        labelbottom=False,
        labeltop=False, labelleft=False, labelright=False) # labels along the bottom edge are off


# In[ ]:


def plot_graphic_Encodeur(fold, nb_train, train_c1, train_c2,train_c3, nb_val, val_c1, val_c2, val_c3, dim, spacing, scale, sigma, nb_parametres, 
                drop_encode, weight_decay, l1_fc1, l2_fc1, learning_rate, lr, patience, epoch, train_micro_f1_score, val_micro_f1_score, 
                val_macro_f1_score, val_weighted_f1_score, train_loss, val_loss,seed) :
    ax = plt.subplot(1, 3, 1)
    plt.text(-0.05, 0.1, ' DATA - FOLD ' + str(fold) + \
                    '\n Train data = ' + str(nb_train) + ' - C1 - C2 - C3 = ' + str(train_c1) + ' - ' + str(train_c2) + ' - ' + str(train_c3) + \
                    '\n Validation data = ' + str(nb_val) + ' - C1 - C2 - C3 = ' + str(val_c1) + ' - ' + str(val_c2) + ' - ' + str(val_c3) + \
                    '\n Image dimensions = ' + str(dim[0]) + ' x '+ str(dim[1]) + ' x ' + str(dim[2]) + ' - Spacing = ' + str(spacing) + \
                    '\n ' + \
                    '\n IMAGE GENERATOR' + \
                    '\n Scale = ' + str(scale[0]) + ' - ' + str(scale[1]) + \
                    '\n Gaussian blur = Random - Sigma = '  + str(sigma[0]) + ' - ' + str(sigma[1]) + \
                    '\n ' + \
                    '\n MODELE CLASSIFICATION DV score - SEGMENTATION - RECONSTRUCTION' + \
                    '\n Nombre de paramètres trainables = ' + str(round(nb_parametres/1e6, 2)) + ' M' + \
                    '\n Deauville : Reconstruction' + \
                    '\n Drop encode = ' + str(drop_encode) + \
                    '\n Optimiseur Adam - Weight decay = ' + str(weight_decay) + \
                    '\n Fc1 Regulateur : L1 = ' + str(l1_fc1) + ' - L2 = ' + str(l2_fc1) + \
                    '\n Learning rate = ' + str(learning_rate) + ' - Learning rate scheduler = ' + str(lr) + ' - Patience = ' + str(patience) + \
                    '\n Seed = ' + str(seed))
    ax.spines['right'].set_visible(False) 
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False, right=False,
        labelbottom=False,
        labeltop=False, labelleft=False, labelright=False) # labels along the bottom edge are off

    ax.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False, right=False,
        labelbottom=False,
        labeltop=False, labelleft=False, labelright=False) # labels along the bottom edge are off

    plt.subplot(1, 3, 2)
    plt.title('Deauville')
    plt.plot(range(epoch),train_micro_f1_score[0:epoch], linestyle='solid', color='green', label='Train F1 score (Accuracy)')
    plt.plot(range(epoch), val_micro_f1_score[0:epoch], linestyle='solid', color='black', label='Val micro F1 score (Accuracy)')
    plt.plot(range(epoch), val_macro_f1_score[0:epoch], linestyle='dashed', color='black', label='Val macro F1 score')
    plt.plot(range(epoch), val_weighted_f1_score[0:epoch], linestyle='dotted', color='black', label='Val weighted F1 score')
    plt.legend(loc='upper left')
    plt.xlabel("Epoch")     
                                
    plt.subplot(1, 3, 3)
    plt.plot(range(epoch),train_loss[0:epoch]/nb_train,linestyle='solid', color='blue', label='Train Loss Total')
    plt.plot(range(epoch),val_loss[0:epoch]/nb_val, linestyle='solid',color='red', label='Val Loss Total')
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")  


# In[ ]:


def plot_graphic_efficient_net(fold, nb_train, train_c1, train_c2,train_c3, nb_val, val_c1, val_c2, val_c3, dim, spacing, scale, sigma, nb_parametres, weight_decay, learning_rate, lr, patience, epoch, train_micro_f1_score, val_micro_f1_score, 
                val_macro_f1_score, val_weighted_f1_score, train_loss, val_loss,seed) :
    ax = plt.subplot(1, 3, 1)
    plt.text(-0.05, 0.1, ' DATA - FOLD ' + str(fold) + \
                    '\n Train data = ' + str(nb_train) + ' - C1 - C2 - C3 = ' + str(train_c1) + ' - ' + str(train_c2) + ' - ' + str(train_c3) + \
                    '\n Validation data = ' + str(nb_val) + ' - C1 - C2 - C3 = ' + str(val_c1) + ' - ' + str(val_c2) + ' - ' + str(val_c3) + \
                    '\n Image dimensions = ' + str(dim[0]) + ' x '+ str(dim[1]) + ' x ' + str(dim[2]) + ' - Spacing = ' + str(spacing) + \
                    '\n ' + \
                    '\n IMAGE GENERATOR' + \
                    '\n Scale = ' + str(scale[0]) + ' - ' + str(scale[1]) + \
                    '\n Gaussian blur = Random - Sigma = '  + str(sigma[0]) + ' - ' + str(sigma[1]) + \
                    '\n ' + \
                    '\n MODELE CLASSIFICATION DV score' + \
                    '\n Nombre de paramètres trainables = ' + str(round(nb_parametres/1e6, 2)) + ' M' + \
                    '\n Optimiseur Adam - Weight decay = ' + str(weight_decay) + \
                    '\n Learning rate = ' + str(learning_rate) + ' - Learning rate scheduler = ' + str(lr) + ' - Patience = ' + str(patience) + \
                    '\n Seed = ' + str(seed))
    ax.spines['right'].set_visible(False) 
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False, right=False,
        labelbottom=False,
        labeltop=False, labelleft=False, labelright=False) # labels along the bottom edge are off

    ax.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False, right=False,
        labelbottom=False,
        labeltop=False, labelleft=False, labelright=False) # labels along the bottom edge are off

    plt.subplot(1, 3, 2)
    plt.title('Deauville')
    plt.plot(range(epoch),train_micro_f1_score[0:epoch], linestyle='solid', color='green', label='Train F1 score (Accuracy)')
    plt.plot(range(epoch), val_micro_f1_score[0:epoch], linestyle='solid', color='black', label='Val micro F1 score (Accuracy)')
    plt.plot(range(epoch), val_macro_f1_score[0:epoch], linestyle='dashed', color='black', label='Val macro F1 score')
    plt.plot(range(epoch), val_weighted_f1_score[0:epoch], linestyle='dotted', color='black', label='Val weighted F1 score')
    plt.legend(loc='upper left')
    plt.xlabel("Epoch")     
                                
    plt.subplot(1, 3, 3)
    plt.plot(range(epoch),train_loss[0:epoch]/nb_train,linestyle='solid', color='blue', label='Train Loss Total')
    plt.plot(range(epoch),val_loss[0:epoch]/nb_val, linestyle='solid',color='red', label='Val Loss Total')
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")  


# In[ ]:


def plot_graphic_diffuse(fold, nb_train, train_c1, train_c2,train_c3, nb_val, val_c1, val_c2, val_c3, dim, spacing, scale, sigma, nb_parametres, score_weight, 
                drop_encode, weight_decay, l1_fc1, l2_fc1,l1_fc1_df, l2_fc1_df, learning_rate, lr, patience, epoch, train_micro_f1_score, val_micro_f1_score, 
                val_macro_f1_score, val_weighted_f1_score,train_micro_f1_score_df,val_micro_f1_score_df, val_macro_f1_score_df, val_weighted_f1_score_df,  
                train_loss, val_loss, train_loss_class, val_loss_class, train_loss_class_diffuse, val_loss_class_diffuse,
                train_dice, val_dice, train_loss_seg, val_loss_seg, seed) :
    ax = plt.subplot(3, 3, 1)
    plt.text(-0.05, 0.1, ' DATA - FOLD ' + str(fold) + \
                    '\n Train data = ' + str(nb_train) + ' - C1 - C2 - C3 = ' + str(train_c1) + ' - ' + str(train_c2) + ' - ' + str(train_c3) + \
                    '\n Validation data = ' + str(nb_val) + ' - C1 - C2 - C3 = ' + str(val_c1) + ' - ' + str(val_c2) + ' - ' + str(val_c3) + \
                    '\n Image dimensions = ' + str(dim[0]) + ' x '+ str(dim[1]) + ' x ' + str(dim[2]) + ' - Spacing = ' + str(spacing) + \
                    '\n ' + \
                    '\n IMAGE GENERATOR' + \
                    '\n Scale = ' + str(scale[0]) + ' - ' + str(scale[1]) + \
                    '\n Gaussian blur = Random - Sigma = '  + str(sigma[0]) + ' - ' + str(sigma[1]) + \
                    '\n ' + \
                    '\n MODELE CLASSIFICATION DV score - CLASSIFICATION Diffusion - SEGMENTATION - RECONSTRUCTION' + \
                    '\n Nombre de paramètres trainables = ' + str(round(nb_parametres/1e6, 2)) + ' M' + \
                    '\n Deauville : Diffision : Segmentation : Reconstruction = ' + str(score_weight) + ' : 1 : 1 : 1' + \
                    '\n Drop encode = ' + str(drop_encode) + \
                    '\n Optimiseur Adam - Weight decay = ' + str(weight_decay) + \
                    '\n Fc1 Regulateur : L1 = ' + str(l1_fc1) + ' - L2 = ' + str(l2_fc1) + \
                    '\n Fc1 Regulateur df : L1 = ' + str(l1_fc1_df) + ' - L2 = ' + str(l2_fc1_df) + \
                    '\n Learning rate = ' + str(learning_rate) + ' - Learning rate scheduler = ' + str(lr) + ' - Patience = ' + str(patience) + \
                    '\n Seed = ' + str(seed))
    ax.spines['right'].set_visible(False) 
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False, right=False,
        labelbottom=False,
        labeltop=False, labelleft=False, labelright=False) # labels along the bottom edge are off

    ax.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False, right=False,
        labelbottom=False,
        labeltop=False, labelleft=False, labelright=False) # labels along the bottom edge are off

    plt.subplot(3, 3, 2)
    plt.title('Deauville')
    plt.plot(range(epoch),train_micro_f1_score[0:epoch], linestyle='solid', color='green', label='Train F1 score (Accuracy)')
    plt.plot(range(epoch), val_micro_f1_score[0:epoch], linestyle='solid', color='black', label='Val micro F1 score (Accuracy)')
    plt.plot(range(epoch), val_macro_f1_score[0:epoch], linestyle='dashed', color='black', label='Val macro F1 score')
    plt.plot(range(epoch), val_weighted_f1_score[0:epoch], linestyle='dotted', color='black', label='Val weighted F1 score')
    plt.legend(loc='upper left')
    plt.xlabel("Epoch")  

    plt.subplot(3, 3, 3)
    plt.title('Diffuse sans LF')
    plt.plot(range(epoch),train_micro_f1_score_df[0:epoch], linestyle='solid', color='green', label='Train F1 score (Accuracy)')
    plt.plot(range(epoch), val_micro_f1_score_df[0:epoch], linestyle='solid', color='black', label='Val micro F1 score (Accuracy)')
    plt.plot(range(epoch), val_macro_f1_score_df[0:epoch], linestyle='dashed', color='black', label='Val macro F1 score')
    plt.plot(range(epoch), val_weighted_f1_score_df[0:epoch], linestyle='dotted', color='black', label='Val weighted F1 score')
    plt.legend(loc='upper left')
    plt.xlabel("Epoch")           
                                
    plt.subplot(3, 3, 4)
    plt.plot(range(epoch),train_loss[0:epoch]/nb_train,linestyle='solid', color='blue', label='Train Loss Total')
    plt.plot(range(epoch),val_loss[0:epoch]/nb_val, linestyle='solid',color='red', label='Val Loss Total')
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")  

    plt.subplot(3, 3, 5)    
    plt.plot(range(epoch),train_loss_class[0:epoch]/nb_train,linestyle='dotted', color='blue', label='Train Loss Class')
    plt.plot(range(epoch),val_loss_class[0:epoch]/nb_val, linestyle='dotted',color='red', label='Val Loss Class')  
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")  

    plt.subplot(3, 3, 6)    
    plt.plot(range(epoch),train_dice[0:epoch], linestyle='solid',color='blue', label='Train Dice')  
    plt.plot(range(epoch),val_dice[0:epoch], linestyle='solid',color='red', label='Val Dice')  
    plt.legend(loc='lower right')
    plt.xlabel("Epoch")  

    plt.subplot(3, 3, 7)    
    plt.plot(range(epoch),train_loss_seg[0:epoch]/nb_train,linestyle='dashed', color='blue', label='Train Loss Seg')
    plt.plot(range(epoch),val_loss_seg[0:epoch]/nb_val, linestyle='dashed',color='red', label='Val Loss Seg')  
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")  

    plt.subplot(3, 3, 8)    
    plt.plot(range(epoch),train_loss_class_diffuse[0:epoch]/nb_train,linestyle='dashed', color='blue', label='Train Loss diffuse')
    plt.plot(range(epoch),val_loss_class_diffuse[0:epoch]/nb_val, linestyle='dashed',color='red', label='Val Loss diffuse')  
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")  

    ax = plt.subplot(3, 3, 9)  
    ax.spines['right'].set_visible(False) 
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False, right=False,
        labelbottom=False,
        labeltop=False, labelleft=False, labelright=False) # labels along the bottom edge are off

    ax.tick_params(
        axis='y',          # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False, right=False,
        labelbottom=False,
        labeltop=False, labelleft=False, labelright=False) # labels along the bottom edge are off


# In[ ]:


def plot_graphic_onlySeg(fold, nb_train, nb_val, dim, spacing, scale, sigma, nb_parametres, 
                                weight_decay, learning_rate, lr, patience, epoch,  
                                train_dice, val_dice, train_loss_seg, val_loss_seg, seed) :
    ax = plt.subplot(1, 3, 1)
    plt.text(-0.05, 0.1, ' DATA - FOLD ' + str(fold) + \
                    '\n Train data = ' + str(nb_train) + \
                    '\n Validation data = ' + str(nb_val) + \
                    '\n Image dimensions = ' + str(dim[0]) + ' x '+ str(dim[1]) + ' x ' + str(dim[2]) + ' - Spacing = ' + str(spacing) + \
                    '\n ' + \
                    '\n IMAGE GENERATOR' + \
                    '\n Scale = ' + str(scale[0]) + ' - ' + str(scale[1]) + \
                    '\n Gaussian blur = Random - Sigma = '  + str(sigma[0]) + ' - ' + str(sigma[1]) + \
                    '\n ' + \
                    '\n MODELE SEGMENTATION' + \
                    '\n Nombre de paramètres trainables = ' + str(round(nb_parametres/1e6, 2)) + ' M' + \
                    '\n Optimiseur Adam - Weight decay = ' + str(weight_decay) + \
                    '\n Learning rate = ' + str(learning_rate) + ' - Learning rate scheduler = ' + str(lr) + ' - Patience = ' + str(patience) + \
                    '\n Seed = ' + str(seed))
    ax.spines['right'].set_visible(False) 
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False, right=False,
        labelbottom=False,
        labeltop=False, labelleft=False, labelright=False) # labels along the bottom edge are off

    ax.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False, right=False,
        labelbottom=False,
        labeltop=False, labelleft=False, labelright=False) # labels along the bottom edge are off
                                  

    plt.subplot(1, 3, 2)    
    plt.plot(range(epoch),train_dice[0:epoch], linestyle='solid',color='blue', label='Train Dice')  
    plt.plot(range(epoch),val_dice[0:epoch], linestyle='solid',color='red', label='Val Dice')  
    plt.legend(loc='lower right')
    plt.xlabel("Epoch")  

    plt.subplot(1, 3, 3)    
    plt.plot(range(epoch),train_loss_seg[0:epoch]/nb_train,linestyle='dashed', color='blue', label='Train Loss Seg')
    plt.plot(range(epoch),val_loss_seg[0:epoch]/nb_val, linestyle='dashed',color='red', label='Val Loss Seg')  
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")  


# In[ ]:


def plot_graphic_MRD_D100(fold, nb_train, train_c1, train_c2, nb_val, val_c1, val_c2, dim, spacing, scale, sigma, nb_parametres, score_weight, 
                drop_encode, weight_decay, l1_fc1, l2_fc1, learning_rate, lr, patience, epoch, train_micro_f1_score, val_micro_f1_score, val_macro_f1_score, 
                val_weighted_f1_score, train_loss, val_loss, train_loss_class, val_loss_class, seed) :
    plt.subplot(2, 2, 1)
    plt.text(0, 0.1, ' DATA - FOLD ' + str(fold) + \
                    '\n Train data = ' + str(nb_train) + ' - C1 - C2 = ' + str(train_c1) + ' - ' + str(train_c2) + ' - ' + \
                    '\n Validation data = ' + str(nb_val) + ' - C1 - C2 = ' + str(val_c1) + ' - ' + str(val_c2) + ' - ' + \
                    '\n Image dimensions = ' + str(dim[0]) + ' x '+ str(dim[1]) + ' x ' + str(dim[2]) + ' - Spacing = ' + str(spacing) + \
                    '\n ' + \
                    '\n IMAGE GENERATOR' + \
                    '\n Scale = ' + str(scale[0]) + ' - ' + str(scale[1]) + \
                    '\n Gaussian blur = Random - Sigma = '  + str(sigma[0]) + ' - ' + str(sigma[1]) + \
                    '\n ' + \
                    '\n MODELE CLASSIFICATION - RECONSTRUCTION' + \
                    '\n Nombre de paramètres trainables = ' + str(round(nb_parametres/1e6, 2)) + ' M' + \
                    '\n Score : Reconstruction = ' + str(score_weight) + ' : 1 : 1' + \
                    '\n Drop encode = ' + str(drop_encode) + \
                    '\n Optimiseur Adam - Weight decay = ' + str(weight_decay) + \
                    '\n Fc1 Regulateur : L1 = ' + str(l1_fc1) + ' - L2 = ' + str(l2_fc1) + \
                    '\n Learning rate = ' + str(learning_rate) + ' - Learning rate scheduler = ' + str(lr) + ' - Patience = ' + str(patience) + \
                    '\n Seed = ' + str(seed))

    plt.subplot(2, 2, 2)
    plt.plot(range(epoch),train_micro_f1_score[0:epoch], linestyle='solid', color='green', label='Train F1 score (Accuracy)')
    plt.plot(range(epoch), val_micro_f1_score[0:epoch], linestyle='solid', color='black', label='Val micro F1 score (Accuracy)')
    plt.plot(range(epoch), val_macro_f1_score[0:epoch], linestyle='dashed', color='black', label='Val macro F1 score')
    plt.plot(range(epoch), val_weighted_f1_score[0:epoch], linestyle='dotted', color='black', label='Val weighted F1 score')
    plt.legend(loc='upper left')
    plt.xlabel("Epoch")          
                                
    plt.subplot(2, 2, 3)
    plt.plot(range(epoch),train_loss[0:epoch]/nb_train,linestyle='solid', color='blue', label='Train Loss Total')
    plt.plot(range(epoch),val_loss[0:epoch]/nb_val, linestyle='solid',color='red', label='Val Loss Total')
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")  

    plt.subplot(2, 2, 4)    
    plt.plot(range(epoch),train_loss_class[0:epoch]/nb_train,linestyle='dotted', color='blue', label='Train Loss Class')
    plt.plot(range(epoch),val_loss_class[0:epoch]/nb_val, linestyle='dotted',color='red', label='Val Loss Class')  
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")  


# In[8]:


def plot_image(id, image_1, image_2, image_label, image_seg, labels, predictions, threshold, dice, nb_lesion, nb_lesion_pred, x, y) : 

    image_1 = np.where(image_1 > 0.2, 0.2, image_1)
    image_2 = np.where(image_2 > 0.2, 0.2, image_2)

    text = 'Patient '+ id[0] + ' - PET'
    mip_PET_1 = plot_image_mip(image_1, 0, text, x, y, 1)
    text = 'Classe = '+ str(labels[0].cpu().numpy())+' - Prediction = '+str(predictions[0].cpu().numpy())
    mip_PET_2 = plot_image_mip(image_1, 90, text, x, y, 1+y)

    text = 'Reconstruction'
    plot_image_mip(image_2, 0, text, x, y, 2)
    plot_image_mip(image_2, 90, '', x, y, 2+y)

    text = 'Label'
    mip_label_1 = plot_image_mip(image_label, 0, text, x, y, 3)
    text = 'Nombre de lésions : '+str(nb_lesion)
    mip_label_2 = plot_image_mip(image_label, 90, text, x, y, 3+y)
    
    text = 'Segmentation - Threshold '+str(threshold)
    mip_seg_1 = plot_image_mip(image_seg, 0, text, x, y, 4)
    text = 'Nombre de lésions détectées : '+str(nb_lesion_pred)
    mip_seg_2 = plot_image_mip(image_seg, 90, text, x, y, 4+y)

    tp_1 = mip_label_1 * mip_seg_1
    tp_2 = mip_label_2 * mip_seg_2

    fp_1 =  mip_seg_1 - tp_1
    fp_2 =  mip_seg_2 - tp_2

    fn_1 = mip_label_1 - tp_1
    fn_2 = mip_label_2 - tp_2

    seg1 = 2*tp_1 + fp_1 + 3*fn_1
    seg2 = 2*tp_2 + fp_2 + 3*fn_2

    seg1 = np.ma.masked_where(seg1 == 0, seg1)
    seg2 = np.ma.masked_where(seg2 == 0, seg2)

    ax = plt.subplot(x, y, 5)
    plt.imshow(mip_PET_1, cmap='gray')
    plt.imshow(seg1, cmap='jet', alpha=0.7)
    plt.title('Dice = ' + str(np.round(dice, 2)))
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    ax = plt.subplot(x, y, 5+y)
    plt.imshow(mip_PET_2, cmap='gray')
    plt.imshow(seg2, cmap='jet', alpha=0.7)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    


# In[ ]:


def plot_image_noSeg(id, image_1, image_2, labels, predictions, x, y) : 

    image_1 = np.where(image_1 > 0.2, 0.2, image_1)
    image_2 = np.where(image_2 > 0.2, 0.2, image_2)

    
    text = 'Patient '+ id[0] + ' - PET'
    mip_PET_1 = plot_image_mip(image_1, 0, text, x, y, 1, color='Greys')
    text = 'Classe = '+ str(labels[0].cpu().numpy())+' - Prediction = '+str(predictions[0].cpu().numpy())
    mip_PET_2 = plot_image_mip(image_1, 90, text, x, y, 1+y, color='Greys')

    text = 'Reconstruction'
    plot_image_mip(image_2, 0, text, x, y, 2)
    plot_image_mip(image_2, 90, '', x, y, 2+y)
    


# In[ ]:


def plot_image_onlySeg(id, image_1, image_label, image_seg, dice, threshold, x, y) : 

    image_1 = np.where(image_1 > 0.2, 0.2, image_1)
    
    text = 'Patient '+ id[0] + ' - PET'
    mip_PET_1 = plot_image_mip(image_1, 0, text, x, y, 1)
    mip_PET_2 = plot_image_mip(image_1, 90, text, x, y, 1+y)

    text = 'Label'
    mip_label_1 = plot_image_mip(image_label, 0, text, x, y, 2)
    mip_label_2 = plot_image_mip(image_label, 90, text, x, y, 2+y)

    text = 'Segmentation - Threshold '+str(threshold)
    mip_seg_1 = plot_image_mip(image_seg, 0, text, x, y, 3)
    mip_seg_2 = plot_image_mip(image_seg, 90, text, x, y, 3+y)

    tp_1 = mip_label_1 * mip_seg_1
    tp_2 = mip_label_2 * mip_seg_2

    fp_1 =  mip_seg_1 - tp_1
    fp_2 =  mip_seg_2 - tp_2

    fn_1 = mip_label_1 - tp_1
    fn_2 = mip_label_2 - tp_2

    seg1 = 2*tp_1 + fp_1 + 3*fn_1
    seg2 = 2*tp_2 + fp_2 + 3*fn_2

    seg1 = np.ma.masked_where(seg1 == 0, seg1)
    seg2 = np.ma.masked_where(seg2 == 0, seg2)

    ax = plt.subplot(x, y, 4)
    plt.imshow(mip_PET_1, cmap='gray')
    plt.imshow(seg1, cmap='jet', alpha=0.7)
    plt.title('Dice = ' + str(np.round(dice, 2)))
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    ax = plt.subplot(x, y, 4+y)
    plt.imshow(mip_PET_2, cmap='gray')
    plt.imshow(seg2, cmap='jet', alpha=0.7)
    ax.set_yticklabels([])
    ax.set_xticklabels([])


# In[ ]:


def plot_image_MRD_D100(id, image_1, image_2, labels, predictions, x, y) : 
    threshold = 0.2
    image_1 = np.where(image_1 > threshold, threshold, image_1)
    image_2 = np.where(image_2 > threshold, threshold, image_2)

    text = 'Patient '+ id[0] + ' - PET'
    plot_image_mip(image_1, 0, text, x, y, 1)
    text = 'Classe = '+ str(labels[0].cpu().numpy())+' - Prediction = '+str(predictions[0].cpu().numpy())
    plot_image_mip(image_1, 90, text, x, y, 3)

    text = 'Reconstruction'
    plot_image_mip(image_2, 0, text, x, y, 2)
    plot_image_mip(image_2, 90, '', x, y, 4)

    ''' ax = plt.subplot(x, y, 3)
    #plt.imshow(mip_PET_1, cmap='gray')
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    ax = plt.subplot(x, y, 3+y)
    #plt.imshow(mip_PET_2, cmap='gray')
    ax.set_yticklabels([])
    ax.set_xticklabels([])'''
    


# In[9]:


def plot_gradcam(id, image, heatmap_0, heatmap_1, image_seg, image_label, labels, predictions, threshold, dice, nb_lesion, nb_lesion_pred, x, y) : 

    image = np.where(image > 0.2, 0.2, image)

    #heatmap_0 = heatmap_0/np.amax(heatmap_0)+image*3
    #heatmap_1 = heatmap_1/np.amax(heatmap_1)+image*3

    text = 'Patient '+ id[0] + ' - PET'
    plot_image_mip(image, 0, text, x, y, 1,'Greys')
    text = 'Classe = '+ str(labels[0].cpu().numpy())+' - Prediction = '+str(predictions[0].cpu().numpy())
    plot_image_mip(image, 90, text, x, y, 1+y,'Greys')

    text = 'GRAD-CAM Heatmap'
    plot_image_mip(image, 0, text, x, y, 2, 'Greys')
    plot_image_mip(heatmap_0, 0, text, x, y, 2,alpha=0.5)
    plot_image_mip(image, 90, '', x, y, 2+y, 'Greys')
    plot_image_mip(heatmap_0, 90, '', x, y, 2+y,alpha=0.5)

    text = 'CAM Heatmap'
    plot_image_mip(image, 0, text, x, y, 3, 'Greys')
    plot_image_mip(heatmap_1, 0, text, x, y, 3,alpha=0.5)
    plot_image_mip(image, 90, '', x, y, 3+y, 'Greys')
    plot_image_mip(heatmap_1, 90, '', x, y, 3+y,alpha=0.5)

    text = 'Segmentation - Threshold '+str(threshold)
    plot_image_mip(image_seg, 0, text, x, y, 4,'Greys')
    text = 'Nombre de lésions segmentées : '+str(nb_lesion_pred)
    plot_image_mip(image_seg, 90, text, x, y, 4+y,'Greys')

    text = 'ground-true '
    plot_image_mip(image_label, 0, text, x, y, 5,'Greys')
    text = 'Nombre de vraies lésions : ' + str(nb_lesion)
    plot_image_mip(image_label, 90, text, x, y, 5+y,'Greys')


# In[ ]:


def plot_gradcam_noSeg(id, image, heatmap_0, heatmap_1, image_label, labels, predictions, x, y) : 

    image = np.where(image > 0.2, 0.2, image)

    #heatmap_0 = heatmap_0/np.amax(heatmap_0)+image*3
    #heatmap_1 = heatmap_1/np.amax(heatmap_1)+image*3

    text = 'Patient '+ id[0] + ' - PET'
    plot_image_mip(image, 0, text, x, y, 1,'Greys')
    text = 'Classe = '+ str(labels[0].cpu().numpy())+' - Prediction = '+str(predictions[0].cpu().numpy())
    plot_image_mip(image, 90, text, x, y, 1+y,'Greys')

    text = 'GRAD-CAM Heatmap'
    plot_image_mip(image, 0, text, x, y, 2, 'Greys')
    plot_image_mip(heatmap_0, 0, text, x, y, 2,alpha=0.5)
    plot_image_mip(image, 90, '', x, y, 2+y, 'Greys')
    plot_image_mip(heatmap_0, 90, '', x, y, 2+y,alpha=0.5)

    text = 'CAM Heatmap'
    plot_image_mip(image, 0, text, x, y, 3, 'Greys')
    plot_image_mip(heatmap_1, 0, text, x, y, 3,alpha=0.5)
    plot_image_mip(image, 90, '', x, y, 3+y, 'Greys')
    plot_image_mip(heatmap_1, 90, '', x, y, 3+y,alpha=0.5)

    text = 'ground-true '
    plot_image_mip(image_label, 0, text, x, y, 4,'Greys')
    plot_image_mip(image_label, 90, text, x, y, 4+y,'Greys')


# In[ ]:


def plot_seg(id, image, image_seg, image_label, threshold, dice, x, y) : 

    image = np.where(image > 0.2, 0.2, image)

    #heatmap_0 = heatmap_0/np.amax(heatmap_0)+image*3
    #heatmap_1 = heatmap_1/np.amax(heatmap_1)+image*3

    text = 'Patient '+ id[0] + ' - PET'
    plot_image_mip(image, 0, text, x, y, 1,'Greys')
    plot_image_mip(image, 90, text, x, y, 1+y,'Greys')

    text = 'Segmentation - Threshold '+str(threshold)
    plot_image_mip(image_seg, 0, text, x, y, 2,'Greys')
    text = 'Dice = '+str(dice)
    plot_image_mip(image_seg, 90, text, x, y, 2+y,'Greys')

    text = 'ground-true '
    plot_image_mip(image_label, 0, text, x, y, 3,'Greys')
    plot_image_mip(image_label, 90, text, x, y, 3+y,'Greys')


# In[ ]:


def plot_gradcam_D100(id, image, heatmap_0, heatmap_1, labels, predictions, x, y) : 
    threshold = 0.2
    alpha = 0.5
    image = np.where(image > threshold, threshold, image)

    #heatmap_0 = heatmap_0/np.amax(heatmap_0)+image*3
    #heatmap_1 = heatmap_1/np.amax(heatmap_1)+image*3

    text = 'Patient '+ id[0] + ' - PET'
    plot_image_mip(image, 0, text, x, y, 1,'Greys')
    text = 'Classe = '+ str(labels[0].cpu().numpy())+' - Prediction = '+str(predictions[0].cpu().numpy())
    plot_image_mip(image, 90, text, x, y, 1+y,'Greys')

    text = 'GRAD CAM Heatmap'
    plot_image_mip(image, 0, text, x, y, 2, 'Greys')
    plot_image_mip(heatmap_0, 0, text, x, y, 2,alpha=alpha)
    plot_image_mip(image, 90, '', x, y, 2+y, 'Greys')
    plot_image_mip(heatmap_0, 90, '', x, y, 2+y,alpha=alpha)

    text = 'CAM Heatmap'
    plot_image_mip(image, 0, text, x, y, 3, 'Greys')
    plot_image_mip(heatmap_1, 0, text, x, y, 3,alpha=alpha)
    plot_image_mip(image, 90, '', x, y, 3+y, 'Greys')
    plot_image_mip(heatmap_1, 90, '', x, y, 3+y,alpha=alpha)


# In[10]:


def data_balance(list_train_ini, train_label_classe_ini) : 
    train_c1 = train_label_classe_ini.tolist().count(0)
    train_c2 = train_label_classe_ini.tolist().count(1)
    train_c3 = train_label_classe_ini.tolist().count(2)

    list_class = [train_c1,train_c2,train_c3]
    ref_index = np.argmax(list_class)
    ratio = [0,0,0]

    for i in range(len(list_class)):
        ratio[i] = 1*list_class[ref_index]/list_class[i]
        if ratio[i] <= 1 and ratio[i] > 0 :
            ratio[i] = 1
        else:
            ratio[i] = round(ratio[i])
            
    list_train = []
    train_label_classe = []
    for i in range(len(list_train_ini)):
        nb = ratio[int(train_label_classe_ini[i])]
        for j in range(nb) : 
            list_train.append(list_train_ini[i])
            train_label_classe.append(train_label_classe_ini[i])
    return list_train, train_label_classe


# In[ ]:


def data_balance_diffuse(list_train_ini, train_label_classe_ini,train_label_classe_diffuse_ini) : 
    train_c1 = train_label_classe_ini.tolist().count(0)
    train_c2 = train_label_classe_ini.tolist().count(1)
    train_c3 = train_label_classe_ini.tolist().count(2)

    list_class = [train_c1,train_c2,train_c3]
    ref_index = np.argmax(list_class)
    ratio = [0,0,0]

    for i in range(len(list_class)):
        ratio[i] = 1*list_class[ref_index]/list_class[i]
        if ratio[i] <= 1 and ratio[i] > 0 :
            ratio[i] = 1
        else:
            ratio[i] = round(ratio[i])
            
    list_train = []
    train_label_classe = []
    train_label_classe_diffuse = []
    for i in range(len(list_train_ini)):
        nb = ratio[int(train_label_classe_ini[i])]
        for j in range(nb) : 
            list_train.append(list_train_ini[i])
            train_label_classe.append(train_label_classe_ini[i])
            train_label_classe_diffuse.append(train_label_classe_diffuse_ini[i])
    return list_train, train_label_classe, train_label_classe_diffuse 


# In[ ]:


def data_balance_diffuse_2(list_train_ini, train_label_classe_ini,train_label_classe_diffuse_ini) : 
    train_label_classe_tmp_ini = train_label_classe_ini.copy()
    for i in range(len(list_train_ini)):
        if train_label_classe_diffuse_ini[i]==1:
            train_label_classe_tmp_ini[i] += 3
        if train_label_classe_diffuse_ini[i]==2:
            train_label_classe_tmp_ini[i] += 6
        if train_label_classe_diffuse_ini[i]==3:
            train_label_classe_tmp_ini[i] += 9
   

    train_c1 = train_label_classe_tmp_ini.tolist().count(0)
    train_c2 = train_label_classe_tmp_ini.tolist().count(1)
    train_c3 = train_label_classe_tmp_ini.tolist().count(2)
    train_c4 = train_label_classe_tmp_ini.tolist().count(3)
    train_c5 = train_label_classe_tmp_ini.tolist().count(4)
    train_c6 = train_label_classe_tmp_ini.tolist().count(5)
    train_c7 = train_label_classe_tmp_ini.tolist().count(6)
    train_c8 = train_label_classe_tmp_ini.tolist().count(7)
    train_c9 = train_label_classe_tmp_ini.tolist().count(8)
    train_c10 = train_label_classe_tmp_ini.tolist().count(9)
    train_c11 = train_label_classe_tmp_ini.tolist().count(10)
    train_c12 = train_label_classe_tmp_ini.tolist().count(11)

    list_class = [train_c1,train_c2,train_c3,train_c4,train_c5,train_c6,train_c7,train_c8,train_c9,train_c10,train_c11,train_c12]
    ref_index = np.argmax(list_class)
    ratio = [0,0,0,0,0,0,0,0,0,0,0,0]

    print('list_class ',list_class)

    for i in range(len(list_class)):
        if list_class[i]!=0:
            ratio[i] = 1*list_class[ref_index]/list_class[i]
            if ratio[i] <= 1 and ratio[i] > 0 :
                ratio[i] = 1
            else:
                ratio[i] = round(ratio[i])
            
    list_train = []
    train_label_classe = []
    train_label_classe_diffuse = []

    for i in range(len(list_train_ini)):
        nb = ratio[int(train_label_classe_tmp_ini[i])]
        for j in range(nb) : 
            list_train.append(list_train_ini[i])
            train_label_classe.append(train_label_classe_ini[i])
            train_label_classe_diffuse.append(train_label_classe_diffuse_ini[i])

    return list_train, train_label_classe, train_label_classe_diffuse 


# In[ ]:


def data_balance_diffuse_3(list_train_ini, train_label_classe_ini,train_label_classe_diffuse_ini) : 
    train_label_classe_tmp_ini = train_label_classe_ini.copy() 

    train_c1 = train_label_classe_tmp_ini.tolist().count(0)
    train_c2 = train_label_classe_tmp_ini.tolist().count(1)
    train_c3 = train_label_classe_tmp_ini.tolist().count(2)

    list_class = [train_c1,train_c2,train_c3]
    ref_index = np.argmax(list_class)
    ratio = [0,0,0]

    print('list_class ',list_class)

    for i in range(len(list_class)):
        if list_class[i]!=0:
            ratio[i] = 1*list_class[ref_index]/list_class[i]
            if ratio[i] <= 1 and ratio[i] > 0 :
                ratio[i] = 1
            else:
                ratio[i] = round(ratio[i])
            
    list_train = []
    train_label_classe = []
    train_label_classe_diffuse = []

    for i in range(len(list_train_ini)):
        nb = ratio[int(train_label_classe_tmp_ini[i])]
        for j in range(nb) : 
            list_train.append(list_train_ini[i])
            train_label_classe.append(train_label_classe_ini[i])
            train_label_classe_diffuse.append(train_label_classe_diffuse_ini[i])

    return list_train, train_label_classe, train_label_classe_diffuse 


# In[ ]:


def data_balance_MRD(list_train_ini, train_label_classe_ini) : 
    train_c1 = train_label_classe_ini.tolist().count(0)
    train_c2 = train_label_classe_ini.tolist().count(1)


    list_class = [train_c1,train_c2]
    ref_index = np.argmax(list_class)
    ratio = [0,0]

    for i in range(len(list_class)):
        ratio[i] = 4*list_class[ref_index]/list_class[i]
        if ratio[i] <= 1 and ratio[i] > 0 :
            ratio[i] = 1
        else:
            ratio[i] = round(ratio[i])
            
    list_train = []
    train_label_classe = []
    for i in range(len(list_train_ini)):
        nb = ratio[int(train_label_classe_ini[i])]
        for j in range(nb) : 
            list_train.append(list_train_ini[i])
            train_label_classe.append(train_label_classe_ini[i])
    return list_train, train_label_classe


# In[ ]:


def data_balance_MRD_diffuse(list_train_ini, train_label_classe_ini,train_label_classe_diffuse_ini) : 
    train_label_classe_tmp_ini = train_label_classe_ini.copy()
    for i in range(len(list_train_ini)):
        if train_label_classe_diffuse_ini[i]==1:
            train_label_classe_tmp_ini[i] += 2

    train_c1 = train_label_classe_tmp_ini.tolist().count(0)
    train_c2 = train_label_classe_tmp_ini.tolist().count(1)
    train_c3 = train_label_classe_tmp_ini.tolist().count(2)
    train_c4 = train_label_classe_tmp_ini.tolist().count(3)

    list_class = [train_c1,train_c2,train_c3,train_c4]
    ref_index = np.argmax(list_class)
    ratio = [0,0,0,0]

    for i in range(len(list_class)):
        ratio[i] = 4*list_class[ref_index]/list_class[i]
        if list_class[i]!=0:
            if ratio[i] <= 1 and ratio[i] > 0 :
                ratio[i] = 1
            else:
                ratio[i] = round(ratio[i])
            
    list_train = []
    train_label_classe = []
    train_label_classe_diffuse = []

    for i in range(len(list_train_ini)):
        nb = ratio[int(train_label_classe_tmp_ini[i])]
        for j in range(nb) : 
            list_train.append(list_train_ini[i])
            train_label_classe.append(train_label_classe_ini[i])
            train_label_classe_diffuse.append(train_label_classe_diffuse_ini[i])
    return list_train, train_label_classe, train_label_classe_diffuse 


# In[ ]:


def data_balance_MRD_diffuse_2(list_train_ini, train_label_classe_ini,train_label_classe_diffuse_ini) : 
    train_label_classe_tmp_ini = train_label_classe_ini.copy()

    for i in range(len(list_train_ini)):
        if train_label_classe_diffuse_ini[i]==1:
            train_label_classe_tmp_ini[i] += 2
        if train_label_classe_diffuse_ini[i]==2:
            train_label_classe_tmp_ini[i] += 4
        if train_label_classe_diffuse_ini[i]==3:
            train_label_classe_tmp_ini[i] += 6

    train_c1 = train_label_classe_tmp_ini.tolist().count(0)
    train_c2 = train_label_classe_tmp_ini.tolist().count(1)
    train_c3 = train_label_classe_tmp_ini.tolist().count(2)
    train_c4 = train_label_classe_tmp_ini.tolist().count(3)
    train_c5 = train_label_classe_tmp_ini.tolist().count(4)
    train_c6 = train_label_classe_tmp_ini.tolist().count(5)
    train_c7 = train_label_classe_tmp_ini.tolist().count(6)
    train_c8 = train_label_classe_tmp_ini.tolist().count(7)

    list_class = [train_c1,train_c2,train_c3,train_c4,train_c5,train_c6,train_c7,train_c8]
    ref_index = np.argmax(list_class)
    ratio = [0,0,0,0,0,0,0,0]

    for i in range(len(list_class)):
        ratio[i] = 3*list_class[ref_index]/list_class[i]
        if list_class[i]!=0:
            if ratio[i] <= 1 and ratio[i] > 0 :
                ratio[i] = 1
            else:
                ratio[i] = round(ratio[i])
            
    list_train = []
    train_label_classe = []
    train_label_classe_diffuse = []

    for i in range(len(list_train_ini)):
        nb = ratio[int(train_label_classe_tmp_ini[i])]
        for j in range(nb) : 
            list_train.append(list_train_ini[i])
            train_label_classe.append(train_label_classe_ini[i])
            train_label_classe_diffuse.append(train_label_classe_diffuse_ini[i])
    return list_train, train_label_classe, train_label_classe_diffuse 


# In[11]:


def get_list(path, fold, dir_p) :

    path_train_val = os.path.join(path, 'Fold'+str(fold))
    list_train_ini = np.load(path_train_val+'/list_train.npy')
    train_label_classe_ini = np.load(path_train_val+'/train_label_classe.npy')
    list_val = list(np.load(path_train_val+'/list_val.npy'))
    val_label_classe = list(np.load(path_train_val+'/val_label_classe.npy'))

    list_train, train_label_classe = data_balance(list_train_ini, train_label_classe_ini)   

    np.save(dir_p + '/list_train.npy', list_train)
    np.save(dir_p + '/train_label_classe.npy', train_label_classe)
    np.save(dir_p + '/list_val.npy', list_val)
    np.save(dir_p + '/val_label_classe.npy', val_label_classe) 

    return list_train, train_label_classe, list_val, val_label_classe


# In[ ]:


def get_list_diffuse(path, fold, dir_p) :

    path_train_val = os.path.join(path, 'Fold'+str(fold))
    list_train_ini = np.load(path_train_val+'/list_train.npy')
    train_label_classe_ini = np.load(path_train_val+'/train_label_classe.npy')
    train_label_classe_diffuse_ini = np.load(path_train_val+'/train_label_classe_diffuse.npy')
    list_val = list(np.load(path_train_val+'/list_val.npy'))
    val_label_classe = list(np.load(path_train_val+'/val_label_classe.npy'))
    val_label_classe_diffuse = list(np.load(path_train_val+'/val_label_classe_diffuse.npy'))

    list_train, train_label_classe, train_label_classe_diffuse = data_balance_diffuse_2(list_train_ini, train_label_classe_ini,train_label_classe_diffuse_ini)
       

    np.save(dir_p + '/list_train.npy', list_train)
    np.save(dir_p + '/train_label_classe.npy', train_label_classe)
    np.save(dir_p + '/list_val.npy', list_val)
    np.save(dir_p + '/val_label_classe.npy', val_label_classe) 
    np.save(dir_p + '/val_label_classe_diffuse.npy', val_label_classe_diffuse)  

    return list_train, train_label_classe, train_label_classe_diffuse, list_val, val_label_classe, val_label_classe_diffuse


# In[ ]:


def get_list_MRD(path, fold, dir_p) :

    path_train_val = os.path.join(path, 'Fold'+str(fold))
    list_train_ini = np.load(path_train_val+'/list_train_MRD.npy')
    train_label_classe_ini = np.load(path_train_val+'/train_label_classe_MRD.npy')
    list_val = list(np.load(path_train_val+'/list_val_MRD.npy'))
    val_label_classe = list(np.load(path_train_val+'/val_label_classe_MRD.npy'))

    list_train, train_label_classe = data_balance_MRD(list_train_ini, train_label_classe_ini)   

    np.save(dir_p + '/list_train_MRD.npy', list_train)
    np.save(dir_p + '/train_label_classe_MRD.npy', train_label_classe)
    np.save(dir_p + '/list_val_MRD.npy', list_val)
    np.save(dir_p + '/val_label_classe_MRD.npy', val_label_classe)   

    return list_train, train_label_classe, list_val, val_label_classe


# In[ ]:


def get_list_MRD_diffuse(path, fold, dir_p) :

    path_train_val = os.path.join(path, 'Fold'+str(fold))
    list_train_ini = np.load(path_train_val+'/list_train.npy')
    train_label_classe_ini = np.load(path_train_val+'/train_label_classe.npy')
    train_label_classe_diffuse_ini = np.load(path_train_val+'/train_label_classe_diffuse.npy')
    list_val = list(np.load(path_train_val+'/list_val.npy'))
    val_label_classe = list(np.load(path_train_val+'/val_label_classe.npy'))
    val_label_classe_diffuse = list(np.load(path_train_val+'/val_label_classe_diffuse.npy'))

    list_train, train_label_classe, train_label_classe_diffuse = data_balance_MRD_diffuse_2(list_train_ini, train_label_classe_ini,train_label_classe_diffuse_ini)
       

    np.save(dir_p + '/list_train.npy', list_train)
    np.save(dir_p + '/train_label_classe.npy', train_label_classe)
    np.save(dir_p + '/list_val.npy', list_val)
    np.save(dir_p + '/val_label_classe.npy', val_label_classe) 
    np.save(dir_p + '/val_label_classe_diffuse.npy', val_label_classe_diffuse)  

    return list_train, train_label_classe, train_label_classe_diffuse, list_val, val_label_classe, val_label_classe_diffuse


# In[ ]:


def get_list_MRD_D100(path, fold, dir_p) :

    path_train_val = os.path.join(path, 'Fold'+str(fold))
    list_train_ini = np.load(path_train_val+'/list_train_MRD_100.npy')
    train_label_classe_ini = np.load(path_train_val+'/train_label_classe_MRD_100.npy')
    list_val = list(np.load(path_train_val+'/list_val_MRD_100.npy'))
    val_label_classe = list(np.load(path_train_val+'/val_label_classe_MRD_100.npy'))

    list_train, train_label_classe = data_balance_MRD(list_train_ini, train_label_classe_ini)   

    np.save(dir_p + '/list_train_MRD_100.npy', list_train)
    np.save(dir_p + '/train_label_classe_MRD_100.npy', train_label_classe)
    np.save(dir_p + '/list_val_MRD_100.npy', list_val)
    np.save(dir_p + '/val_label_classe_MRD_100.npy', val_label_classe)   

    return list_train, train_label_classe, list_val, val_label_classe


# In[12]:


def save_resultat(dir_p, train_loss, val_loss, train_loss_class, val_loss_class, train_loss_seg, val_loss_seg, train_loss_reconst, val_loss_reconst, 
                    train_micro_f1_score, val_micro_f1_score, val_macro_f1_score, val_weighted_f1_score, train_dice, val_dice) :     
    np.save(dir_p+'/train_loss.npy', train_loss)
    np.save(dir_p+'/val_loss.npy', val_loss)
    np.save(dir_p+'/train_loss_class.npy', train_loss_class)
    np.save(dir_p+'/val_loss_class.npy', val_loss_class)
    np.save(dir_p+'/train_loss_seg.npy', train_loss_seg)
    np.save(dir_p+'/val_loss_seg.npy', val_loss_seg)
    np.save(dir_p+'/train_loss_reconst.npy', train_loss_reconst)
    np.save(dir_p+'/val_loss_reconst.npy', val_loss_reconst)
    np.save(dir_p+'/train_micro_f1_score.npy', train_micro_f1_score)
    np.save(dir_p+'/val_micro_f1_score.npy', val_micro_f1_score)
    np.save(dir_p+'/val_macro_f1_score.npy', val_macro_f1_score)
    np.save(dir_p+'/val_weighted_f1_score.npy', val_weighted_f1_score)
    np.save(dir_p+'/train_dice.npy', train_dice)
    np.save(dir_p+'/val_dice.npy', val_dice)


# In[13]:


def count_FROC(patient, nb_lesion, image_seg, image_label, dim, spacing) : 
    smooth = 1e-6
    true_positive,false_positive = 0, 0    
    _, _, positions, s_patient = load_image_and_label(patient)  
    seg_mask, nb_lesion_pred = label(image_seg)    
    label_mask, nb_lesion_label = label(image_label)    
    nb_lesion_pred_ini = nb_lesion_pred

    for i in range(1, nb_lesion_pred+1) :
        seg = np.where(seg_mask==i, 1, 0)
        if np.sum(seg) < 10 : 
            image_seg = np.where(seg == 1, 0, image_seg)
    seg_mask, nb_lesion_pred = label(image_seg) 
    nb_lesion_pred_postprocess  = nb_lesion_pred
        
    for l in range(1, nb_lesion+1) :
        img_nrrd, _ = nrrd.read(os.path.join(base_nrrd, patient,'majorityLabel'+str(l)+'.nrrd')) 
        img_nrrd, _ = preprocessing_image(img=img_nrrd, positions=positions, s_patient=s_patient,flip=False,blur=False, 
                                            scale=(1, 1, 1), sigma=(1, 1, 1), dim=dim, spacing=spacing)
        img_nrrd = np.where(img_nrrd > 0, 1, 0)
        intersection = image_seg * img_nrrd         
        if np.sum(intersection)/(np.sum(img_nrrd)+smooth) >= 0.5 :
            true_positive +=1
        elif np.sum(intersection) > 0 and np.sum(intersection)/(np.sum(img_nrrd)+smooth) < 0.5 : 
            for i in range(1, nb_lesion_pred+1) :
                seg = np.where(seg_mask==i, 1, 0)
                intersection = seg * image_label
                if np.sum(intersection)==(np.sum(seg)) : 
                    image_seg = np.where(seg == 1, 0, image_seg)
                    seg_mask, nb_lesion_pred = label(image_seg) 
                    #print('Lésion détectée mais trop petite')
                    break

    false_positive = nb_lesion_pred - true_positive
    print(patient, nb_lesion_pred_ini-true_positive, nb_lesion_pred_postprocess-true_positive, false_positive)

    return true_positive, false_positive


# In[14]:


def count_FROC_center(image_seg, image_label, threshold_dist) : 
    smooth = 1e-6
    true_positive = 0      
    seg_mask, nb_lesion_pred = label(image_seg)    
    label_mask, nb_lesion = label(image_label)   
  
    if nb_lesion > 0 :
        for i in range(1, nb_lesion_pred+1) :
            seg = np.where(seg_mask==i, 1, 0)
            if np.sum(seg) < 2 : 
                image_seg = np.where(seg == 1, 0, image_seg)
        seg_mask, nb_lesion_pred = label(image_seg) 
            
        for l in range(1, nb_lesion+1) :
            
            img_lesion = np.where(label_mask==l,1,0)
            
            intersection = image_seg * img_lesion
            
            center_all_seg = center_of_mass(image_seg,seg_mask,np.array([i+1 for i in range(nb_lesion_pred)]))
            center_lesion = center_of_mass(img_lesion)
            
            if np.sum(intersection)>0 :
                
                center_seg = center_all_seg[np.amax(intersection*seg_mask)-1]          
                dist = np.sqrt( (center_lesion[0]-center_seg[0])**2 + (center_lesion[1]-center_seg[1])**2 + (center_lesion[2]-center_seg[2])**2)
                
                if dist <= threshold_dist:
                       true_positive +=1

        #false_positive = nb_lesion_pred - true_positive
        

    return true_positive, nb_lesion, nb_lesion_pred 
    
   


# In[ ]:


def FROC_per_FOLD(path_seg, nb_threshold,threshold_dist):
    all_image_label = np.load(os.path.join(path_seg, 'all_image_label.npy'))
    all_image_seg = np.load(os.path.join(path_seg, 'all_image_seg.npy'))
    sensitivity_avg, false_positive_avg= np.zeros(nb_threshold), np.zeros(nb_threshold)

    for t in tqdm(range(nb_threshold)) :
        threshold = 1/(nb_threshold+1)*(t+1)
        count = 0
        #print(f't={t}')
        #for i in [1] : 
        
        FP, TP, list_image_seg, list_image_label, list_nb_lesion, list_nb_lesion_pred = [], [], [], [], [], []
        for i in range(all_image_seg.shape[0]) :
                
            image_label = all_image_label[i]
            image_seg_prob = all_image_seg[i]

            image_seg = np.where(image_seg_prob > threshold, 1, 0)
            true_positive, nb_lesion, nb_lesion_pred = count_FROC_center(image_seg, image_label, threshold_dist)
            if (nb_lesion>0):
                list_image_seg.append(image_seg)
                list_image_label.append(image_label)        
                false_positive = nb_lesion_pred - true_positive
                TP.append(true_positive)
                FP.append(false_positive)
                list_nb_lesion.append(nb_lesion)
                list_nb_lesion_pred.append(nb_lesion_pred)    
                sensitivity_avg[t] += true_positive/nb_lesion
                false_positive_avg[t] += false_positive
                count +=1

            
    sensitivity_avg /= count
    false_positive_avg /= count 
    return sensitivity_avg, false_positive_avg, list_image_seg, list_image_label, list_nb_lesion_pred, list_nb_lesion, FP, TP 

    


# In[ ]:


def check_image_FROC(list_image_seg,list_image_label,list_nb_lesion_pred,list_nb_lesion):
    for i in range(len(TP)):
        image_seg = list_image_seg[i]
        image_label = list_image_label[i]
        intersection = image_seg*image_label
        nb_lesion_pred = list_nb_lesion_pred[i]
        nb_lesion = list_nb_lesion[i]


        axis = 1
        mip_label=np.amax(image_label,axis=axis)
        mip_seg = np.amax(image_seg,axis=axis)
        mip_intersection = np.amax(intersection,axis=axis)


        plt.figure(figsize=(20,10))
        plt.subplot(2,3,1)
        plt.imshow(mip_label, cmap='Greys',alpha=0.8)
        plt.title(f'Label, P={nb_lesion}')

        plt.subplot(2,3,2)
        plt.imshow(mip_seg, cmap='Greys',alpha=0.8)
        plt.title(f'Seg, Pred={nb_lesion_pred}')

        plt.subplot(2,3,3)
        plt.imshow(mip_intersection, cmap='Greys',alpha=0.8)
        plt.title(f'Seg, TP={TP[i]}, FP={FP[i]}')

        axis = 0
        mip_label=np.amax(image_label,axis=axis)
        mip_seg = np.amax(image_seg,axis=axis)
        mip_intersection = np.amax(intersection,axis=axis)

        plt.subplot(2,3,4)
        plt.imshow(mip_label, cmap='Greys',alpha=0.8)
        plt.title(f'Label, P={nb_lesion}')

        plt.subplot(2,3,5)
        plt.imshow(mip_seg, cmap='Greys',alpha=0.8)
        plt.title(f'Seg, Pred={nb_lesion_pred}')

        plt.subplot(2,3,6)
        plt.imshow(mip_intersection, cmap='Greys',alpha=0.8)
        plt.title(f'Seg, TP={TP[i]}, FP={FP[i]}')

        plt.savefig(f'/home/nguyen-k/Bureau/segCassiopet/Comparatif/FigTest/img{i}.jpeg')
        plt.close('all')


# In[ ]:


def display_FROC_FOLD(false_positive_avg, sensitivity_avg,color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12):
    ax = plt.figure(figsize=(20, 10))
    plt.plot(false_positive_avg, sensitivity_avg, color,marker,linestyle,linewidth,markersize)
    ax.xticks(fontsize=25)
    ax.yticks(fontsize=25)


# In[ ]:


if __name__=="__main__":
    from tqdm import tqdm

    dim = (32*4, 32*3, 32*6)
    spacing = 4
    path_seg = '/home/nguyen-k/Bureau/segCassiopet/Comparatif/Archives/Multitask_PET_CD_L0/Fold1/Fig_seg_val/'
    all_image_label = np.load(os.path.join(path_seg, 'all_image_label.npy'))
    all_image_seg = np.load(os.path.join(path_seg, 'all_image_seg.npy'))

    path_list = '/home/nguyen-k/Bureau/segCassiopet/List_Patient_0/Fold1'
    list_val = np.load(os.path.join(path_list, 'list_val.npy'))

    nb_threshold = 10
    threshold_dist = 5
    sensitivity_avg, false_positive_avg= np.zeros(nb_threshold), np.zeros(nb_threshold)

    for t in tqdm(range(nb_threshold)) :
        threshold = 1/(nb_threshold+1)*(t+1)
        count = 0
        #print(f't={t}')
        #for i in [1] : 
        
        FP, TP, list_image_seg, list_image_label, list_nb_lesion, list_nb_lesion_pred = [], [], [], [], [], []
        for i in range(all_image_seg.shape[0]) :
                
            image_label = all_image_label[i]
            image_seg_prob = all_image_seg[i]

            image_seg = np.where(image_seg_prob > threshold, 1, 0)
            true_positive, nb_lesion, nb_lesion_pred = count_FROC_center(image_seg, image_label, threshold_dist)
            if (nb_lesion>0):
                list_image_seg.append(image_seg)
                list_image_label.append(image_label)        
                false_positive = nb_lesion_pred - true_positive
                TP.append(true_positive)
                FP.append(false_positive)
                list_nb_lesion.append(nb_lesion)
                list_nb_lesion_pred.append(nb_lesion_pred)    
                sensitivity_avg[t] += true_positive/nb_lesion
                false_positive_avg[t] += false_positive
                count +=1

            
    sensitivity_avg /= count
    false_positive_avg /= count 


    for i in range(len(TP)):
        image_seg = list_image_seg[i]
        image_label = list_image_label[i]
        intersection = image_seg*image_label
        nb_lesion_pred = list_nb_lesion_pred[i]
        nb_lesion = list_nb_lesion[i]


        axis = 1
        mip_label=np.amax(image_label,axis=axis)
        mip_seg = np.amax(image_seg,axis=axis)
        mip_intersection = np.amax(intersection,axis=axis)


        plt.figure(figsize=(20,10))
        plt.subplot(2,3,1)
        plt.imshow(mip_label, cmap='Greys',alpha=0.8)
        plt.title(f'Label, P={nb_lesion}')

        plt.subplot(2,3,2)
        plt.imshow(mip_seg, cmap='Greys',alpha=0.8)
        plt.title(f'Seg, Pred={nb_lesion_pred}')

        plt.subplot(2,3,3)
        plt.imshow(mip_intersection, cmap='Greys',alpha=0.8)
        plt.title(f'Seg, TP={TP[i]}, FP={FP[i]}')

        axis = 0
        mip_label=np.amax(image_label,axis=axis)
        mip_seg = np.amax(image_seg,axis=axis)
        mip_intersection = np.amax(intersection,axis=axis)

        plt.subplot(2,3,4)
        plt.imshow(mip_label, cmap='Greys',alpha=0.8)
        plt.title(f'Label, P={nb_lesion}')

        plt.subplot(2,3,5)
        plt.imshow(mip_seg, cmap='Greys',alpha=0.8)
        plt.title(f'Seg, Pred={nb_lesion_pred}')

        plt.subplot(2,3,6)
        plt.imshow(mip_intersection, cmap='Greys',alpha=0.8)
        plt.title(f'Seg, TP={TP[i]}, FP={FP[i]}')

        plt.savefig(f'/home/nguyen-k/Bureau/segCassiopet/Comparatif/FigTest/img{i}.jpeg')
        plt.close('all')


    plt.figure(figsize=(20, 10))
    plt.plot(false_positive_avg, sensitivity_avg, 'o-')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

