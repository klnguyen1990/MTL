#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import torch
import torchvision 
import nibabel as nib
import skimage.filters
import random
import SimpleITK as sitk
import matplotlib.pyplot as plt 
from pandas_ods_reader import read_ods
import pandas as pd
from skimage.transform import pyramid_laplacian, pyramid_expand, resize


# In[2]:


base = '/media/nguyen-k/A75D-9A3A/CASSIOPET/dcm_baseline'
base_D100 = '/media/nguyen-k/A75D-9A3A/CASSIOPET/dcm_target'
path_squelette = '/home/nguyen-k/Bureau/segCassiopet/label_squelette'
path_label = '/home/nguyen-k/Bureau/segCassiopet/labels_seg'
base_CT = '/home/nguyen-k/Bureau/segCassiopet/imagesCT'
seuil = 150000
base_nrrd = '/home/nguyen-k/Bureau/segCassiopet/dcm_test'


# In[3]:


def resize_by_axis_pytorch(image, dim1, dim2, ax):
    
    resized_list = []
    
    image = torch.from_numpy(image)
    unstack_img_depth_list = torch.unbind(image, dim = ax)    
    unstack_img_depth_list = [e.numpy() for e in unstack_img_depth_list]
    
    data_transform = torchvision.transforms.Resize((dim1, dim2))
    for i in unstack_img_depth_list:        
        h,w,c = i.shape
        i = torch.tensor(i)
        i = torch.reshape(i, (c,h,w))
        i = data_transform(i)
        c,h,w = i.size(dim=0), i.size(dim=1),i.size(dim=2)
        i = torch.reshape(i, (h,w,c))  
        resized_list.append(i)
    
    stack_img = torch.stack(resized_list, dim = ax)    

    return stack_img


# In[4]:


def load_image_PET(p) :

    #Image patient
    img = nib.load(os.path.join(base, p,'patient_SUV.nii')).get_fdata() 
        
    path = os.path.join(path_squelette, 'cassiopet_' + p.replace('-','')+'.nii.gz')
    if os.path.isfile(path) : 
        img_squelette = nib.load(path).get_fdata()

        #Position de la tête
        img_tete = np.where(img_squelette == 2, 1, 0)
        pos_tete_max = img.shape[2] - 1
        while np.max(img_tete[:,:,pos_tete_max]) == 0 and pos_tete_max > 0:
            pos_tete_max = pos_tete_max - 1
        if pos_tete_max == 0 :
            pos_tete_max = img.shape[2] - 1
            pos_tete_min = img.shape[2] - 1
        else : 
            pos_tete_min = pos_tete_max - 50
            while np.max(img_tete[:,:,pos_tete_min]) == 0 :
                pos_tete_min = pos_tete_min + 1
    else :
        pos_tete_max, pos_tete_min = img.shape[2] - 1, img.shape[2] - 1

    position = pos_tete_min, pos_tete_max

    #Get Spacing
    img_sitk = sitk.ReadImage(os.path.join(base, p,'patient.nii'))  
    s_patient = img_sitk.GetSpacing()

    return img, position, s_patient


# In[5]:


def load_image_CT(p) :

    #Image patient
    path = os.path.join(base_CT, 'cassiopet_'+p.replace('-', '')+'_0001.nii')
    img = nib.load(path).get_fdata() 
    max_HU = np.amax(img)
    img /= max_HU

    #Get Spacing
    img_sitk = sitk.ReadImage(path)  
    s_patient = img_sitk.GetSpacing()

    return img, s_patient


# In[ ]:


def load_image_CT_2(p,min_HU) :

    #Image patient
    path = os.path.join(base_CT, 'cassiopet_'+p.replace('-', '')+'_0001.nii')
    img_0 = nib.load(path).get_fdata()
    img_0 = np.where(img_0>min_HU,img_0,0) 
    img = np.where(img_0>0,1,img_0)
    max_HU = np.amax(img)
    img /= max_HU

    #Get Spacing
    img_sitk = sitk.ReadImage(path)  
    s_patient = img_sitk.GetSpacing()

    return img, img_0/np.amax(img_0), s_patient


# In[6]:


def load_image_and_label(p) :

    #Image patient
    img = nib.load(os.path.join(base, p,'patient_SUV.nii')).get_fdata()     
    path = os.path.join(path_squelette, 'cassiopet_' + p.replace('-','')+'.nii.gz')

    #Position de la tête
    if os.path.isfile(path) : 
        img_squelette = nib.load(path).get_fdata()
        img_tete = np.where(img_squelette == 2, 1, 0)
        pos_tete_max = img.shape[2] - 1
        while np.max(img_tete[:,:,pos_tete_max]) == 0 and pos_tete_max > 0:
            pos_tete_max = pos_tete_max - 1
        if pos_tete_max == 0 :
            pos_tete_max = img.shape[2] - 1
            pos_tete_min = img.shape[2] - 1
        else : 
            pos_tete_min = pos_tete_max - 50
            while np.max(img_tete[:,:,pos_tete_min]) == 0 :
                pos_tete_min = pos_tete_min + 1
    else :
        pos_tete_max, pos_tete_min = img.shape[2] - 1, img.shape[2] - 1
    
    position = pos_tete_min, pos_tete_max
    
    #Label lésion
    path_1 = os.path.join(path_label, 'cassiopet_' + p.replace('-','')+'.nii.gz')
    if os.path.isfile(path_1) :
        img_label = nib.load(path_1).get_fdata()
        img_squelette = np.where(img_label == 1, 1, img_squelette)
    else :
        img_label = img * 0

    #Get Spacing
    img_sitk = sitk.ReadImage(os.path.join(base, p,'patient.nii'))  
    s_patient = img_sitk.GetSpacing()

    return img, img_label, position, s_patient 


# In[8]:


def preprocessing_image(img, positions, s_patient,flip,blur, scale, sigma, dim, spacing) :
    
    pos_tete_min, _ = positions   
    l_max = pos_tete_min-10

    img = img[ : , :, 0:l_max]

    #NORMALISATION SPACING
    new_dim = [0, 0, 0]
    for i in range(len(s_patient)) :
        new_dim[i] = int(img.shape[i]*s_patient[i]/spacing*scale[i])    
    
    img = np.expand_dims(img, axis = -1)
    img = resize_by_axis_pytorch(img, new_dim[0], new_dim[1], 2)    
    img = resize_by_axis_pytorch(img.numpy(), new_dim[0], new_dim[2], 1)
    img = np.squeeze(img).numpy()    

    #Crop
    if new_dim[0] > dim[0] :
        img = img[int((new_dim[0]-dim[0])/2):int((new_dim[0]+dim[0])/2), :, :]

    if new_dim[1] > dim[1] :
        img = img[:, int((new_dim[1]-dim[1])/2):int((new_dim[1]+dim[1])/2), :]

    if new_dim[2] > dim[2] :
        img = img[:, :, new_dim[2] - dim[2] : new_dim[2]]

    #Padding
    if new_dim[0] < dim[0] : 
        temp = np.zeros((dim[0], img.shape[1], img.shape[2]))
        temp[int((dim[0]-new_dim[0])/2):int((new_dim[0]+dim[0])/2), :, :] = img
        img = temp.copy()

    if new_dim[1] < dim[1] : 
        temp = np.zeros((img.shape[0], dim[1], img.shape[2]))
        temp[:, int((dim[1]-new_dim[1])/2):int((new_dim[1]+dim[1])/2), :] = img
        img = temp.copy()       

    if new_dim[2] < dim[2] : 
        temp = np.zeros((img.shape[0], img.shape[1], dim[2]))
        temp[:, :, dim[2] - new_dim[2] : dim[2]] = img
        img = temp.copy()

    if flip :
        img = np.flip(img, 0)
    if blur :
        img = skimage.filters.gaussian(img, sigma=(sigma[0], sigma[1], sigma[2]), truncate=3.5, channel_axis=True)
            
    return img, new_dim


# In[9]:


def preprocessing_image_and_label(img, img_label, positions, s_patient, scale, flip, blur, sigma, dim, spacing) :

    pos_tete_min, _ = positions   
    l_max = pos_tete_min-10
    
    img, new_dim = preprocessing_image(img, positions, s_patient, flip=flip, blur=blur, scale=scale, sigma=sigma, dim = dim, spacing = spacing)

    #LABEL

    if np.amax(img_label) == 0 :
        img_label = img * 0
    else : 
        img_label = img_label[ : , :, 0:l_max]

        img_label = np.expand_dims(img_label, axis = -1)
        img_label = resize_by_axis_pytorch(img_label, new_dim[0], new_dim[1], 2)    
        img_label = resize_by_axis_pytorch(img_label.numpy(), new_dim[0], new_dim[2], 1)    
        img_label = np.squeeze(img_label).numpy()
        img_label = np.where(img_label > 0.5, 1, 0)   

        #Crop
        if new_dim[0] > dim[0] :
            img_label = img_label[int((new_dim[0]-dim[0])/2):int((new_dim[0]+dim[0])/2), :, :]

        if new_dim[1] > dim[1] :
            img_label = img_label[:, int((new_dim[1]-dim[1])/2):int((new_dim[1]+dim[1])/2), :]

        if new_dim[2] > dim[2] :
            img_label = img_label[:, :, new_dim[2] - dim[2] : new_dim[2]]

        #Padding
        if new_dim[0] < dim[0] : 
            temp = np.zeros((dim[0], img_label.shape[1], img_label.shape[2]))
            temp[int((dim[0]-new_dim[0])/2):int((new_dim[0]+dim[0])/2), :, :] = img_label
            img_label = temp.copy()
        
        if new_dim[1] < dim[1] :
            temp = np.zeros((img_label.shape[0], dim[1], img_label.shape[2]))
            temp[:, int((dim[1]-new_dim[1])/2):int((new_dim[1]+dim[1])/2), :] = img_label
            img_label = temp.copy()   
        
        if new_dim[2] < dim[2] : 
            temp = np.zeros((img_label.shape[0], img_label.shape[1], dim[2]))
            temp[:, :, dim[2] - new_dim[2] : dim[2]] = img_label
            img_label = temp.copy()
        #Flip
        if flip :
            img_label = np.flip(img_label, 0)       
        
        #img_label = np.expand_dims(img_label, axis = 0)  
    #BLUR        
    if blur : 
        img = skimage.filters.gaussian(img, sigma=(sigma[0], sigma[1], sigma[2]), truncate=3.5, channel_axis=True)
        
    return img, img_label


# In[10]:


def score_de_deauville(task=1) : 
    reponse_path = '/home/nguyen-k/Bureau/segCassiopet/Deauville_DL_Allods.ods'
    reponse_data = read_ods(reponse_path, 1)
    reponse_data = reponse_data[reponse_data['Reponse'].notna()] 
    reponse_data.reset_index(drop=True, inplace=True)

    base = '/media/nguyen-k/A75D-9A3A/CASSIOPET/dcm_baseline'
    list_dir = os.listdir(base)

    list_patient = []
    label_classe = []

    if task == 3 : 

        for i in range(0, len(list_dir)) :
            ref = 0
            while ref < reponse_data.shape[0] and reponse_data['Patient'][ref]!= list_dir[i]:
                ref = ref + 1  

            path_PET = os.path.join(base, list_dir[i],'patient_SUV.nii')
            path_CT = os.path.join(base_CT, 'cassiopet_'+list_dir[i].replace('-', '')+'_0001.nii')
            if ref != reponse_data.shape[0] and os.path.isfile(path_PET) and os.path.isfile(path_CT) :
                list_patient.append(reponse_data['Patient'][ref])
                label_classe.append(reponse_data['Reponse'][ref]-1)
    
    else :

        for i in range(0, len(list_dir)) :
            ref = 0
            while ref < reponse_data.shape[0] and reponse_data['Patient'][ref]!= list_dir[i]:
                ref = ref + 1  
            if task == 1 : 
                path_PET = os.path.join(base, list_dir[i],'patient_SUV.nii')
            else : 
                path_PET = os.path.join(base, list_dir[i],'patient.nii')

            if ref != reponse_data.shape[0] and os.path.isfile(path_PET)  :
                list_patient.append(reponse_data['Patient'][ref])
                label_classe.append(reponse_data['Reponse'][ref]-1)
            
    return list_patient, label_classe


# In[ ]:


def score_de_deauville_2() : 
    path1 = '/home/nguyen-k/Bureau/segCassiopet/CASSIOPET BJ_v5.ods'
    cols = ['Subject Identifier for the Study (SUBJID)','Bone Marrow (BM) Uptake Appearance (BMAPPEAR)','Presence of Focal Lesion (FL) (FLPRES)']
    data1 = pd.read_excel(path1,'PET')
    data1 = data1[data1[cols[1]].notna()] 
    data1.reset_index(drop=True, inplace=True)

    data1 = data1[cols]
    data1 = data1.rename(columns={cols[0]: "Patient"})

    path2 = '/home/nguyen-k/Bureau/segCassiopet/Deauville_DL_Allods.ods'
    data2 = read_ods(path2, 1)
    data2 = data2[data2['Reponse'].notna()] 
    data2.reset_index(drop=True, inplace=True)

    reponse_data = pd.merge(data1, data2, how ='inner', on =['Patient'])
    is_diffuse = []
    for i in range(reponse_data.shape[0]):
        if (str(reponse_data[cols[1]][i]) == 'Homogeneous') & (str(reponse_data[cols[2]][i]) == 'No'): 
                is_diffuse.append(1) 
        else:
            is_diffuse.append(0)
    reponse_data = reponse_data.assign(Diffuse_Homo_NoLF=is_diffuse)

    list_dir = os.listdir(base)

    list_patient = []
    label_classe_DV = []
    label_classe_diff_Homo_NoLF = []
    label_classe_diffuse = []#reponse_data[cols[1]]
    label_classe_isfocal = []#reponse_data[cols[2]]
    print(len(label_classe_isfocal))

    for i in range(0, len(list_dir)) :
        ref = 0
        while ref < reponse_data.shape[0] and reponse_data['Patient'][ref]!= list_dir[i]:
            ref = ref + 1  

        path_PET = os.path.join(base, list_dir[i],'patient_SUV.nii')
        #path_CT = os.path.join(base_CT, 'cassiopet_'+list_dir[i].replace('-', '')+'_0001.nii')
        #if ref != reponse_data.shape[0] and os.path.isfile(path_PET) and os.path.isfile(path_CT) :
        if ref != reponse_data.shape[0] and os.path.isfile(path_PET):
            list_patient.append(reponse_data['Patient'][ref])
            label_classe_DV.append(reponse_data['Reponse'][ref]-1)
            label_classe_diff_Homo_NoLF.append(reponse_data['Diffuse_Homo_NoLF'][ref])
            label_classe_diffuse.append(reponse_data[cols[1]][ref])
            label_classe_isfocal.append(reponse_data[cols[2]][ref])
            
    return list_patient, label_classe_DV, label_classe_diff_Homo_NoLF ,label_classe_diffuse, label_classe_isfocal


# In[ ]:


def score_de_deauville_3() : 
    path1 = '/home/nguyen-k/Bureau/segCassiopet/CASSIOPET BJ_v5.ods'
    cols = ['Subject Identifier for the Study (SUBJID)','Bone Marrow (BM) Uptake Appearance (BMAPPEAR)','Presence of Focal Lesion (FL) (FLPRES)']
    data1 = pd.read_excel(path1,'PET')
    data1 = data1[data1[cols[1]].notna()] 
    data1.reset_index(drop=True, inplace=True)

    data1 = data1[cols]
    data1 = data1.rename(columns={cols[0]: "Patient"})

    path2 = '/home/nguyen-k/Bureau/segCassiopet/Deauville_DL_Allods.ods'
    data2 = read_ods(path2, 1)
    data2 = data2[data2['Reponse'].notna()] 
    data2.reset_index(drop=True, inplace=True)

    reponse_data = pd.merge(data1, data2, how ='inner', on =['Patient'])
    #is_diffuse = np.zeros((1,reponse_data.shape[0]))
    is_diffuse = []
    print(reponse_data.shape[0])
    for i in range(reponse_data.shape[0]):
        print('i=',i)
        if (str(reponse_data[cols[1]][i]) == 'Heterogeneous') and (str(reponse_data[cols[2]][i]) == 'Yes'): 
            is_diffuse.append(0)
        elif (str(reponse_data[cols[1]][i]) == 'Homogeneous') and (str(reponse_data[cols[2]][i]) == 'Yes'):
            is_diffuse.append(1)
        elif (str(reponse_data[cols[1]][i]) == 'Heterogeneous') and (str(reponse_data[cols[2]][i]) == 'No'):
            is_diffuse.append(2)
        elif (str(reponse_data[cols[1]][i]) == 'Homogeneous') and (str(reponse_data[cols[2]][i]) == 'No'):
            is_diffuse.append(3)
        else : 
            print('unkown...')
    
    reponse_data = reponse_data.assign(Diffuse_Homo_NoLF=is_diffuse)
    #reponse_data.to_excel('resp.xlsx')

    list_dir = os.listdir(base)

    list_patient = []
    label_classe_DV = []
    label_classe_diff_LFNLF = []
    label_classe_diffuse = []
    label_classe_isfocal = []
    print(len(label_classe_isfocal))

    for i in range(0, len(list_dir)) :
        ref = 0
        while ref < reponse_data.shape[0] and reponse_data['Patient'][ref]!= list_dir[i]:
            ref = ref + 1  

        path_PET = os.path.join(base, list_dir[i],'patient_SUV.nii')
        #path_CT = os.path.join(base_CT, 'cassiopet_'+list_dir[i].replace('-', '')+'_0001.nii')
        #if ref != reponse_data.shape[0] and os.path.isfile(path_PET) and os.path.isfile(path_CT) :
        if ref != reponse_data.shape[0] and os.path.isfile(path_PET):
            list_patient.append(reponse_data['Patient'][ref])
            label_classe_DV.append(reponse_data['Reponse'][ref]-1)
            label_classe_diff_LFNLF.append(reponse_data['Diffuse_Homo_NoLF'][ref])
            label_classe_diffuse.append(reponse_data[cols[1]][ref])
            label_classe_isfocal.append(reponse_data[cols[2]][ref])
            
    return list_patient, label_classe_DV, label_classe_diff_LFNLF ,label_classe_diffuse, label_classe_isfocal


# In[11]:


def Minimal_residual_disease(sheet='MRD2_CASSIOPEIA',exam='baseline'):
    reponse_path = '/home/nguyen-k/Bureau/segCassiopet/MRD2_PETRA.xlsx'
    data = pd.ExcelFile(reponse_path)
    reponse_data = pd.read_excel(data, sheet)
    for i in range(len(reponse_data)):
        if reponse_data['MRD'][i] == 'DETECTABLE':
            reponse_data['MRD'][i] = 1
        else:
            reponse_data['MRD'][i] = 0

    assert exam in ['baseline', 'D100'], f'exam has to be ethier baseline or D100 but got {exam}'
    if exam == 'baseline':
        base = '/media/nguyen-k/A75D-9A3A/CASSIOPET/dcm_baseline'
    else:
        base = '/media/nguyen-k/A75D-9A3A/CASSIOPET/dcm_target'
    list_dir = os.listdir(base)

    list_patient = []
    label_classe = []

    for i in range(0, len(list_dir)) :
        ref = 0
        while ref < reponse_data.shape[0] and reponse_data['N°'][ref]!= list_dir[i]:
            ref = ref + 1  

        path_PET = os.path.join(base, list_dir[i],'patient_SUV.nii')
    
        if ref != reponse_data.shape[0] and os.path.isfile(path_PET):
            list_patient.append(reponse_data['N°'][ref])
            label_classe.append(reponse_data['MRD'][ref])
            
    return list_patient, label_classe


# In[ ]:


def Minimal_residual_disease_2(sheet='MRD2_CASSIOPEIA',exam='baseline'):
    path1 = '/home/nguyen-k/Bureau/segCassiopet/CASSIOPET BJ_v5.ods'
    cols = ['Subject Identifier for the Study (SUBJID)','Bone Marrow (BM) Uptake Appearance (BMAPPEAR)','Presence of Focal Lesion (FL) (FLPRES)']
    data1 = pd.read_excel(path1,'PET')
    data1 = data1[data1[cols[1]].notna()] 
    data1.reset_index(drop=True, inplace=True)
    data1 = data1[cols]
    data1 = data1.rename(columns={cols[0]: "Patient"})
    data1.to_excel('data1.xlsx')

    path2 = '/home/nguyen-k/Bureau/segCassiopet/MRD2_PETRA.xlsx'
    data2 = pd.ExcelFile(path2)
    data2 = pd.read_excel(data2, sheet)
    for i in range(len(data2)):
        if data2['MRD'][i] == 'DETECTABLE':
            data2['MRD'][i] = 1
        else:
            data2['MRD'][i] = 0
    data2.drop('Patient', inplace=True, axis=1)
    data2 = data2.rename(columns={'N°':'Patient'})
    data2.to_excel('data2.xlsx')

    reponse_data = pd.merge(data1, data2, how ='inner', on =['Patient'])
    is_diffuse = []
    for i in range(reponse_data.shape[0]):
        if (str(reponse_data[cols[1]][i]) == 'Homogeneous') & (str(reponse_data[cols[2]][i]) == 'No'): 
                is_diffuse.append(1) 
        else:
            is_diffuse.append(0)
    reponse_data = reponse_data.assign(Diffuse_Homo_NoLF=is_diffuse)

    assert exam in ['baseline', 'D100'], f'exam has to be ethier baseline or D100 but got {exam}'
    if exam == 'baseline':
        base = '/media/nguyen-k/A75D-9A3A/CASSIOPET/dcm_baseline'
    else:
        base = '/media/nguyen-k/A75D-9A3A/CASSIOPET/dcm_target'
        
    list_dir = os.listdir(base)

    list_patient = []
    label_classe_MRD = []
    label_classe_diff_Homo_NoLF = []
    label_classe_diffuse = []#reponse_data[cols[1]]
    label_classe_isfocal = []#reponse_data[cols[2]

    for i in range(0, len(list_dir)) :
        ref = 0
        while ref < reponse_data.shape[0] and reponse_data['Patient'][ref]!= list_dir[i]:
            ref = ref + 1  

        path_PET = os.path.join(base, list_dir[i],'patient_SUV.nii')
    
        if ref != reponse_data.shape[0] and os.path.isfile(path_PET):
            list_patient.append(reponse_data['Patient'][ref])
            label_classe_MRD.append(reponse_data['MRD'][ref])
            label_classe_diff_Homo_NoLF.append(reponse_data['Diffuse_Homo_NoLF'][ref])
            label_classe_diffuse.append(reponse_data[cols[1]][ref])
            label_classe_isfocal.append(reponse_data[cols[2]][ref])
            
    return list_patient, label_classe_MRD, label_classe_diff_Homo_NoLF ,label_classe_diffuse, label_classe_isfocal


# In[ ]:


def Minimal_residual_disease_3(sheet='MRD2_CASSIOPEIA',exam='baseline'):
    path1 = '/home/nguyen-k/Bureau/segCassiopet/CASSIOPET BJ_v5.ods'
    cols = ['Subject Identifier for the Study (SUBJID)','Bone Marrow (BM) Uptake Appearance (BMAPPEAR)','Presence of Focal Lesion (FL) (FLPRES)']
    data1 = pd.read_excel(path1,'PET')
    data1 = data1[data1[cols[1]].notna()] 
    data1.reset_index(drop=True, inplace=True)
    data1 = data1[cols]
    data1 = data1.rename(columns={cols[0]: "Patient"})
    data1.to_excel('data1.xlsx')

    path2 = '/home/nguyen-k/Bureau/segCassiopet/MRD2_PETRA.xlsx'
    data2 = pd.ExcelFile(path2)
    data2 = pd.read_excel(data2, sheet)
    for i in range(len(data2)):
        if data2['MRD'][i] == 'DETECTABLE':
            data2['MRD'][i] = 1
        else:
            data2['MRD'][i] = 0
    data2.drop('Patient', inplace=True, axis=1)
    data2 = data2.rename(columns={'N°':'Patient'})

    reponse_data = pd.merge(data1, data2, how ='inner', on =['Patient'])
    is_diffuse = []
    print(reponse_data.shape[0])
    for i in range(reponse_data.shape[0]):
        print('i=',i)
        if (str(reponse_data[cols[1]][i]) == 'Heterogeneous') and (str(reponse_data[cols[2]][i]) == 'Yes'): 
            is_diffuse.append(0)
            print(0)
        elif (str(reponse_data[cols[1]][i]) == 'Homogeneous') and (str(reponse_data[cols[2]][i]) == 'Yes'):
            is_diffuse.append(1)
            print(1)
        elif (str(reponse_data[cols[1]][i]) == 'Heterogeneous') and (str(reponse_data[cols[2]][i]) == 'No'):
            is_diffuse.append(2)
            print(2)
        elif (str(reponse_data[cols[1]][i]) == 'Homogeneous') and (str(reponse_data[cols[2]][i]) == 'No'):
            is_diffuse.append(3)
            print(3)
        else : 
            print('unkown...')

    reponse_data = reponse_data.assign(Diffuse_Homo_NoLF=is_diffuse)
    reponse_data.to_excel('resp.xlsx')

    assert exam in ['baseline', 'D100'], f'exam has to be ethier baseline or D100 but got {exam}'
    if exam == 'baseline':
        base = '/media/nguyen-k/A75D-9A3A/CASSIOPET/dcm_baseline'
    else:
        base = '/media/nguyen-k/A75D-9A3A/CASSIOPET/dcm_target'
        
    list_dir = os.listdir(base)

    list_patient = []
    label_classe_MRD = []
    label_classe_diff_Homo_NoLF = []
    label_classe_diffuse = []#reponse_data[cols[1]]
    label_classe_isfocal = []#reponse_data[cols[2]

    for i in range(0, len(list_dir)) :
        ref = 0
        while ref < reponse_data.shape[0] and reponse_data['Patient'][ref]!= list_dir[i]:
            ref = ref + 1  

        path_PET = os.path.join(base, list_dir[i],'patient_SUV.nii')
    
        if ref != reponse_data.shape[0] and os.path.isfile(path_PET):
            list_patient.append(reponse_data['Patient'][ref])
            label_classe_MRD.append(reponse_data['MRD'][ref])
            label_classe_diff_Homo_NoLF.append(reponse_data['Diffuse_Homo_NoLF'][ref])
            label_classe_diffuse.append(reponse_data[cols[1]][ref])
            label_classe_isfocal.append(reponse_data[cols[2]][ref])
            
    return list_patient, label_classe_MRD, label_classe_diff_Homo_NoLF ,label_classe_diffuse, label_classe_isfocal


# In[12]:


def get_clinic_data(patient) :
    survie_data = read_ods('/home/nguyen-k/Bureau/segCassiopet/Clinics.ods', 1)
    ref = 0
    while ref < survie_data.shape[0] and survie_data['Patient'][ref]!= patient:
        ref = ref + 1 
    if ref != survie_data.shape[0] :
        age = survie_data['Age'][ref]/100
        if survie_data['Sex'][ref] == 'Male' :
            sex = 1
        else : sex = 0
        if survie_data['Risk_Result'][ref] == 'STANDARD RISK' : 
            risk = 0
        else : risk = 1
    else :
        print(patient, 'not found')
        age = 0.5
        sex = 2
        risk = 2

    clinic_data = np.array([age, sex, risk])

    return clinic_data     


# In[13]:


def Laplace_pyr_fusion(img_PET,img_CT,max_layer=5,downscale=2):
    
    lapl_pyr_CT = tuple(pyramid_laplacian(img_CT, max_layer=max_layer, downscale=downscale))
    lapl_pyr_PET = tuple(pyramid_laplacian(img_PET, max_layer=max_layer, downscale=downscale))

    fused_level = []
    for i in range(len(lapl_pyr_CT)):
        fused = (lapl_pyr_PET[i] + lapl_pyr_CT[i])/2
        fused_level.append(fused)

    orig = fused_level[len(lapl_pyr_CT)-1]
    for i in range(len(lapl_pyr_CT)-1,0,-1):
        up = pyramid_expand(orig, upscale=downscale)
        up = resize(up,fused_level[i-1].shape)
        orig = up + fused_level[i-1]

    return orig


# In[14]:


'''if __name__ == '__main__':
    
    i = '050-11'  
    img_patient_ini, img_label_ini, positions, s_patient = load_image_and_label(i) 
    img_patient, img_label = preprocessing_image_and_label(img_patient_ini, img_label_ini, positions, s_patient, scale=(1, 1, 1), flip=False, 
                                                            blur=False, sigma=(1, 1, 1), dim = (32*4, 32*3, 32*6), spacing=4)
    img_patient = np.squeeze(img_patient)
    img_label = np.squeeze(img_label)
    print(img_patient.shape)
    plt.figure(figsize=(20, 10))
    plt.subplot(1,2,1)
    plt.imshow(np.rot90(img_patient[:, int(img_patient.shape[1]/2), :]))
    plt.subplot(1,2,2)
    plt.imshow(np.rot90(img_patient[int(img_patient.shape[0]/2), :, :]))
    plt.figure(figsize=(20, 10))
    plt.subplot(1,2,1)
    plt.imshow(np.rot90(img_label[:, int(img_label.shape[1]/2), :]))
    plt.subplot(1,2,2)
    plt.imshow(np.rot90(img_label[int(img_label.shape[0]/2), :, :]))'''



# In[15]:


'''if __name__ == '__main__':
    p = '050-11'
    img_patient = nib.load(os.path.join(base, p,'patient.nii')).get_fdata() 
    img_patient[160:172, :, :] = 0

    img_tete = img_patient[:, :, 400:450]
    img_tete = np.where(img_tete > 8000, 1, 0)
    plt.figure(figsize=(20,10))
    plt.subplot(1, 3, 1)
    plt.imshow(img_tete[int(img_tete.shape[0]/2), :, :])
    plt.subplot(1, 3, 2)
    plt.imshow(img_tete[:, int(img_tete.shape[1]/2), :])
    plt.subplot(1, 3, 3)
    plt.imshow(img_tete[:, :, int(img_tete.shape[2]/2)])

    img_squelette = img_patient > 200
    img_tete_1 = img_patient * 0
    img_tete_1[:, :, 400:450] = img_tete
    img_squelette = np.where(img_tete_1>0, 2, img_squelette)
    plt.figure(figsize=(20,10))
    plt.subplot(1, 3, 1)
    plt.imshow(img_squelette[int(img_squelette.shape[0]/2), :, :])
    plt.subplot(1, 3, 2)
    plt.imshow(img_squelette[:, int(img_squelette.shape[1]/2), :])
    plt.subplot(1, 3, 3)
    plt.imshow(img_squelette[:, :, int(img_squelette.shape[2]/2)])

    img_squelette = nib.Nifti1Image(img_squelette,None)
    path = os.path.join(path_squelette, 'cassiopet_' + p.replace('-','')+'.nii.gz')
    nib.save(img_squelette, path)'''


# In[22]:


if __name__=='__main__':
    from tqdm import tqdm
    from Pytorch_utils import plot_image_mip
    save_path = '/home/nguyen-k/Bureau/segCassiopet/Baseline_D100_Fig'
    try:
        os.mkdir(save_path)
    except:
        print('Directory already exists')

    patients = os.listdir(base)
    x,y=1,4
    threshold = 0.2

    for p in tqdm(patients):
        path_bl = os.path.join(base_D100, p,'patient_SUV.nii')
        path_D100 = os.path.join(base, p,'patient_SUV.nii')
        
        if os.path.isfile(path_bl) & os.path.isfile(path_D100):

            img_D100 = nib.load(path_bl).get_fdata()
            img_D100 = img_D100/np.amax(img_D100)
            img_bl =  nib.load(path_D100).get_fdata()

            img_bl = img_bl/np.amax(img_bl)
            img_bl = np.where(img_bl > threshold, 0.2, img_bl)
            img_D100 = np.where(img_D100 > threshold, 0.2, img_D100)

            path_1 = os.path.join(path_label, 'cassiopet_' + p.replace('-','')+'.nii.gz')
            if os.path.isfile(path_1) :
                img_label = nib.load(path_1).get_fdata()
            else :
                img_label = img_bl * 0

            plt.figure(figsize=(20,10))
            text = 'Patient '+ p + ' - PET baseline'
            plot_image_mip(img_bl, 0, text, x, y, 1,color='Greys')
            plot_image_mip(img_label, 0, text, x, y, 1,color='Greens',alpha=0.5)
            plot_image_mip(img_bl, 90, '', x, y, 2,color='Greys')
            plot_image_mip(img_label, 90, '', x, y, 2,color='Greens',alpha=0.5)

            text = 'Patient '+ p + ' - PET D100'
            mip_D100_1 = plot_image_mip(img_D100, 0, text, x, y, 3,color='Greys')
            mip_D100_2 = plot_image_mip(img_D100, 90, '', x, y, 4,color='Greys')
            plt.savefig(os.path.join(save_path,p+'.jpeg'),bbox_inches='tight')
            plt.close('all')


          

