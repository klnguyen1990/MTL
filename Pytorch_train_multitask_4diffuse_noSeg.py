#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import f1_score
from sklearn import metrics
#from scipy.ndimage import label
import scipy.ndimage as nd

os.system('jupyter nbconvert --to python Pytorch_dataloader.ipynb')
from Pytorch_dataloader import dataloader_PET_Diffuse

os.system('jupyter nbconvert --to python gradcam.ipynb')
from gradcam import *

os.system('jupyter nbconvert --to python Pytorch_utils.ipynb')
from Pytorch_utils import *

os.system('jupyter nbconvert --to python roc_Precision_Recall.ipynb')
from roc_Precision_Recall import *

# In[2]:


smooth = 1e-8
threshold = 0.2
classes = [0,1,2]


# In[3]:


class LRScheduler():
    def __init__(
        self, optimizer, patience=20, min_lr=1e-6, factor=0.5
    ):
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True
            )
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)
        


# In[5]:


def calcul_loss(model, l1_fc1, l2_fc1, l1_fc1_df, l2_fc1_df, prob, labels, prob_diffuse, labels_diffuse,decode, inputs, score_weight) : 

    all_fc1_params = torch.cat([x.view(-1) for x in model.fc1.parameters()])
    l1_regularization_fc1 = l1_fc1 * torch.norm(all_fc1_params, 1)
    l2_regularization_fc1 = l2_fc1 * torch.norm(all_fc1_params, 2)
    all_fc1_params_df = torch.cat([x.view(-1) for x in model.fc1_df.parameters()])
    l1_regularization_fc1_df = l1_fc1_df * torch.norm(all_fc1_params_df, 1)
    l2_regularization_fc1_df = l2_fc1_df * torch.norm(all_fc1_params_df, 2)

    loss_classif = torch.nn.CrossEntropyLoss()(prob, labels) + l1_regularization_fc1 + l2_regularization_fc1
    loss_classif_diffuse = torch.nn.CrossEntropyLoss()(prob_diffuse, labels_diffuse) + l1_regularization_fc1_df + l2_regularization_fc1_df
    loss_reconst = torch.nn.MSELoss()(decode, inputs)

    loss = loss_classif * score_weight + loss_classif_diffuse + loss_reconst
    
    return loss, loss_classif, loss_classif_diffuse 


# In[6]:


def train(fold, model, nb_epoch, score_weight, l1_fc1, l2_fc1, l1_fc1_df, l2_fc1_df, dim, spacing, scale, sigma,
        num_workers, drop_encode, batch_size, learning_rate, patience, weight_decay, dir_p, path_list, seed) :
    
    list_train, train_label_classe, train_label_classe_diffuse, list_val, val_label_classe, val_label_classe_diffuse = get_list_diffuse(path_list, fold, dir_p)
    print(train_label_classe.count(0),train_label_classe.count(1),train_label_classe.count(2))
    print(train_label_classe_diffuse.count(0), train_label_classe_diffuse.count(1),train_label_classe_diffuse.count(2),train_label_classe_diffuse.count(3))

    nb_train, nb_val = len(list_train), len(list_val)         
    #list_train, train_label_classe, train_label_classe_diffuse = [list_train[i] for i in range(10)], [train_label_classe[i] for i in range(10)], [train_label_classe_diffuse[i] for i in range(10)]#
    #list_val, val_label_classe, val_label_classe_diffuse  = [list_val[i] for i in range(10)], [val_label_classe[i] for i in range(10)], [val_label_classe_diffuse[i] for i in range(10)]#
    
    [train_c1, train_c2, train_c3] = [train_label_classe.count(i) for i in range(3)]
    [val_c1, val_c2, val_c3] = [val_label_classe.count(i) for i in range(3)]  
    
    train_dataset = dataloader_PET_Diffuse(patient_id = list_train, classe = train_label_classe, diffuse = train_label_classe_diffuse, 
                                            dfo = [], isTransform=True, scale=scale, sigma=sigma, dim=dim, spacing=spacing)
    val_dataset = dataloader_PET_Diffuse(patient_id = list_val, classe = val_label_classe, diffuse = val_label_classe_diffuse, 
                                            dfo = [], isTransform=False, scale=(1,1), sigma=(1,1), dim=dim, spacing=spacing)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=num_workers, drop_last = False)    
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,shuffle=True, num_workers=num_workers)
    
    [train_loss,val_loss,train_loss_class,val_loss_class,train_loss_class_diffuse,val_loss_class_diffuse] = [np.zeros(nb_epoch) for i in range(6)]
    [train_micro_f1_score, val_micro_f1_score, val_weighted_f1_score, val_macro_f1_score] = [np.zeros(nb_epoch) for i in range(4)]
    [train_micro_f1_score_df, val_micro_f1_score_df, val_weighted_f1_score_df, val_macro_f1_score_df] = [np.zeros(nb_epoch) for i in range(4)]

    nb_parametres = count_parameters(model) 
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  

    lr_scheduler = LRScheduler(optimizer, patience=patience, min_lr=1e-6, factor=0.5)

    for epoch in range(nb_epoch) :

        #TRAIN
        model.train() 
        lab, pred, pred_diffuse, lab_diffuse = [], [], [], []
        count = 0
        
        for data in trainloader:
        
            image, labels_seg, labels, labels_diffuse,_, id = data
            image = image.float().cuda()
            labels = labels.long().cuda()
            labels_diffuse = labels_diffuse.long().cuda()
            labels_seg = labels_seg.float().cuda()
            
            optimizer.zero_grad() 
            prob, prob_diffuse, reconst = model(image)
            loss, loss_classif, loss_classif_diffuse = calcul_loss(model, l1_fc1, l2_fc1, l1_fc1_df, l2_fc1_df, prob, 
                                                                    labels, prob_diffuse,labels_diffuse, reconst, image, score_weight)
            loss.backward()
            optimizer.step() 

            train_loss[epoch] += loss.item()
            train_loss_class[epoch] += loss_classif.item()
            train_loss_class_diffuse[epoch] += loss_classif_diffuse.item()
        
            _, predictions = torch.max(prob, 1)
            _, predictions_diffuse = torch.max(prob_diffuse, 1)

            for label, prediction, label_diffuse, prediction_diffuse in zip(labels, predictions, labels_diffuse, predictions_diffuse) :                
                lab.append(label.int().item())
                pred.append(prediction.int().item())
                pred_diffuse.append(prediction_diffuse.int().item())
                lab_diffuse.append(label_diffuse.int().item())

        lab = np.array(lab) 
        pred = np.array(pred)
        train_micro_f1_score[epoch] = f1_score(y_true=lab, y_pred=pred, average='micro')
        train_micro_f1_score_df[epoch] = f1_score(y_true=lab_diffuse, y_pred=pred_diffuse, average='micro')

        lr_scheduler(train_loss[epoch])
        lr = get_lr(optimizer)

        #EVALUATION

        model.eval()
        lab, pred, lab_diffuse, pred_diffuse = [], [], [], []
        count = 0
        
         
        for data in valloader :
            
            image, _, labels, labels_diffuse,_, id = data
            image = image.float().cuda()
            labels = labels.long().cuda()
            labels_diffuse = labels_diffuse.long().cuda()

            prob, prob_diffuse, reconst = model(image)
    
            loss, loss_classif, loss_classif_diffuse = calcul_loss(model, l1_fc1, l2_fc1, l1_fc1_df, l2_fc1_df, prob, labels, prob_diffuse,labels_diffuse, reconst, image, score_weight)

            val_loss[epoch] += loss.item()
            val_loss_class[epoch] += loss_classif.item()
            val_loss_class_diffuse[epoch] += loss_classif_diffuse.item()

            #_, predictions = torch.max(prob, 1)
            train_loss_class_diffuse[epoch] += loss_classif_diffuse.item()
        
            _, predictions = torch.max(prob, 1)
            _, predictions_diffuse = torch.max(prob_diffuse, 1)

            for label, prediction, label_diffuse, prediction_diffuse in zip(labels, predictions, labels_diffuse, predictions_diffuse) :                
                lab.append(label.int().item())
                pred.append(prediction.int().item())
                lab_diffuse.append(label_diffuse.int().item())
                pred_diffuse.append(prediction_diffuse.int().item())   

        lab = np.array(lab)   
        pred = np.array(pred)        
        val_micro_f1_score[epoch] = f1_score(y_true=lab, y_pred=pred, average='micro')
        val_macro_f1_score[epoch] = f1_score(y_true=lab, y_pred=pred, average='macro')
        val_weighted_f1_score[epoch] = f1_score(y_true=lab, y_pred=pred, average='weighted')  
        val_micro_f1_score_df[epoch] = f1_score(y_true=lab_diffuse, y_pred=pred_diffuse, average='micro')
        val_macro_f1_score_df[epoch] = f1_score(y_true=lab_diffuse, y_pred=pred_diffuse, average='macro')
        val_weighted_f1_score_df[epoch] = f1_score(y_true=lab_diffuse, y_pred=pred_diffuse, average='weighted')

        #GRAPHIC
        
        fig = plt.figure(figsize=(30, 20))
        fig.patch.set_facecolor('xkcd:white')
        subfigs = fig.subfigures(2, 1, hspace = 0.005)

        subfigs[0].subplots(2, 3, sharex=True)

        plot_graphic_diffuse_noSeg(fold, nb_train, train_c1, train_c2, train_c3, nb_val, val_c1, val_c2, val_c3, dim, spacing, scale, sigma, nb_parametres, score_weight, 
                drop_encode, weight_decay, l1_fc1, l2_fc1, l1_fc1_df, l2_fc1_df, learning_rate, lr, patience, epoch, train_micro_f1_score, val_micro_f1_score, 
                val_macro_f1_score, val_weighted_f1_score,train_micro_f1_score_df,val_micro_f1_score_df, val_macro_f1_score_df, val_weighted_f1_score_df,  
                train_loss, val_loss, train_loss_class, val_loss_class, train_loss_class_diffuse, val_loss_class_diffuse,seed)
        

        subfigs[1].subplots(2, 2, sharex=False,sharey=True)   
        image = image[0, 0, :, :, :].cpu().numpy()
        reconst = reconst[0, 0, :, :, :].cpu().detach().numpy()
        
        plot_image_noSeg(id, image, reconst, labels, predictions, 2, 2) 
        fig.savefig(dir_p + '/Recap.png', facecolor=fig.get_facecolor(),bbox_inches='tight')
        plt.close('all')

    np.save(dir_p+'/train_loss.npy', train_loss)
    np.save(dir_p+'/val_loss.npy', val_loss)
    np.save(dir_p+'/train_loss_class.npy', train_loss_class)
    np.save(dir_p+'/val_loss_class.npy', val_loss_class)
    np.save(dir_p+'/train_micro_f1_score.npy', train_micro_f1_score)
    np.save(dir_p+'/val_micro_f1_score.npy', val_micro_f1_score)
    np.save(dir_p+'/val_macro_f1_score.npy', val_macro_f1_score)
    np.save(dir_p+'/val_weighted_f1_score.npy', val_weighted_f1_score)


# In[7]:


def evaluation(model, list_patient, label_classe, label_classe_diffuse, label_classe_dfo, dim, spacing, num_workers, dir_p_1) : 
    model.eval()
    #list_patient, label_classe, label_classe_diffuse = [list_patient[i] for i in range(10)], [label_classe[i] for i in range(10)], [label_classe_diffuse[i] for i in range(10)]#
    prob, prob_df = np.empty((len(list_patient),3)), np.empty((len(list_patient),4))
    pred, lab, pred_df, lab_df, lab_dfo = [], [], [], [], []
    all_image_label = np.empty((len(list_patient),dim[0],dim[1],dim[2]))

    val_dataset = dataloader_PET_Diffuse(patient_id = list_patient, classe = label_classe, diffuse = label_classe_diffuse, 
                                            dfo = label_classe_dfo, isTransform=False, scale=(1,1), sigma=(1,1), dim=dim, spacing=spacing)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=1,shuffle=False, num_workers=num_workers)

    dice_lesion  = np.zeros(len(list_patient))
    count, nb_lesion_total, nb_lesion_pred_total = 0, 0, 0

    #with torch.no_grad() : 
    for i, data in enumerate(valloader) :
        
        image, labels_seg, labels, labels_diffuse, labels_dfo, id = data
        image = image.float().cuda()
        labels = labels.long().cuda()
        labels_diffuse = labels_diffuse.long().cuda()
        labels_seg = labels_seg.float().cuda()
        
        proba, proba_diffuse, reconst = model(image)
        _, predictions = torch.max(proba, 1)
        _, predictions_diffuse = torch.max(proba_diffuse, 1)

        heatmap_0 = gradcam(model,image,dim)
        heatmap_1 = cam(model,image,dim)
        
        proba = np.array(proba.cpu().detach().numpy())
        prob[i] = proba
        proba_diffuse = np.array(proba_diffuse.cpu().detach().numpy())
        prob_df[i] = proba_diffuse

        image = image[0, 0, :, :, :].cpu().numpy()
        reconst = reconst[0, 0, :, :, :].cpu().detach().numpy()
        image_label = labels_seg[0, 0, :, :, :].cpu().numpy()

        all_image_label[i] = image_label

        fig = plt.figure(figsize=(25,20))
        fig.patch.set_facecolor('xkcd:white')        
       
        plot_gradcam_noSeg(id, image, heatmap_0, heatmap_1, image_label,labels, predictions, 2, 4)
        fig.savefig(dir_p_1 +'/Patient-' + id[0] + '.png', facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close('all')      
        
       
        for label, prediction, label_DF, pred_DF, label_dfo in zip(labels, predictions, labels_diffuse, predictions_diffuse, labels_dfo):
            pred.append(prediction.int().item())
            lab.append(label.int().item())
            pred_df.append(pred_DF.int().item())
            lab_df.append(label_DF.int().item())
            lab_dfo.append(label_dfo.int().item())

    pred = np.array(pred)
    lab = np.array(lab)
    

    print('Classification')
    print('Micro F1 score ', f1_score(y_true=lab, y_pred=pred, average='micro'))
    print('Macro F1 score ', f1_score(y_true=lab, y_pred=pred, average='macro'))
    print('Weighted F1 score ', f1_score(y_true=lab, y_pred=pred, average='weighted'))

    print('Micro F1 score df ', f1_score(y_true=lab_df, y_pred=pred_df, average='micro'))
    print('Macro F1 score df', f1_score(y_true=lab_df, y_pred=pred_df, average='macro'))
    print('Weighted F1 score df ', f1_score(y_true=lab_df, y_pred=pred_df, average='weighted'))

    print(' ')
    print('Segmentation')
    print('Détection : ', np.round(nb_lesion_pred_total/(nb_lesion_total+smooth), 2))
    print('Dice : ', np.round(dice_lesion.sum()/(count+smooth), 2))

    mat_label = np.zeros((len(lab),3))
    for i in range(len(lab)) :
        mat_label[i,lab[i]] = 1
    
    roc_auc, fpr, tpr = compute_ROC_auc(y_label=mat_label, y_predicted=prob, n_classes=3)
    print('roc_auc',roc_auc)
    
    plt.clf()
    plot_ROC_curve(fpr,tpr,roc_auc,classe=0,color='blue')
    plot_ROC_curve(fpr,tpr,roc_auc,classe=1,color='red')
    plot_ROC_curve(fpr,tpr,roc_auc,classe=2,color='black')
    plt.savefig(dir_p_1+'/ROC.png')

    mat_label_df = np.zeros((len(lab_df),4))
    for i in range(len(lab_df)) :
        mat_label_df[i,lab_df[i]] = 1
    
    roc_auc_df, fpr_df, tpr_df = compute_ROC_auc(y_label=mat_label_df, y_predicted=prob_df, n_classes=4)
    print('roc_auc df',roc_auc_df)
    
    plt.clf()
    plot_ROC_curve(fpr_df,tpr_df,roc_auc_df,classe=0,color='blue')
    plot_ROC_curve(fpr_df,tpr_df,roc_auc_df,classe=1,color='red')
    plot_ROC_curve(fpr_df,tpr_df,roc_auc_df,classe=2,color='black')
    plot_ROC_curve(fpr_df,tpr_df,roc_auc_df,classe=3,color='green')
    plt.savefig(dir_p_1+'/ROC_df.png')

    plt.close('all')

    np.save(dir_p_1+'/all_image_label.npy', all_image_label)

    return prob, prob_df

