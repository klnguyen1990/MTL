#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import f1_score

os.system('jupyter nbconvert --to python Pytorch_dataloader.ipynb')
from Pytorch_dataloader import dataloader_PET

os.system('jupyter nbconvert --to python Pytorch_utils.ipynb')
from Pytorch_utils import *

os.system('jupyter nbconvert --to python roc_Precision_Recall.ipynb')
from roc_Precision_Recall import *


# In[ ]:


smooth = 1e-8
threshold = 0.2
classes = [0,1,2]


# In[ ]:


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
        


# In[ ]:


class DiceBCELoss(torch.nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets):
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        bce = torch.nn.functional.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = bce + dice_loss
        
        return Dice_BCE


# In[ ]:


def train(fold, model, nb_epoch, score_weight, l1_fc1, l2_fc1, dim, spacing, scale, sigma,
        num_workers, drop_encode, batch_size, learning_rate, patience, weight_decay, dir_p, path_list, seed) :
    
    list_train, train_label_classe, list_val, val_label_classe = get_list(path_list, fold, dir_p)
    nb_train, nb_val = len(list_train), len(list_val)         
    #list_train, train_label_classe, = [list_train[i] for i in range(5)], [train_label_classe[i] for i in range(5)], 
    #list_val, val_label_classe  = [list_val[i] for i in range(5)], [val_label_classe[i] for i in range(5)]
    [train_c1, train_c2, train_c3] = [train_label_classe.count(i) for i in range(3)]
    [val_c1, val_c2, val_c3] = [val_label_classe.count(i) for i in range(3)]        

    train_dataset = dataloader_PET(patient_id = list_train, classe = train_label_classe, isTransform=True, 
                                            scale=scale, sigma=sigma, dim=dim, spacing=spacing)
    val_dataset = dataloader_PET(patient_id = list_val, classe = val_label_classe, isTransform=False, 
                                            scale=scale, sigma=sigma, dim=dim, spacing=spacing)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=num_workers, drop_last = False)    
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,shuffle=True, num_workers=num_workers)
    
    [train_loss,val_loss,train_loss_seg,val_loss_seg,train_loss_class,val_loss_class] = [np.zeros(nb_epoch) for i in range(6)]
    [train_dice, val_dice] = [np.zeros(nb_epoch) for i in range(2)]
    [train_micro_f1_score, val_micro_f1_score, val_weighted_f1_score, val_macro_f1_score] = [np.zeros(nb_epoch) for i in range(4)]

    nb_parametres = count_parameters(model) 
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  

    lr_scheduler = LRScheduler(optimizer, patience=patience, min_lr=1e-5, factor=0.5)

    for epoch in range(nb_epoch) :

        #TRAIN
        model.train() 
        lab, pred = [], []
        
        for data in trainloader  :
            image, _, labels, id = data
            image = image.float().cuda()
            labels = labels.long().cuda()
            
            optimizer.zero_grad() 
            prob  = model(image)

            all_fc1_params = torch.cat([x.view(-1) for x in model.fc1.parameters()])
            l1_regularization_fc1 = l1_fc1 * torch.norm(all_fc1_params, 1)
            l2_regularization_fc1 = l2_fc1 * torch.norm(all_fc1_params, 2)
            loss = torch.nn.CrossEntropyLoss()(prob, labels) + l1_regularization_fc1 + l2_regularization_fc1

            loss.backward()
            optimizer.step() 

            train_loss[epoch] += loss.item()
        
            _, predictions = torch.max(prob, 1)

            for label, prediction in zip(labels, predictions) :                
                lab.append(label.int().item())
                pred.append(prediction.int().item())

        lab = np.array(lab) 
        pred = np.array(pred)
        train_micro_f1_score[epoch] = f1_score(y_true=lab, y_pred=pred, average='micro')

        lr_scheduler(train_loss[epoch])
        lr = get_lr(optimizer)

        #EVALUATION

        model.eval()
        lab, pred = [], []
        
        with torch.no_grad() : 
            for data in valloader :
            
                image, _, labels, id = data
                image = image.float().cuda()
                labels = labels.long().cuda()

                prob = model(image)

                all_fc1_params = torch.cat([x.view(-1) for x in model.fc1.parameters()])
                l1_regularization_fc1 = l1_fc1 * torch.norm(all_fc1_params, 1)
                l2_regularization_fc1 = l2_fc1 * torch.norm(all_fc1_params, 2)
                loss = torch.nn.CrossEntropyLoss()(prob, labels) + l1_regularization_fc1 + l2_regularization_fc1
            
                val_loss[epoch] += loss.item()

                _, predictions = torch.max(prob, 1)

                for label, prediction in zip(labels, predictions):                
                    lab.append(label.int().item())
                    pred.append(prediction.int().item())

        lab = np.array(lab)   
        pred = np.array(pred)        
        val_micro_f1_score[epoch] = f1_score(y_true=lab, y_pred=pred, average='micro')
        val_macro_f1_score[epoch] = f1_score(y_true=lab, y_pred=pred, average='macro')
        val_weighted_f1_score[epoch] = f1_score(y_true=lab, y_pred=pred, average='weighted')  

        #GRAPHIC
        
        fig = plt.figure(figsize=(30, 20))
        fig.patch.set_facecolor('xkcd:white')
        subfigs = fig.subfigures(2, 1, hspace = 0.005)

        subfigs[0].subplots(2, 3, sharex=True)

        plot_graphic(fold, nb_train, train_c1, train_c2, train_c3, nb_val, val_c1, val_c2, val_c3, dim, spacing, scale, sigma, nb_parametres, score_weight, 
                drop_encode, weight_decay, l1_fc1, l2_fc1, learning_rate, lr, patience, epoch, train_micro_f1_score, val_micro_f1_score, val_macro_f1_score, 
                val_weighted_f1_score, train_loss, val_loss, train_loss_class, val_loss_class, train_dice, val_dice, train_loss_seg, val_loss_seg, seed)

        subfigs[1].subplots(2, 5, sharex=False,sharey=True)   

        image = image[0, 0, :, :, :].cpu().numpy()

        dice = 0
        nb_lesion, nb_lesion_pred = 0, 0
            
        plot_image(id, image, image, image, image, labels, predictions, threshold, dice, nb_lesion, nb_lesion_pred, 2, 5) 
        fig.savefig(dir_p + '/Recap.png', facecolor=fig.get_facecolor(),bbox_inches='tight')
        plt.close('all')

    np.save(dir_p+'/train_loss.npy', train_loss)
    np.save(dir_p+'/val_loss.npy', val_loss)
    np.save(dir_p+'/train_loss_class.npy', train_loss_class)
    np.save(dir_p+'/val_loss_class.npy', val_loss_class)
    np.save(dir_p+'/train_loss_seg.npy', train_loss_seg)
    np.save(dir_p+'/val_loss_seg.npy', val_loss_seg)
    np.save(dir_p+'/train_micro_f1_score.npy', train_micro_f1_score)
    np.save(dir_p+'/val_micro_f1_score.npy', val_micro_f1_score)
    np.save(dir_p+'/val_macro_f1_score.npy', val_macro_f1_score)
    np.save(dir_p+'/val_weighted_f1_score.npy', val_weighted_f1_score)
    np.save(dir_p+'/train_dice.npy', train_dice)
    np.save(dir_p+'/val_dice.npy', val_dice)


# In[ ]:


def evaluation(model, list_patient, label_classe, scale, sigma, dim, spacing, num_workers, dir_p_1) : 
    model.eval()
    #list_patient, label_classe = [list_patient[i] for i in range(5)], [label_classe[i] for i in range(5)]
    prob = np.empty((len(list_patient),3))
    pred, lab = [], []

    val_dataset = dataloader_PET(patient_id = list_patient, classe = label_classe, isTransform=False, 
                                            scale=scale, sigma=sigma, dim=dim, spacing=spacing)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=1,shuffle=False, num_workers=num_workers)

    with torch.no_grad() : 
        for i, data in enumerate(valloader) :

            image, _, labels, id = data
            image = image.float().cuda()
            labels = labels.long().cuda()
            
            proba = model(image)
            _, predictions = torch.max(proba, 1)

            proba = np.array(proba.cpu())
            prob[i] = proba

            image = image[0, 0, :, :, :].cpu().numpy()

            fig = plt.figure(figsize=(25,20))
            fig.patch.set_facecolor('xkcd:white')
            plot_image(id, image, image, image, image, labels, predictions, threshold, 0, 0, 0, 2, 5)
            fig.savefig(dir_p_1 +'/img-' + id[0] + '.png', facecolor=fig.get_facecolor(), bbox_inches='tight')
            plt.close('all')
            
            for label, prediction in zip(labels, predictions):
                pred.append(prediction.int().item())
                lab.append(label.int().item())

    pred = np.array(pred)
    lab = np.array(lab)
    
    print('Classification')
    print('Micro F1 score ', f1_score(y_true=lab, y_pred=pred, average='micro'))
    print('Macro F1 score ', f1_score(y_true=lab, y_pred=pred, average='macro'))
    print('Weighted F1 score ', f1_score(y_true=lab, y_pred=pred, average='weighted'))

    mat_label = np.zeros((len(lab),3))
    for i in range(len(lab)) :
        mat_label[i,lab[i]] = 1
    
    roc_auc, fpr, tpr = compute_ROC_auc(y_label=mat_label, y_predicted=prob, n_classes=3)
    plt.clf()
    plot_ROC_curve(fpr,tpr,roc_auc,classe=0,color='blue')
    plot_ROC_curve(fpr,tpr,roc_auc,classe=1,color='red')
    plot_ROC_curve(fpr,tpr,roc_auc,classe=2,color='black')
    plt.savefig(dir_p_1+'/ROC.png')

    precision, recall,average_precision = compute_precision_recall(y_label=mat_label,y_predicted=prob,n_classes=3)
    plot_precision_recall_curve(precision, recall, average_precision,n_classes=3,color=['blue','red','black'])
    plt.savefig(dir_p_1+'/AUC.png')

    plt.close('all')

    return prob

