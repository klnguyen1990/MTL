# MTL
Main program : 
- Pytorch_multitask_nnunet.ipynb (MTL-3) : Treatment reponse classification + seg + autoencoder
- Pytorch_multitask_nnunet_4diffuse_noSeg.ipynb (MTL-3DF) : Treatment reponse classification + diffuse classification + autoencoder
- Pytorch_multitask_nnunet_4diffuse (MTL-4) :  Treatment reponse classification + diffuse classification + seg + autoencoder
- Pytorch_encodeur.ipynb (base model) : Treatment reponse classification

- Pytorch_train_****.py : contain train and evaluation function for each model
- Pytorch_model.py : containt architecture of models
- Pytorch_dataloader.py : load data into minibatch + data augmentation

Other program : 
- list_patient.ipynb : train/test split for treatment reponse classification
- list_patient_diff.ipynb : train/test split for diffuse classification
- Pytorch_utils : contains posprocessing tools, display graphics, get balance data, functions to calculate FROC 
- preprocessing_Pytorch : image preprocessing, get list patient/label functions
- Stats.py / P_value : compute p-value using Delong method
- roc_Precision_Recall.py : compute AUCROC score and display ROC curves 
- plot_froc.ipynb : display FROC curves
- gradcam.py : CAM/ GRad CAM functions
