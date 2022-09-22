# MTL
Wrokspace : 
- '/home/nguyen-k/Bureau/segCassiopet2'
- Structure of the workspace directory : 
  * scripts : pretrained nnUnet, main program, other program (listed bellow)
  * sub-directories : 
   - List_Patient_0 : contains list of train/val/test set for the Treatment reponse classification over 5 folds
   - List_Patient_diff_0 : contains list of train/val/test set for Treatment reponse classification and diffuse classification over 5 folds
   - Comparatif : stocks results of several MTL models
      * Archives_MTL3 (MTL-3) : for Treatment reponse classification + diffuse classification + autoencoder
      * Archives_4diffuse_noSeg (MTL-3DF) : for Treatment reponse classification + diffuse classification + autoencoder
      * Archives_4diffuse (MTL-4) : for Treatment reponse classification + diffuse classification + seg + autoencoder
      * Archives_Encodeur (base model) : for Treatment reponse classification only  
   - imagesPET_Positif/Negatif : stock compressed niffti images for patient avec ou sans focal lesion (use in list_patient.ipynb)
   - label_seg : stock focal lesion grounthTrue in compressed niffti format (used in prepocessing_Pytorch.py)

Pretrained nnUnet: run all python scripts below in fllowing order
- setOrig.py : to set all images towards the same original and pixel spacing.
- makejson.ipynb : to generate data JSON file which containt all information about data (labels, train/val dataset)
- run 'nnUNet_plan_and_preprocess -t $task --verify_dataset_integrity' to check where dataset respect to nnUnet format for input.
  Replace $task by 1 for PET images with SUV normalization
- run 'nnUNet_train 3d_fullres nnUNetTrainerV2 $task $fold' to obtain a pretrained model. This pretrained model will be stocked in '/media/nguyen-k/nnUNet_trained_models/nnUNet/3d_fullres/Task001/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/model_best.model.pkl'

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
