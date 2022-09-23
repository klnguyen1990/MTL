#!/bin/bash

task=1

#Set Origine and Spacing
jupyter nbconvert --to python setOrig.ipynb
path_Origine='/home/nguyen-k/Bureau/segCassiopet2/setOrig.py'
python3 $path_Origine --task $task

#Générer fichier data.json
jupyter nbconvert --to python makejson.ipynb
path_json='/home/nguyen-k/Bureau/segCassiopet2/makejson.py'
python3 $path_json --task $task

#Preprocess
nnUNet_plan_and_preprocess -t $task --verify_dataset_integrity

# training and validation
for fold in 0 1 2 3 4
do
    nnUNet_train 3d_fullres nnUNetTrainerV2 $task $fold
done