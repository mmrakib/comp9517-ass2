#### COMP9517 23T3 Assignment 2

This is the offical code repository for group *Adult Supervision*.

**Developers**
- Mohammad M. Rakib (z5361151)
- adrianne_sun (z5373505)
- Huan Jie (Jay) Choo (z5367538)
- Plushii (z5404816)
- roterex (z5169764)

**Setup**
    there is a requierments.txt file provided that contains all requiered libs except for torch and torchvision
Setup a virtual enviroment with that and the models should run. Torch and TorchVision can be installed seperatly
for cuda support (a command has been provided in the requierments file). Either python 3.9.4 or 3.9.3 for uni
machines is recommended.

    all files for the elpv dataset should be placed directly in the src/elpv/ folder. Such that labels.csv has
the path ./src/elpv/labels.csv

    models should be placed in models/ and histories in histories/ (you may need to fix a file path
or 2 though i did my best to get them correct)

**Description**
    The objective of this project is to utilise computer vision algorithms on electroluminescence (EL) images to analyse the health of
photo-voltaic (PV) cells on solar panels.

The dataset (both for training and testing) used for this project is: https://github.com/zae-bayern/elpv-dataset

**Models**
KNN:                        src/SimpleKNN.ipynb
VGG-19 all at once:         src/AllAtOnce_VGG19.ipynb
VGG-19 by type:             src/cnn-test.py
ResNet:                     src/resnet.ipynb
Edge Detection:             src/EdgeDetect-test.ipynb

**Trained Model Files**
https://drive.google.com/drive/folders/1Vmo-yltTKYTgKgp2X27KkTd0YbgfeYWd?usp=sharing