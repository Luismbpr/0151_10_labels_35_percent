# Food Image Classification App
### 0151_Image_Classification_Using_PyTorch_10_Labels_35_Percent
#### 0151_10_labels_35_percent

## Table of contents
* [About](#about)
* [General Information](#general-information)
* [Libraries Used](#libraries-used)
* [Dataset Information](#dataset-information)
* [Model Creation](#model-creation)
* [Side Notes](#side-notes)
* [Resources](#resources)
* [Other](#other)
* [Aknowledgement](#aknowledgement)
* [License](#license)

## About
Multilabel Image Classification for 10 labels trained on 35 percent of total data

<br>

## General Information
Image classification web app that can predict 10 different types of food given an input image.

The app uses a Feature Extractor (EfficientNet) to predict whether an image belongs to one of the following food images:
- French fries
- Hamburger
- Hot dog
- Lasagna
- Nachos
- Pizza
- Samosa
- Steak
- Sushi
- Tacos

<br>


## Libraries Used
Some of the libraries used for this project:
- Pytorch
- Gradio
- Pillow


<br>

## Dataset Information

Labels	Avg-Images-Per-Label	Total-Images	Percentage
101	1000	101000	100
        3500	3.5
            
Labels	Avg-Images-Per-Label	Total-Images	Percentage
10	1000	10000	100
        3500	35


Total-Images	Percentage
3500	100
2625	75
875	25

Whole Dataset 101000 - 100\%
         Train: 2625 -  2.60\%
         Test:   875 -  0.87\%

[INFO] creating image split for train...
[INFO] Getting random subset of 2625 images for train...
[INFO] creating image split for test...
[INFO] Getting random subset of 875 images for test...
Creating directory:
/content/drive/MyDrive/Colab Notebooks/0151_PTCH_FDLN/0151_011_Model_Deployment/data/food_101_more_classes_35_percent

amount to get 0.35

target_dir:
/content/drive/MyDrive/Colab Notebooks/0151_PTCH_FDLN/0151_011_Model_Deployment/data/food_101_more_classes_35_percent

Train: samosa: 270
Train: hamburger: 262
Train: french_fries: 252
Train: sushi: 250
Train: lasagna: 272
Train: hot_dog: 249
Train: steak: 279
Train: tacos: 254
Train: pizza: 259
Train: nachos: 278
Test: pizza: 89
Test: steak: 84
Test: sushi: 93
Test: samosa: 90
Test: hot_dog: 88
Test: nachos: 86
Test: lasagna: 100
Test: tacos: 83
Test: hamburger: 77
Test: french_fries: 85

Downloading: "https://download.pytorch.org/models/efficientnet_b2_rwightman-c35c1473.pth" to /root/.cache/torch/hub/checkpoints/efficientnet_b2_rwightman-c35c1473.pth
100%|██████████| 35.2M/35.2M [00:00<00:00, 85.2MB/s]

============================================================================================================================================
Layer (type (var_name))                                      Input Shape          Output Shape         Param #              Trainable
============================================================================================================================================
EfficientNet (EfficientNet)                                  [1, 3, 224, 224]     [1, 10]              --                   Partial
├─Sequential (features)                                      [1, 3, 224, 224]     [1, 1408, 7, 7]      --                   False
│    └─Conv2dNormActivation (0)                              [1, 3, 224, 224]     [1, 32, 112, 112]    --                   False
│    │    └─Conv2d (0)                                       [1, 3, 224, 224]     [1, 32, 112, 112]    (864)                False
│    │    └─BatchNorm2d (1)                                  [1, 32, 112, 112]    [1, 32, 112, 112]    (64)                 False
│    │    └─SiLU (2)                                         [1, 32, 112, 112]    [1, 32, 112, 112]    --                   --
│    └─Sequential (1)                                        [1, 32, 112, 112]    [1, 16, 112, 112]    --                   False
│    │    └─MBConv (0)                                       [1, 32, 112, 112]    [1, 16, 112, 112]    (1,448)              False
│    │    └─MBConv (1)                                       [1, 16, 112, 112]    [1, 16, 112, 112]    (612)                False
│    └─Sequential (2)                                        [1, 16, 112, 112]    [1, 24, 56, 56]      --                   False
│    │    └─MBConv (0)                                       [1, 16, 112, 112]    [1, 24, 56, 56]      (6,004)              False
│    │    └─MBConv (1)                                       [1, 24, 56, 56]      [1, 24, 56, 56]      (10,710)             False
│    │    └─MBConv (2)                                       [1, 24, 56, 56]      [1, 24, 56, 56]      (10,710)             False
│    └─Sequential (3)                                        [1, 24, 56, 56]      [1, 48, 28, 28]      --                   False
│    │    └─MBConv (0)                                       [1, 24, 56, 56]      [1, 48, 28, 28]      (16,518)             False
│    │    └─MBConv (1)                                       [1, 48, 28, 28]      [1, 48, 28, 28]      (43,308)             False
│    │    └─MBConv (2)                                       [1, 48, 28, 28]      [1, 48, 28, 28]      (43,308)             False
│    └─Sequential (4)                                        [1, 48, 28, 28]      [1, 88, 14, 14]      --                   False
│    │    └─MBConv (0)                                       [1, 48, 28, 28]      [1, 88, 14, 14]      (50,300)             False
│    │    └─MBConv (1)                                       [1, 88, 14, 14]      [1, 88, 14, 14]      (123,750)            False
│    │    └─MBConv (2)                                       [1, 88, 14, 14]      [1, 88, 14, 14]      (123,750)            False
│    │    └─MBConv (3)                                       [1, 88, 14, 14]      [1, 88, 14, 14]      (123,750)            False
│    └─Sequential (5)                                        [1, 88, 14, 14]      [1, 120, 14, 14]     --                   False
│    │    └─MBConv (0)                                       [1, 88, 14, 14]      [1, 120, 14, 14]     (149,158)            False
│    │    └─MBConv (1)                                       [1, 120, 14, 14]     [1, 120, 14, 14]     (237,870)            False
│    │    └─MBConv (2)                                       [1, 120, 14, 14]     [1, 120, 14, 14]     (237,870)            False
│    │    └─MBConv (3)                                       [1, 120, 14, 14]     [1, 120, 14, 14]     (237,870)            False
│    └─Sequential (6)                                        [1, 120, 14, 14]     [1, 208, 7, 7]       --                   False
│    │    └─MBConv (0)                                       [1, 120, 14, 14]     [1, 208, 7, 7]       (301,406)            False
│    │    └─MBConv (1)                                       [1, 208, 7, 7]       [1, 208, 7, 7]       (686,868)            False
│    │    └─MBConv (2)                                       [1, 208, 7, 7]       [1, 208, 7, 7]       (686,868)            False
│    │    └─MBConv (3)                                       [1, 208, 7, 7]       [1, 208, 7, 7]       (686,868)            False
│    │    └─MBConv (4)                                       [1, 208, 7, 7]       [1, 208, 7, 7]       (686,868)            False
│    └─Sequential (7)                                        [1, 208, 7, 7]       [1, 352, 7, 7]       --                   False
│    │    └─MBConv (0)                                       [1, 208, 7, 7]       [1, 352, 7, 7]       (846,900)            False
│    │    └─MBConv (1)                                       [1, 352, 7, 7]       [1, 352, 7, 7]       (1,888,920)          False
│    └─Conv2dNormActivation (8)                              [1, 352, 7, 7]       [1, 1408, 7, 7]      --                   False
│    │    └─Conv2d (0)                                       [1, 352, 7, 7]       [1, 1408, 7, 7]      (495,616)            False
│    │    └─BatchNorm2d (1)                                  [1, 1408, 7, 7]      [1, 1408, 7, 7]      (2,816)              False
│    │    └─SiLU (2)                                         [1, 1408, 7, 7]      [1, 1408, 7, 7]      --                   --
├─AdaptiveAvgPool2d (avgpool)                                [1, 1408, 7, 7]      [1, 1408, 1, 1]      --                   --
├─Sequential (classifier)                                    [1, 1408]            [1, 10]              --                   True
│    └─Dropout (0)                                           [1, 1408]            [1, 1408]            --                   --
│    └─Linear (1)                                            [1, 1408]            [1, 10]              14,090               True
============================================================================================================================================
Total params: 7,715,084
Trainable params: 14,090
Non-trainable params: 7,700,994
Total mult-adds (M): 657.65
============================================================================================================================================
Input size (MB): 0.60
Forward/backward pass size (MB): 156.80
Params size (MB): 30.86
Estimated Total Size (MB): 188.26
============================================================================================================================================

EfficientNetB2 Feature Extractor was trained on [Food101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/).
[Food-101 dataset](https://pytorch.org/vision/main/generated/torchvision.datasets.Food101.html) was downloaded From PyTorch. A percentage of the entire dataset was randomly selected and gathered for specific labels to retrain the model.


<br>


<br>

## Model Creation

There were several model iterations previous to this model.
For more information please see the model report.

####
The first iteration of the model was trained on 3 classes with 15% of the data.
The second iteration was using the same 3 classes with 20% of the data.
The third iteration was to select 10 labels and train them on 20% of the data
The fourth iteration was to select 10 labels and train them on 35 % of the data.
The fourth iteration resulted on the best-performing model for the 10 labels selected.
####

This iteration consisted on creating a model with 46 labels using 50% of the whole dataset, which consists on around 1000 images per label.

##### Splits

| | Train | Test |
|--|--|--|
| | 80 % | 20 % |


- Example Images: Performed a last test with some random sample images gathered from the test set. Some of those images were used for the example dataset


<br>

## Side Notes
For more information about this project please see 'main_report.pdf'. This reports contains more information regarding the scope of this project along with the decisions taken, the project implementation and some of its results.

<br>

## Resources

##### Some of the resources used:
*Note: For the list of complete resources and aknowledgments please see 'main_report.pdf'.*

[EfficientNet](https://pytorch.org/vision/main/models/efficientnet.html)

[Food-101 Dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)

[Food-101 Dataset on PyTorch](https://pytorch.org/vision/main/generated/torchvision.datasets.Food101.html)

@inproceedings{bossard14,
  title = {Food-101 -- Mining Discriminative Components with Random Forests},
  author = {Bossard, Lukas and Guillaumin, Matthieu and Van Gool, Luc},
  booktitle = {European Conference on Computer Vision},
  year = {2014}
}

[Daniel Bourke's PyTorch For Deep Learning YouTube](https://www.youtube.com/watch?v=V_xro1bcAuA&t=4695s)


## Other

Local Virtual Environment Information
- Environment Name: venv_0151_Deployment_310_002
- Python Version: 3.10.12

## Aknowledgement

I would like to express my gratitude. To PyTorch for the amount of information and amount of work they have done. To Gradio for making deployment simple; To the people who created and released the dataset used (Food101).

I would like to express my gratitude to Mr. Daniel Bourke, especially for the resources he has put available.

Thank you to all who make information, resources and knowledge available, especially to everyone who do it without any paywall.

Thank you to everyone who has contributed towards the improvement of this fascinating area of Artificial Intelligence.


## License
MIT License








Epoch & Train-loss & Train-acc & Test-loss & Test-acc\\
1 & 0.5501 & 0.8415 & 0.6448 & 0.8018\\
2 & 0.5287 & 0.8535 & 0.6715 & 0.7939\\
3 & 0.5318 & 0.8494 & 0.6324 & 0.8062\\
4 & 0.5263 & 0.8539 & 0.6598 & 0.7939\\
5 & 0.5102 & 0.8584 & 0.6354 & 0.8084\\
6 & 0.493 & 0.8626 & 0.6344 & 0.8006\\
7 & 0.495 & 0.863 & 0.6497 & 0.7919\\
8 & 0.4633 & 0.8652 & 0.6265 & 0.7985\\
9 & 0.4461 & 0.8727 & 0.6535 & 0.7974\\
10 & 0.4621 & 0.869 & 0.6346 & 0.8006\\
11 & 0.5244 & 0.8358 & 0.6401 & 0.8074\\
12 & 0.508 & 0.8434 & 0.6488 & 0.7928\\
13 & 0.5071 & 0.8336 & 0.6123 & 0.824\\
14 & 0.4993 & 0.8475 & 0.6494 & 0.8106\\
15 & 0.494 & 0.8441 & 0.6309 & 0.7994\\
16 & 0.4841 & 0.8419 & 0.6305 & 0.8151\\
17 & 0.4474 & 0.8577 & 0.6422 & 0.8119\\
18 & 0.4792 & 0.8479 & 0.6183 & 0.8084\\
19 & 0.4752 & 0.8468 & 0.6217 & 0.8128\\
20 & 0.4354 & 0.8554 & 0.6438 & 0.8128\\
