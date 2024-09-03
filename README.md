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

| Labels | Avg-Images-Per-Label | Total-Images | Percentage |
|--|--|--|--|
| 101 | 1000 | 101000 | 100 |
| | | 3500 | 3.5 |

| Labels | Avg-Images-Per-Label | Total-Images | Percentage |
|--|--|--|--|
| 10 | 1000 | 10000 | 100|
| | | 3500 | 35 |


| Total-Images | Percentage |
|--|--|
| 3500 | 100 |
| 2625 | 75 |
| 875 | 25 |

| | | Total-Images | Percentage |
|--|--|--|--|
| Whole Dataset | 101000 | 100 % |
| |Train | 2625 | 2.60 % |
| |Test | 875 | 0.87 % |


| Split | Label | Images Per Label |
|--|--|--|
| Train | samosa | 270 |
| Train | hamburger | 262 |
| Train | french_fries | 252 |
| Train | sushi | 250 |
| Train | lasagna | 272 |
| Train | hot_dog | 249 |
| Train | steak | 279 |
| Train | tacos | 254 |
| Train | pizza | 259 |
| Train | nachos | 278 |
| Test | pizza | 89 |
| Test | steak | 84 |
| Test | sushi | 93 |
| Test | samosa | 90 |
| Test | hot_dog | 88 |
| Test | nachos | 86 |
| Test | lasagna | 100 |
| Test | tacos | 83 |
| Test | hamburger | 77 |
| Test | french_fries | 85 |

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

| Train | Test |
|--|--|
| 80 % | 20 % |


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