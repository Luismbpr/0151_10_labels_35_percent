
## Turning Gradio App into a Python Script (app.py)
###- Setup
##    - Import libraries
import gradio as gr
import os
import torch
#from torch import nn
#import torchvision

from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict

## Python File Creation - model.py
## Creates an EfficientNetB2 model with best weights and model transforms
import torch
import torchvision
from torch import nn

#def create_effnetb2_model(num_classes:int=10,
#                          seed:int=42):
#    """
#    Creates an EfficientNetB2 feature extractor model
#    with it's default (best-performing) weights and an EfficientNetB2 transforms.
#    
#    Used to create an instance of feature extractor and transform the input images.
#    
#    
#    Args:
#        num_classes: integer (optional): Number of classes in the classifier head.
#                        Default=10.
#        seed: integer (optional). Random seed value. Default=42
#    
#    Returns: 
#        Model, Transforms
#        Model: (torch.nn.Module) EffNetB2 Feature Extractor model.
#        Transforms: (torchvision.transforms): EffNetB2 image transforms.
#    
#    """
#    ## 1, 2, 3 Create EffNetB2 Pretrained weights, transforms and model
#    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
#    transforms = weights.transforms()
#    model = torchvision.models.efficientnet_b2(weights=weights)
#    
#    ## Freeze all layers in the base model
#    for param in model.parameters():
#        param.requires_grad = False
#    
#    ##Modify classifier head according to number of classes on the output
#    torch.manual_seed(seed)
#    model.classifier = nn.Sequential(
#        nn.Dropout(p=0.3, inplace=True),
#        nn.Linear(in_features=1408, out_features=num_classes)
#    )
#    return model, transforms


##    - Setup class names
class_names = ["french_fries", "hamburger", "hot_dog", "lasagna", "nachos", "pizza", "samosa", "steak", "sushi", "tacos"]
#print(f"class_names: {class_names}")
## - Preparation
##    - Model, model weights and transforms preparation
effnetb2, effnetb2_transforms = create_effnetb2_model(num_classes=len(class_names),
                                                      seed=42)

#create_effnetb2_model(num_classes=10, seed=42)
#effnetb2, effnetb2_transforms = create_effnetb2_model(num_classes=10)

## Load trained state_dict weights
effnetb2.load_state_dict(
    torch.load(f="001_001_pretrained_effnetb2_feature_extractor_10_labels_35_percent_20_epochs.pth",
               map_location=torch.device("cpu")
              )
)

#effnetb2.load_state_dict(
#    torch.load(f="/Users/luis/Documents/Proyects_To_Export/0151_FoodImageClassification_10Labels/venv_0151_Deployment_312_001/001_001_pretrained_effnetb2_feature_extractor_10_labels_35_percent_20_epochs.pth",
#               map_location=torch.device("cpu")
#              )
#)


# Load saved weights
#effnetb2.load_state_dict(
#    torch.load(
#        f="001_001_pretrained_effnetb2_feature_extractor_10_labels_35_percent_20_epochs.pth",
#        map_location=torch.device("cpu"),  # load to CPU
#    )
#)

##- Predict Function
##- Inference
def predict(img) -> Tuple[Dict, float]:
    """
    Transforms and performs prediction on the input image. Returns prediction label and the time it took to perform the prediction.
    
    Args:
        img: Image (PIL)
    
    Returns: 
        Prediction: Prediction Label and prediction probability.
        Prediction Time: Time it took to perform the prediction (seconds).
    
    """
    ## Start a timer
    start_time = timer()
    
    ## Transform the input image for use with EffNetB2
    img = effnetb2_transforms(img).unsqueeze(0) ## unsqueeze - Add a batch dimension on 0th index
    
    ## Put model into eval model and inference mode
    effnetb2.eval()
    with torch.inference_mode():
        
        ## Forward Pass on Transformed image
        ## Pred logits -> Prediction probabilities
        pred_probs = torch.softmax(effnetb2(img), dim=1)
    
    ## Create a prediction label and predction probability dictionary
    #pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    
    ## Calculate pred time
    end_time = timer()
    pred_time = round(end_time - start_time, 5)
    
    ## Return pred dict and pred time
    return pred_labels_and_probs, pred_time

## - Gradio App
## Create title, description, article
title = "Food Image Classification"
##description = "An [EfficientNetB2 Feature Extractor](https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b2.html#torchvision.models.efficientnet_b2) computer vision model that classifies images for 10 labels: french fries hamburger, hot dog, lasagna, nachos, pizza, samosa, steak, sushi, tacos "
description = "The model can predict the following food labels: french fries, hamburger, hot dog, lasagna, nachos, pizza, samosa, steak, sushi, tacos."
article = "Using an EfficientNet Feature Extractor [aided by](https://github.com/mrdbourke)"

## Example list
example_list = [["examples/" + example] for example in os.listdir("examples")]
#print(example_list)
#example_list = [["examples/" + example] for example in os.listdir("examples")]

##    - Gradio interface (Inputs, Outputs)
gradio_app = gr.Interface(fn=predict,
                          inputs=gr.Image(type='pil'),
                          outputs=[gr.Label(num_top_classes=10, label="Predictions"),
                                   gr.Number(label="Prediction Time (seconds)")],
                          examples=example_list,
                          title=title,
                          description=description,
                          article=article)

##    - Gradio Launch
## App Launch
gradio_app.launch()


