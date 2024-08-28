## Python File Creation - model.py
## Creates an EfficientNetB2 model with best weights and model transforms
import torch
import torchvision
from torch import nn

def create_effnetb2_model(num_classes:int=10,
                          seed:int=42):
    """
    Creates an EfficientNetB2 feature extractor model
    with it's default (best-performing) weights and an EfficientNetB2 transforms.
    
    Used to create an instance of feature extractor and transform the input images.
    
    
    Args:
        num_classes: integer (optional): Number of classes in the classifier head.
                        Default=10.
        seed: integer (optional). Random seed value. Default=42
    
    Returns: 
        Model, Transforms
        Model: (torch.nn.Module) EffNetB2 Feature Extractor model.
        Transforms: (torchvision.transforms): EffNetB2 image transforms.
    
    """
    ## 1, 2, 3 Create EffNetB2 Pretrained weights, transforms and model
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    transforms = weights.transforms()
    model = torchvision.models.efficientnet_b2(weights=weights)
    
    ## Freeze all layers in the base model
    for param in model.parameters():
        param.requires_grad = False
    
    ##Modify classifier head according to number of classes on the output
    torch.manual_seed(seed)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1408, out_features=num_classes)
    )
    
    return model, transforms

