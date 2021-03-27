import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import copy
import math
import random

import timm
import PIL
import re

IMAGE_SIZE = 224
import os
import random
seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device('cpu') # Model Evaluated on CPU
# Load in the Unique Classes and Ingredients 
to_tensor = torchvision.transforms.ToTensor()
Unique_Ingredients = torch.load("./Ingredients.pth")
Unique_Classes = torch.load("./Classes.pth")

test_transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(),
    ToTensorV2()
])
# Load the Model
class ConvBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, padding, groups):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size = kernel_size, padding = padding, groups = groups)
        self.bn = nn.BatchNorm2d(out_features)
        self.act1 = nn.SiLU(inplace = True)
    def forward(self, x):
        return self.bn(self.act1(self.conv(x)))
class SqueezeExcite(nn.Module):
    def __init__(self, in_channels, inner_channels):
        super().__init__()
        self.in_channels = in_channels
        self.inner_channels = inner_channels
        self.Squeeze = nn.Linear(self.in_channels, self.inner_channels)
        self.act1 = nn.SiLU(inplace = True)
        self.Excite = nn.Linear(self.inner_channels, self.in_channels)
    def forward(self, x):
        avg_pool = torch.mean(x, dim = -1)
        avg_pool = torch.mean(avg_pool, dim = -1)
        squeezed = self.act1(self.Squeeze(avg_pool))
        excited = torch.sigmoid(self.Excite(squeezed)).unsqueeze(-1).unsqueeze(-1)
        return excited * x
class BottleNeck(nn.Module):
    def __init__(self, in_channels, inner_channels, device):
        super().__init__()
        self.device = device 
        self.in_channels = in_channels
        self.inner_channels = inner_channels
        
        self.Squeeze = ConvBlock(self.in_channels, self.inner_channels, 1, 0, 1)
        self.Process = ConvBlock(self.inner_channels, self.inner_channels, 3, 1, 1)
        self.Expand = ConvBlock(self.inner_channels, self.in_channels, 1, 0, 1)
        self.SE = SqueezeExcite(self.in_channels, self.in_channels // 16)
        self.gamma = nn.Parameter(torch.zeros((1), device = self.device))
    def forward(self, x):
        squeezed = self.Squeeze(x) 
        processed = self.Process(squeezed)
        expanded = self.Expand(processed)
        excited = self.SE(expanded)
        return self.gamma * excited + x
class ModifiedResNetQT(nn.Module):
    def freeze(self, x):
        for parameter in x.parameters():
            parameter.requires_grad = False
    def unfreeze(self, x):
        for parameter in x.parameters():
            parameter.requires_grad = True
    def __init__(self, in_dim, device, drop_prob = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.drop_prob = drop_prob
        self.device = device
        
        self.model_name = "resnet200d"
        self.model = timm.create_model(self.model_name, pretrained = False)
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        self.freeze(self.model)
        # Extract Layers
        self.conv1 = self.model.conv1
        self.bn1 = self.model.bn1
        self.act1 = self.model.act1
        self.pool = self.model.maxpool
        
        self.layer1 = self.model.layer1
        self.layer2 = self.model.layer2
        self.layer3 = self.model.layer3
        self.layer4 = self.model.layer4
        #self.unfreeze(self.layer3)
        self.unfreeze(self.layer4)
        
        
        self.Attention1 = nn.Identity()#SqueezeExcite(1024, 128)
        self.Attention2 = SqueezeExcite(2048, 256)
        self.Attention3 = SqueezeExcite(2048, 256)
        self.features_extract = nn.Sequential(*[
            BottleNeck(2048, 512, self.device) for i in range(7)
        ])
        self.proj = ConvBlock(2048, self.in_dim, 1, 0, 1)
        self.dropout = nn.Dropout2d(self.drop_prob)
        
    def forward(self, x):
        features0 = self.pool(self.bn1(self.act1(self.conv1(x))))
        layer1 = self.layer1(features0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        # Attention1
        layer3 = self.Attention1(layer3)
        layer4 = self.layer4(layer3)
        # Attention2
        layer4 = self.Attention2(layer4)
        # Dropout
        layer4 = self.dropout(layer4)
        # Additional Processing
        features = self.features_extract(layer4)
        features = self.Attention3(features)
        features = self.proj(features)
        return features
class FeatureExtractor(nn.Module):
    def __init__(self, out_dim, device):
        super().__init__()
        self.out_dim = out_dim
        self.device = device
        self.ResNet = ModifiedResNetQT(self.out_dim, self.device)
    def forward(self, images):
        features = self.ResNet(images)
        return features # (B, 384, 7, 7)
class Pyra_Class(nn.Module):
    def __init__(self, num_classes, device, drop_prob = 0.2):
        # File_path: the path to the pretrained feature extractor
        # Im_size: the size of the image after encoding(7x7 = 49), allows us to precompute positional encodings
        super().__init__()
        self.device = device
        self.drop_prob = drop_prob 
        self.in_dim = 4096
        self.num_classes = num_classes
        self.Class_Encoder = FeatureExtractor(self.in_dim, self.device)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.Dropout = nn.Dropout(self.drop_prob)
        self.ClassLinear = nn.Linear(self.in_dim, self.num_classes) # Class Classification Head
    def save(self):
        torch.save(self.state_dict(), "./Class.pth")
    def forward(self, x):
        encoded_class = self.Class_Encoder(x)
        processed_class = torch.squeeze(self.avg_pool(encoded_class))
        if len(processed_class.shape) == 1:
            processed_class = processed_class.unsqueeze(0)
        processed_class = self.Dropout(processed_class)
        Class = self.ClassLinear(processed_class)
        return Class
class Pyra_Ing(nn.Module):
    def __init__(self, num_ing_classes, device, drop_prob = 0.2):
        # File_path: the path to the pretrained feature extractor
        # Im_size: the size of the image after encoding(7x7 = 49), allows us to precompute positional encodings
        super().__init__()
        self.device = device
        self.drop_prob = drop_prob 
        self.in_dim = 4096
        self.num_ing_classes = num_ing_classes
        self.Ing_Encoder = FeatureExtractor(self.in_dim, self.device)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.Dropout = nn.Dropout(self.drop_prob)
        self.IngLinear = nn.Linear(self.in_dim, self.num_ing_classes) # Ingredient Classification Head
    def save(self):
        torch.save(self.state_dict(), "./Ingredient.pth")
    def forward(self, x):
        encoded_ing = self.Ing_Encoder(x)
        processed_ing = torch.squeeze(self.avg_pool(encoded_ing))
        if len(processed_ing.shape) == 1:
            processed_ing = processed_ing.unsqueeze(0)
        processed_ing = self.Dropout(processed_ing)
        IngClass= self.IngLinear(processed_ing)
        return IngClass
class Pyra(nn.Module):
    def __init__(self, num_ing_classes, num_classes, device, drop_prob = 0.2):
        # File_path: the path to the pretrained feature extractor
        # Im_size: the size of the image after encoding(7x7 = 49), allows us to precompute positional encodings
        super().__init__()
        self.device = device
        self.drop_prob = drop_prob 
        self.num_classes = num_classes
        self.num_ing_classes = num_ing_classes
        self.PyraClass = Pyra_Class(self.num_classes, self.device, drop_prob = self.drop_prob)
        self.PyraIng = Pyra_Ing(self.num_ing_classes, self.device, drop_prob = self.drop_prob)
    def load_ing(self, path):
        self.PyraIng.load_state_dict(torch.load(path, map_location = self.device))
    def load_class(self, path):
        self.PyraClass.load_state_dict(torch.load(path, map_location = self.device))
    def save_ing(self):
        self.PyraIng.save()
    def save_class(self):
        self.PyraClass.save()
    def forward_ing(self, x):
        return self.PyraIng(x) 
    def forward_class(self, x):
        return self.PyraClass(x)
    def forward(self, x):
        return self.PyraIng(x), self.PyraClass(x)
class PyraSolver(nn.Module):
    def __init__(self, num_ing_classes, num_classes, device):
        super().__init__()
        self.device = device
        self.num_ing_classes = num_ing_classes
        self.num_classes = num_classes
        self.Pyra = Pyra(self.num_ing_classes, self.num_classes, self.device)
    def forward(self, images):
        '''
        Test Time Inference on the Model
        '''
        self.eval()
        with torch.no_grad():
            IngPred, ClassPred= self.Pyra(images)
            IngPred = torch.sigmoid(IngPred)
            ClassPred = F.softmax(ClassPred)
            # Grab Top 15 ingredients(may have some false positives)
            _, sort = torch.sort(IngPred, descending = True)
            B, C = sort.shape
            sort = sort[:, :15]
            _, selected_classes = torch.max(ClassPred, dim = -1)
            indices = torch.zeros((B, C), dtype = torch.bool)
            for b in range(B):
                for idx in range(15):
                    index = sort[b, idx].item()
                    indices[b, index] = True
            return indices, selected_classes
def load_model():
    model = PyraSolver(len(Unique_Ingredients), len(Unique_Classes), device)
    model.load_state_dict(torch.load('./Pyra.pth', map_location = device))
    return model
def decode_ing(pred):
    '''
    pred: Tensor(C)
    '''
    C = pred.shape[0]
    ingredients = []
    for c in range(C):
        if pred[c] == True:
            ingredients += [Unique_Ingredients[c]]
    return ingredients
def decode(predIng, predClass):
    '''
    Decodes the Predictions from the model
    '''
    class_selected = Unique_Classes[predClass.item()]
    ingredients = decode_ing(predIng)
    return ingredients, class_selected
def inference(model, image):
    # Load in Image and Augment TTA
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = test_transforms(image = image)['image'].unsqueeze(0)
    predIng, predClass = model(image)
    predIng = torch.squeeze(predIng)
    predClass = torch.squeeze(predClass)
    return decode(predIng, predClass)
