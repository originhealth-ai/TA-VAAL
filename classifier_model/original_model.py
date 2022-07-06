import torchvision
import torchvision.transforms.functional as F
from torchvision import transforms
from torch import nn
import torch
import glob
import os
from PIL import Image
import numpy as np
from torchvision import transforms
from copy import deepcopy
import cv2
from typing import List, Tuple
from tqdm.notebook import tqdm
import pdb
import classifier_model.components.classificationmodels as clf


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class OriginalClassificationModel(torch.nn.Module):
    def __init__(self, backbone,classifiers) -> None:
        super().__init__()
        

        self._arch={'vgg16': clf.vgg16_bn, 'vgg19': clf.vgg19_bn, 'resnet50': clf.resnet50 ,
                    'resnet101': clf.resnet101 ,'resnet152': clf.resnet152 ,'resnext50': clf.resnext50_32x4d ,
                    'resnext101': clf.resnext101_32x8d,'wideresnet50': clf.wide_resnet50_2 ,'wideresnet101': clf.wide_resnet101_2}
        
        self.classifier_model = self._arch[backbone](num_classes=classifiers['n_class'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        pred = self.classifier_model(x)

        return pred