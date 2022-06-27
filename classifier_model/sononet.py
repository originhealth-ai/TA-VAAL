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
from classifier_model.components.inceptionbackbones import inceptionv4,inception_resnet_v2
import classifier_model.components.attentionheads as atn
import classifier_model.components.classificationmodels as clf


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SononetModel(torch.nn.Module):
    def __init__(self, classifiers) -> None:
        super().__init__()
        
        self.classifier_model = clf.load_sononet_model_weights(num_classes=classifiers['n_class'],ckpt_path="classifier_framework/saved_models/SN64.pth")

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        pred = self.classifier_model(x)

        return pred