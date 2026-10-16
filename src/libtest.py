# Tests des librairies

# Segment anything
from segment_anything import SamPredictor, sam_model_registry

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

# YOLO
from ultralytics import YOLO

# Albumentations
import albumentations as A

# print
print("All libraries can be imported !")