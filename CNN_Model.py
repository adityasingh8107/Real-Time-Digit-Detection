#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as tf


# In[2]:


class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1,32,kernel_size=5, stride=1,padding=2)
        self.conv2 = nn.Conv2d(32,64,kernel_size=5,stride=1,padding=2)
        self.fc1 = nn.Linear(64*7*7,64)
        self.fc2 = nn.Linear(64,10)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        x = x.view(-1,64*7*7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return F.softmax(x)


# In[ ]:




