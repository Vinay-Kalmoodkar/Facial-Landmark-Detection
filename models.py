## Defining the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        print('Hello')
        super(Net, self).__init__()
        
        # First convolution layer
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # 224---> (W-F)/S +1 --> (224-5)+1 --->32,220,220-->MaxPooling-->32,110,110
        self.conv1 = nn.Conv2d(1, 32, 5)

        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Second convolution layer
        # 32,110,110 ---> (110-5)+1 ---> 32,106,106--->MaxPooling---> 32,53,53
        self.conv2 = nn.Conv2d(32,32, 5)

        # Third convolution layer
        # 32,53,53 ------> 53-3 +1 ----> 16,51,51---->MaxPooling--> 16,25,25
        self.conv3 = nn.Conv2d(32, 16, 3)

        # Fourth convolution layer
        # 32,53,53 ------> 25-5 +1 ----> 16,21,21---->MaxPooling--> 16,10,10
        self.conv4 = nn.Conv2d(16, 16, 5)
        
        # First Linear/Dense layer which expects the inputs in 1D
        self.fc1 = nn.Linear(16*10*10,50)

        # Dropout layer to avoid over fitting
        self.fc1_drop = nn.Dropout(p=0.4)
        
        # finally, create 136 output channels (for the 10 classes)
        self.fc2 = nn.Linear(50, 136)

        
    def forward(self, x):
        ## x is the input image
        ## 4 sets of Convolution + Activation + Maxpooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        #flattening
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
