import torch as T
import torch.functional as F
import torch.nn as nn
from torchsummary import summary

"""
Legend:
    c_n -> 2D convolutional layer number n
    bn_n -> 2D batch normalization layer number n
    relu -> rectified linear unit activation function
    mp -> 2D max pooling layer
    ds_layer -> downsampling layer
    ap -> average pooling layer
    
"""

class Bottleneck_block(nn.Module):
    def __init__(self, n_inCh, n_outCh, stride = 1, ds_layer = None, expansion = 4):
        super(Bottleneck_block,self).__init__()
        self.expansion = expansion

        # 1st block
        self.c_1 = nn.Conv2d(n_inCh, n_outCh, kernel_size=1, stride=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(n_outCh)
        
        # 2nd block
        self.c_2 = nn.Conv2d(n_outCh, n_outCh, kernel_size=3, stride=stride, padding=1)
        self.bn_2 = nn.BatchNorm2d(n_outCh)
        
        # 3rd block
        self.c_3 = nn.Conv2d(n_outCh, n_outCh*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(n_outCh*self.expansion)
        
        # relu as a.f. for each block 
        self.relu = nn.ReLU()
        
        self.ds_layer  = ds_layer
        self.stride = stride
        
    
    def forward(self, x):
        x_init = x.clone()  # identity shortcuts
        
        # forwarding into the bottleneck layer
        
        x = self.relu(self.bn_1(self.c_1(x)))
        x = self.relu(self.bn_2(self.c_2(x)))
        x = self.bn_3(self.c_3(x))
        
        #downsample identity whether necessary and sum 
        if self.ds_layer is not None:
            x_init = self.ds_layer(x_init)
        x+=x_init
        x=self.relu(x)
        
        return x
        
        

# basic block 2 operations: conv + batch normalization
# bottleneck block 3 operations: conv + batch normalization + ReLU

class ResNet101(nn.Module):
    
    def __init__(self, n_channels = 3, n_classes = 81):
        super(ResNet101,self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.expansion = 4 # output/inpiut feature dimension ratio in bottleneck
        self.input_ch= 64
        self.bottleneck_struct = [3,4,23,3]
        self.fm_dim = [64,128,256,512] #feature map dimension
        
        self._create_net()
    
    
    def _create_net(self):
        # first block
        self.c_1 = nn.Conv2d(self.n_channels, self.fm_dim[0], 7, stride = 2, padding = 3, bias = False) # 7-> kernel size
        self.bn_1 = nn.BatchNorm2d(self.fm_dim[0])
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(self.fm_dim[0], stride = 7//2, padding= 1)
        
        # body blocks
        self.l1 = self._buildLayers(self.bottleneck_struct[0], self.fm_dim[0])
        self.l2 = self._buildLayers(self.bottleneck_struct[1], self.fm_dim[1], stride = 2)
        self.l3 = self._buildLayers(self.bottleneck_struct[2], self.fm_dim[2], stride = 2)
        self.l4 = self._buildLayers(self.bottleneck_struct[3], self.fm_dim[3], stride = 2)
        
        
        # last block
        self.ap = nn.AdaptiveAvgPool2d((1,1)) # (1,1) output dimension
        self.fc = nn.Linear(512*self.expansion, self.n_classes)
        
        self.af_out = nn.Sigmoid() # to use 
        
    def _buildLayers(self, n_blocks, n_fm, stride = 1):
        
        if stride != 1 or self.input_ch != n_fm*self.expansion:
            ds_layer = nn.Sequential(
                nn.Conv2d(self.input_ch, n_fm*self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(n_fm**self.expansion)
                )
        else:
            ds_layer = None
        list_layers = []
        
        list_layers.append(Bottleneck_block(self.input_ch, n_fm, ds_layer= ds_layer, stride = stride))
        self.input_ch = n_fm * self.expansion
        
        for index in range(n_blocks -1):
            list_layers.append(Bottleneck_block(self.input_ch, n_fm))
        
        return nn.Sequential(*list_layers)
        
    
        
    def forward(self, x):
        # first block
        x = self.mp(self.relu(self.bn_1(self.c_1(x))))
        
        # body blocks
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
    
        # last block
        x = self.ap(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x
    
# a = ResNet101()

        
        

        
        