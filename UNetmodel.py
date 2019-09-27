#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 13:11:11 2019

@author: wei
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class DownConv(nn.Module):
    """
    Encoder block has 2 convolution layers and 1 maxpooling layer.
    Encoder block will extract features and down-sample the feature maps.
    Note: if the length and width of feature map is odd number, then add 
    padding in maxpooling layer so that the size of feature maps is matching 
    in decoder.
    
    """
    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        feature_map = x

        if self.pooling:
            if x.size()[2]%2 == 1 or x.size()[3]%2 == 1:
                x = F.max_pool2d(x, kernel_size=2, stride=2, padding=1)
            else:
                x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        return x, feature_map

class UpConv(nn.Module):
    """
    Decoder block has 1 transpose convolution layer and 2 convolution layers.
    Decoder block will up-sample feature maps.
    Skip-connection is performed by concatenating feature maps from encoder to 
    coresponding decoder.

    """
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(2*self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)


    def forward(self, down_maps, up_maps):
        """ 
        Parameters:
            down_maps: feature maps extracted from the corsponding encoder block
            up_maps: feature maps generated in the decoder block
        """
        
        up_maps = self.upconv(up_maps)

        #crop the up image size to be the same as feature in encoder if the
        #size of feature is odd number
        x = torch.cat((up_maps[:,:,:down_maps.size()[2],:down_maps.size()[3]], down_maps), 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x
    

class UNet(nn.Module):
    """ 
    UNet implementation is based on https://arxiv.org/abs/1505.04597 and adapted
    from https://github.com/jaxony/unet-pytorch.
    
    Modifications to the original paper:
    (1) To keep the spacial dimension, padding is used in all convolution layers
    (2) Skip connectiong works for any size of feature maps 
    """

    def __init__(self, num_classes, in_channels=3, depth=5, start_filts=64):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor. If input
            is RGB image, the channel is 3.
            depth: int, number of decoders in the network
            start_filts: int, number of convolutional filters in the first encoder block.
        """
        super(UNet, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth

        self.down_convs = []
        self.up_convs = []

        # Create a encoder block list saving all the encoder operations
        for i in range(depth):
            if i == 0:
                temp_in_channels = self.in_channels
            else:
                 temp_in_channels = temp_out_channels   
            temp_out_channels = self.start_filts*(2**i)
            pooling = True if i < depth-1 else False
            encoder_block = DownConv(temp_in_channels, temp_out_channels, pooling=pooling)
            self.down_convs.append(encoder_block)
  
        # Create a dncoder block list saving all the dncoder operations
        # Decoder path has depth-1 blocks (one less than encoder path)
        for k in range(depth-1):
            temp_in_channels = temp_out_channels
            temp_out_channels = temp_in_channels // 2
            decoder_block = UpConv(temp_in_channels, temp_out_channels)
            self.up_convs.append(decoder_block)

        self.conv_final = nn.Conv2d(temp_out_channels, self.num_classes, kernel_size=1)

        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)


    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)


    def forward(self, x):
        encoder_outs = []
        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, feature_map = module(x)
            encoder_outs.append(feature_map)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)
        
        x = self.conv_final(x)
        
        return x   


