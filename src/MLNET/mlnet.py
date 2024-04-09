""" This file is modified from:
https://github.com/Cogito2012/DRIVE/blob/master/src/saliency/mlnet.py
"""
import torch
import torch.nn as nn
from torchvision import models


class MLNet(nn.Module):
    """
    Referenced from: https://github.com/immortal3/MLNet-Pytorch/blob/master/MLNet_Pytorch.ipynb
    """

    def __init__(self, input_shape):
        super(MLNet, self).__init__()
        self.input_shape = input_shape
        self.output_shape = [int(input_shape[0] / 8), int(input_shape[1] / 8)]
        self.scale_factor = 10
        self.prior_size = [
            int(self.output_shape[0] / self.scale_factor),
            int(self.output_shape[1] / self.scale_factor),
        ]

        # loading pre-trained vgg16 model and removing last max pooling layer (Conv5-3 pooling)
        # 16: conv3-3 pool (1/8), 23: conv4-3 pool (1/16), 30: conv5-3 (1/16)
        vgg16_model = models.vgg16(pretrained=True)
        self.freeze_params(vgg16_model, 21)
        features = list(vgg16_model.features)[:-1]

        # making same spatial size  by calculation :)
        # in pytorch there was problem outputing same size in maxpool2d
        features[23].stride = 1
        features[23].kernel_size = 5
        features[23].padding = 2

        self.features = nn.ModuleList(features).eval()
        # adding dropout layer
        self.fddropout = nn.Dropout2d(p=0.5)
        # adding convolution layer to down number of filters 1280 ==> 64
        self.int_conv = nn.Conv2d(
            1280, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.pre_final_conv = nn.Conv2d(
            64, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
        )
        # prior initialized to ones
        self.prior = nn.Parameter(
            torch.ones(
                (1, 1, self.prior_size[0], self.prior_size[1]), requires_grad=True
            )
        )

        # bilinear upsampling layer
        self.bilinearup = torch.nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)

        # initialize new parameters
        self.init_new_params()

    def freeze_params(self, model, last_freeze_layer):
        # freezing Layer
        for i, param in enumerate(model.parameters()):
            if i <= last_freeze_layer:
                param.requires_grad = False

    def init_new_params(self):
        def zero_params(tensor):
            if tensor is not None:
                tensor.data.fill_(0)

        nn.init.kaiming_normal_(
            self.int_conv.weight, mode="fan_out", nonlinearity="relu"
        )
        zero_params(self.int_conv.bias)
        nn.init.kaiming_normal_(
            self.pre_final_conv.weight, mode="fan_out", nonlinearity="relu"
        )
        zero_params(self.pre_final_conv.bias)
        torch.nn.init.xavier_normal_(self.prior)

    def forward(self, x, return_bottom=False):
        results = []
        for ii, model in enumerate(self.features):
            # model = model.to(x.device)
            x = model(x)
            if ii in {16, 23, 29}:
                results.append(x)

        # concat to get 1280 = 512 + 512 + 256
        x = torch.cat((results[0], results[1], results[2]), 1)

        # adding dropout layer with dropout set to 0.5 (default)
        x = self.fddropout(x)

        # 64 filters convolution layer
        bottom = self.int_conv(x)
        # 1*1 convolution layer
        x = self.pre_final_conv(bottom)

        upscaled_prior = self.bilinearup(self.prior)

        # dot product with prior
        x = x * upscaled_prior
        # x = torch.sigmoid(x)
        x = torch.nn.functional.relu(x, inplace=True)

        if return_bottom:
            return x, bottom
        return x
