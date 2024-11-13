import torch
import torch.nn as nn
from torchvision import models

class VGGFeatures(nn.Module):
    def __init__(self, layers=['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'], device=None, requires_grad=False):
        super(VGGFeatures, self).__init__()
        if device==None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.to(device).eval()
        self.layers = layers
        self.layer_name_mapping = {
            '1': "relu1_1",
            '3': "relu1_2",
            '6': "relu2_1",
            '8': "relu2_2",
            '11': "relu3_1",
            '13': "relu3_2",
            '15': "relu3_3",
            '17': "relu3_4",
            '20': "relu4_1",
            '22': "relu4_2",
            '24': "relu4_3",
            '26': "relu4_4",
            '29': "relu5_1",
            '31': "relu5_2",
            '33': "relu5_3",
            '35': "relu5_4",
        }
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layer_name_mapping:
                layer_name = self.layer_name_mapping[name]
                if layer_name in self.layers:
                    features[layer_name] = x
            if len(features) == len(self.layers):
                break
        return features
