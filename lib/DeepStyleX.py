import torch
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn as nn
import io

# Save the image as compressed bytes
def save_image_as_bytes(image, format: str = "JPEG"):
    # Save image to a byte buffer
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    # Convert to bytes and save with torch.save
    return buffer.getvalue()

def load_image_from_bytes(bytes):
    # Convert bytes back to a PIL image
    buffer = io.BytesIO(bytes)
    image = Image.open(buffer)
    return image

class CustomConvolutionalLayer(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=1, stride=1):
        super(CustomConvolutionalLayer, self).__init__()
        ref_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(ref_padding)
        self.convolution2d = nn.Conv2d(in_size, out_size, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.convolution2d(out)
        return out


class ResidualBlock_BatchNorm(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock_BatchNorm, self).__init__()
        self.conv_layer1 = CustomConvolutionalLayer(channels, channels, kernel_size=3, stride=1)
        self.batch_norm1 = nn.BatchNorm2d(channels, affine=True)
        self.relu = nn.ReLU()
        self.conv_layer2 = CustomConvolutionalLayer(channels, channels, kernel_size=3, stride=1)
        self.batch_norm2 = nn.BatchNorm2d(channels, affine=True)

    def forward(self, x):
        out = self.relu(self.batch_norm1(self.conv_layer1(x)))
        out = self.batch_norm2(self.conv_layer2(out))
        out = out + x
        return out

class ResidualBlock_InstanceNorm(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock_InstanceNorm, self).__init__()
        self.conv_layer1 = CustomConvolutionalLayer(channels, channels, kernel_size=3, stride=1)
        self.inst_norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()
        self.conv_layer2 = CustomConvolutionalLayer(channels, channels, kernel_size=3, stride=1)
        self.inst_norm2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        out = self.relu(self.inst_norm1(self.conv_layer1(x)))
        out = self.inst_norm2(self.conv_layer2(out))
        out = out + x
        return out


class UpsampleConvolutionalLayer(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, stride=1, upsample=None):
        super(UpsampleConvolutionalLayer, self).__init__()
        self.upsample = upsample
        ref_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(ref_padding)
        self.convolution2d = nn.Conv2d(in_size, out_size, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.convolution2d(out)
        return out


class DeepStyleX(torch.nn.Module):
    def __init__(self, downsample_layers=[(3,9,1),(32,3,2),(64,3,2),(128,)], num_residual_blocks=5, upsample_layers=[(128,3,2),(64,3,2),(32,9,2),(3,)], batch_norm = False):
        """
        Params:
        - downsample_layers: list of (size, kernel_size, stride)
        """
        super(DeepStyleX, self).__init__()
        for i in range(len(upsample_layers)):
            if type(upsample_layers[i])!=list and type(upsample_layers[i])!=tuple:
                upsample_layers[i] = (upsample_layers[i],)
        for i in range(len(downsample_layers)):
            if type(downsample_layers[i])!=list and type(downsample_layers[i])!=tuple:
                downsample_layers[i] = (downsample_layers[i],)
        assert downsample_layers[-1][0]==upsample_layers[0][0]

        self.style_image = None

        # Non-linearities
        self.relu = nn.ReLU()

        # Initial convolution layers
        self.downsample_layers = downsample_layers
        self.batch_norm = batch_norm
        self.down_layers = nn.ModuleList()
        for i in range(1, len(downsample_layers)):
            self.down_layers.append(
                CustomConvolutionalLayer(downsample_layers[i-1][0], downsample_layers[i][0], kernel_size=downsample_layers[i-1][1], stride=downsample_layers[i-1][2])
                )
            if batch_norm:
                self.down_layers.append(nn.BatchNorm2d(downsample_layers[i][0], affine=True))
            else:
                self.down_layers.append(nn.InstanceNorm2d(downsample_layers[i][0], affine=True))
            self.down_layers.append(self.relu)
        # Residual layers
        self.residual_layers = nn.ModuleList()
        self.num_residual_blocks = num_residual_blocks
        for i in range(num_residual_blocks):
            if batch_norm:
                self.residual_layers.append(ResidualBlock_BatchNorm(upsample_layers[0][0]))
            else:
                self.residual_layers.append(ResidualBlock_InstanceNorm(upsample_layers[0][0]))
        # Upsampling Layers
        self.up_layers = nn.ModuleList()
        self.upsample_layers = upsample_layers
        for i in range(1, len(upsample_layers)):
            if(i<len(upsample_layers)-1):
                self.up_layers.append(
                    UpsampleConvolutionalLayer(upsample_layers[i-1][0], upsample_layers[i][0], kernel_size=upsample_layers[i-1][1], stride=1, upsample=upsample_layers[i-1][2])
                )
                if batch_norm:
                    self.up_layers.append(nn.BatchNorm2d(upsample_layers[i][0], affine=True))
                else:
                    self.up_layers.append(nn.InstanceNorm2d(upsample_layers[i][0], affine=True))
                self.up_layers.append(self.relu)
            else:
                self.up_layers.append(
                    CustomConvolutionalLayer(upsample_layers[i-1][0], upsample_layers[i][0], kernel_size=upsample_layers[i-1][1], stride=1)
                )

    def forward(self, X):
        y = X
        for layer in self.down_layers:
            y = layer(y)

        for layer in self.residual_layers:
            y = layer(y)

        for layer in self.up_layers:
            y = layer(y)

        return y
    
    def save(self, path, optimizer=None, style_image=None):
        obj = dict()
        obj['params'] = self.state_dict()
        obj['downsample_layers'] = self.downsample_layers
        obj['num_residual_blocks'] = self.num_residual_blocks
        obj['upsample_layers'] = self.upsample_layers
        obj['batch_norm'] = self.batch_norm
        if style_image==None:
            style_image = self.style_image
        if style_image!=None:
            obj['style_image'] = save_image_as_bytes(style_image)
        else:
            obj['style_image'] = None
        obj['opti'] = optimizer
        torch.save(obj, path)

    @classmethod
    def load(cls, path, map_location=None):
        if map_location==None:
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        obj = torch.load(path, weights_only=True, map_location=map_location)
        batch_norm = False
        if 'batch_norm' in obj.keys():
            batch_norm = obj["batch_norm"]
        instance = cls(downsample_layers=obj['downsample_layers'],num_residual_blocks=obj['num_residual_blocks'],upsample_layers=obj['upsample_layers'], batch_norm=batch_norm)
        instance.load_state_dict(obj['params'])
        if 'style_image' in obj.keys():
            try:
                if type(obj["style_image"]) == torch.Tensor:
                    instance.style_image = ToPILImage()(obj["style_image"])
                else:
                    instance.style_image = load_image_from_bytes(obj["style_image"])
            except:
                print("Couldn't load style image from the model")
        optimizer = None
        if 'opti' in obj.keys():
            optimizer = obj["opti"]
        return instance, optimizer