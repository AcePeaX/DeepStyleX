import torch
from torchvision import transforms

# Define normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std =[0.229, 0.224, 0.225])
def preprocess(image, device=None, resize=True):
    if device==None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Assume 'image' is a PIL Image
    if resize:
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    return transform(image).to(device)  # Add batch dimension

def deprocess(tensor):
    # Denormalize and convert tensor to PIL Image
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std =[1/0.229, 1/0.224, 1/0.225]
    )
    if len(tensor.shape)==4:
        tensor = tensor.squeeze(0)
    tensor = tensor.cpu() #inv_normalize(tensor.squeeze(0).cpu())
    tensor = torch.clamp(tensor, 0, 1)
    return transforms.ToPILImage()(tensor)

def gram_matrix(X: torch.Tensor):
    """
    Compute the Gram matrix for a batch of feature maps.

    Args:
        X (torch.Tensor): Input tensor of shape (b, c, h, w), where
                                b = batch size,
                                c = number of channels,
                                h = height,
                                w = width.

    Returns:
        torch.Tensor: Gram matrix for each image in the batch, of shape (b, c, c).
    """

    if len(X.shape)==4:
        (b, ch, h, w) = X.shape
        features = X.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
    else:
        (ch, h, w) = X.shape
        features = X.view(ch, w * h)
        features_t = features.transpose(0, 1)
        gram = features.matmul(features_t) / (ch * h * w)
    return gram


