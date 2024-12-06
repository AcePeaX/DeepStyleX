import torch
from torchvision import transforms
from PIL import Image

def normalize(X):
    mean = X.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = X.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    X = X.div_(255.0)
    return (X - mean) / std

def preprocess(image, device=None, resize=True):
    if device==None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Assume 'image' is a PIL Image
    if resize:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            #transforms.Lambda(lambda x: x.mul(255))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Lambda(lambda x: x.mul(255))
        ])

    return 255*transform(image).to(device)  # Add batch dimension

def deprocess(tensor):
    # Denormalize and convert tensor to PIL Image
    if len(tensor.shape)==4:
        tensor = tensor.squeeze(0)
    tensor = tensor.cpu()
    tensor = tensor.clone().clamp(0, 255).detach().numpy().transpose(1, 2, 0).astype("uint8")
    #tensor = torch.clamp(tensor, 0, 1)
    return transforms.ToPILImage()(tensor)

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std

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



def resize_image_with_max_resolution(image: Image.Image, max_resolution: int) -> Image.Image:
    """
    Resize a PIL image to ensure its resolution (width * height) does not exceed max_resolution.
    Maintains the original aspect ratio. Only reduces size if needed.

    Args:
        image (Image.Image): Input PIL image.
        max_resolution (int): Maximum allowed resolution (width * height).

    Returns:
        Image.Image: Resized PIL image with the same aspect ratio.
    """
    # Get the original dimensions of the image
    original_width, original_height = image.size
    original_resolution = original_width * original_height

    # Check if resizing is needed
    if original_resolution <= max_resolution:
        return image  # No resizing needed

    # Compute the scaling factor to fit within max_resolution
    scaling_factor = (max_resolution / original_resolution) ** 0.5

    # Calculate the new dimensions while maintaining aspect ratio
    new_width = int(original_width * scaling_factor)
    new_height = int(original_height * scaling_factor)

    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    return resized_image


import mimetypes

def get_file_type(file_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type:
        if mime_type.startswith('image'):
            return "image"
        elif mime_type.startswith('video'):
            return "video"
    return "unknown"

