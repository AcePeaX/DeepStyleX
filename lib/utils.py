import torch
from torchvision import transforms

# Define normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std =[0.229, 0.224, 0.225])
def preprocess(image, device=None):
    if device==None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Assume 'image' is a PIL Image
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        #normalize
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
