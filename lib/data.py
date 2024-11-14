import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms



class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif', 'webp'))]

    def __len__(self):
        return len(self.image_files)
    
    def get_raw(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.folder_path, img_name)
        
        # Load image
        image = Image.open(img_path).convert("RGB")  # Ensure 3 channels

        return image

    def __getitem__(self, idx):        
        # Load image
        image = self.get_raw(idx)
        
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
        
        return image