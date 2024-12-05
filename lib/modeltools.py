import torch
from utils import gram_matrix
import datetime

def append_metadata(base_name, epoch=None, batch_id=None, separator="_"):
    """
    Appends specific metadata (datetime, epoch, batch_id) to a given string.
    
    Args:
        base_name (str): The base string to which metadata is appended.
        epoch (int, optional): Epoch number to append.
        batch_id (int, optional): Batch ID to append.
        separator (str, optional): Separator to use between the base name and metadata. Default is "_".
    
    Returns:
        str: The modified string with appended metadata.
    """
    # Get the current datetime
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Initialize the list with the base name
    if base_name==None:
        parts = [current_datetime]
    else:
        parts = [base_name, current_datetime]
    
    # Add epoch if provided, padded to 4 digits
    if epoch is not None:
        parts.append(f"epoch{epoch:04d}")
    
    # Add batch ID if provided, padded to 4 digits
    if batch_id is not None:
        parts.append(f"batch{batch_id:04d}")
    
    # Combine all parts with the separator
    return separator.join(parts)


# define the loss
def loss_function(output_features, original_features, style_gram_features, criterion: torch.nn.Module, content_weight=1, style_weight=1, vgg_layers=None):
    style_loss = 0.
    weights = {'relu1_2': 1, 'relu2_2': 1, 'relu3_3': 2, 'relu4_3': 5, 'relu5_3': 10}


    content_loss = content_weight * criterion(output_features['relu2_2'], original_features['relu2_2'])
    c=0
    feature_iterator = output_features.keys()
    if vgg_layers!=None:
        feature_iterator = vgg_layers
    for feature in feature_iterator:
        gm_y = gram_matrix(output_features[feature])
        style_loss += criterion(gm_y, style_gram_features[feature]) * weights[feature]
        c+=1
    if c!=0:
        style_loss *= style_weight/c

    return content_loss + style_loss