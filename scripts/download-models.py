import requests
from tqdm import tqdm
import os

def download_from_github_release(url, output_path):
    """
    Downloads a file from a GitHub release URL with a progress bar.
    
    Args:
        url (str): The direct download URL of the file.
        output_path (str): The path where the file will be saved.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))  # Get file size from headers
    
    with open(output_path, 'wb') as file, tqdm(
        desc=f"Downloading {url.split('/')[-1]}",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
            bar.update(len(chunk))


if __name__ == "__main__":
    url = "https://github.com/AcePeaX/DeepStyleX/releases/download/models-v1.0/models.zip"
    output_path = "data/saves/models-v1.0.zip"
    save_directory = os.path.dirname(output_path)
    if not os.path.exists(save_directory):
        # Create the folder (and any necessary intermediate directories)
        os.makedirs(save_directory)
        print("Created directory:",save_directory)
    print("Downloading from:",url)
    download_from_github_release(url, output_path)
    print("Sucessfully downloaded the models into:", output_path)
    print("To use the models, decompress the archive into data/saves")
