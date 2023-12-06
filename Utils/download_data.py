import os
import zipfile
import sys
import validators
from pathlib import Path

import requests

# Read command line input
folder_name = sys.argv[1]
github_link = sys.argv[2] # Only getting datasets from github for now


if os.path.isdir("/PyTorch/Datasets/" + folder_name):
    print("Dataset already downloaded!") # Check if folder name already exists
elif not validators.url(github_link):
    print("Github does not exist!") # Check if the url is valid
else:
    # Create folder
    download_path = Path("/PyTorch/Datasets/" + folder_name)
    download_path.mkdir()
    
    # Download pizza, steak, sushi data
    with open(download_path / "the_zip_file.zip", "wb") as f:
        request = requests.get(github_link)
        print("Downloading data...")
        f.write(request.content)

    # Unzip pizza, steak, sushi data
    with zipfile.ZipFile(download_path / "the_zip_file.zip", "r") as zip_ref:
        print("Unzipping data...") 
        zip_ref.extractall(download_path)
        
    # Remove zip file
    os.remove(download_path / "the_zip_file.zip")
