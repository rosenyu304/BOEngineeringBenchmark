import torch
import gzip
import shutil
import os# Define the paths
compressed_model_path = 'model.pt.gz'
decompressed_model_path = 'model.pt'# Decompress the .gz file
with gzip.open(compressed_model_path, 'rb') as f_in:
    with open(decompressed_model_path, 'wb') as f_out:
       shutil.copyfileobj(f_in, f_out)# Load the decompressed model
model = torch.load(decompressed_model_path)# Optionally, clean up the decompressed file if no longer needed
os.remove(decompressed_model_path)

