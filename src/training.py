from torch import nn
import torch
import os

model_folder_path = "models/"

def save_model(model: nn.Module, filename):
    os.makedirs(model_folder_path, exist_ok=True)
    torch.save(model.state_dict(), model_folder_path + filename)

def load_model(empty_model: nn.Module, filename):
    empty_model.load_state_dict(torch.load(model_folder_path + filename, weights_only=True))
    empty_model.eval()
    return empty_model