import torch_directml

device = torch_directml.device() if torch_directml.is_available() else "cpu"
