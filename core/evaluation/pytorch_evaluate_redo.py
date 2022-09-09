from utils import get_project_root
from core.creating_features_and_preprocessing.create_raw_features import get_features
import numpy as np
import torch
from core.training.deep_KNN_model import KNNNet

classes = ["0 - blues", "1 - classical", "2 - country", "3 - disco", "4 - hiphop",
           "5 - jazz", "6 - metal", "7 - pop", "8 - reggae", "9 - rock", ]

# directories
root_directory = get_project_root()
preprocessing_directory =  f"{root_directory}/data/preprocessing_parameters/"
model_save_directory = f"{root_directory}/saved_models/pytorch_model/"

# get parameters for rescaling
stds = np.load(f"{preprocessing_directory}/stds.npy")
means = np.load(f"{preprocessing_directory}/means.npy")

# file path for target file
sample_file_path = f"{root_directory}/data/genres/reggae/reggae.00020.au"

# create model instance
# device = torch.device("cuda" if use_cuda else "cpu")
device = "cpu"
model = KNNNet().to(device)
model_file_path = f"{model_save_directory}/deep_KNN.pt"
model.load_state_dict(torch.load(model_file_path))
model.eval()


# run model on data point and get class
x = get_features(sample_file_path)
x = torch.tensor((x - means) / stds).to(torch.float32)
x = torch.reshape(x,(1,26))
x = x.to(device)
output = model(x)
output_class = torch.argmax(output).item()
print(output)
print(classes[output_class])