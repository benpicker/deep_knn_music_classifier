import torch
from core.training.deep_KNN_model import KNNNet
from utils import get_project_root
from core.creating_features_and_preprocessing.create_raw_features import get_features
from sklearn.preprocessing import StandardScaler
import numpy as np
import keras
import pandas as pd
from utils import get_project_root, GTZANDerivedDataDataset

classes = ["0 - blues", "1 - classical", "2 - country", "3 - disco", "4 - hiphop",
           "5 - jazz", "6 - metal", "7 - pop", "8 - reggae", "9 - rock", ]
# device = torch.device("cuda" if use_cuda else "cpu")
device = "cpu"

root_directory = get_project_root()
model_save_directory = f"{root_directory}/saved_models/keras_model/"

model = keras.models.load_model(model_save_directory)


# read in feature_data and labels
root_directory = get_project_root()
data_file_directory = f"{root_directory}/data/feature_data/"
data = pd.read_csv(f'{data_file_directory}/data.csv')


genre = "blues"
file_name = f"{genre}.00012.au"
# sample_file_path = f"{root_directory}/data/genres/{genre}/{file_name}"

idx = list(data.index[data['filename'] == file_name])[0]
data_file_directory = f"{root_directory}/data/feature_data/"
genre_label_file = f"{data_file_directory}/y_knn_predicted_labels.csv"
feature_data_file_path = f"{data_file_directory}/X_features.csv"

# create data loaders
data = GTZANDerivedDataDataset(genre_label_file, feature_data_file_path)
x, y = data.__getitem__(idx)
x, y = np.array(x).reshape(-1,26), np.array(y)

output = model.predict(x)
print(output)
output_class = np.argmax(output)
print(output)
print(classes[output_class])

