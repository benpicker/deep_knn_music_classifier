from utils import get_project_root
from core.creating_features_and_preprocessing.create_raw_features import get_features
import numpy as np
import keras

classes = ["0 - blues", "1 - classical", "2 - country", "3 - disco", "4 - hiphop",
           "5 - jazz", "6 - metal", "7 - pop", "8 - reggae", "9 - rock", ]

# directories
root_directory = get_project_root()
preprocessing_directory =  f"{root_directory}/data/preprocessing_parameters/"
model_save_directory = f"{root_directory}/saved_models/keras_model/"

# get parameters for rescaling
stds = np.load(f"{preprocessing_directory}/stds.npy")
means = np.load(f"{preprocessing_directory}/means.npy")

# file path for target file
sample_file_path = f"{root_directory}/data/genres/blues/blues.00020.au"

model = keras.models.load_model(model_save_directory)

x = get_features(sample_file_path)
x = (x - means) / stds
x = np.array(x).reshape(-1,26)

output = model.predict(x)
output_class = np.argmax(output)
print(classes[output_class])