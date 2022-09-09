from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import pandas as pd
from utils import get_project_root
import os

# read in feature_data and labels
root_directory = get_project_root()
data_file_directory = f"{root_directory}/data/feature_data/"
preprocessing_directory =  f"{root_directory}/data/preprocessing_parameters/"
data = pd.read_csv(f'{data_file_directory}/data.csv')

# drop unnecessary columns
data = data.drop(['filename'], axis=1)

# create labels column
# makes sure the labels are between 0 and n-1, for n classes
y = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# create feature_data columns
# feature_data: 'chroma_stft', 'rmse', 'spectral_centroid', 'spectral_bandwidth',
#           'rolloff', 'zero_crossing_rate', MFCC 1 through 20
# rescales the data so that it is mean 0 and unit variance
X_unscaled = np.array(data.iloc[:, :-1], dtype=float)
means = np.mean(X_unscaled, axis=0)
stds = np.std(X_unscaled, axis=0)
X = (X_unscaled-means)/stds

# save parameters
np.save(f"{preprocessing_directory}/means.npy", means)
np.save(f"{preprocessing_directory}/stds.npy", stds)

# # save feature_data to file
pd.DataFrame(X).to_csv(f"{data_file_directory}/X_features.csv", index=False, header=False)
pd.DataFrame(y).to_csv(f"{data_file_directory}/y_labels.csv", index=False, header=False)
# # os.remove(f"{data_file_directory}/data.csv")
