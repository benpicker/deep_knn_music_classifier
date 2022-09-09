import pandas as pd
from utils import get_project_root, GTZANDerivedDataDataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# get feature_data and labels
root_directory = get_project_root()
data_file_directory = f"{root_directory}/data/feature_data/"
X_path = f"{data_file_directory}/X_features.csv"
y_path = f"{data_file_directory}/y_labels.csv"
X = np.array(pd.read_csv(X_path,header=None))
y = np.array(pd.read_csv(y_path, header=None))

# split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
y_train = np.reshape(y_train,-1)
y_test = np.reshape(y_test,-1)

# make knn
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

test_predictions = knn.predict(X_test)
test_accuracy = np.sum(np.equal(y_test,test_predictions))/len(y_test)
print(f"Standard KNN Test Set Accuracy: {test_accuracy}")

# generate and save labels
knn_predicted_labels = knn.predict(X)
pd.DataFrame(knn_predicted_labels).to_csv(f"{data_file_directory}/y_knn_predicted_labels.csv", index=False, header=False)
