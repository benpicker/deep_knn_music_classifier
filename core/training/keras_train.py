import keras
from keras import models
from keras import layers
import numpy as np
from utils import get_project_root
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import get_accuracy

# loading training and testing
root_directory = get_project_root()
data_file_directory = f"{root_directory}/data/feature_data/"
model_save_directory = f"{root_directory}/saved_models"
genre_label_path = f"{data_file_directory}/y_labels.csv"
knn_predicted_genre_label_path = f"{data_file_directory}y_knn_predicted_labels.csv"
audio_data_path = f"{data_file_directory}/X_features.csv"

X = pd.read_csv(audio_data_path, header=None)
y = pd.read_csv(knn_predicted_genre_label_path, header=None)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# creating a model
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train,
                    y_train,
                    epochs=20,
                    batch_size=45)



# calculate accuracy
test_loss, test_acc = model.evaluate(X_test, y_test)
print('test_acc: ', test_acc)

# predictions of KNN labels
predictions = model.predict(X_test)
np.argmax(predictions[0])

# save model
model.save(f"{model_save_directory}/keras_model")
