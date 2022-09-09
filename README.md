# Deep KNN Music Classifier 


### Choices for implementing KNN 

Fully connected neural networks are termed "universal function approximators" because they can in theory represent any function. At its core, a function is just a mapping from inputs to outputs and a neural network successfully replicates the behavior of a function if it can reliably match the function's input output mapping. Thus, the basic idea is we can use fully connected networks to replicate the behavior of a KNN. 


1. ***Getting KNN Predictions*** I first trained a standard KNN on the audio features and original genre labels. This gave me the predictions of the KNN. 
2. ***Training DNN on KNN predictions*** I then trained the deep KNN on the audio features and KNN predicted labels, allowing me to predict the KNN output using a DNN. My neural network used fully connected networks with ReLU activations. For the PyTorch version, I used batch normalizations to mitigate so called "internal covariate shift." The batch normalizing yielded a 10-20% improvement in results and greater consistency. 


* Standard KNN: 57.7% accuracy 
* Deep KNN with Keras: 73.5% accuracy 
* Deep KNN in PyTorch: 73% accuracy 


### Files 

The files are organized into folders 
* `core` -- the coding files 
* `data` -- the original data for the project and any data used for computing stuff
* `saved_models` -- the model files for the keras and pytorch models 
* `tests` -- various unit tests 


Within `core`, we have 
* `creating_features_and_preprocessing` -- the files needed to generate data for training 
    * `create_raw_features.py` -- this creates the features without normalizing them 
    * `preprocessing_features.py` -- this normalizes the features 
    * `standard_KNN.py` -- this generates the KNN predictions used for training 
* `training` -- the files used to train the models 
    * `deep_KNN_model.py` -- this is the PyTorch neural network class 
    * `pytorch_train.py` -- this trains the PyTorch model 
    * `keras_train.py` -- this trains the keras model 

* `evaluation` -- files that can run the model on sample files 
    * `keras_evaluation.py` -- evaluates input points using Keras model 
    * `pytorch_evaluation.py` -- evaluates input points using Pytorch model 
