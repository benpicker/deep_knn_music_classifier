# Deep KNN Music Classifier 

Fully connected neural networks are termed "universal function approximators" because they can in theory represent any function. At its core, a function is just a mapping from inputs to outputs and a neural network successfully replicates the behavior of a function if it can reliably match the function's input output mapping. Thus, the basic idea is we can use fully connected networks to replicate the behavior of a KNN. 


1. ***Getting KNN Predictions*** I first trained a standard KNN on the audio features and original genre labels. This gave me the predictions of the KNN. 
2. ***Training DNN on KNN predictions*** I then trained the deep KNN on the audio features and KNN predicted labels, allowing me to predict the KNN output using a DNN. My neural network used fully connected networks with ReLU activations. For the PyTorch version, I used batch normalizations to mitigate so called "internal covariate shift." The batch normalizing yielded a 10-20% improvement in results and greater consistency. 


* Standard KNN: 57.7% accuracy 
* Deep KNN with Keras: 73.5% accuracy 
* Deep KNN in PyTorch: 73% accuracy 



