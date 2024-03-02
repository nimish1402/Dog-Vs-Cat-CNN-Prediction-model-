# Dog-Vs-Cat-CNN-Prediction-model-
This repository contains a Convolutional Neural Network (CNN) model implemented in TensorFlow and Keras for predicting whether an image contains a cat or a dog. The model utilizes techniques such as Conv2D layers, Dropout, and BatchNormalization to handle overfitting of the data.

**Dataset**

The model is trained on the popular Cat vs Dog dataset, which consists of thousands of images of cats and dogs. The dataset is divided into training and testing sets to evaluate the performance of the model.
Dataset link : https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset

**Model Architecture**

The CNN model architecture consists of multiple Conv2D layers followed by MaxPooling layers for feature extraction. BatchNormalization is applied to normalize the activations of the network, and Dropout layers are used to prevent overfitting during training. The final layer is a Dense layer with a softmax activation function to output probabilities of the input image belonging to each class (cat or dog).
