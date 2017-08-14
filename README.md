# python_trendmarketplace

# Python Libraries for Image Recognition
### Sklearn & TensorFlow are used for image recognition of different complexity
## Our Motivation
As future data analysts, we want to be able to master mainstream data analysis tools especially machine learning libraries. 

Among all options out there in the market, scikit-learn is the most easy-to-use and general-purpose machine learning platform in Python. 
We will take this opportunity to start out our ML journey by implementing a primary image clustering task using sklearn’s packages. As a supplement, TensorFlow’s advanced image recognition package will also be introduced.
## Our Hypothesis
TensorFlow can outbeat sklearn in Minnesota scenery image recognition since TensorFlow supports more complexity computation.
## Our Dataset
2000 pictures of ‘Minnesota natural scenery’ parsed from Google Image
The raw images are converted a simple format called RGB 
## Our Testing

### Sklearn’s image clustering using RGB features
* **K-Means**

Package

from sklearn.cluster import KMeans

KMeans(n_clusters = n)

In a 3D space with R, G, B axis, initialize K centroids and cluster all data points around the nearest centroids based on Euclidean distance.

K-Means lacks accuracy when a certain data point is plot between cluster boundaries. It can only make hard assignment on each data point, while the data point may contain information which lead to multiple clusters.

* **Gaussian Mixture Model**

Package

from sklearn import mixture

mixture.GMM(n_components=4, covariance_type='full')

multivariate_normal.pdf(data, mean, covariance) * weight

Sklearn.mixture package is applied in learn, sample and estimate Gaussian Mixture Model, which is a probabilistic model  assuming data points from a finite number of Gaussian distributions. 

The model converts pictures with the feature of R, G, B and then cluster observations according to probabilities of R, G, B distribution in GMM and nearest mean in K-means. The package is used to fix and predict models in two approaches.Two thousand Minnesota scenery pictures are clustered into three folders, with snow, sunset and forest as common characteristics. The sklearn packages provides a simple means of rough classification with limited features to train.


### TensorFlow’s CNN image classification using Inception v3 model

TensorFlow is an open source software library for machine learning written in Python and C++. Inception is a pre-trained 22 layers deep neural network built on TensorFlow. By training on a 1000 categories labeled dataset ImageNet, Inception has high accuracy in image recognition. 

Inception-v3 is trained for the ImageNet Large Visual Recognition Challenge using the data from 2012. This is a standard task in computer vision, where models try to classify entire images into 1000 classes, like "Zebra", "Dalmatian", and "Dishwasher".

The model is used for recognize day-to-day object and consists of 22 layers.

We use the pre-trained Inception v3 model to recognize object. The model gives the output of the probability of an object is in the image by using softmax regression. CNN can generate different filters (also known as neuron or kernel) through the training process and use them to create activation map or so-called feature map. The features that a CNN can recognise is determined by the numbers of layers and the number of filters in each layer.

CNN can also perform non-linear algorithm by using ReLU function in the convolution layer, which couldn’t be performed by GMM model and K-means model.
