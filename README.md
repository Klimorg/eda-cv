# eda-cv

Simple api with some useful functions for Exploratory Data Analysis for Computer Vision.

Docker image available at [dockerhub](https://hub.docker.com/repository/docker/vorphus/eda-cv/tags?page=1&ordering=last_updated)

Right now, the endpoints avalaible in the API are :

* Computing the mean and standard deviation of each channel of an RGB image.
* Computing the color histogram of each channel of an RGB image.
* Computing the mean image of an image dataset.
* Computing a mean vs std scatterplot of an image dataset.
* Embeddings via CNNs trained on ImageNet + plots with t-SNE and Umap,

TODO:

* add Eigenfaces,
* add diff between an image and the mean image of the dataset.
* build an UI with prettier rendering of graphs.
