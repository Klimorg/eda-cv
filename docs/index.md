# Welcome on **eda-cv** documentation

This API summarizes some useful function when doing Exeploratory Data Analysis with images.

Right now, the endpoints avalaible in the API are :

* Computing the mean and standard deviation of each channel of an RGB image.
* Computing the color histogram of each channel of an RGB image.
* Computing the mean image of an image dataset.
* Computing a mean vs std scatterplot of an image dataset.


## How to use it

A docker image has been made available on [DockerHub](https://hub.docker.com/repository/docker/vorphus/eda-cv/general). To use it,

* Download the image

```shell
docker pull vorphus/eda-cv:latest
```

* Create a directory named `eda-cv` (or whatever the name you want).
* Run the following command.

```shell
docker run -it --rm --name eda-cv -p 8080:8080 -v absolute_path_to_eda-cv:/opt vorphus/eda-cv:latest
```

The image will then create 4 directories in eda-cv :

* A directory named `data`.
* A directory named `results` with 3 subirectories : `histograms`, `mean_image`, `scatterplots`.

The `data` directory is supposed to store the image dataset on which you want to compute the mean image or the mean vs std scatterplot. The `results/histograms`, `results/mean_image`, `results/scatterplots` will store the image resulting of such functions.

!!! attention "Attention"

    1. As of now, this API is in early stage, so the `data` directory supports only "**one class datasets**". If your dataset has two or more classes, you'll have to do analysis one class at a time.
    2. To compute the mean image of an image dataset, **all images must have the same height and width**.
