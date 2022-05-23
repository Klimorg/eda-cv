from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from PIL import Image

from app.config import settings


def read_imagefile(data: bytes) -> Image.Image:
    """Read an image stored in bytes format.

    Args:
        data (bytes): The images stored in bytes format.

    Returns:
        Image.Image: The read image.
    """
    return Image.open(BytesIO(data))


def load_image_into_numpy_array(data: bytes) -> np.ndarray:
    """Load an image stored in bytes format in a numpy array.

    Args:
        data (bytes): The images stored in bytes format.

    Returns:
        np.ndarray: The np.array associated to the image.
    """
    return np.array(Image.open(BytesIO(data)))


def compute_channels_mean(image: np.ndarray) -> Tuple[float, float, float]:
    """Compute the mean over each channels of an RGB images.

    Args:
        image (np.ndarray): The image, as a np.array, for which you want to compute the means.

    Returns:
        Tuple[float, float, float]: The RGB means.
    """

    red_mean_value = image[:, :, 0].mean()
    green_mean_value = image[:, :, 1].mean()
    blue_mean_value = image[:, :, 2].mean()

    return red_mean_value, green_mean_value, blue_mean_value


def compute_channels_std(image: np.ndarray) -> Tuple[float, float, float]:
    """Compute the standard deviation over each channels of an RGB images.

    Args:
        image (np.ndarray): The image, as a np.array, for which you want to compute the stds.

    Returns:
        Tuple[float, float, float]: The RGB stds.
    """

    red_std_value = image[:, :, 0].std()
    green_std_value = image[:, :, 1].std()
    blue_std_value = image[:, :, 2].std()

    return red_std_value, green_std_value, blue_std_value


def compute_histograms_channels(
    image: np.ndarray,
    filename: str,
    timestamp: str,
) -> Path:
    """Compute the channels normed histograms of an image.

    The bins of the histograms are all of width 1, meaning that the normed histogram here defines a
    Probability mass function on each channels, i.e. the sum of all values for each channels is equal to 1.

    Args:
        image (np.ndarray): The image, as a np.array, for which you want to compute the channels normed histograms.
        filename (str): The name of the image file.
        timestamp (str): The timestamp at which the endpoint has been called.

    Returns:
        Path: The path to the histogram.
    """
    colors = ("red", "green", "blue")
    channel_ids = (0, 1, 2)

    pixel_range_value = 255
    bins = np.arange(0, pixel_range_value)

    # create the histogram plot, with three lines, one for
    # each color
    plt.figure()
    plt.xlim([0, pixel_range_value])
    for channel_id, color in zip(channel_ids, colors):
        # bins = 256

        histogram, bin_edges = np.histogram(
            image[:, :, channel_id],
            bins=bins,
            range=(0, pixel_range_value),
            density=True,
        )
        plt.plot(bin_edges[0:-1], histogram, color=color)

    plt.title(f"Color Histogram of {filename}")
    plt.xlabel("Color value")
    plt.ylabel("Pixel density")

    saved_image_path = Path(
        f"{settings.histograms_dir}/{filename}_{timestamp}.png",
    ).resolve()

    plt.savefig(saved_image_path)

    return saved_image_path


def compute_mean_image(images_list: List[np.ndarray], timestamp: str) -> Path:
    """Compute the mean image of an image dataset.

    Args:
        images_list (List[np.ndarray]): The image dataset on which you compute the mean image.
        timestamp (str): The timestamp at which the endpoint has been called.

    Returns:
        Path: The path to the mean image.
    """
    # Assuming all images are the same size, get dimensions of first image
    height, width, _ = images_list[0].shape
    num_images = len(images_list)
    # Create a numpy array of floats to store the average (assume RGB images)
    arr = np.zeros((height, width, 3), dtype=np.float32)

    # Build up average pixel intensities, casting each image as an array of floats
    arr = sum(images_list) / num_images

    # Round values in array and cast as 8-bit integer
    arr = np.array(np.round(arr), dtype=np.uint8)

    # Generate, save final image
    out = Image.fromarray(arr, mode="RGB")

    saved_image_path = Path(
        f"{settings.mean_image_dir}/average_{timestamp}.png",
    ).resolve()

    out.save(saved_image_path)

    return saved_image_path


def get_items_list(directory: str, extension: str) -> List[Path]:
    """
    The code above does the following:

    1. Creates a list of all the files in the directory.
    2. Applies a filter to the list to only include files with the given extension.
    3. Sorts the list by file name.
    4. Returns the list.
    """
    return sorted(
        Path(file).absolute()
        for file in Path(directory).glob(f"**/*{extension}")
        if file.is_file()
    )


def compute_scatterplot(images_list: List[np.ndarray], timestamp: str) -> Path:
    """Compute the mean vs std scatterplot of an image dataset

    Args:
        images_list (List[np.ndarray]): The image dataset on which you compute the mean vs std scatterplot.
        timestamp (str): The timestamp at which the endpoint has been called.

    Returns:
        Path: The path to the scatterplot.
    """

    colors = ("red", "green", "blue")
    channel_ids = (0, 1, 2)

    for channel, color in zip(channel_ids, colors):
        means = [compute_channels_mean(image)[channel] for image in images_list]
        stds = [compute_channels_std(image)[channel] for image in images_list]

        plt.scatter(means, stds, c=color, alpha=0.5)

    plt.title("Mean-std scatterplot. Pixel values in [0,1]")
    plt.xlabel("means")
    plt.ylabel("stds")

    saved_image_path = Path(
        f"{settings.scatterplots_dir}/scatter_{timestamp}.png",
    ).resolve()

    plt.savefig(saved_image_path)

    return saved_image_path
