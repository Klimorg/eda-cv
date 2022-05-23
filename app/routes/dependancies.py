from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from app.config import settings


def read_imagefile(data: bytes) -> Image.Image:
    """_summary_

    Args:
        data (bytes): _description_

    Returns:
        Image.Image: _description_
    """
    return Image.open(BytesIO(data))


def load_image_into_numpy_array(data: bytes) -> np.ndarray:
    """_summary_

    Args:
        data (bytes): _description_

    Returns:
        np.ndarray: _description_
    """
    return np.array(Image.open(BytesIO(data)))


def compute_channels_mean(image: np.ndarray) -> Tuple[float, float, float]:
    """_summary_

    Args:
        image (np.ndarray): _description_

    Returns:
        Tuple[float, float, float]: _description_
    """

    red_mean_value = image[:, :, 0].mean()
    green_mean_value = image[:, :, 1].mean()
    blue_mean_value = image[:, :, 2].mean()

    return red_mean_value, green_mean_value, blue_mean_value


def compute_channels_std(image: np.ndarray) -> Tuple[float, float, float]:
    """_summary_

    Args:
        image (np.ndarray): _description_

    Returns:
        Tuple[float, float, float]: _description_
    """

    red_std_value = image[:, :, 0].std()
    green_std_value = image[:, :, 1].std()
    blue_std_value = image[:, :, 2].std()

    return red_std_value, green_std_value, blue_std_value


def compute_histograms_channels(
    image: np.ndarray,
    filename: str,
    timestamp: str,
    normalize: bool,
) -> Path:
    """_summary_

    Args:
        image (np.ndarray): _description_
        filename (str): _description_
        timestamp (str): _description_
    """
    colors = ("red", "green", "blue")
    channel_ids = (0, 1, 2)
    pixel_range_value = 255

    if normalize:
        image = image / 255
        pixel_range_value = 1

    # create the histogram plot, with three lines, one for
    # each color
    plt.figure()
    plt.xlim([0, pixel_range_value])
    for channel_id, c in zip(channel_ids, colors):
        histogram, bin_edges = np.histogram(
            image[:, :, channel_id],
            bins=256,
            range=(0, pixel_range_value),
        )
        plt.plot(bin_edges[0:-1], histogram, color=c)

    plt.title(f"Color Histogram of {filename}")
    plt.xlabel("Color value")
    plt.ylabel("Pixel count")

    saved_image_path = Path(f"{settings.histograms_dir}/{filename}_{timestamp}.png")

    plt.savefig(saved_image_path)

    return saved_image_path


def compute_mean_image(images_list: List[np.ndarray], timestamp: str) -> Path:
    """_summary_

    Args:
        images_list (List[np.ndarray]): _description_
        timestamp (str): _description_
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

    saved_image_path = Path(f"{settings.mean_image_dir}/average_{timestamp}.png")

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
    """_summary_

    Args:
        images_list (List[np.ndarray]): _description_
        timestamp (str): _description_

    Returns:
        Path: _description_
    """

    colors = ("red", "green", "blue")
    channel_ids = (0, 1, 2)

    for channel, color in zip(channel_ids, colors):
        means = [compute_channels_mean(image)[channel] for image in images_list]
        stds = [compute_channels_std(image)[channel] for image in images_list]

        plt.scatter(means, stds, c=color, alpha=0.5)

    plt.title("Mean-std scatterplot")
    plt.xlabel("means")
    plt.ylabel("stds")

    saved_image_path = Path(f"{settings.scatterplots_dir}/scatter_{timestamp}.png")

    plt.savefig(saved_image_path)

    return saved_image_path
