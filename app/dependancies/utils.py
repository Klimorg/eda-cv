from io import BytesIO
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image


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


def generate_batch(lst, batch_size):
    """Yields batch of specified size"""
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]
