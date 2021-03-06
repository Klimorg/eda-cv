from io import BytesIO
from pathlib import Path

import arrow
import numpy as np
from fastapi import APIRouter, File, UploadFile, status
from fastapi.responses import FileResponse, Response
from loguru import logger
from PIL import Image

from app.config import settings
from app.dependancies.eda_functions import (
    compute_channels_mean,
    compute_channels_std,
    compute_histograms_channels,
    compute_mean_image,
    compute_scatterplot,
)
from app.dependancies.utils import (
    get_items_list,
    load_image_into_numpy_array,
    read_imagefile,
)
from app.pydantic_models import Extension, FeatureReport

router = APIRouter()


@router.post(
    "/mean_values",
    response_model=FeatureReport,
    status_code=status.HTTP_200_OK,
    tags=["CV"],
)
async def get_mean_values(file: UploadFile = File(...)):
    """Return the mean and standard deviation over each channels of an RGB images."""
    image = load_image_into_numpy_array(await file.read())

    red_mean_value, green_mean_value, blue_mean_value = compute_channels_mean(image)

    red_std_value, green_std_value, blue_std_value = compute_channels_std(image)

    return FeatureReport(
        red_mean_value=red_mean_value,
        green_mean_value=green_mean_value,
        blue_mean_value=blue_mean_value,
        red_std_value=red_std_value,
        green_std_value=green_std_value,
        blue_std_value=blue_std_value,
    )


@router.post(
    "/return_image",
    tags=["CV"],
    status_code=status.HTTP_200_OK,
)
async def image_endpoint(file: UploadFile = File(...)):
    """Placeholder. Just return the given image."""
    image = read_imagefile(await file.read())

    # here you can do whatever you want with your image

    # send the final result
    bytes_image = BytesIO()
    image.save(bytes_image, format="PNG")
    result = {"filename": file.filename}

    return Response(
        content=bytes_image.getvalue(),
        headers=result,
        media_type="image/png",
    )


@router.post(
    "/channels_histogram",
    tags=["CV"],
    status_code=status.HTTP_200_OK,
    operation_id="compute_histograms_channels",
)
async def get_histograms_channels(
    file: UploadFile = File(...),
):
    """Compute the channels normed histograms of an image."""

    timestamp = arrow.now().format("YYYY-MM-DD_HH-mm-ss")

    filename = Path(file.filename).stem
    image = load_image_into_numpy_array(await file.read())
    logger.info(f"image loaded : {image.shape}")

    saved_image_path = compute_histograms_channels(
        image=image,
        filename=filename,
        timestamp=timestamp,
    )

    result = {"filename": file.filename}

    return FileResponse(saved_image_path, headers=result)


@router.get(
    "/dataset_mean_image",
    tags=["CV"],
    status_code=status.HTTP_200_OK,
)
async def get_dataset_mean_image(extension: Extension):
    """Compute the mean image of an image dataset."""

    # TODO : check for image size and resize if necessary

    timestamp = arrow.now().format("YYYY-MM-DD_HH-mm-ss")

    images_paths = get_items_list(
        directory=settings.data_dir,
        extension=extension.value,
    )

    images_list = [
        np.array(Image.open(image), dtype=np.float32) for image in images_paths
    ]
    saved_image_path = compute_mean_image(images_list=images_list, timestamp=timestamp)

    return FileResponse(saved_image_path)


@router.get(
    "/mean_std_scatterplot",
    tags=["CV"],
    status_code=status.HTTP_200_OK,
)
async def get_mean_std_scatterplot(extension: Extension):
    """Compute the mean vs std scatterplot of an image dataset."""
    timestamp = arrow.now().format("YYYY-MM-DD_HH-mm-ss")

    images_paths = get_items_list(
        directory=settings.data_dir,
        extension=extension.value,
    )

    saved_image_path = compute_scatterplot(
        images_paths=images_paths,
        timestamp=timestamp,
    )

    return FileResponse(saved_image_path)
