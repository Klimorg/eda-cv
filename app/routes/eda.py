from io import BytesIO
from pathlib import Path

import arrow
import numpy as np
from fastapi import APIRouter, File, UploadFile, status
from fastapi.responses import FileResponse, Response
from PIL import Image

from app.config import settings
from app.pydantic_models import Extension, FeatureReport
from app.routes.dependancies import (
    compute_channels_mean,
    compute_channels_std,
    compute_histograms_channels,
    compute_mean_image,
    compute_scatterplot,
    get_items_list,
    load_image_into_numpy_array,
    read_imagefile,
)

router = APIRouter()


@router.post(
    "/mean_values",
    response_model=FeatureReport,
    status_code=status.HTTP_200_OK,
    tags=["CV"],
)
async def get_mean_values(file: UploadFile = File(...)):
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
    normalize: bool = False,
):

    timestamp = arrow.now().format("YYYY-MM-DD_HH:mm:ss")

    filename = Path(file.filename).stem
    image = load_image_into_numpy_array(await file.read())

    saved_image_path = compute_histograms_channels(
        image=image,
        filename=filename,
        timestamp=timestamp,
        normalize=normalize,
    )

    result = {"filename": file.filename}

    return FileResponse(saved_image_path, headers=result)


@router.get(
    "/dataset_mean_image",
    tags=["CV"],
    status_code=status.HTTP_200_OK,
)
async def get_dataset_mean_image(extension: Extension):

    # TODO : check for image size and resize if necessary

    timestamp = arrow.now().format("YYYY-MM-DD_HH:mm:ss")

    if extension == Extension.jpg:
        images_paths = get_items_list(directory=settings.data_dir, extension=".jpg")
    elif extension == Extension.png:
        images_paths = get_items_list(directory=settings.data_dir, extension=".png")
    else:
        pass

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
    timestamp = arrow.now().format("YYYY-MM-DD_HH:mm:ss")

    if extension == Extension.jpg:
        images_paths = get_items_list(directory=settings.data_dir, extension=".jpg")
    elif extension == Extension.png:
        images_paths = get_items_list(directory=settings.data_dir, extension=".png")
    else:
        pass

    images_list = [
        np.array(Image.open(image), dtype=np.float32) for image in images_paths
    ]

    saved_image_path = compute_scatterplot(images_list=images_list, timestamp=timestamp)

    return FileResponse(saved_image_path)
