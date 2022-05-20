from io import BytesIO
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, UploadFile, status
from fastapi.responses import FileResponse, Response

from app.pydantic_models import FeatureReport
from app.routes.dependancies import (
    compute_histograms_channels,
    compute_mean_image,
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

    red_mean_value = image[:, :, 0].mean()
    green_mean_value = image[:, :, 1].mean()
    blue_mean_value = image[:, :, 2].mean()

    return FeatureReport(
        red_mean_value=red_mean_value,
        green_mean_value=green_mean_value,
        blue_mean_value=blue_mean_value,
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
        content=bytes_image.getvalue(), headers=result, media_type="image/png"
    )


@router.post(
    "/channels_histogram",
    tags=["CV"],
    status_code=status.HTTP_200_OK,
    operation_id="compute_histograms_channels",
)
async def get_histograms_channels(file: UploadFile = File(...)):
    filename = Path(file.filename).stem
    image = load_image_into_numpy_array(await file.read())

    compute_histograms_channels(image=image, filename=filename)
    result = {"filename": file.filename}

    return FileResponse(f"histograms/{filename}.png", headers=result)


@router.post(
    "/mean_image",
    tags=["CV"],
    status_code=status.HTTP_200_OK,
)
async def get_mean_image(files: List[UploadFile] = File(...)):

    images_list = [load_image_into_numpy_array(await file.read()) for file in files]

    compute_mean_image(images_list=images_list)

    return FileResponse("mean_image/average.png")
