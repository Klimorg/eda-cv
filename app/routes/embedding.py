from io import BytesIO
from pathlib import Path

import arrow
import numpy as np
from fastapi import APIRouter, File, UploadFile, status
from fastapi.responses import FileResponse, Response
from loguru import logger
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE

from app.config import settings
from app.dependancies.eda_functions import (
    get_items_list,
    load_image_into_numpy_array,
    read_imagefile,
)
from app.dependancies.embedding_function import EmbeddingEngine
from app.pydantic_models import Embeddings, EmbeddingsModel, Extension, Providers

router = APIRouter()


@router.get(
    "/embeddings",
    status_code=status.HTTP_200_OK,
    tags=["embedding"],
)
async def get_cnn_embedding(
    model: EmbeddingsModel,
    provider: Providers,
    extension: Extension,
):

    timestamp = arrow.now().format("YYYY-MM-DD_HH-mm-ss")

    images_paths = get_items_list(
        directory=settings.data_dir,
        extension=extension.value,
    )
    images_labels = [Path(image_path).parent.stem for image_path in images_paths[:5]]

    engine = EmbeddingEngine(model=model, provider=provider)

    logits = engine.infer(images_paths)

    X_embedded = TSNE(
        n_components=2,
        learning_rate="auto",
        init="random",
    ).fit_transform(logits)

    logger.info(f"{X_embedded.shape}")

    plt.scatter(means, stds, c=color, alpha=0.5)

    plt.title("Mean-std scatterplot. Pixel values in [0,1]")
    plt.xlabel("means")
    plt.ylabel("stds")

    saved_image_path = Path(
        f"{settings.scatterplots_dir}/scatter_{timestamp}.png",
    ).resolve()

    plt.savefig(saved_image_path)

    return saved_image_path

    return [
        Embeddings(inferences=list(logits[idx, ...])) for idx, _ in enumerate(logits)
    ]
