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
from app.dependancies.embedding_function import EmbeddingEngine
from app.dependancies.utils import (
    get_items_list,
    load_image_into_numpy_array,
    read_imagefile,
)
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

    engine = EmbeddingEngine(model=model, provider=provider)

    logits = engine.infer(images_paths)

    images_labels = [Path(image_path).parent.stem for image_path in images_paths]
    labels_dict = {label: idx for idx, label in enumerate(sorted(set(images_labels)))}
    tags = [labels_dict[image_label] for image_label in images_labels]
    N = len(set(images_labels))

    # define the colormap
    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list("Custom cmap", cmaplist, cmap.N)

    X_embedded = TSNE(
        n_components=2,
        learning_rate="auto",
        init="random",
    ).fit_transform(logits)

    logger.info(f"{X_embedded.shape}")

    scat = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=tags, cmap=cmap)

    plt.title("tSNE scatterplot with ResNet50v2 preprocess")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(
        handles=scat.legend_elements()[0],
        labels=sorted(set(images_labels)),
        title="species",
    )
    # create the colorbar
    # define the bins and normalize
    bounds = np.linspace(0, N, N + 1)
    cb = plt.colorbar(scat, spacing="proportional", ticks=bounds)
    cb.set_label("Custom cbar")

    saved_image_path = Path(
        f"{settings.scatterplots_dir}/scatter_{timestamp}.png",
    ).resolve()

    plt.savefig(saved_image_path)

    return FileResponse(saved_image_path)
