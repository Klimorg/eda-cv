import arrow
import numpy as np
from fastapi import APIRouter, status
from fastapi.responses import FileResponse
from loguru import logger
from tqdm import tqdm

from app.config import settings
from app.dependancies.embedding_function import EmbeddingEngine
from app.dependancies.utils import generate_batch, get_items_list
from app.pydantic_models import ClusteringMode, EmbeddingsModel, Extension, Providers

router = APIRouter()


@router.get(
    "/clustering",
    status_code=status.HTTP_200_OK,
    tags=["clustering"],
)
async def get_cnn_embedding(
    model: EmbeddingsModel,
    provider: Providers,
    extension: Extension,
    mode: ClusteringMode,
    batch_size: int = 32,
):

    timestamp = arrow.now().format("YYYY-MM-DD_HH-mm-ss")

    images_paths = get_items_list(
        directory=settings.data_dir,
        extension=extension.value,
    )

    engine = EmbeddingEngine(model=model, provider=provider)
    logger.info("Model loaded.")

    batches = generate_batch(lst=images_paths, batch_size=batch_size)
    logits = []

    for batch in tqdm(batches, total=np.ceil(len(images_paths) / batch_size)):
        batch_logits = engine.infer(batch)
        logits.append(batch_logits)
    logits = np.vstack(logits)
    logger.info("Inference done.")

    X_embedded = engine.compute_clustering(logits=logits, mode=mode)
    logger.info("Computing clustering.")

    saved_image_path = engine.plot(
        logits=X_embedded,
        images_paths=images_paths,
        timestamp=timestamp,
        mode=mode,
    )

    config = {
        "model": model.value,
        "provider": provider.value,
        "extension": extension.value,
        "mode": mode.value,
        "timestamp": timestamp,
    }

    return FileResponse(saved_image_path, headers=config)
