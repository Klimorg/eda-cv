import arrow
from fastapi import APIRouter, status
from fastapi.responses import FileResponse

from app.config import settings
from app.dependancies.embedding_function import EmbeddingEngine
from app.dependancies.utils import get_items_list
from app.pydantic_models import EmbeddingsModel, Extension, Providers

router = APIRouter()


@router.get(
    "/tsne",
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

    X_embedded = engine.compute_tsne(logits=logits)

    saved_image_path = engine.plot(
        logits=X_embedded,
        images_paths=images_paths,
        timestamp=timestamp,
    )

    return FileResponse(saved_image_path)
