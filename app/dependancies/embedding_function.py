from functools import lru_cache

import numpy as np
import onnxruntime as rt
from loguru import logger
from PIL import Image

from app.pydantic_models import EmbeddingsModel, Providers


class EmbeddingEngine:
    def __init__(
        self, model: EmbeddingsModel, provider: Providers, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model = model.value
        self.provider = [provider.value]

        self.loaded_model = rt.InferenceSession(self.model, providers=self.provider)

    def infer(self, images_paths):

        images_list = [Image.open(image).resize((224, 224)) for image in images_paths]

        images = [np.asarray(image, dtype="float32") / 255 for image in images_list]

        images = np.reshape(images, (-1, 224, 224, 3))
        logger.info(f"{images.shape}")

        logits = self.loaded_model.run(["avg_pool"], {"input": images})
        logger.info(f"{type(logits)}")

        return logits[0]


# model = "MobileViT-XXS_FPN_2021-10-13_17-37-15.onnx"

# providers = ["CPUExecutionProvider"]
# m = rt.InferenceSession(model, providers=providers)
# onnx_pred = m.run(["avg_pool"], {"input": batch})

# onnx_pred[0].shape


# image = Image.open(image_path).resize((1024, 1024))
# print(f"{image.size}")
# width, height = image.size
