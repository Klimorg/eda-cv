from functools import lru_cache

import onnxruntime as rt

from app.pydantic_models import EmbeddingsModel, Providers


class EmbeddingEngine:
    def __init__(
        self, model: EmbeddingsModel, provider: Providers, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model = model.value
        self.provider = [provider.value]

    @lru_cache()
    async def load_model(self):
        return rt.InferenceSession(self.model, providers=self.provider)

    async def infer(self, loaded_model, batch):
        logits = await loaded_model.run(["avg_pool"], {"input": batch})
        return logits[0]


# model = "MobileViT-XXS_FPN_2021-10-13_17-37-15.onnx"

# providers = ["CPUExecutionProvider"]
# m = rt.InferenceSession(model, providers=providers)
# onnx_pred = m.run(["avg_pool"], {"input": batch})

# onnx_pred[0].shape
