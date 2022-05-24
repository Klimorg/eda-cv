from enum import Enum

from pydantic import BaseModel


class FeatureReport(BaseModel):
    red_mean_value: float
    green_mean_value: float
    blue_mean_value: float
    red_std_value: float
    green_std_value: float
    blue_std_value: float


class Extension(Enum):
    png = ".png"
    jpg = ".jpg"


class EmbeddingsModel(Enum):
    resnet50v2 = "resnet50v2"


class Providers(Enum):
    cpu = "CPUExecutionProvider"
    gpu = "CUDAExecutionProvider"


class ClusteringMode(Enum):
    tsne = "tSNE"
    umap = "UMAP"
