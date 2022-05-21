from enum import Enum

from pydantic import BaseModel, BaseSettings


class FeatureReport(BaseModel):
    red_mean_value: float
    green_mean_value: float
    blue_mean_value: float


class Extension(Enum):
    png = ".png"
    jpg = ".jpg"
