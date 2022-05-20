from typing import Mapping

from pydantic import BaseModel


class FeatureReport(BaseModel):
    red_mean_value: float
    green_mean_value: float
    blue_mean_value: float
