from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from loguru import logger

from app.config import settings
from app.routes import eda, embedding

app = FastAPI(
    title="Basic API for Computer Vision EDA",
    description="TBA",
    version="0.1.0",
)

app.include_router(eda.router, prefix="/eda")
app.include_router(embedding.router, prefix="/embedding")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def create_directories():
    logger.info("Creating data directory.")
    Path(f"{settings.data_dir}").mkdir(parents=True, exist_ok=True)
    logger.info("Creating results directories.")
    Path(f"{settings.histograms_dir}").mkdir(parents=True, exist_ok=True)
    Path(f"{settings.mean_image_dir}").mkdir(parents=True, exist_ok=True)
    Path(f"{settings.scatterplots_dir}").mkdir(parents=True, exist_ok=True)


@app.get(
    "/",
    tags=["Startup"],
    description="démarrage de l'API sur la page de documentation.",
)
def main():
    return RedirectResponse(url="/docs/")


@app.get("/healthcheck", tags=["Healthcheck"])
def get_api_status():
    return {"Status": "ok"}
