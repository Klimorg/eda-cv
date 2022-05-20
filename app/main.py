from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.routes import eda

app = FastAPI(
    title="Basic API for Computer Vision EDA",
    description="""TBA""",
    version="0.1.0",
)

app.include_router(eda.router, prefix="/eda")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
