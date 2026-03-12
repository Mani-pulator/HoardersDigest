from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.models.database import engine
from app.models.models import SQLModel

from app.routers import digest

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    SQLModel.metadata.create_all(engine)
    yield
    # SQLModel.metadata.drop_all(engine)

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(digest.router, prefix="/api")


@app.get("/api")
def api_root():
    return {"message": "hello from backend"}
