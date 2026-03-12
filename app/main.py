from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import digest

app = FastAPI()

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
