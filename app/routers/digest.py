import json

from fastapi import APIRouter, UploadFile

from app.schemas.collection import parse_fb_collections

router = APIRouter(prefix="/digest", tags=["digest"])


@router.get("/collections")
def get_collections():
    collections = [
        {"name": "Career", "saves": []},
        {"name": "Recipe", "saves": []},
    ]
    return {"collections": collections}


@router.post("/upload")
async def upload(file: UploadFile):
    content = await file.read()
    data = json.loads(content)
    collections = parse_fb_collections(data)
    return {"collections": [c.model_dump() for c in collections]}
