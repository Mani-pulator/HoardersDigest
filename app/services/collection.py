from sqlmodel import Session
from app.models.database import engine
from app.models.models import Collection, Save, SaveSummary
from app.schemas.collection import FacebookCollection


def create_collections(collections: list[FacebookCollection]):
    with Session(engine) as session:
        for c in collections:
            collection = Collection(name=c.name)
            session.add(collection)
            session.flush()
            
            for s in c.saves:
                save = Save(media_url=s.url, category=s.category, collection_id=collection.id)
                session.add(save)
                
        session.commit()