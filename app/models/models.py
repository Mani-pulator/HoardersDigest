from datetime import datetime
from enum import Enum

from sqlalchemy import Column, Text
from sqlmodel import Field, SQLModel

class ProcessingStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    done = "done"
    failed = "failed"

class Collection(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    # upload_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    name: str
    status: ProcessingStatus = ProcessingStatus.pending
    summary: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    

class Save(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    collection_id: int = Field(foreign_key="collection.id")
    media_url: str
    category: str # e.g. "post", "reel", "story"


class SaveSummary(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    save_id: int = Field(foreign_key="save.id")
    transcript: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    visual_description: str | None = Field(default=None, sa_column=Column(Text, nullable=True))




