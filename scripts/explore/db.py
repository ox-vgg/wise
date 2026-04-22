import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .models import Base

def init_explore_db(project_dir: Path):
    explore_db_path = project_dir / "metadata" / "explore.db"
    
    dburi = f"sqlite:///{explore_db_path.absolute()}"
    engine = create_engine(dburi, connect_args={"check_same_thread": False})
    
    # Create all tables if they don't exist
    Base.metadata.create_all(engine)
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return engine, SessionLocal
