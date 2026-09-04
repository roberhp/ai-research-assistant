import os

import pytest
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from ai_research_assistant.database import Base

load_dotenv()


@pytest.fixture
def db_session():
    database_url = os.environ["DATABASE_TEST_URL"]

    engine = create_engine(database_url)

    with engine.begin() as connection:
        connection.execute(
            text("CREATE EXTENSION IF NOT EXISTS vector")
        )

    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    session = Session(engine)

    try:
        yield session
    finally:
        session.rollback()
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()