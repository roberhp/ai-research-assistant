import os

import pytest
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session


load_dotenv()


@pytest.fixture
def db_session():
    database_url = os.environ["DATABASE_TEST_URL"]

    engine = create_engine(database_url)
    session = Session(engine)

    try:
        yield session
    finally:
        session.rollback()
        session.close()
        engine.dispose()