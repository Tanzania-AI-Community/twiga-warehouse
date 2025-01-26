import os
from time import time
import logging
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

logger = logging.getLogger(__name__)


def get_database_url() -> str:
    """Get formatted database URL from settings"""
    database_uri = urlparse(DATABASE_URL)

    assert database_uri.hostname

    if "neon.tech" in str(database_uri.hostname):
        return f"postgresql+asyncpg://{database_uri.username}:{database_uri.password}@{database_uri.hostname}{database_uri.path}?ssl=require"

    return f"postgresql+asyncpg://{database_uri.username}:{database_uri.password}@{database_uri.hostname}:{database_uri.port}{database_uri.path}"
