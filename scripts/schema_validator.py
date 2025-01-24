# twiga-warehouse/scripts/schema_validator.py

from sqlalchemy import MetaData, create_engine
from sqlalchemy.schema import CreateTable
import requests
import hashlib

# TODO: This is just a claude script. Should modify it to work here so that we validate that we have a schema match when uploading content.


def get_schema_hash(schema_sql: str) -> str:
    """Generate deterministic hash of schema definition"""
    return hashlib.sha256(schema_sql.encode()).hexdigest()


async def fetch_twiga_schemas(branch: str = "main") -> dict:
    """Fetch schema definitions from twiga repository"""
    # Could be from raw GitHub content or a dedicated endpoint
    SCHEMA_URL = f"https://raw.githubusercontent.com/Tanzania-AI-Community/twiga/{branch}/app/database/models.py"
    response = requests.get(SCHEMA_URL)
    return response.text


async def validate_schemas():
    """Validate that warehouse schemas match twiga schemas"""
    # Get twiga schemas
    twiga_schemas = await fetch_twiga_schemas()
    twiga_hash = get_schema_hash(twiga_schemas)

    # Compare with local schemas
    with open("schemas/current_hash.txt", "r") as f:
        local_hash = f.read().strip()

    if local_hash != twiga_hash:
        raise ValueError(
            "Schema mismatch detected! Please update warehouse schemas from twiga."
        )


def update_schemas():
    """Update local schemas from twiga"""
    schema_text = fetch_twiga_schemas()

    # Update local files
    with open("schemas/models.py", "w") as f:
        f.write(schema_text)

    # Update hash
    with open("schemas/current_hash.txt", "w") as f:
        f.write(get_schema_hash(schema_text))
