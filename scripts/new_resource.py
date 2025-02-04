from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
import json
from sqlmodel import select
import logging
from typing import Any
import yaml

# Import all your models
import database.models as models
from database.utils import get_database_url

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_resource(file_path: str, resource_identifier: str):
    try:
        with open(file_path) as f:
            data = yaml.safe_load(f)

        engine = create_async_engine(get_database_url())

        async with AsyncSession(engine) as session:
            resource_data = data[resource_identifier]
            stmt = select(models.Resource).where(
                models.Resource.name == resource_data["name"]
            )
            result = await session.execute(stmt)
            existing_resource = result.scalar_one_or_none()

            if existing_resource:
                logger.info(
                    f"Subject already exists: {existing_resource.name} (ID: {existing_resource.id})"
                )
                return existing_resource

            resource = models.Resource(name=resource_data["name"])
            session.add(resource)
            await session.flush()
            logger.info(f"Created resource: {resource.name} (ID: {resource.id})")

            await session.commit()
    except Exception as e:
        logger.error(f"Error injecting sample data: {str(e)}")
        raise
    finally:
        await engine.dispose()


async def process_chunks(
    session: AsyncSession,
    json_data: list[dict[str, Any]],
    resource_id: int,
    batch_size: int = 30,
):
    """Process chunks to create Chunks in the db"""
    try:
        total_chunks = len(json_data)
        for start_idx in range(0, total_chunks, batch_size):
            end_idx = min(start_idx + batch_size, total_chunks)
            batch_items = json_data[start_idx:end_idx]

            for item in batch_items:
                metadata = item["metadata"]
                chunk = models.Chunk(
                    resource_id=resource_id,
                    content=item["chunk"],
                    chunk_type=metadata["chunk_type"],
                    top_level_section_index=metadata["chapter_number"],
                    top_level_section_title=metadata["chapter"],
                    embedding=item["embedding"],
                )
                session.add(chunk)

            await session.commit()
            logger.info(
                f"Processed and saved chunks {start_idx + 1} to {end_idx} of {total_chunks}"
            )

    except Exception as e:
        logger.error(f"Error processing chunks: {str(e)}")
        raise


async def create_chunks(file_path: str, resource_identifier: str, chunks_path: str):
    try:
        engine = create_async_engine(get_database_url())
        with open(file_path) as f:
            yaml_data = yaml.safe_load(f)
            resource_name = yaml_data[resource_identifier]["name"]

        async with AsyncSession(engine) as session:
            stmt = select(models.Resource).where(models.Resource.name == resource_name)
            result = await session.execute(stmt)
            resource = result.scalar_one_or_none()

            if not resource:
                raise ValueError(
                    "Resource not found. Please create it before adding chunks."
                )

            # Check for existing chunks
            chunks_stmt = select(models.Chunk).where(
                models.Chunk.resource_id == resource.id
            )
            chunks_result = await session.execute(chunks_stmt)
            existing_chunks = chunks_result.scalars().all()

            if existing_chunks:
                logger.info(
                    f"Found {len(existing_chunks)} existing chunks for resource {resource_name}"
                )
                return

            # Load and process chunks if none exist
            with open(chunks_path, "r") as f:
                chunks_data = json.load(f)

            await process_chunks(
                session=session, json_data=chunks_data, resource_id=resource.id
            )
            logger.info("Vector data injection complete.")

    except Exception as e:
        logger.error(f"Error injecting vector data: {str(e)}")
        raise
    finally:
        await engine.dispose()


async def connect_resource_to_class(
    yaml_path: str, class_identifier: str, resource_identifier: str
):
    try:
        engine = create_async_engine(get_database_url())

        # Load data from YAML
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        resource_name = data[resource_identifier]["name"]
        class_data = data[class_identifier]

        async with AsyncSession(engine) as session:
            # Get resource and class IDs
            resource_stmt = select(models.Resource).where(
                models.Resource.name == resource_name
            )
            resource = (await session.execute(resource_stmt)).scalar_one_or_none()

            if not resource:
                raise ValueError(f"Resource {resource_name} not found")

            class_stmt = select(models.Class).where(
                models.Class.grade_level == class_data["grade_level"]
            )
            class_obj = (await session.execute(class_stmt)).scalar_one_or_none()

            if not class_obj:
                raise ValueError(
                    f"Class with grade level {class_data['grade_level']} not found"
                )

            # Check if relationship exists
            rel_stmt = select(models.ClassResource).where(
                models.ClassResource.class_id == class_obj.id,
                models.ClassResource.resource_id == resource.id,
            )
            existing = (await session.execute(rel_stmt)).scalar_one_or_none()

            if existing:
                logger.info("Class-resource relationship already exists")
                return

            # Create relationship
            class_resource = models.ClassResource(
                class_id=class_obj.id, resource_id=resource.id
            )
            session.add(class_resource)
            await session.commit()
            logger.info(
                f"Created class-resource relationship (ID: {class_resource.id})"
            )

    except Exception as e:
        logger.error(f"Error connecting resource to class: {str(e)}")
        raise
    finally:
        await engine.dispose()
