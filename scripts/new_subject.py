from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from pathlib import Path
import logging
from sqlmodel import select
import yaml

# Import all your models
import database.models as models
from database.utils import get_database_url

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_subject(file_path: str, subject_identifier: str):
    try:
        """Load sample data into the database."""
        # Load sample data from YAML
        with open(file_path) as f:
            data = yaml.safe_load(f)

        engine = create_async_engine(get_database_url())

        async with AsyncSession(engine) as session:
            # Get the subject data
            subject_data = data[subject_identifier]

            # Check if subject already exists
            stmt = select(models.Subject).where(
                models.Subject.name == subject_data["name"]
            )
            result = await session.execute(stmt)
            existing_subject = result.scalar_one_or_none()

            if existing_subject:
                logger.info(
                    f"Subject already exists: {existing_subject.name} (ID: {existing_subject.id})"
                )
                return existing_subject

            # Create new subject
            subject = models.Subject(
                name=subject_data["name"],
            )
            session.add(subject)
            await session.flush()
            logger.info(f"Created new subject: {subject.name} (ID: {subject.id})")

            await session.commit()
            return subject

    except Exception as e:
        logger.error(f"Error injecting sample data: {str(e)}")
        raise
    finally:
        await engine.dispose()
