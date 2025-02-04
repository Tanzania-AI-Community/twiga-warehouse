from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
import logging
from sqlmodel import select
import yaml

import database.models as models
from database.utils import get_database_url


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_class(file_path: str, class_identifier: str, subject_identifier: str):
    try:
        """Load sample data into the database."""
        with open(file_path) as f:
            data = yaml.safe_load(f)

        engine = create_async_engine(get_database_url())

        async with AsyncSession(engine) as session:
            subject_data = data[subject_identifier]
            stmt = select(models.Subject).where(
                models.Subject.name == subject_data["name"]
            )
            result = await session.execute(stmt)
            subject = result.scalar_one_or_none()

            if not subject:
                raise Exception(
                    "The subject related to this class doesn't exist, you must create it first."
                )
            else:
                logger.info(
                    f"Found existing subject: {subject.name} (ID: {subject.id})"
                )

            assert subject.id

            # Check if the class already exists
            class_data = data[class_identifier]
            stmt = select(models.Class).where(
                models.Class.subject_id == subject.id,
                models.Class.grade_level == class_data["grade_level"],
            )
            result = await session.execute(stmt)
            existing_class = result.scalar_one_or_none()

            if existing_class:
                raise Exception("This class already exists.")
            else:
                logger.info(
                    f"No existing class found with name: {class_data['name']}, proceeding with creation."
                )

            # Create the class
            class_obj = models.Class(
                subject_id=subject.id,  # Use the actual subject ID
                grade_level=class_data["grade_level"],
                status=class_data["status"],
            )
            session.add(class_obj)
            await session.flush()
            logger.info(
                f"Created class: Grade level: {class_obj.grade_level}, Subject ID: {class_obj.subject_id}, ID: {class_obj.id})"
            )

            await session.commit()
    except Exception as e:
        logger.error(f"Error injecting sample data: {str(e)}")
        raise
    finally:
        await engine.dispose()
