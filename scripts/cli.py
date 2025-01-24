import typer
import asyncio
import logging
from typing import Optional

from scripts.new_subject import create_subject
from scripts.new_resource import (
    create_resource,
    create_chunks,
    connect_resource_to_class,
)
from scripts.new_class import create_class

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command()
def create_new_subject(
    yaml_path: str = typer.Option(..., help="Path to YAML file with subject data"),
    subject_identifier: str = typer.Option(
        ..., help="Identifier for subject in YAML file"
    ),
):
    """Create a new subject from YAML data"""
    try:
        asyncio.run(create_subject(yaml_path, subject_identifier))
        logger.info(f"Successfully created subject from {subject_identifier}")
    except Exception as e:
        logger.error(f"Failed to create subject: {str(e)}")
        raise typer.Exit(1)


@app.command()
def create_new_class(
    yaml_path: str = typer.Option(..., help="Path to YAML file with resource data"),
    class_identifier: str = typer.Option(
        ..., help="The identifier for the class in the YAML file"
    ),
    subject_identifier: str = typer.Option(
        ..., help="The identifier for the subject in the YAML file"
    ),
):
    """Create a new class from YAML data"""
    try:
        asyncio.run(create_class(yaml_path, class_identifier, subject_identifier))
        logger.info(f"Successfully created class from {class_identifier}")
    except Exception as e:
        logger.error(f"Failed to create class: {str(e)}")
        raise typer.Exit(1)


@app.command()
def create_new_resource(
    yaml_path: str = typer.Option(..., help="Path to YAML file with resource data"),
    resource_identifier: str = typer.Option(
        ..., help="Identifier for resource in YAML file"
    ),
    chunks_path: Optional[str] = typer.Option(
        None, help="Path to JSON file with chunks data"
    ),
    class_identifier: Optional[str] = typer.Option(
        None, help="Identifier for the class in YAML file"
    ),
):
    """Create a new resource and optionally add chunks and connect to class"""
    try:
        # Create the resource
        asyncio.run(create_resource(yaml_path, resource_identifier))

        # If chunks path provided, create chunks
        if chunks_path:
            asyncio.run(create_chunks(yaml_path, resource_identifier, chunks_path))

        # If class_id provided, connect resource to class
        if class_identifier:
            # Get resource ID from yaml data
            asyncio.run(
                connect_resource_to_class(
                    yaml_path, class_identifier, resource_identifier
                )
            )

        logger.info(f"Successfully created resource from {resource_identifier}")
    except Exception as e:
        logger.error(f"Failed to create resource: {str(e)}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

""" Examples

To create a new class geography form 6 (note that the subject must already exist)
python -m scripts.cli create-new-class --yaml-path /Users/victoroldensand/Documents/Hack/twiga-stuff/twiga-warehouse/assets/data.yaml --class-identifier geography_class_form6 --subject-identifier geography_subject

To create a new resource for a specific class
python -m scripts.cli create-new-resource --yaml-path assets/data.yaml --resource-identifier geography_resource_form6 --chunks-path assets/chunks/geography_form_6_wiki_content_chunks.json --class-identifier geography_class_form6

"""
