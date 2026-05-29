from pathlib import Path

import yaml

from src.config.settings import settings
from src.models import (
    BookDefinition,
    BookMetadata,
    BookPagination,
    BookSourcePaths,
    ClassMetadata,
    ResourceMetadata,
    SubjectMetadata,
)


def parse_comma_separated_ints(value: str) -> list[int]:
    try:
        return [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise ValueError(f"'{value}' is not a comma-separated list of integers") from exc


def normalize_page_numbers(value: int | str | list[int]) -> int | list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        parsed_values = parse_comma_separated_ints(value=value)
        if len(parsed_values) == 1:
            return parsed_values[0]
        return parsed_values
    raise ValueError(f"Unsupported page number format: {value!r}")


def derive_input_file_name(subject_name: str, form: str) -> str:
    return f"{subject_name}_{normalize_form_name(form=form)}.pdf"


def resolve_book_paths(
    input_dir: Path,
    input_file_name: str,
    output_file_name: str,
    input_root: Path | None = None,
    output_root: Path | None = None,
) -> BookSourcePaths:
    books_root = input_root or Path(settings.INPUT_BOOKS_PATH)
    outputs_root = output_root or Path(settings.OUTPUT_BOOKS_PATH)

    resolved_input_dir = books_root / input_dir
    return BookSourcePaths(
        input_dir=input_dir,
        info_path=resolved_input_dir / "info.yaml",
        input_path=resolved_input_dir / input_file_name,
        output_path=outputs_root / output_file_name,
        checkpoints_path=resolved_input_dir / "checkpoints",
    )


def load_info_yaml(info_path: Path) -> dict[str, object]:
    with info_path.open(mode="r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML document at {info_path}: expected a mapping")

    return data


def build_book_definition(paths: BookSourcePaths) -> BookDefinition:
    yaml_data = load_info_yaml(info_path=paths.info_path)

    resource_data = yaml_data.get("resource", {})
    class_data = yaml_data.get("class", {})
    subject_data = yaml_data.get("subject", {})
    book_config = yaml_data.get("book_config", {})

    metadata = BookMetadata(
        resource=ResourceMetadata(
            name=resource_data.get("name", ""),
            type=resource_data.get("type", "textbook"),
            authors=list(resource_data.get("authors", [])),
        ),
        class_=ClassMetadata(
            name=class_data.get("name", ""),
            grade_level=class_data.get("grade_level", ""),
            status=class_data.get("status", ""),
        ),
        subject=SubjectMetadata(name=subject_data.get("name", "")),
    )

    if "table_of_contents_page_number" not in book_config:
        raise ValueError(f"Missing 'book_config.table_of_contents_page_number' in {paths.info_path}")

    if "first_page_number" not in book_config:
        raise ValueError(f"Missing 'book_config.first_page_number' in {paths.info_path}")

    return BookDefinition(
        metadata=metadata,
        source_paths=paths,
        pagination=BookPagination(
            table_of_contents_page_numbers=normalize_page_numbers(
                value=book_config["table_of_contents_page_number"]
            ),
            first_page_number=int(book_config["first_page_number"]),
            last_page_number=book_config.get("last_page_number"),
        ),
    )


def normalize_form_name(form: str) -> str:
    form_mapping = {
        "form_1": "form_one",
        "form_2": "form_two",
        "form_3": "form_three",
        "form_4": "form_four",
        "form_5": "form_five",
        "form_6": "form_six",
    }
    return form_mapping.get(form, form)
