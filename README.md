# Twiga Warehouse

This repository provides tools for extracting text and important information from textbooks, preparing them for ingestion into a vector database for use with Large Language Models (LLMs) in its parent project, [Twiga](https://github.com/Tanzania-AI-Community/twiga). Currently, the extraction pipeline only supports text from PDF files, chunking and splitting content using different strategies, and outputting structured data suitable for downstream LLM applications.

## Features

- Extracts text from PDF textbooks.
- Supports multiple chunking strategies (e.g., LangChain, LLM-based).
- Outputs structured JSON for easy ingestion into vector databases.
- Modular design for easy extension and integration.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Tanzania-AI-Community/twiga-warehouse.git
    cd twiga-warehouse
    ```

2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

You can also install the dependencies via the `pyptoject.toml` file, if preferred.

3. Set up your environment variables:
    ```sh
    cp .env.template .env
    # Edit .env as needed
    ```

    The two Unstructured env variables are not mandatory for running the pipeline.

## Before running the pipeline

Parsing a PDF and splitting its contents in chunks is not an easy task. For this pipeline to work, we are making some assumptions on how we must store the book so that we can parse it.

1. Where should the books placed for their parsing?

You can place the books wherever you want. The only think to take into account is that we are assuming that you will have a directory where all the books will be placed. This directory path should be your `INPUT_BOOKS_PATH` env variable. Inside that directory, you will create a folder for each book to parse. This folder name or relative path will be your `--input_dir` argument. There, you will have 2 files: the book in PDF format (you will use it in `--input_file_name` argument) and a YAML file named `info.yaml`.

2. Which format should the PDF have?

The answer to this question depends on the type of chunker to use. Currently, we only supoort PDFs where text can be easily extracted by a basic PDF reader. As some PDF textbooks are scanned using a printer, the text cannot be read by these chunkers. For this, we suggest using [OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF) to add the text layer that we need.

3. Which format should the `info.yaml` file have?

This file stores information about the subject, the authors, the grade level... This is an example of a valid `info.yaml` file:

```yaml
resource:
  name: "History for Secondary Schools Student's Book Form Three"
  type: "textbook"
  authors:
    - "John Doe"
    - "Doe John"

subject:
  name: "history"

class:
  grade_level: "os3"
  status: "active"
  name: "History Form Three"

book_config:
  table_of_contents_page_number: "4,5,6"
  first_page_number: 12
  last_page_number: 145
```

Understanding how [Twiga's](https://github.com/Tanzania-AI-Community/twiga) database looks like is very important for knowing the needed information of this file. As the pipeline extracts the table of contents of the textbook, we are adding the lines where the table of contents is stored. Also, the initial and last page of the document are needed so that we parse that window, avoiding possible errors in the chunking process.

## Usage

You can run the main extraction pipeline using the following commands:

```sh
python3 -m src.main --chunker_type langchain --input_dir "form_4/biology/" --input_file_name "biology_form_4.pdf" --output_file_name "biology_form_4.json"
```

```sh
python3 -m src.main --chunker_type llm --input_dir "form_3/history/" --input_file_name "history_form_3.pdf" --output_file_name "history_test_4.json" --page_batch_size 2
```

An explanation of the arguments allowed can be found in `src.main.py`.

## Project Structure

- `src/`: Main source code for extraction and processing.
- `database/`: Database models, enums, and utilities. Currently not used.
- `legacy/`: Legacy scripts for various parsing strategies.
- `scripts/`: Utility scripts for CLI and schema validation. Currently not used.
- `data/`: Input and output data directories.

## Contributing

Contributions are welcome! Please, consider opening issues before submitting pull requests.

## License

See [LICENSE](LICENSE) for details.

---

For more information, see the [Twiga main repository](https://github.com/Tanzania-AI-Community/twiga).
