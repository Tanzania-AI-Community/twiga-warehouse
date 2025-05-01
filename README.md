Warehouse repository for Twiga. See more information here: https://github.com/Tanzania-AI-Community/twiga

Test call:
- python -m src.main --title test --author me --input_path <book>.pdf --output_path <output>.json --chunker_type unstructured
- python -m src.main --title test-title --author me --input_path <book>.pdf --output_path test-output.json --table_of_contents_page_number 6 --first_page_number 10 --chunker_type unstructured