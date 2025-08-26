import json
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import re
import pathlib

pdf_path = r"C:\Users\ADMIN\Desktop\KTHAIS\twiga-warehouse\data\parsed\text.md"

with open(pdf_path, "r", encoding="utf-8") as md_file:
    md_doc = md_file.read()

headers_to_split_on = [
    ("####", "Header 1"), # Chapter section
    ("######", "Header 2"), # Acknowledgements, table of contents, etc.
    ("#", "Header 3"), # Bold
    ("#####", "Header 4"), # Bold + italic
]

def preprocess_md(md_doc):
    """Modify md document so that bold and bold + italic text is recognized as headers."""

    md_doc = re.sub(r'\n\*\*(.*?)\*\*\n', r'\n# \1\n', md_doc) # Bold 
    md_doc = re.sub(r'\n\*\*_([^*]+)_\*\*\n', r'\n##### \1\n', md_doc) # Bold + italic

    return md_doc

def md_split(md_doc):
    """Split document based on md headers."""

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    md_header_splits = markdown_splitter.split_text(md_doc)

    return md_header_splits

def recursive_split(md_header_splits):
    """Split document recursively."""

    chunk_size = 250
    chunk_overlap = 30
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    splits = text_splitter.split_documents(md_header_splits)

    return splits

# Change bold & bold + italic text to headers
md_doc = preprocess_md(md_doc)

# Split data
md_header_splits = md_split(md_doc)
#character_splits = recursive_split(md_header_splits)

# Append metadata
splits_data = []
for split in md_header_splits:
    splits_data.append({
        "content": split.page_content,
        "metadata": split.metadata
    })

# Save JSON output
output_dir = pathlib.Path(r"C:\Users\ADMIN\Desktop\KTHAIS\twiga-warehouse\data\parsed")
output_dir.mkdir(exist_ok=True)

output_json_path = output_dir / "text.json"
with open(output_json_path, "w", encoding="utf-8") as json_file:
    json.dump(splits_data, json_file, ensure_ascii=False, indent=4)

print(f"Markdown splits saved to {output_json_path}")