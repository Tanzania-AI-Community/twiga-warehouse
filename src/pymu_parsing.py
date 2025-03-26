import fitz
import pathlib
import pymupdf4llm

# If output directories do not exist, create them
output_dir = pathlib.Path(r"C:\Users\ADMIN\Desktop\KTHAIS\twiga-warehouse\data\parsed")
images_dir = output_dir / "images"
output_dir.mkdir(exist_ok=True)
images_dir.mkdir(exist_ok=True)

def text_only_extraction(pdf_path, output_md):
    """Only extract text from pdf, ignore images."""

    md_text = pymupdf4llm.to_markdown(pdf_path)
    pathlib.Path("pymu_output/output.md").write_bytes(md_text.encode())

def save_image(image_data, xref, ext):
    """Save image to the images directory."""

    image_file = images_dir / f"img_{xref}.{ext}"
    with open(image_file, "wb") as img_file:
        img_file.write(image_data)
    return image_file

def extract_images(doc, page_number):
    """Extract images from a specific page."""

    images_md = []
    for img in doc.get_page_images(page_number):
        xref = img[0]
        image = doc.extract_image(xref)
        if not image:
            continue
        image_file = save_image(image['image'], xref, image['ext'])
        images_md.append(f"![Image {xref}]({image_file})\n")
    return "\n".join(images_md)

def text_and_image_extraction(pdf_path, output_md):
    """Extract text and image page by page."""

    for page_number in range(len(doc)):
        page_text = pymupdf4llm.to_markdown(pdf_path, pages=[page_number])
        page_images = extract_images(doc, page_number)
        
        markdown_content.append(f"## Page {page_number + 1}\n")
        markdown_content.append(page_text.strip())
        if page_images:
            markdown_content.append(page_images)

    # Save md output
    with open(output_md, "w", encoding="utf-8") as md_file:
        md_file.write("\n\n".join(markdown_content))

    print(f"Markdown with text and images saved to: {output_md}")

# Initialize md content
markdown_content = []

# Load pdf
pdf_path = r"C:\Users\ADMIN\Desktop\KTHAIS\twiga-warehouse\data\geo_form2.pdf"
doc = fitz.open(pdf_path)
output_md = output_dir / "text.md"

# Extract text and images
text_and_image_extraction(pdf_path, output_md)