
import hashlib
import json
import re
import shutil
from pathlib import Path

import pymupdf

from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
)
from docling_core.types.doc import TextItem

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input_pdfs"
OUTPUT_DIR = BASE_DIR / "outputs"

# Images smaller than this (either side) or thinner than this
# aspect ratio are decoration (lines, bullets, spacers).
MIN_IMAGE_SIDE = 64
MAX_ASPECT_RATIO = 15

# An image covering more than this share of the page is treated
# as a scanned page, so OCR text inside it is kept.
SCANNED_PAGE_COVERAGE = 0.8

options = PdfPipelineOptions()

# OCR is kept for scanned pages; OCR text that falls inside
# a normal embedded image is dropped below (the image gets
# its own description, so the text would be duplicated).
options.do_ocr = True

# Tables are extracted by PyMuPDF below; Docling text inside
# table areas is dropped so tables are not indexed twice.
options.do_table_structure = False

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=options
        )
    }
)

HEADING_LABELS = {"section_header", "title"}


# ============================================================
# CLEAN EXTRACTED PARAGRAPH
# ============================================================

def normalize_paragraph(text):
    """
    Remove line wrapping inside one paragraph and join words
    hyphenated across lines.
    """

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    text = re.sub(r"(?<=[A-Za-z])-\n(?=[a-z])", "", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


# ============================================================
# GEOMETRY HELPERS (top-left origin, PDF points)
# ============================================================

def center_inside(box, region):
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]


def box_area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


# ============================================================
# FIND TABLES AND IMAGES WITH PYMUPDF
# ============================================================

def find_page_tables(page):
    """
    Read every table while the page is still open: PyMuPDF
    Table objects stop working once their page is released.
    """

    tables = []

    for table in page.find_tables().tables:

        rows = table.extract()
        header = table.header

        tables.append({
            "rows": rows,
            "header_names": list(header.names) if header and header.names else None,
            "header_external": bool(header and header.external),
            "bbox": tuple(table.bbox),
        })

    return tables


def collect_page_images(page, seen_hashes):
    """
    Return useful images on one page: decoration and images
    already seen in this document (logos, repeated banners)
    are dropped.
    """

    kept = []

    for block in page.get_text("dict").get("blocks", []):

        if block.get("type") != 1:
            continue

        image_bytes = block.get("image")

        if not image_bytes:
            continue

        width = block.get("width", 0)
        height = block.get("height", 0)

        if min(width, height) < MIN_IMAGE_SIDE:
            continue

        if max(width, height) / max(1, min(width, height)) > MAX_ASPECT_RATIO:
            continue

        digest = hashlib.sha256(image_bytes).hexdigest()

        if digest in seen_hashes:
            continue

        seen_hashes.add(digest)

        kept.append({
            "bytes": image_bytes,
            "ext": block.get("ext", "png").lower(),
            "bbox": list(block.get("bbox", [])),
            "sha256": digest,
        })

    return kept


# ============================================================
# EXTRACT TEXT BY PAGE IN DOCUMENT READING ORDER
# ============================================================

def extract_document_text(pdf_path, page_info):
    """
    Run Docling once on the whole document.

    page_info[page_no] = {"height", "tables", "images", "area"}
    with boxes in top-left coordinates.

    Headings are written as "# Heading" so the chunker can keep
    section context. Text inside tables, and OCR text inside
    ordinary images, is skipped to avoid duplicates.
    """

    print("Running Docling layout analysis...")

    document = converter.convert(str(pdf_path)).document

    page_text = {}

    for item, _level in document.iterate_items():

        if not isinstance(item, TextItem) or not item.prov:
            continue

        label = str(getattr(item.label, "value", item.label)).lower()

        if label in {"page_header", "page_footer"}:
            continue

        paragraph = normalize_paragraph(item.text)

        if not paragraph:
            continue

        prov = item.prov[0]
        page_number = prov.page_no
        info = page_info.get(page_number)

        if info:
            box = prov.bbox.to_top_left_origin(info["height"])
            box = (box.l, box.t, box.r, box.b)

            if any(center_inside(box, t) for t in info["tables"]):
                continue

            if any(
                center_inside(box, img)
                and box_area(img) < SCANNED_PAGE_COVERAGE * info["area"]
                for img in info["images"]
            ):
                continue

        if label in HEADING_LABELS:
            paragraph = f"# {paragraph}"

        page_text.setdefault(page_number, []).append(paragraph)

    return {
        page: "\n\n".join(paragraphs)
        for page, paragraphs in page_text.items()
    }


# ============================================================
# SAVE TABLES AS ROW / COLUMN TEXT
# ============================================================

def save_page_tables(tables, page_number, pdf_name, tables_dir):

    saved = []

    for table_index, table in enumerate(tables, start=1):

        rows = table["rows"]

        if not rows:
            continue

        if table["header_names"]:
            headers = [str(v or "").strip() for v in table["header_names"]]
        else:
            headers = [str(v or "").strip() for v in rows[0]]

        # Header row inside the table body is not a data row.
        if table["header_external"]:
            data_rows = rows
        else:
            data_rows = rows[1:]

        lines = [
            f"Source PDF: {pdf_name}",
            f"Page Number: {page_number}",
            f"Table Number: {table_index}",
            ""
        ]

        for row_index, row in enumerate(data_rows, start=1):

            lines.append(f"Row {row_index}:")

            for column_index, value in enumerate(row):

                column_name = (
                    headers[column_index]
                    if column_index < len(headers)
                    else ""
                ) or f"Column {column_index + 1}"

                lines.append(
                    f"{column_name}: {str(value or '').strip()}"
                )

            lines.append("")

        filename = f"page_{page_number:03d}_table_{table_index:03d}.txt"
        table_path = tables_dir / filename
        table_path.write_text("\n".join(lines), encoding="utf-8")

        saved.append({
            "filename": filename,
            "path": table_path.relative_to(BASE_DIR).as_posix(),
            "rows": len(data_rows),
            "bbox": list(table["bbox"])
        })

    return saved


def save_page_images(images, page_number, images_dir):

    saved = []

    for index, image in enumerate(images, start=1):

        filename = f"page_{page_number:03d}_img_{index:03d}.{image['ext']}"
        output_path = images_dir / filename
        output_path.write_bytes(image["bytes"])

        saved.append({
            "filename": filename,
            "path": output_path.relative_to(BASE_DIR).as_posix(),
            "bbox": image["bbox"],
            "sha256": image["sha256"],
        })

    return saved


def reset_folder(folder):
    """
    Remove stale files from an earlier run. The folder itself
    is kept: on Windows, deleting and recreating a folder at
    once can fail while the delete is still pending.
    """
    folder.mkdir(parents=True, exist_ok=True)
    for path in folder.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def process_pdf(pdf_path):

    document_name = pdf_path.stem
    document_folder = OUTPUT_DIR / document_name

    text_dir = document_folder / "text"
    images_dir = document_folder / "images"
    tables_dir = document_folder / "tables"

    # Image descriptions are NOT deleted: describe_images.py
    # matches each description to its image by SHA-256 and
    # regenerates or removes the ones that no longer match.
    for folder in (text_dir, images_dir, tables_dir):
        reset_folder(folder)

    print(f"\nProcessing: {pdf_path.name}")

    with pymupdf.open(pdf_path) as pdf:

        # Pass 1: tables and images (PyMuPDF, top-left boxes).
        page_tables = {}
        page_images = {}
        page_info = {}
        seen_hashes = set()

        for page_index, page in enumerate(pdf):

            page_number = page_index + 1
            tables = find_page_tables(page)
            images = collect_page_images(page, seen_hashes)

            page_tables[page_number] = tables
            page_images[page_number] = images
            page_info[page_number] = {
                "height": page.rect.height,
                "area": page.rect.width * page.rect.height,
                "tables": [t["bbox"] for t in tables],
                "images": [tuple(i["bbox"]) for i in images if len(i["bbox"]) == 4],
            }

        # Pass 2: text in reading order (Docling).
        page_text = extract_document_text(pdf_path, page_info)

        manifest = {
            "source_pdf": pdf_path.name,
            "document_id": document_name,
            "extraction_engine": "docling",
            "pages": []
        }

        for page_index in range(len(pdf)):

            page_number = page_index + 1
            text = page_text.get(page_number, "")

            text_path = text_dir / f"page_{page_number:03d}.txt"
            text_path.write_text(text, encoding="utf-8")

            images = save_page_images(
                page_images[page_number], page_number, images_dir
            )

            tables = save_page_tables(
                page_tables[page_number], page_number,
                pdf_path.name, tables_dir
            )

            manifest["pages"].append({
                "page_number": page_number,
                "text_file": text_path.relative_to(BASE_DIR).as_posix(),
                "character_count": len(text),
                "images": images,
                "tables": tables
            })

            print(
                f"Page {page_number}: {len(text)} characters, "
                f"{len(images)} images, {len(tables)} tables"
            )

    (document_folder / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"Completed: {document_name}")


def main():

    if not INPUT_DIR.exists():
        print("input_pdfs folder not found.")
        return

    pdf_files = sorted(INPUT_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDFs found.")
        return

    for pdf_path in pdf_files:
        process_pdf(pdf_path)

    print("\nPDF EXTRACTION COMPLETED")


if __name__ == "__main__":
    main()
