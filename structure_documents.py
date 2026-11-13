
import json
import hashlib
import re
from pathlib import Path
from urllib.parse import quote


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FOLDER = BASE_DIR / "outputs"

STRUCTURED_FOLDER = BASE_DIR / "structured"

CONTENT_TYPES = [
    "text",
    "image_descriptions",
    "tables"
]

# Reference block written by older describe_images.py runs.
OLD_IMAGE_FORMAT = re.compile(
    r"^Image Description:\s*\n(.*?)\n\s*\nImage Reference:",
    re.S
)


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def get_page_number(filename):
    """
    Extract page number from filenames like:
    page_001.txt, page_002_img_001.txt, page_003_table_001.txt
    """
    match = re.search(r"page_(\d+)", filename)
    return int(match.group(1)) if match else None


def create_record_id(document_name, content_type, path):
    """Stable ID from document, content type and file path."""
    key = f"{document_name}|{content_type}|{path}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def read_image_description(file_path, document_folder):
    """
    Return (description, extra_metadata) for one image.

    Only the description is embedded; the image path and URL
    go into metadata. Descriptions whose image is gone are
    skipped.
    """

    content = file_path.read_text(encoding="utf-8").strip()

    match = OLD_IMAGE_FORMAT.match(content)
    if match:
        content = match.group(1).strip()

    meta_file = file_path.with_suffix(".json")
    meta = {}

    if meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except ValueError:
            meta = {}

    image_path = meta.get("image_path")

    if not image_path:
        # Old run without a sidecar: find the image by name.
        candidates = sorted(
            (document_folder / "images").glob(f"{file_path.stem}.*")
        )
        if candidates:
            image_path = candidates[0].relative_to(BASE_DIR).as_posix()

    if not image_path or not (BASE_DIR / image_path).exists():
        return None, None

    extra = {
        "image_path": image_path,
        "image_url": meta.get("image_url") or (
            f"/api/images/{quote(document_folder.name, safe='')}"
            f"/{quote(Path(image_path).name, safe='')}"
        ),
    }

    return content, extra


# --------------------------------------------------
# STRUCTURE ONE DOCUMENT
# --------------------------------------------------

def process_document(document_folder, output_file):

    document_name = document_folder.name
    total_records = 0

    for content_type in CONTENT_TYPES:

        source_folder = document_folder / content_type

        if not source_folder.exists():
            continue

        files = sorted(
            path for path in source_folder.rglob("*")
            if path.is_file() and path.suffix.lower() == ".txt"
        )

        for file_path in files:

            extra = {}

            if content_type == "image_descriptions":
                content, extra = read_image_description(
                    file_path, document_folder
                )
                if content is None:
                    print(f"  Skipping orphan description: {file_path.name}")
                    continue
            else:
                content = file_path.read_text(encoding="utf-8").strip()

            if not content:
                continue

            record = {
                "record_id": create_record_id(
                    document_name,
                    content_type,
                    file_path.relative_to(document_folder).as_posix()
                ),
                "document_id": document_name,
                "page_number": get_page_number(file_path.name),
                "content_type": content_type,
                "source_file": file_path.relative_to(BASE_DIR).as_posix(),
                "content": content,
                **extra,
            }

            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_records += 1

    return total_records


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def main():

    STRUCTURED_FOLDER.mkdir(parents=True, exist_ok=True)

    structured_path = STRUCTURED_FOLDER / "documents.jsonl"
    total_records = 0

    # Overwrites previous structured output on each run.
    with structured_path.open("w", encoding="utf-8") as output_file:

        for document_folder in sorted(
            f for f in OUTPUT_FOLDER.iterdir() if f.is_dir()
        ):
            count = process_document(document_folder, output_file)
            total_records += count
            print(f"{document_folder.name}: {count} records")

    print("\nDocument structuring completed!")
    print(f"Total records: {total_records}")
    print(f"Saved to: {structured_path}")


if __name__ == "__main__":
    main()
