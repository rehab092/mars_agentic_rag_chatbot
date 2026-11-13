
# ============================================================
# DESCRIBE EXTRACTED IMAGES WITH QWEN-VL
# ============================================================
#
# For every image in outputs/<doc>/images/ this writes:
#   image_descriptions/<image>.txt   description text only
#   image_descriptions/<image>.json  image path, URL, page,
#                                    SHA-256 of the image
#
# The SHA-256 ties each description to the exact image, so a
# description is regenerated when its image changed, and
# deleted when its image no longer exists.
# ============================================================

import io
import os
import re
import json
import base64
import hashlib
import mimetypes
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv
from openai import OpenAI, AuthenticationError

try:
    from PIL import Image  # pip install pillow
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

OUTPUTS_DIR = BASE_DIR / "outputs"

QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "").strip().rstrip("/")
QWEN_API_KEY = (os.getenv("QWEN_API_KEY") or "").strip()
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3.5-4b")

SUPPORTED_IMAGES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

MAX_IMAGE_SIZE_MB = 10
MAX_IMAGE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024

# Longest side sent to the server (images use context tokens).
MAX_IMAGE_SIDE = 1024

# Qwen-VL rejects tiny or extremely thin images.
MIN_IMAGE_SIDE = 32
MAX_ASPECT_RATIO = 50

# Characters of the page's text sent as context with the image.
PAGE_CONTEXT_CHARS = 1500

MAX_CONSECUTIVE_FAILURES = 5

# Existing up-to-date descriptions are not regenerated.
OVERWRITE_EXISTING = False


class SkipImage(Exception):
    """Image is not worth describing (e.g. a 1px line)."""


# ------------------------------------------------------------
# QWEN CLIENT
# ------------------------------------------------------------

def create_qwen_client():

    if not QWEN_BASE_URL:
        raise ValueError("QWEN_BASE_URL is missing from .env")

    if not QWEN_API_KEY:
        raise ValueError("QWEN_API_KEY is missing from .env")

    # The OpenAI library appends /chat/completions, so the
    # base URL must end with /v1.
    base_url = QWEN_BASE_URL
    if not base_url.endswith("/v1"):
        base_url += "/v1"

    return OpenAI(
        base_url=base_url,
        api_key=QWEN_API_KEY,
        # Caddy checks this header; vLLM never sees it.
        default_headers={"X-Qwen-Key": QWEN_API_KEY},
        timeout=180.0,
        max_retries=2
    )


def check_endpoint(client):
    """Fail fast if the server, tunnel or key is wrong."""

    try:
        client.models.list()
    except AuthenticationError:
        raise SystemExit(
            "Endpoint rejected the API key (401). Check that "
            "QWEN_API_KEY in .env matches the key on the server, "
            "and that Caddy and vLLM are running."
        )
    except Exception as error:
        raise SystemExit(
            f"Cannot reach the Qwen endpoint: {error}\n"
            "Check QWEN_BASE_URL (the tunnel URL changes when "
            "cloudflared restarts) and that the server is running."
        )

    print("Endpoint check: OK")


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get_page_number(image_path):
    match = re.search(r"page_(\d+)", image_path.name)
    return int(match.group(1)) if match else None


def build_metadata(image_path, document_folder, sha256):

    document_name = quote(document_folder.name, safe="")
    image_name = quote(image_path.name, safe="")

    return {
        "image_path": image_path.relative_to(BASE_DIR).as_posix(),
        "image_url": f"/api/images/{document_name}/{image_name}",
        "document_id": document_folder.name,
        "page_number": get_page_number(image_path),
        "image_sha256": sha256,
        "model": QWEN_MODEL,
    }


def page_context(document_folder, page_number):
    """Text of the page the image sits on (captions etc.)."""

    if page_number is None:
        return ""

    page_file = document_folder / "text" / f"page_{page_number:03d}.txt"

    if not page_file.exists():
        return ""

    text = page_file.read_text(encoding="utf-8").strip()
    return text[:PAGE_CONTEXT_CHARS]


# ------------------------------------------------------------
# CONVERT IMAGE TO BASE64 DATA URL
# ------------------------------------------------------------

def image_to_data_url(image_path):

    image_bytes = image_path.read_bytes()

    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError(f"Image exceeds {MAX_IMAGE_SIZE_MB} MB")

    mime_type = mimetypes.guess_type(image_path.name)[0]

    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError(f"Unsupported image format: {image_path.suffix}")

    if PIL_AVAILABLE:

        with Image.open(io.BytesIO(image_bytes)) as img:

            img.load()

            w, h = img.size
            if min(w, h) < MIN_IMAGE_SIDE or max(w, h) / min(w, h) > MAX_ASPECT_RATIO:
                raise SkipImage(f"too small/thin ({w}x{h})")

            img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))

            buffer = io.BytesIO()

            if mime_type == "image/jpeg":
                img.convert("RGB").save(buffer, format="JPEG", quality=90)
            else:
                # PNG keeps text and diagrams sharp.
                mime_type = "image/png"
                if img.mode not in ("RGB", "RGBA", "L"):
                    img = img.convert("RGBA")
                img.save(buffer, format="PNG")

            image_bytes = buffer.getvalue()

    encoded = base64.b64encode(image_bytes).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


# ------------------------------------------------------------
# DESCRIBE ONE IMAGE WITH QWEN
# ------------------------------------------------------------

PROMPT = """
You are a visual document analysis assistant
specializing in NASA Mars exploration missions.

Carefully analyze the supplied image. The text of the
document page the image appears on is given below as
context; use it to identify what the image shows (captions,
instrument names), but describe only what is visible.

Describe:

1. The main visible objects, diagrams or equipment.
2. The structure and components, where visible.
3. Any readable labels, numbers, measurements
   or scientific information.
4. Relationships between visible components.
5. The purpose illustrated by the image,
   but only when supported by visual evidence or the caption.

Important rules:
- Be accurate and detailed.
- Do not invent missing labels or specifications.
- If a label is unreadable, say it is unreadable.
- Distinguish clearly between visible facts
  and uncertain interpretations.
- Use plain text with meaningful paragraphs.
- Do not include Markdown bold or heading markers.
- Do not generate image references or URLs.
- Do not assume every picture depicts a rover.

Return only the image description.

Page text (context):
\"\"\"
{context}
\"\"\"
"""


def generate_description(client, image_path, context):

    image_url = image_to_data_url(image_path)

    response = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text",
                 "text": PROMPT.format(context=context or "(none)")},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }],
        temperature=0.1,
        max_tokens=900,
        # Spend tokens on the description, not on reasoning.
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    )

    description = response.choices[0].message.content or ""

    if not isinstance(description, str):
        raise ValueError("Qwen returned an unsupported response format.")

    description = description.strip()

    if not description:
        raise ValueError("Qwen returned an empty image description.")

    return description


# ------------------------------------------------------------
# OLD FORMAT MIGRATION
# ------------------------------------------------------------

OLD_FORMAT = re.compile(
    r"^Image Description:\s*\n(.*?)\n\s*\nImage Reference:",
    re.S
)


def migrate_old_description(txt_file, image_path, document_folder, sha256):
    """
    Earlier runs wrote the reference block into the .txt file.
    Keep the description text and move the rest into the .json
    sidecar, so the image is not re-described.
    """

    text = txt_file.read_text(encoding="utf-8")
    match = OLD_FORMAT.match(text)

    if not match:
        return False

    txt_file.write_text(match.group(1).strip() + "\n", encoding="utf-8")

    meta = build_metadata(image_path, document_folder, sha256)
    meta["migrated"] = True
    txt_file.with_suffix(".json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return True


def is_up_to_date(txt_file, sha256):

    meta_file = txt_file.with_suffix(".json")

    if not (txt_file.exists() and meta_file.exists()):
        return False

    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except ValueError:
        return False

    return meta.get("image_sha256") == sha256


def remove_orphans(description_folder, image_stems):
    """Delete descriptions whose image no longer exists."""

    removed = 0

    if not description_folder.exists():
        return removed

    for path in description_folder.iterdir():
        if path.suffix in (".txt", ".json") and path.stem not in image_stems:
            path.unlink()
            removed += 1

    return removed


# ------------------------------------------------------------
# PROCESS ALL EXTRACTED IMAGES
# ------------------------------------------------------------

def process_images():

    if not OUTPUTS_DIR.exists():
        raise FileNotFoundError(f"Outputs folder not found: {OUTPUTS_DIR}")

    client = create_qwen_client()

    print("Qwen model:", QWEN_MODEL)

    if not PIL_AVAILABLE:
        print(
            "Warning: Pillow is not installed, so images are sent "
            "at full size. Run: pip install pillow"
        )

    check_endpoint(client)

    print("Starting image description generation...\n")

    completed = skipped = failed = migrated = orphans = 0
    consecutive_failures = 0

    for document_folder in sorted(OUTPUTS_DIR.iterdir()):

        if not document_folder.is_dir():
            continue

        images_folder = document_folder / "images"

        if not images_folder.exists():
            continue

        description_folder = document_folder / "image_descriptions"
        description_folder.mkdir(parents=True, exist_ok=True)

        images = sorted(
            path for path in images_folder.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGES
        )

        orphans += remove_orphans(
            description_folder, {p.stem for p in images}
        )

        print(f"\nDocument: {document_folder.name}")
        print(f"Images found: {len(images)}")

        for image_path in images:

            txt_file = description_folder / f"{image_path.stem}.txt"
            sha256 = file_sha256(image_path)

            if txt_file.exists() and not txt_file.with_suffix(".json").exists():
                if migrate_old_description(
                    txt_file, image_path, document_folder, sha256
                ):
                    migrated += 1

            if not OVERWRITE_EXISTING and is_up_to_date(txt_file, sha256):
                skipped += 1
                continue

            try:

                print(f"Describing: {image_path.name}")

                description = generate_description(
                    client, image_path,
                    page_context(document_folder, get_page_number(image_path))
                )

                txt_file.write_text(description + "\n", encoding="utf-8")
                txt_file.with_suffix(".json").write_text(
                    json.dumps(
                        build_metadata(image_path, document_folder, sha256),
                        indent=2, ensure_ascii=False
                    ),
                    encoding="utf-8"
                )

                print(f"Saved: {txt_file.name}")

                completed += 1
                consecutive_failures = 0

            except SkipImage as reason:

                print(f"Skipping {image_path.name}: {reason}")
                # Remove a stale description for a skipped image.
                txt_file.unlink(missing_ok=True)
                txt_file.with_suffix(".json").unlink(missing_ok=True)
                skipped += 1

            except AuthenticationError:

                print(
                    "\nAuthentication failed (401). Stopping. "
                    "Check QWEN_API_KEY and that Caddy and vLLM "
                    "are running with the same key."
                )
                raise SystemExit(1)

            except Exception as error:

                print(f"Failed: {image_path.name}")
                print(f"Reason: {error}")

                failed += 1
                consecutive_failures += 1

                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print(
                        f"\n{MAX_CONSECUTIVE_FAILURES} failures in a row. "
                        "Stopping. Fix the problem above, then run again: "
                        "finished images are skipped automatically."
                    )
                    break

        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            break

    print("\n================================")
    print("QWEN IMAGE PROCESSING COMPLETED")
    print("================================")
    print("Descriptions generated:", completed)
    print("Images skipped (up to date or unusable):", skipped)
    print("Old-format descriptions migrated:", migrated)
    print("Orphan description files removed:", orphans)
    print("Images failed:", failed)


if __name__ == "__main__":
    process_images()
