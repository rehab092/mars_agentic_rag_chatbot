
# ============================================================
# RUN THE FULL RAG INGESTION PIPELINE
# ============================================================
#
# Usage:
#   python run_pipeline.py                  run every step
#   python run_pipeline.py --from chunk     start at a step
#   python run_pipeline.py --only store     run one step
#   python run_pipeline.py --list           show step names
#
# Stops at the first step that fails.
# ============================================================

import argparse
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

STEPS = [
    ("extract",   "extract_pdf.py"),
    ("describe",  "describe_images.py"),
    ("clean",     "clean_text.py"),
    ("structure", "structure_documents.py"),
    ("chunk",     "smart_chunking.py"),
    ("embed",     "generate_embeddings.py"),
    ("store",     "store_chromadb.py"),
]


def main():

    names = [name for name, _ in STEPS]

    parser = argparse.ArgumentParser(description="Run the RAG ingestion pipeline.")
    parser.add_argument("--from", dest="start", choices=names, help="first step to run")
    parser.add_argument("--only", choices=names, help="run just this step")
    parser.add_argument("--list", action="store_true", help="list steps and exit")
    args = parser.parse_args()

    if args.list:
        for name, script in STEPS:
            print(f"{name:10} {script}")
        return

    if args.only:
        steps = [s for s in STEPS if s[0] == args.only]
    elif args.start:
        steps = STEPS[names.index(args.start):]
    else:
        steps = STEPS

    total_start = time.time()

    for number, (name, script) in enumerate(steps, start=1):

        print("\n" + "=" * 60)
        print(f"STEP {number}/{len(steps)}: {name} ({script})")
        print("=" * 60, flush=True)

        step_start = time.time()

        # Same Python interpreter (your venv) that runs this file.
        result = subprocess.run(
            [sys.executable, str(BASE_DIR / script)],
            cwd=BASE_DIR,
        )

        minutes = (time.time() - step_start) / 60

        if result.returncode != 0:
            print(f"\nFAILED at step '{name}' (exit code {result.returncode}) "
                  f"after {minutes:.1f} min.")
            print(f"Fix the error, then resume with:  python run_pipeline.py --from {name}")
            sys.exit(result.returncode)

        print(f"\nStep '{name}' finished in {minutes:.1f} min.")

    print("\n" + "=" * 60)
    print(f"PIPELINE COMPLETED in {(time.time() - total_start) / 60:.1f} min")
    print("=" * 60)


if __name__ == "__main__":
    main()
