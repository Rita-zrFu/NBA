"""Verify retained course files and extracted figures without running analysis."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Optional original courses/585 directory")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "docs/migration_manifest.json").read_text())
    errors = []

    def check(path, expected, label):
        if not path.is_file():
            errors.append(f"Missing: {label}")
        elif sha256(path) != expected:
            errors.append(f"SHA-256 mismatch: {label}")

    for entry in manifest["copies"]:
        check(root / entry["destination_relative"], entry["sha256"], entry["destination_relative"])
        if args.source is not None:
            check(args.source / entry["source_relative"], entry["sha256"], f"source/{entry['source_relative']}")
    figures = json.loads((root / "results/legacy/figure_manifest.json").read_text())
    for entry in figures:
        check(root / entry["path"], entry["sha256"], entry["path"])
    if errors:
        for error in errors:
            print(error)
        return 1
    print(f"PASS: {len(manifest['copies'])} retained course files and {len(figures)} historical figures match their hashes.")
    if args.source is not None:
        print(f"PASS: {len(manifest['copies'])} original source files also match the recorded hashes.")
    print("No analysis or model training was executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
