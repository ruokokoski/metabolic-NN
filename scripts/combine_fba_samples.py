import argparse
import csv
import sys
from pathlib import Path


def combine_csv(file_one: Path, file_two: Path, output: Path, skip_header_check: bool) -> None:
    if not file_one.exists():
        raise FileNotFoundError(f"File not found: {file_one}")
    if not file_two.exists():
        raise FileNotFoundError(f"File not found: {file_two}")

    output.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    with output.open("w", newline="") as out_f:
        writer = csv.writer(out_f)

        with file_one.open("r", newline="") as f1:
            r1 = csv.reader(f1)
            header1 = next(r1, None)
            if header1 is None:
                raise ValueError(f"{file_one} is empty")
            writer.writerow(header1)
            for row in r1:
                writer.writerow(row)
                rows_written += 1

        with file_two.open("r", newline="") as f2:
            r2 = csv.reader(f2)
            header2 = next(r2, None)
            if header2 is None:
                # Nothing to append
                print(f"Warning: {file_two} is empty", file=sys.stderr)
                return
            if not skip_header_check and header2 != header1:
                print(
                    "Warning: headers differ between files. "
                    "Using header from file_one and appending file_two rows.",
                    file=sys.stderr,
                )
            for row in r2:
                writer.writerow(row)
                rows_written += 1

    print(f"Wrote {rows_written} rows to {output}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Combine two CSVs by keeping the header from the first file and "
            "appending all rows from both files (skipping the second header)."
        )
    )
    parser.add_argument("file_one", help="First CSV (header kept)")
    parser.add_argument("file_two", help="Second CSV (header skipped)")
    parser.add_argument("output", help="Output CSV path")
    parser.add_argument(
        "--skip-header-check",
        action="store_true",
        help="Do not warn if headers differ",
    )

    args = parser.parse_args()

    try:
        combine_csv(
            Path(args.file_one),
            Path(args.file_two),
            Path(args.output),
            args.skip_header_check,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
