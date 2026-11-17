"""Audit the relationship between questionnaire scores and released labels."""

import argparse
import csv
import json
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "data" / "cleaned_autism_data.csv"
ITEMS = tuple(f"A{i}_Score" for i in range(1, 11))
THRESHOLD = 7


def audit_dataset(path=DATA_PATH):
    """Return descriptive counts for a fixed rule over the supplied CSV."""
    counts = {score: [0, 0] for score in range(11)}
    rows = 0
    matches = 0
    with Path(path).open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        required = {*ITEMS, "result", "class_asd"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("CSV must contain all ten item scores, result, and class_asd")
        for row in reader:
            try:
                items = [int(row[column]) for column in ITEMS]
                score = int(row["result"])
                label = int(row["class_asd"])
                if any(item not in (0, 1) for item in items) or label not in (0, 1):
                    raise ValueError("Item scores and labels must be binary")
                if score != sum(items):
                    raise ValueError("result must equal the sum of the item scores")
            except (TypeError, ValueError) as error:
                raise ValueError(f"CSV line {reader.line_num}: {error}") from error
            counts[score][label] += 1
            rows += 1
            matches += (score >= THRESHOLD) == label
    if not rows:
        raise ValueError("CSV has no data rows")
    negatives = sum(pair[0] for pair in counts.values())
    positives = rows - negatives
    return {
        "rows": rows,
        "threshold": THRESHOLD,
        "positive_labels": positives,
        "negative_labels": negatives,
        "rule_matches": matches,
        "rule_agreement": matches / rows,
        "majority_agreement": max(positives, negatives) / rows,
        "by_score": {
            score: {"negative": pair[0], "positive": pair[1]}
            for score, pair in counts.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    args = parser.parse_args()
    try:
        report = audit_dataset(args.data)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Audit failed: {error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
