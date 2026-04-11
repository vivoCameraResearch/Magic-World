import argparse
import csv
import json
from pathlib import Path


PICKUP_VERBS = {"take", "pick", "pickup", "pick-up", "pick up"}


def normalize_noun(noun):
    return noun.replace("_", " ")


def build_pickup_manifest(rows, allowed_nouns=None):
    normalized_allowed_nouns = None
    if allowed_nouns is not None:
        normalized_allowed_nouns = {
            normalize_noun(str(noun)) for noun in allowed_nouns
        }

    manifest = []
    for row in rows:
        if str(row["verb"]).strip().lower() not in PICKUP_VERBS:
            continue

        object_name = normalize_noun(str(row["noun"]))
        if (
            normalized_allowed_nouns is not None
            and object_name not in normalized_allowed_nouns
        ):
            continue

        manifest_row = {
            "source_dataset": "epic_kitchens_100",
            "video_id": row["video_id"],
            "participant_id": row["participant_id"],
            "start_frame": row["start_frame"],
            "end_frame": row["end_frame"],
            "event_type": "pick_up",
            "object_name": object_name,
            "event_text": f"pick up the {object_name}",
        }
        if "source_video_path" in row:
            manifest_row["source_video_path"] = row["source_video_path"]

        manifest.append(manifest_row)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def load_epic_rows(input_csv):
    with open(input_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        has_source_video_path = "source_video_path" in (reader.fieldnames or [])
        return [
            {
                "video_id": row["video_id"],
                "participant_id": row["participant_id"],
                "start_frame": int(row["start_frame"]),
                "end_frame": int(row["stop_frame"]),
                "verb": row["verb"],
                "noun": row["noun"],
                **(
                    {"source_video_path": row.get("source_video_path")}
                    if has_source_video_path
                    else {}
                ),
            }
            for row in reader
        ]


def main():
    args = parse_args()
    manifest = build_pickup_manifest(load_epic_rows(args.input_csv))
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


if __name__ == "__main__":
    main()
