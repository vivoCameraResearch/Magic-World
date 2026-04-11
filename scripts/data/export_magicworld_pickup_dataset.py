import argparse
import json
import shutil
from pathlib import Path


def _relative_export_path(path, root_dir_name):
    path = Path(path)
    if not path.is_absolute():
        return str(path)

    parts = path.parts
    try:
        root_index = parts.index(root_dir_name)
    except ValueError:
        return path.name

    return str(Path(*parts[root_index:]))


def load_manifest_rows(manifest_json_path):
    with Path(manifest_json_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_sample_id(manifest_row):
    return (
        f"{manifest_row['video_id']}_{int(manifest_row['start_frame']):06d}"
        f"_{int(manifest_row['end_frame']):06d}"
    )


def build_magicworld_metadata_entry(clip_path, control_path, manifest_row):
    return {
        "id": build_sample_id(manifest_row),
        "file_path": _relative_export_path(clip_path, "clips"),
        "control_file_path": _relative_export_path(control_path, "controls"),
        "text": "",
        "event_text": manifest_row["event_text"],
        "event_type": manifest_row["event_type"],
        "object_name": manifest_row["object_name"],
        "type": "video",
    }


def write_metadata_json(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run the smoke export path: copies source bytes into clips/, "
            "copies source bytes into controls/, and skips real clipping."
        ),
    )
    return parser


def export_pickup_samples(manifest_rows, output_dir):
    metadata_rows = []
    clips_dir = output_dir / "clips"
    controls_dir = output_dir / "controls"

    for manifest_row in manifest_rows:
        source_video_path = manifest_row.get("source_video_path")
        if not source_video_path:
            raise ValueError(
                "Manifest row is missing source_video_path for sample "
                f"{build_sample_id(manifest_row)}"
            )

        sample_id = build_sample_id(manifest_row)
        clip_path = clips_dir / f"{sample_id}.mp4"
        control_path = controls_dir / f"{sample_id}.mp4"

        shutil.copyfile(source_video_path, clip_path)
        shutil.copyfile(source_video_path, control_path)
        metadata_rows.append(
            build_magicworld_metadata_entry(clip_path, control_path, manifest_row)
        )

    return metadata_rows


def main(argv=None):
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)

    if not args.dry_run:
        raise SystemExit(
            "--dry-run performs a smoke export that copies source bytes, "
            "copies source bytes into controls/, and skips real clipping"
        )

    manifest_rows = load_manifest_rows(args.manifest_json)

    if not manifest_rows:
        raise SystemExit("No pickup samples were exported")

    (output_dir / "clips").mkdir(parents=True, exist_ok=True)
    (output_dir / "controls").mkdir(parents=True, exist_ok=True)
    metadata_rows = export_pickup_samples(manifest_rows, output_dir)
    write_metadata_json(metadata_rows, output_dir / "metadata.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
