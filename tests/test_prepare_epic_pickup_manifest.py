import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.data.prepare_epic_pickup_manifest import (
    build_pickup_manifest,
    load_epic_rows,
)


class BuildPickupManifestTest(unittest.TestCase):
    def test_keeps_only_pickup_actions_and_freezes_v0_schema(self):
        rows = [
            {
                "video_id": "P01_01",
                "participant_id": "P01",
                "start_frame": 10,
                "end_frame": 20,
                "verb": "pick up",
                "noun": "cup",
            },
            {
                "video_id": "P01_02",
                "participant_id": "P01",
                "start_frame": 30,
                "end_frame": 40,
                "verb": "open",
                "noun": "fridge",
            },
        ]

        manifest = build_pickup_manifest(rows)

        self.assertEqual(
            manifest,
            [
                {
                    "source_dataset": "epic_kitchens_100",
                    "video_id": "P01_01",
                    "participant_id": "P01",
                    "start_frame": 10,
                    "end_frame": 20,
                    "event_type": "pick_up",
                    "object_name": "cup",
                    "event_text": "pick up the cup",
                }
            ],
        )
        self.assertEqual(
            sorted(manifest[0].keys()),
            [
                "end_frame",
                "event_text",
                "event_type",
                "object_name",
                "participant_id",
                "source_dataset",
                "start_frame",
                "video_id",
            ],
        )

    def test_normalizes_noun_by_replacing_underscores_with_spaces(self):
        rows = [
            {
                "video_id": "P01_03",
                "participant_id": "P01",
                "start_frame": 50,
                "end_frame": 60,
                "verb": "pickup",
                "noun": "tea__cup",
            },
        ]

        manifest = build_pickup_manifest(rows)

        self.assertEqual(len(manifest), 1)
        self.assertEqual(manifest[0]["video_id"], "P01_03")
        self.assertEqual(manifest[0]["object_name"], "tea  cup")
        self.assertEqual(manifest[0]["event_text"], "pick up the tea  cup")

    def test_keeps_take_rows_because_they_are_in_pickup_verbs(self):
        rows = [
            {
                "video_id": "P01_05",
                "participant_id": "P01",
                "start_frame": 90,
                "end_frame": 100,
                "verb": "take",
                "noun": "mug",
            }
        ]

        manifest = build_pickup_manifest(rows)

        self.assertEqual(
            manifest,
            [
                {
                    "source_dataset": "epic_kitchens_100",
                    "video_id": "P01_05",
                    "participant_id": "P01",
                    "start_frame": 90,
                    "end_frame": 100,
                    "event_type": "pick_up",
                    "object_name": "mug",
                    "event_text": "pick up the mug",
                }
            ],
        )

    def test_normalizes_verb_whitespace_and_case_before_filtering(self):
        rows = [
            {
                "video_id": "P01_11",
                "participant_id": "P01",
                "start_frame": 210,
                "end_frame": 220,
                "verb": "  Pick Up  ",
                "noun": "bottle",
            }
        ]

        manifest = build_pickup_manifest(rows)

        self.assertEqual(len(manifest), 1)
        self.assertEqual(manifest[0]["event_text"], "pick up the bottle")

    def test_filters_to_allowed_nouns_when_allowlist_is_provided(self):
        rows = [
            {
                "video_id": "P01_06",
                "participant_id": "P01",
                "start_frame": 110,
                "end_frame": 120,
                "verb": "pick up",
                "noun": "cup",
            },
            {
                "video_id": "P01_07",
                "participant_id": "P01",
                "start_frame": 130,
                "end_frame": 140,
                "verb": "pick up",
                "noun": "plate",
            },
        ]

        manifest = build_pickup_manifest(rows, allowed_nouns={"cup"})

        self.assertEqual(len(manifest), 1)
        self.assertEqual(manifest[0]["object_name"], "cup")

    def test_normalizes_allowed_nouns_before_filtering(self):
        rows = [
            {
                "video_id": "P01_09",
                "participant_id": "P01",
                "start_frame": 170,
                "end_frame": 180,
                "verb": "pick up",
                "noun": "tea__cup",
            }
        ]

        manifest = build_pickup_manifest(rows, allowed_nouns={"tea__cup"})

        self.assertEqual(len(manifest), 1)
        self.assertEqual(manifest[0]["object_name"], "tea  cup")

    def test_preserves_source_video_path_when_present(self):
        rows = [
            {
                "video_id": "P01_12",
                "participant_id": "P01",
                "start_frame": 230,
                "end_frame": 240,
                "verb": "pick up",
                "noun": "cup",
                "source_video_path": "/dataset/videos/P01/P01_12.MP4",
            }
        ]

        manifest = build_pickup_manifest(rows)

        self.assertEqual(len(manifest), 1)
        self.assertEqual(
            manifest[0]["source_video_path"], "/dataset/videos/P01/P01_12.MP4"
        )
        self.assertEqual(
            sorted(manifest[0].keys()),
            [
                "end_frame",
                "event_text",
                "event_type",
                "object_name",
                "participant_id",
                "source_dataset",
                "source_video_path",
                "start_frame",
                "video_id",
            ],
        )


class LoadEpicRowsTest(unittest.TestCase):
    def test_reads_optional_source_video_path_column_when_present(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_csv = Path(tmp_dir) / "input.csv"
            input_csv.write_text(
                "video_id,start_frame,stop_frame,verb,noun,participant_id,source_video_path\n"
                "P01_13,250,260,pick up,cup,P01,/dataset/videos/P01/P01_13.MP4\n",
                encoding="utf-8",
            )

            rows = load_epic_rows(input_csv)

        self.assertEqual(
            rows,
            [
                {
                    "video_id": "P01_13",
                    "participant_id": "P01",
                    "start_frame": 250,
                    "end_frame": 260,
                    "verb": "pick up",
                    "noun": "cup",
                    "source_video_path": "/dataset/videos/P01/P01_13.MP4",
                }
            ],
        )

    def test_omits_source_video_path_when_column_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_csv = Path(tmp_dir) / "input.csv"
            input_csv.write_text(
                "video_id,start_frame,stop_frame,verb,noun,participant_id\n"
                "P01_14,270,280,pick up,cup,P01\n",
                encoding="utf-8",
            )

            rows = load_epic_rows(input_csv)

        self.assertEqual(
            rows,
            [
                {
                    "video_id": "P01_14",
                    "participant_id": "P01",
                    "start_frame": 270,
                    "end_frame": 280,
                    "verb": "pick up",
                    "noun": "cup",
                }
            ],
        )


class PreparePickupManifestCliTest(unittest.TestCase):
    def test_cli_writes_pickup_manifest_json(self):
        script_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "data"
            / "prepare_epic_pickup_manifest.py"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_csv = tmp_path / "input.csv"
            output_json = tmp_path / "output.json"
            input_csv.write_text(
                "video_id,start_frame,stop_frame,verb,noun,participant_id\n"
                "P01_08,150,160,pick up,cup,P01\n",
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--input-csv",
                    str(input_csv),
                    "--output-json",
                    str(output_json),
                ],
                check=True,
            )

            manifest = json.loads(output_json.read_text(encoding="utf-8"))

        self.assertEqual(manifest[0]["event_text"], "pick up the cup")
        self.assertNotIn("source_video_path", manifest[0])

    def test_cli_preserves_legacy_manifest_schema_without_source_video_path(self):
        script_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "data"
            / "prepare_epic_pickup_manifest.py"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_csv = tmp_path / "input.csv"
            output_json = tmp_path / "output.json"
            input_csv.write_text(
                "video_id,start_frame,stop_frame,verb,noun,participant_id\n"
                "P01_15,290,300,pick up,bottle,P01\n",
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--input-csv",
                    str(input_csv),
                    "--output-json",
                    str(output_json),
                ],
                check=True,
            )

            manifest = json.loads(output_json.read_text(encoding="utf-8"))

        self.assertEqual(
            manifest,
            [
                {
                    "source_dataset": "epic_kitchens_100",
                    "video_id": "P01_15",
                    "participant_id": "P01",
                    "start_frame": 290,
                    "end_frame": 300,
                    "event_type": "pick_up",
                    "object_name": "bottle",
                    "event_text": "pick up the bottle",
                }
            ],
        )

    def test_cli_creates_parent_directories_for_output_json(self):
        script_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "data"
            / "prepare_epic_pickup_manifest.py"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_csv = tmp_path / "input.csv"
            output_json = tmp_path / "nested" / "output.json"
            input_csv.write_text(
                "video_id,start_frame,stop_frame,verb,noun,participant_id\n"
                "P01_10,190,200,pick up,cup,P01\n",
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--input-csv",
                    str(input_csv),
                    "--output-json",
                    str(output_json),
                ],
                check=True,
            )

            manifest = json.loads(output_json.read_text(encoding="utf-8"))

        self.assertEqual(manifest[0]["event_text"], "pick up the cup")
