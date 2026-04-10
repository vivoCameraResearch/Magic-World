import importlib
import json
import tempfile
import unittest
from pathlib import Path


def load_export_module(test_case):
    try:
        return importlib.import_module("scripts.data.export_magicworld_pickup_dataset")
    except ModuleNotFoundError as exc:
        test_case.fail(f"export module is missing: {exc}")


class BuildMagicWorldMetadataEntryTest(unittest.TestCase):
    def test_builds_magicworld_metadata_entry_with_expected_schema(self):
        export_module = load_export_module(self)

        entry = export_module.build_magicworld_metadata_entry(
            clip_path=Path("clips/P01_01_000100_000160.mp4"),
            control_path=Path("controls/P01_01_000100_000160.mp4"),
            manifest_row={
                "video_id": "P01_01",
                "start_frame": 100,
                "end_frame": 160,
                "event_text": "pick up the cup",
                "event_type": "pick_up",
                "object_name": "cup",
            },
        )

        self.assertEqual(
            entry,
            {
                "id": "P01_01_000100_000160",
                "file_path": "clips/P01_01_000100_000160.mp4",
                "control_file_path": "controls/P01_01_000100_000160.mp4",
                "text": "",
                "event_text": "pick up the cup",
                "event_type": "pick_up",
                "object_name": "cup",
                "type": "video",
            },
        )

    def test_uses_relative_paths_in_metadata_even_for_absolute_inputs(self):
        export_module = load_export_module(self)

        entry = export_module.build_magicworld_metadata_entry(
            clip_path=Path("/tmp/export/clips/P01_01_000100_000160.mp4"),
            control_path=Path("/tmp/export/controls/P01_01_000100_000160.mp4"),
            manifest_row={
                "video_id": "P01_01",
                "start_frame": 100,
                "end_frame": 160,
                "event_text": "pick up the cup",
                "event_type": "pick_up",
                "object_name": "cup",
            },
        )

        self.assertEqual(entry["file_path"], "clips/P01_01_000100_000160.mp4")
        self.assertEqual(
            entry["control_file_path"], "controls/P01_01_000100_000160.mp4"
        )


class ManifestHelpersTest(unittest.TestCase):
    def test_build_sample_id_uses_video_and_frame_range(self):
        export_module = load_export_module(self)

        self.assertEqual(
            export_module.build_sample_id(
                {"video_id": "P01_01", "start_frame": 100, "end_frame": 160}
            ),
            "P01_01_000100_000160",
        )

    def test_load_manifest_rows_reads_json_rows(self):
        export_module = load_export_module(self)

        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_json = Path(tmp_dir) / "manifest.json"
            manifest_json.write_text(
                json.dumps([{"video_id": "P01_01", "start_frame": 100, "end_frame": 160}]),
                encoding="utf-8",
            )

            self.assertEqual(
                export_module.load_manifest_rows(manifest_json),
                [{"video_id": "P01_01", "start_frame": 100, "end_frame": 160}],
            )


class WriteMetadataJsonTest(unittest.TestCase):
    def test_writes_json_metadata_payload(self):
        export_module = load_export_module(self)
        rows = [
            {
                "id": "P01_01_000100_000160",
                "file_path": "clips/P01_01_000100_000160.mp4",
                "control_file_path": "controls/P01_01_000100_000160.mp4",
                "text": "",
                "event_text": "pick up the cup",
                "event_type": "pick_up",
                "object_name": "cup",
                "type": "video",
            }
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "nested" / "metadata.json"

            export_module.write_metadata_json(rows, output_path)

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(payload[0]["text"], "")
        self.assertEqual(payload[0]["event_text"], "pick up the cup")
        self.assertEqual(payload[0]["id"], "P01_01_000100_000160")
        self.assertEqual(
            payload[0]["control_file_path"], "controls/P01_01_000100_000160.mp4"
        )


class ExportPickupSamplesTest(unittest.TestCase):
    def test_fails_clearly_when_manifest_row_is_missing_source_video_path(self):
        export_module = load_export_module(self)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "export"

            with self.assertRaisesRegex(
                ValueError,
                "source_video_path.*P01_01_000100_000160",
            ):
                export_module.export_pickup_samples(
                    [
                        {
                            "video_id": "P01_01",
                            "start_frame": 100,
                            "end_frame": 160,
                            "event_text": "pick up the cup",
                            "event_type": "pick_up",
                            "object_name": "cup",
                        }
                    ],
                    output_dir,
                )


class MainTest(unittest.TestCase):
    def test_main_fails_when_export_has_no_rows(self):
        export_module = load_export_module(self)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "export"
            manifest_json = Path(tmp_dir) / "manifest.json"
            manifest_json.write_text("[]", encoding="utf-8")

            with self.assertRaises(SystemExit) as exc:
                export_module.main(
                    [
                        "--manifest-json",
                        str(manifest_json),
                        "--output-dir",
                        str(output_dir),
                        "--dry-run",
                    ]
                )

            self.assertFalse(output_dir.exists())

        self.assertIn("No pickup samples were exported", str(exc.exception))

    def test_main_requires_dry_run_for_smoke_export(self):
        export_module = load_export_module(self)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "export"
            manifest_json = Path(tmp_dir) / "manifest.json"
            source_video_path = Path(tmp_dir) / "source.mp4"
            source_video_path.write_bytes(b"fake-mp4-bytes")
            manifest_json.write_text(
                json.dumps(
                    [
                        {
                            "video_id": "P01_01",
                            "start_frame": 100,
                            "end_frame": 160,
                            "source_video_path": str(source_video_path),
                            "event_text": "pick up the cup",
                            "event_type": "pick_up",
                            "object_name": "cup",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                SystemExit,
                "--dry-run performs a smoke export that copies source bytes",
            ):
                export_module.main(
                    [
                        "--manifest-json",
                        str(manifest_json),
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            self.assertFalse(output_dir.exists())

    def test_dry_run_exports_smoke_sample(self):
        export_module = load_export_module(self)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "export"
            manifest_json = Path(tmp_dir) / "manifest.json"
            source_video_path = Path(tmp_dir) / "source.mp4"
            source_video_path.write_bytes(b"fake-mp4-bytes")
            manifest_json.write_text(
                json.dumps(
                    [
                        {
                            "video_id": "P01_01",
                            "start_frame": 100,
                            "end_frame": 160,
                            "source_video_path": str(source_video_path),
                            "event_text": "pick up the cup",
                            "event_type": "pick_up",
                            "object_name": "cup",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            exit_code = export_module.main(
                [
                    "--manifest-json",
                    str(manifest_json),
                    "--output-dir",
                    str(output_dir),
                    "--dry-run",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "clips").is_dir())
            self.assertTrue((output_dir / "controls").is_dir())
            self.assertTrue((output_dir / "metadata.json").is_file())
            payload = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(
                payload,
                [
                    {
                        "id": "P01_01_000100_000160",
                        "file_path": "clips/P01_01_000100_000160.mp4",
                        "control_file_path": "controls/P01_01_000100_000160.mp4",
                        "text": "",
                        "event_text": "pick up the cup",
                        "event_type": "pick_up",
                        "object_name": "cup",
                        "type": "video",
                    }
                ],
            )
            self.assertEqual(
                (output_dir / "clips" / "P01_01_000100_000160.mp4").read_bytes(),
                b"fake-mp4-bytes",
            )
            self.assertEqual(
                (output_dir / "controls" / "P01_01_000100_000160.mp4").read_bytes(),
                b"fake-mp4-bytes",
            )

    def test_parser_describes_dry_run_as_smoke_export(self):
        export_module = load_export_module(self)

        parser = export_module.build_parser()

        dry_run_action = next(
            action for action in parser._actions if "--dry-run" in action.option_strings
        )

        self.assertIn("smoke export", dry_run_action.help)
        self.assertIn("copies source bytes", dry_run_action.help)
        self.assertIn("controls/", dry_run_action.help)
