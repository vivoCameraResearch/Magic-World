import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.modules.setdefault("albumentations", types.ModuleType("albumentations"))

if "func_timeout" not in sys.modules:
    func_timeout = types.ModuleType("func_timeout")

    class _FunctionTimedOut(Exception):
        pass

    def _func_timeout(_timeout, func, args=None, kwargs=None):
        return func(*(args or ()), **(kwargs or {}))

    func_timeout.FunctionTimedOut = _FunctionTimedOut
    func_timeout.func_timeout = _func_timeout
    sys.modules["func_timeout"] = func_timeout

if "decord" not in sys.modules:
    decord = types.ModuleType("decord")

    class _UnusedVideoReader:
        def __init__(self, *args, **kwargs):
            raise AssertionError("VideoReader should not be used in this test")

    decord.VideoReader = _UnusedVideoReader
    sys.modules["decord"] = decord

import videox_fun.data.dataset_image_video as dataset_image_video_module
from scripts.data import export_magicworld_pickup_dataset as export_module
from videox_fun.data.dataset_image_video import (
    ImageVideoCameraControlDataset,
    ImageVideoDataset,
    compose_training_text,
)


class ComposeTrainingTextTest(unittest.TestCase):
    def setUp(self):
        self.sample = {
            "text": "a person in a kitchen",
            "event_text": "pick up the cup",
        }

    def test_original_mode_returns_text(self):
        self.assertEqual(compose_training_text(self.sample), "a person in a kitchen")

    def test_event_prefix_mode_joins_event_and_text(self):
        self.assertEqual(
            compose_training_text(self.sample, mode="event_prefix"),
            "pick up the cup. a person in a kitchen",
        )

    def test_event_only_mode_returns_event_text(self):
        self.assertEqual(
            compose_training_text(self.sample, mode="event_only"),
            "pick up the cup",
        )

    def test_event_only_mode_returns_event_text_without_base_text(self):
        self.assertEqual(
            compose_training_text({"event_text": "pick up the cup"}, mode="event_only"),
            "pick up the cup",
        )

    def test_event_prefix_mode_omits_separator_when_base_text_is_empty(self):
        self.assertEqual(
            compose_training_text({"text": "", "event_text": "pick up the cup"}, mode="event_prefix"),
            "pick up the cup",
        )

    def test_event_prefix_mode_preserves_base_text_punctuation(self):
        self.assertEqual(
            compose_training_text(
                {"text": "a person in a kitchen.", "event_text": "pick up the cup"},
                mode="event_prefix",
            ),
            "pick up the cup. a person in a kitchen.",
        )


class ImageVideoDatasetTextCompositionTest(unittest.TestCase):
    def test_dataset_keeps_original_text_by_default(self):
        sample = self._load_sample()

        self.assertEqual(sample["text"], "a person in a kitchen")

    def test_dataset_uses_configured_event_only_mode_before_text_drop(self):
        sample = self._load_sample(text_composition_mode="event_only")

        self.assertEqual(sample["text"], "pick up the cup")

    def _load_sample(self, text_composition_mode="original"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "frame.png"
            metadata_path = tmp_path / "dataset.json"

            Image.new("RGB", (8, 8), color="white").save(image_path)
            metadata_path.write_text(
                json.dumps(
                    [
                        {
                            "file_path": str(image_path),
                            "text": "a person in a kitchen",
                            "event_text": "pick up the cup",
                            "type": "image",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            dataset = ImageVideoDataset(
                str(metadata_path),
                text_drop_ratio=0.0,
                text_composition_mode=text_composition_mode,
                image_sample_size=8,
            )

            return dataset[0]


class ImageVideoCameraControlDatasetTextCompositionTest(unittest.TestCase):
    def test_camera_control_dataset_uses_configured_event_prefix_mode_for_images(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "frame.png"
            control_image_path = tmp_path / "control.png"
            metadata_path = tmp_path / "dataset.json"

            Image.new("RGB", (8, 8), color="white").save(image_path)
            Image.new("RGB", (8, 8), color="black").save(control_image_path)
            metadata_path.write_text(
                json.dumps(
                    [
                        {
                            "file_path": str(image_path),
                            "control_file_path": str(control_image_path),
                            "text": "a person in a kitchen",
                            "event_text": "pick up the cup",
                            "type": "image",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            dataset = ImageVideoCameraControlDataset(
                str(metadata_path),
                text_drop_ratio=0.0,
                text_composition_mode="event_prefix",
                image_sample_size=8,
            )

            self.assertEqual(dataset[0]["text"], "pick up the cup. a person in a kitchen")

    def test_camera_control_dataset_image_samples_keep_training_schema_keys(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "frame.png"
            control_image_path = tmp_path / "control.png"
            metadata_path = tmp_path / "dataset.json"

            Image.new("RGB", (8, 8), color="white").save(image_path)
            Image.new("RGB", (8, 8), color="black").save(control_image_path)
            metadata_path.write_text(
                json.dumps(
                    [
                        {
                            "file_path": str(image_path),
                            "control_file_path": str(control_image_path),
                            "text": "a person in a kitchen",
                            "event_text": "pick up the cup",
                            "type": "image",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            dataset = ImageVideoCameraControlDataset(
                str(metadata_path),
                text_drop_ratio=0.0,
                image_sample_size=8,
            )

            sample = dataset[0]

            self.assertIn("render_mask_values", sample)
            self.assertIn("render_pixel_values", sample)
            self.assertIsNone(sample["render_mask_values"])
            self.assertIsNone(sample["render_pixel_values"])

    def test_camera_control_dataset_video_rows_allow_missing_render_metadata_fields(self):
        class _FakeVideoReader:
            def __len__(self):
                return 4

        class _FakeVideoReaderContext:
            def __init__(self, *_args, **_kwargs):
                self.reader = _FakeVideoReader()

            def __enter__(self):
                return self.reader

            def __exit__(self, exc_type, exc, tb):
                return False

        def _fake_get_video_reader_batch(_reader, batch_index):
            return np.zeros((len(batch_index), 8, 8, 3), dtype=np.uint8)

        original_contextmanager = dataset_image_video_module.VideoReader_contextmanager
        original_get_batch = dataset_image_video_module.get_video_reader_batch

        try:
            dataset_image_video_module.VideoReader_contextmanager = _FakeVideoReaderContext
            dataset_image_video_module.get_video_reader_batch = _fake_get_video_reader_batch

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                metadata_path = tmp_path / "dataset.json"
                metadata_path.write_text(
                    json.dumps(
                        [
                            {
                                "file_path": "clips/sample.mp4",
                                "control_file_path": "controls/sample.mp4",
                                "text": "a person in a kitchen",
                                "event_text": "pick up the cup",
                                "event_type": "pick_up",
                                "object_name": "cup",
                                "type": "video",
                            }
                        ]
                    ),
                    encoding="utf-8",
                )

                dataset = ImageVideoCameraControlDataset(
                    str(metadata_path),
                    text_drop_ratio=0.0,
                    video_sample_size=8,
                    video_sample_stride=1,
                    video_sample_n_frames=2,
                    enable_bucket=True,
                )

                batch = dataset.get_batch(0)

            self.assertIsNone(batch[1])
            self.assertIsNone(batch[2])
            self.assertEqual(batch[5], "a person in a kitchen")
            self.assertEqual(batch[6], "video")
        finally:
            dataset_image_video_module.VideoReader_contextmanager = original_contextmanager
            dataset_image_video_module.get_video_reader_batch = original_get_batch

    def test_camera_control_dataset_accepts_exported_smoke_metadata_control_video(self):
        class _TrackingVideoReader:
            def __init__(self, path):
                self.path = path

            def __len__(self):
                return 4

        class _TrackingVideoReaderContext:
            opened_paths = []

            def __init__(self, path, *_args, **_kwargs):
                self.reader = _TrackingVideoReader(path)

            def __enter__(self):
                self.opened_paths.append(self.reader.path)
                return self.reader

            def __exit__(self, exc_type, exc, tb):
                return False

        def _fake_get_video_reader_batch(_reader, batch_index):
            return np.zeros((len(batch_index), 8, 8, 3), dtype=np.uint8)

        original_contextmanager = dataset_image_video_module.VideoReader_contextmanager
        original_get_batch = dataset_image_video_module.get_video_reader_batch

        try:
            dataset_image_video_module.VideoReader_contextmanager = _TrackingVideoReaderContext
            dataset_image_video_module.get_video_reader_batch = _fake_get_video_reader_batch

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                output_dir = tmp_path / "export"
                source_video_path = tmp_path / "source.mp4"
                source_video_path.write_bytes(b"fake-mp4-bytes")
                (output_dir / "clips").mkdir(parents=True)
                (output_dir / "controls").mkdir(parents=True)

                metadata_rows = export_module.export_pickup_samples(
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
                    ],
                    output_dir,
                )
                metadata_path = output_dir / "metadata.json"
                export_module.write_metadata_json(metadata_rows, metadata_path)

                dataset = ImageVideoCameraControlDataset(
                    str(metadata_path),
                    data_root=str(output_dir),
                    text_drop_ratio=0.0,
                    video_sample_size=8,
                    video_sample_stride=1,
                    video_sample_n_frames=2,
                    enable_bucket=True,
                )

                batch = dataset.get_batch(0)

            self.assertEqual(batch[6], "video")
            self.assertEqual(
                _TrackingVideoReaderContext.opened_paths,
                [
                    str(output_dir / "clips" / "P01_01_000100_000160.mp4"),
                    str(output_dir / "controls" / "P01_01_000100_000160.mp4"),
                ],
            )
        finally:
            dataset_image_video_module.VideoReader_contextmanager = original_contextmanager
            dataset_image_video_module.get_video_reader_batch = original_get_batch


class ExportedPickupMetadataTextCompositionTest(unittest.TestCase):
    def test_exported_pickup_metadata_avoids_duplicate_prompt_in_event_prefix_mode(self):
        exported_row = export_module.build_magicworld_metadata_entry(
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

        self.assertEqual(exported_row["text"], "")
        self.assertEqual(
            compose_training_text(exported_row, mode="event_prefix"),
            "pick up the cup",
        )
