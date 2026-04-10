import unittest
from pathlib import Path


RUNBOOK_PATH = Path(__file__).resolve().parents[1] / "docs" / "pickup_event_smoke_runbook.md"


class PickupSmokeRunbookTest(unittest.TestCase):
    def test_runbook_contains_required_sections(self):
        content = RUNBOOK_PATH.read_text(encoding="utf-8")

        for section in (
            "## Export Command",
            "## Training Command",
            "## Expected Artifacts",
            "## Acceptance Checklist",
        ):
            with self.subTest(section=section):
                self.assertIn(section, content)

    def test_runbook_documents_required_smoke_flags_and_manifest_field(self):
        content = RUNBOOK_PATH.read_text(encoding="utf-8")

        for required_text in (
            "--dry-run",
            "--config_path",
            "--smoke_run",
            "--max_train_samples",
            "source_video_path",
        ):
            with self.subTest(required_text=required_text):
                self.assertIn(required_text, content)

    def test_runbook_aligns_event_prefix_training_with_empty_base_text_export(self):
        content = RUNBOOK_PATH.read_text(encoding="utf-8")

        for required_text in (
            "--enable_event_text --text_composition_mode event_prefix",
            "`text` to an empty string",
            "keeps `event_text` populated",
        ):
            with self.subTest(required_text=required_text):
                self.assertIn(required_text, content)

    def test_runbook_distinguishes_completed_verification_from_next_operator_step(self):
        content = RUNBOOK_PATH.read_text(encoding="utf-8")

        for required_text in (
            "verified in this branch",
            "focused tests + dry-run export",
            "next operator step",
            "still requires local model assets",
        ):
            with self.subTest(required_text=required_text):
                self.assertIn(required_text, content)


if __name__ == "__main__":
    unittest.main()
