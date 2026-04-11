# Pickup Event Smoke Runbook

Use this path to verify the pickup export and fine-tuning closure with the current smoke-only exporter behavior.

Status boundary: the repository-level verification verified in this branch is limited to focused tests + dry-run export. The next operator step still requires local model assets to run the smoke training command below.

## Export Command

```bash
python scripts/data/export_magicworld_pickup_dataset.py \
  --manifest-json tmp/pickup_manifest.json \
  --output-dir tmp/pickup_smoke_export \
  --dry-run
```

`--dry-run` is required. Each manifest row must include `source_video_path` because the exporter hard-fails when that field is missing. The smoke path copies the source video bytes into `clips/`, copies the same source video bytes into `controls/` as control `.mp4` artifacts, and emits `metadata.json`.

The exported metadata sets `text` to an empty string and keeps `event_text` populated. That keeps the documented `--enable_event_text --text_composition_mode event_prefix` training path aligned with the dataset contract and avoids composing duplicated prompts such as `pick up the cup. pick up the cup`.

## Training Command

```bash
python scripts/train_magicworld_v1.5.py \
  --pretrained_model_name_or_path ckpt/Wan2.1-T2V-1.3B \
  --pretrained_transformer_path ckpt/Wan2.1-T2V-1.3B \
  --config_path config/wan2.1/wan_civitai.yaml \
  --train_data_dir tmp/pickup_smoke_export \
  --train_data_meta tmp/pickup_smoke_export/metadata.json \
  --output_dir tmp/pickup_smoke_train \
  --train_mode control \
  --train_batch_size 1 \
  --max_train_steps 1 \
  --checkpointing_steps 1 \
  --smoke_run \
  --max_train_samples 1 \
  --enable_event_text \
  --text_composition_mode event_prefix
```

`--config_path` is required because the training script loads it before building the run config. `--smoke_run` also requires `--max_train_samples`, and the sample count must stay at or below the script limit.

This is the next operator step, not a run already completed in this branch.

## Expected Artifacts

- `tmp/pickup_smoke_export/metadata.json`
- `tmp/pickup_smoke_export/clips/<sample_id>.mp4`
- `tmp/pickup_smoke_export/controls/<sample_id>.mp4`
- `tmp/pickup_smoke_train/` after the next operator runs the smoke train locally
- `tmp/pickup_smoke_train/checkpoint-1/` after the one-step smoke train completes locally

## Acceptance Checklist

- Export command exits successfully with `--dry-run`.
- Input manifest rows include `source_video_path` so export does not fail before writing artifacts.
- Export output contains `metadata.json`, one clip, and one control video artifact.
- `metadata.json` rows keep `text`, `event_text`, `event_type`, `object_name`, and `control_file_path`, with `text` empty and `event_text` carrying the pickup prompt.
- Repository-level verification in this branch covers focused tests and the export dry-run only.
- Training command uses `--config_path`, `--smoke_run`, and `--max_train_samples 1` as the next operator step once local model assets are available.
- Training command enables event text with `--enable_event_text --text_composition_mode event_prefix`.
