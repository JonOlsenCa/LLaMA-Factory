# V4 Cleanup and Isolation Plan

Goal: isolate V4 SQLCoder workflow, archive non-V4 assets, and verify via dry-run that only V4 paths are needed.

## Keep (V4-critical)
- Core code/runtime: `src/`, `pyproject.toml`, `requirements.txt`, `Makefile` (trim later), `.env` if used.
- V4 configs: `automation/configs/vgpt2_v4/stage1_sft.yaml`, `automation/configs/vgpt2_v4/stage2_dpo.yaml` (if DPO kept).
- V4 scripts: `training/01_start_sft_v4.ps1`, `training/02_start_dpo_v4.ps1` (optional), `scripts/vgpt2_v4/**`, `scripts/expand_v4_training_data.py`.
- V4 data: `data/vgpt2_v4_sft_expanded_clean.json` (and any DDL/context files if required by generators).
- V4 docs: `docs/V4_TRAINING_STRATEGY.md`, `training/V4_TRAINING.md`, `docs/DEEP_DIVE_ANALYSIS.md` (rationale), `CONTINUATION.md`.
- V4 outputs: `saves/vgpt2_v4/` (checkpoints), `output/` (probes/evals). READMEs added to describe expected contents.

## Archive (staging, not deleted yet)
Move to `archive/` once dry-run confirms no dependency:
- Legacy training/data/configs: v2/v3 assets, old datasets (`data/vgpt2_v3*`, etc.), old automation configs, legacy docs not referenced above.
- Legacy scripts: `scripts/vgpt2_v3/**`, other non-V4 helpers.
- Legacy checkpoints: `saves/vgpt2_v2_*`, `saves/vgpt2_v3_*` (or move entire dirs).
- Old reports/outputs unrelated to V4.

## Dry-Run Checklist (no training)
1) Ensure dataset exists: `data/vgpt2_v4_sft_expanded_clean.json`. Regenerate only if missing with `scripts/expand_v4_training_data.py`.
2) Run: `pwsh -File training/01_start_sft_v4.ps1 -DryRun`.
   - Verify no path leaks outside V4 keep set.
   - Confirm it references only stage1_sft.yaml and the V4 dataset.
3) Add a V4-only probe script (e.g., `scripts/vgpt2_v4/probe.py`) to validate checkpoints without touching legacy paths. Output to `output/probe_sft.json`.

## Training/Validation Flow (target state)
- Train: `pwsh -File training/01_start_sft_v4.ps1` (optionally point output to `saves/vgpt2_v4/sft_rerun` to avoid clobbering).
- Probe: `python scripts/vgpt2_v4/probe.py --model saves/vgpt2_v4/sft --output output/probe_sft.json`.
- (Optional) DPO: `pwsh -File training/02_start_dpo_v4.ps1`.

## Deletion Step (after full validation)
- Only after successful train + probe + eval confirm zero references to archived items, delete contents of `archive/`.

## Notes
- Archive is staging; nothing has been moved yet. Moves/deletions must be explicit and traceable.
- Be aggressive in archiving, but restore from `archive/` if dry-run shows missing dependencies.
