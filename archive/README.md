# Archive Staging (Do Not Delete Yet)

This folder is the holding area for non-V4 assets. After V4 dry-run and full validation confirm zero dependencies, the contents here may be permanently deleted.

## Intended Archive Targets (not yet moved)
- Legacy training code, data, and configs: v2/v3 assets, older automation configs, old datasets, unused docs.
- Old checkpoints: saves/vgpt2_v2_*, saves/vgpt2_v3_*.
- Legacy scripts: scripts/vgpt2_v3/** and other non-v4 helpers.
- Any docs unrelated to V4/SQLCoder flow.

## Process
1) Move candidates into this folder (or a dated subfolder) instead of deleting.
2) Run V4 dry-run and full train/validate. If any missing dependency is detected, restore from here.
3) After validation, delete archived items.

## Notes
- No files have been moved yet. This is a staging placeholder.
- All destructive moves/deletes should be logged in git for traceability.
