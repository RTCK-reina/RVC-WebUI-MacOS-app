## Imported Claude Cowork project instructions

## Build / test / verify commands (verified 2026-09)

- Test suite (stub harness, no ML deps): `.venv/bin/python -m pytest tests`
- Real-torch equivalence suite: `tools/run_equivalence_tests.sh`
  (tests/conftest.py force-stubs torch; the script runs the file outside
  tests/ against build/python_env)
- Format: CI runs `black .` on push; local: `.venv/bin/python -m black .`
- App build: `DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer ./build_app.sh --skip-conda`
  - xcode-select points at CommandLineTools; full Xcode exists only as
    Xcode-beta.app — DEVELOPER_DIR is REQUIRED or xcodebuild fails
  - `--skip-conda` reuses build/python_env; `--skip-xcode` reuses the last
    xcodebuild output for signing/packaging iteration
  - ad-hoc signing is the default (CODE_SIGN_IDENTITY unset). Do NOT add
    `--options runtime` to bundled executables for ad-hoc builds — it
    enables library validation keyed on Team ID and the interpreter then
    rejects every bundled .so ("different Team IDs"). See 643f167.
- Bundled-env smoke: `build/RVC-WebUI.app/Contents/Resources/python/bin/python`

## Environment notes

- Bundled python env (build/python_env): python 3.10, torch 2.11.0, MPS ok.
- Dev venv (.venv): python 3.14, pytest + black only, NO torch — the suite
  relies on tests/conftest.py stubs.
- Training opt-ins (configs/*/train): `grad_clip_norm` (default 0.0=off),
  `warmup_epochs` (default 0=off, absolute-batch linear ramp, resume-safe).
- docs/conditional_review.md tracks the C-1..C-5 decisions.
