# In-package runtime data

Files here ship inside the `hermes_katana` wheel (unlike `training/`, which is
source-only and absent from installs).

`runtime_artifact_manifest.json` — when a hermetic deployment pins a verified
artifact set, the training pipeline writes the manifest here so
`hermes_katana.runtime_artifacts` can find it after `pip install`. Its absence
is a normal state for a source checkout and is reported as a warning, not a
failure (audit finding F1).
