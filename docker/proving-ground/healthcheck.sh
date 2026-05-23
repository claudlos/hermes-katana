#!/bin/bash
set -euo pipefail

python -c "import hermes_katana; import hermes_katana.proving_ground" >/dev/null
katana --help >/dev/null
katana proving-ground list-tasks >/dev/null
katana artifacts status >/dev/null

echo "healthcheck PASS"
