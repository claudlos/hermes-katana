#!/usr/bin/env bash
#
# Layer-2 weekly check: full regression sweep + headline evals + calibration
# + external baselines + DB record + regression gate + leaderboard refresh.
#
# Designed for cron / scheduled GH Actions, but safe to run locally any
# time. Idempotent at the DB level (record_metrics.py skips already-seen
# source paths unless --force).
#
# Stages (in order):
#
#   1. end_to_end_sweep.py            — 19-stage regression sweep
#   2. eval v14 confirmed_only_v1     — w/ bootstrap CIs (headline)
#   3. eval v14 test split            — w/ bootstrap CIs
#   4. v14_calibration_analysis.py    — ECE + Brier + per-family
#   5. run_external_baselines.py      — deepset + protectai
#   6. record_metrics.py --auto-ingest — append fresh runs to DB
#   7. update_leaderboard.py          — refresh LEADERBOARD.md
#   8. check_regression.py            — gate; exit code propagates
#
# Output: results/weekly_check_<ts>/
#   stage_logs/<stage>.{out,err}
#   summary.md
#
# Env:
#   HERMES_KATANA_DEVICE   cpu|cuda (default: auto-detect; force cpu on
#                          hosts with broken NVIDIA driver)
#   SKIP_STAGES            space-separated stage IDs to skip
#                          (e.g. "external_baselines headline_test")
#   HK_PYTHON              python interpreter (default: .venv/bin/python)
#
# Exit codes:
#   0  all stages green, no regression
#   1  one or more stages failed
#   2  stages ran but regression detector flagged a drop past gates

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

HK_PYTHON="${HK_PYTHON:-.venv/bin/python}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="results/weekly_check_${TS}"
LOG_DIR="${OUT_DIR}/stage_logs"
mkdir -p "${LOG_DIR}"

# ---- helpers -------------------------------------------------------------

log()  { printf '[weekly %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '[weekly %s] WARN: %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

declare -A STAGE_STATUS    # stage_id -> pass|fail|skip
declare -A STAGE_DURATION  # stage_id -> seconds (int)
STAGE_ORDER=()

run_stage() {
    local id="$1"; shift
    local desc="$1"; shift
    STAGE_ORDER+=("${id}")

    if [[ " ${SKIP_STAGES:-} " == *" ${id} "* ]]; then
        log "SKIP ${id}: ${desc} (in SKIP_STAGES)"
        STAGE_STATUS["${id}"]="skip"
        STAGE_DURATION["${id}"]=0
        return 0
    fi

    log "RUN  ${id}: ${desc}"
    local t0 t1
    t0=$(date +%s)
    if "$@" \
        >"${LOG_DIR}/${id}.out" \
        2>"${LOG_DIR}/${id}.err"; then
        t1=$(date +%s)
        STAGE_STATUS["${id}"]="pass"
        STAGE_DURATION["${id}"]=$((t1 - t0))
        log "PASS ${id} ($((t1 - t0))s)"
    else
        t1=$(date +%s)
        STAGE_STATUS["${id}"]="fail"
        STAGE_DURATION["${id}"]=$((t1 - t0))
        warn "FAIL ${id} ($((t1 - t0))s) — see ${LOG_DIR}/${id}.err"
    fi
}

# ---- stages --------------------------------------------------------------

# Stage 1 — regression sweep
run_stage "sweep" "end-to-end regression sweep" \
    "${HK_PYTHON}" scripts/end_to_end_sweep.py \
        --out-dir "${OUT_DIR}/sweep"

# Stage 2 — v14 on confirmed_only_v1 (headline) w/ bootstrap
run_stage "headline_confirmed" "v14 eval on confirmed_only_v1 (bootstrap 1000)" \
    "${HK_PYTHON}" training/eval_katana_v11.py \
        --checkpoint training/checkpoints/katana_v14/best \
        --data evals/benchmarks/confirmed_only_v1/test.jsonl \
        --config training/configs/katana_v14.yaml \
        --bootstrap 1000 \
        --out-dir "${OUT_DIR}/eval_v14_confirmed_only"

# Stage 3 — v14 on test split w/ bootstrap
run_stage "headline_test" "v14 eval on test split (bootstrap 1000)" \
    "${HK_PYTHON}" training/eval_katana_v11.py \
        --checkpoint training/checkpoints/katana_v14/best \
        --data training/data_v7/splits/test.jsonl \
        --config training/configs/katana_v14.yaml \
        --bootstrap 1000 \
        --out-dir "${OUT_DIR}/eval_v14_test"

# Stage 4 — calibration
run_stage "calibration" "v14 calibration + per-family analysis" \
    "${HK_PYTHON}" scripts/v14_calibration_analysis.py \
        --checkpoint training/checkpoints/katana_v14/best \
        --data evals/benchmarks/confirmed_only_v1/test.jsonl \
        --config training/configs/katana_v14.yaml \
        --out-dir "${OUT_DIR}/v14_calibration"

# Stage 5 — external baselines (deepset + protectai).
# run_external_baselines.py defaults to CPU "to avoid GPU contention" but
# during the weekly check this is the only workload, so use cuda when the
# user hasn't pinned device=cpu. Saves ~7 minutes per run vs CPU.
ext_device="cuda"
if [[ "${HERMES_KATANA_DEVICE:-}" == "cpu" ]]; then
    ext_device="cpu"
fi
run_stage "external_baselines" "deepset + protectai on confirmed_only_v1 (${ext_device})" \
    "${HK_PYTHON}" scripts/run_external_baselines.py \
        --include-protectai \
        --device "${ext_device}" \
        --out-dir "${OUT_DIR}/external_baselines"

# Stage 6 — DB record (always run; idempotent)
# Note: record_metrics.py --auto-ingest scans results/ for *all* eval/calib/
# baseline outputs that match its known shapes — including the ones the
# previous stages just wrote into OUT_DIR (which is under results/).
run_stage "record" "append fresh metrics to history DB" \
    "${HK_PYTHON}" scripts/record_metrics.py --auto-ingest

# Stage 7 — leaderboard refresh
run_stage "leaderboard" "refresh confirmed_only_v1 LEADERBOARD.md" \
    "${HK_PYTHON}" scripts/update_leaderboard.py

# Stage 8 — regression gate (last; sets overall exit code)
# NB: run_stage swallows the non-zero from check_regression.py into
# STAGE_STATUS=fail; we look at that explicitly to distinguish a real
# regression from a stage that errored. Use --vs-best as the stricter
# gate (compare against historical best, not just last run).
run_stage "regression" "regression check vs prior best" \
    "${HK_PYTHON}" scripts/check_regression.py --vs-best

# ---- summary -------------------------------------------------------------

SUMMARY="${OUT_DIR}/summary.md"

n_pass=0; n_fail=0; n_skip=0
for id in "${STAGE_ORDER[@]}"; do
    case "${STAGE_STATUS[$id]}" in
        pass) ((n_pass++)) ;;
        fail) ((n_fail++)) ;;
        skip) ((n_skip++)) ;;
    esac
done

# Distinguish "stage hard-failed" from "regression detected but stages OK"
overall_status="ok"
regression_failed=0
hard_failed=0

if [[ "${STAGE_STATUS[regression]:-skip}" == "fail" ]]; then
    regression_failed=1
fi
for id in "${STAGE_ORDER[@]}"; do
    if [[ "$id" == "regression" ]]; then continue; fi
    if [[ "${STAGE_STATUS[$id]}" == "fail" ]]; then
        hard_failed=1
    fi
done

if (( hard_failed )); then
    overall_status="hard-failed"
elif (( regression_failed )); then
    overall_status="regressed"
fi

{
    echo "# Weekly check — ${TS}"
    echo
    echo "- host: $(hostname)"
    echo "- device: ${HERMES_KATANA_DEVICE:-auto}"
    echo "- HK_PYTHON: ${HK_PYTHON}"
    echo "- git: $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a') ($(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'n/a'))"
    echo "- overall: **${overall_status}**  (${n_pass} pass / ${n_fail} fail / ${n_skip} skip)"
    echo
    echo "## Stage results"
    echo
    echo "| stage | status | duration |"
    echo "| --- | --- | ---: |"
    for id in "${STAGE_ORDER[@]}"; do
        echo "| ${id} | ${STAGE_STATUS[$id]} | ${STAGE_DURATION[$id]}s |"
    done
    echo
    echo "## Headline numbers"
    echo
    if [[ -f "${OUT_DIR}/eval_v14_confirmed_only/metrics.json" ]]; then
        "${HK_PYTHON}" - <<EOF
import json, pathlib
m = json.loads(pathlib.Path("${OUT_DIR}/eval_v14_confirmed_only/metrics.json").read_text())
bin = m.get("binary_attack_vs_benign", {})
bs = m.get("bootstrap_95ci", {}).get("macro_f1", {})
print(f"- confirmed_only_v1 macro F1: {m.get('macro_f1'):.4f}  "
      f"95% CI [{bs.get('ci_lo', 0):.4f}, {bs.get('ci_hi', 0):.4f}]")
print(f"- confirmed_only_v1 binary F1: {bin.get('f1', 0):.4f}  "
      f"FPR {bin.get('fpr', 0) * 100:.2f}%")
EOF
    fi
    if [[ -f "${OUT_DIR}/eval_v14_test/metrics.json" ]]; then
        "${HK_PYTHON}" - <<EOF
import json, pathlib
m = json.loads(pathlib.Path("${OUT_DIR}/eval_v14_test/metrics.json").read_text())
bin = m.get("binary_attack_vs_benign", {})
bs = m.get("bootstrap_95ci", {}).get("macro_f1", {})
print(f"- test split macro F1: {m.get('macro_f1'):.4f}  "
      f"95% CI [{bs.get('ci_lo', 0):.4f}, {bs.get('ci_hi', 0):.4f}]")
print(f"- test split binary F1: {bin.get('f1', 0):.4f}  "
      f"FPR {bin.get('fpr', 0) * 100:.2f}%")
EOF
    fi
    if [[ -f "${OUT_DIR}/v14_calibration/calibration.json" ]]; then
        "${HK_PYTHON}" - <<EOF
import json, pathlib
m = json.loads(pathlib.Path("${OUT_DIR}/v14_calibration/calibration.json").read_text())
print(f"- ECE (top-label, 15 bins): {m.get('top_label_ece_15bins')}")
print(f"- Brier macro: {m.get('brier_macro')}")
EOF
    fi
    echo
    echo "## Regression check"
    echo
    if [[ -f "${LOG_DIR}/regression.out" ]]; then
        cat "${LOG_DIR}/regression.out"
    fi
} >"${SUMMARY}"

log "summary -> ${SUMMARY}"
log "stage logs -> ${LOG_DIR}/"
log "overall: ${overall_status}"

if (( hard_failed )); then exit 1; fi
if (( regression_failed )); then exit 2; fi
exit 0
