#!/usr/bin/env bash
# Run every gate under a hard memory cap.
#
# Why the cap: this box has 15GB RAM and zram swap (compressed, lives IN RAM). Overcommitting
# doesn't page to disk and get OOM-killed — it burns CPU compressing pages into memory that is
# already gone, and the desktop locks up. A cgroup cap turns that hang into a clean crash.
# Qwen2.5-1.5B in fp32 is ~6.2GB per instance; never hold two at once.
set -u
CAP="${VIDCAP_MEM_CAP:-7G}"

# Kaggle HAS systemd-run but no D-Bus session, so `command -v` is not enough —
# probe that it actually runs something before relying on it.
if systemd-run --user --scope -q -p "MemoryMax=$CAP" true >/dev/null 2>&1; then
  CGROUP=1
else
  CGROUP=0
  echo "note: no usable cgroup cap (expected on Kaggle) — relying on container limits"
fi

run() {
  echo "--- $* ---"
  if [ "$CGROUP" = 1 ]; then
    systemd-run --user --scope -q -p "MemoryMax=$CAP" -p MemorySwapMax=0 "$@"
  else
    "$@"
  fi
}

fail=0
for t in test_phase0.py test_model.py test_pipeline.py test_train.py; do
  run python3 "$t" || { echo "FAILED: $t"; fail=1; }
done
run python3 -m scripts.validate_metrics || fail=1

[ "$fail" = 0 ] && echo "=== all gates passed ===" || echo "=== FAILURES ==="
exit "$fail"
