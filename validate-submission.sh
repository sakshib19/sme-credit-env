#!/usr/bin/env bash
#
# validate-submission.sh — SME Credit Risk OpenEnv Submission Validator
# ======================================================================
# Checks: HF Space responds to /reset, docker build succeeds, openenv validate passes.
#
# Prerequisites:
#   - Docker       https://docs.docker.com/get-docker/
#   - openenv-core pip install 'openenv-core[core]'
#   - curl         (pre-installed on most systems)
#
# Usage:
#   chmod +x validate-submission.sh
#   ./validate-submission.sh <hf_space_url> [repo_dir]
#
# Examples:
#   ./validate-submission.sh https://YOUR_USERNAME-sme-credit-env.hf.space
#   ./validate-submission.sh https://YOUR_USERNAME-sme-credit-env.hf.space ./sme-credit-env
#
# On Windows use Git Bash:
#   bash validate-submission.sh https://YOUR_USERNAME-sme-credit-env.hf.space .

set -uo pipefail

DOCKER_BUILD_TIMEOUT=600

# Colour codes (disabled if not a terminal)
if [ -t 1 ]; then
  RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
  BOLD='\033[1m';   NC='\033[0m'
else
  RED=''; GREEN=''; YELLOW=''; BOLD=''; NC=''
fi

# ── Helpers ───────────────────────────────────────────────────────────────
run_with_timeout() {
  local secs="$1"; shift
  if command -v timeout &>/dev/null;   then timeout "$secs" "$@"
  elif command -v gtimeout &>/dev/null; then gtimeout "$secs" "$@"
  else
    "$@" &
    local pid=$!
    ( sleep "$secs" && kill "$pid" 2>/dev/null ) &
    local watcher=$!
    wait "$pid" 2>/dev/null; local rc=$?
    kill "$watcher" 2>/dev/null; wait "$watcher" 2>/dev/null
    return $rc
  fi
}

portable_mktemp() {
  mktemp "${TMPDIR:-/tmp}/${1:-validate}-XXXXXX" 2>/dev/null || mktemp
}

CLEANUP_FILES=()
cleanup() { rm -f "${CLEANUP_FILES[@]+"${CLEANUP_FILES[@]}"}"; }
trap cleanup EXIT

log()  { printf "[%s] %b\n"   "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; PASS=$((PASS + 1)); }
fail() { log "${RED}FAILED${NC} -- $1"; }
hint() { printf "  ${YELLOW}Hint:${NC} %b\n" "$1"; }
stop_at() {
  printf "\n${RED}${BOLD}Stopped at %s.${NC} Fix the above issue then re-run.\n\n" "$1"
  exit 1
}

# ── Arguments ─────────────────────────────────────────────────────────────
PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  printf "Usage: %s <hf_space_url> [repo_dir]\n\n" "$0"
  printf "  hf_space_url  e.g. https://yourname-sme-credit-env.hf.space\n"
  printf "  repo_dir      path to local repo (default: current directory)\n"
  exit 1
fi

if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  printf "Error: directory '%s' not found\n" "${2:-.}"; exit 1
fi

PING_URL="${PING_URL%/}"
export PING_URL
PASS=0

# ── Header ────────────────────────────────────────────────────────────────
printf "\n${BOLD}==========================================${NC}\n"
printf "${BOLD}  OpenEnv Submission Validator${NC}\n"
printf "${BOLD}  SME Credit Risk RL Environment${NC}\n"
printf "${BOLD}==========================================${NC}\n"
log "Repo:     $REPO_DIR"
log "Space:    $PING_URL"
printf "\n"

# ── Step 1: Ping HF Space ─────────────────────────────────────────────────
log "${BOLD}Step 1/3  Pinging HF Space${NC} ($PING_URL/reset) ..."

CURL_OUT=$(portable_mktemp "validate-curl")
CLEANUP_FILES+=("$CURL_OUT")

HTTP_CODE=$(curl -s -o "$CURL_OUT" -w "%{http_code}" \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{}' \
  "$PING_URL/reset" \
  --max-time 30 2>/dev/null || printf "000")

if [ "$HTTP_CODE" = "200" ]; then
  pass "HF Space is live and /reset returned 200"
elif [ "$HTTP_CODE" = "000" ]; then
  fail "HF Space unreachable (timeout or DNS failure)"
  hint "Is the Space running? Visit $PING_URL in your browser."
  hint "Try: curl -v -X POST -H 'Content-Type: application/json' -d '{}' $PING_URL/reset"
  stop_at "Step 1"
else
  fail "/reset returned HTTP $HTTP_CODE (expected 200)"
  hint "Check Space build logs at https://huggingface.co/spaces/YOUR_USERNAME/sme-credit-env"
  hint "Common causes: wrong port (need 8000 inside container), missing COPY tasks/ in Dockerfile"
  cat "$CURL_OUT" 2>/dev/null && printf "\n"
  stop_at "Step 1"
fi

# ── Step 2: Docker build ───────────────────────────────────────────────────
log "${BOLD}Step 2/3  docker build${NC} ..."

if ! command -v docker &>/dev/null; then
  fail "docker not found"
  hint "Install Docker: https://docs.docker.com/get-docker/"
  stop_at "Step 2"
fi

# Find Dockerfile
if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR/server"
else
  fail "No Dockerfile found in $REPO_DIR or $REPO_DIR/server/"
  stop_at "Step 2"
fi
log "  Dockerfile: $DOCKER_CONTEXT/Dockerfile"

BUILD_OK=false
run_with_timeout "$DOCKER_BUILD_TIMEOUT" \
  docker build "$DOCKER_CONTEXT" -t sme-credit-env-validate:latest \
  && BUILD_OK=true

if $BUILD_OK; then
  pass "docker build succeeded"
else
  fail "docker build failed"
  hint "Run: docker build $DOCKER_CONTEXT 2>&1 | tail -40"
  hint "Common cause: uv.lock out of sync with pyproject.toml — run: uv lock"
  stop_at "Step 2"
fi

# ── Step 3: openenv validate ───────────────────────────────────────────────
log "${BOLD}Step 3/3  openenv validate${NC} ..."

# Find openenv binary
OPENENV_CMD=""
for candidate in \
    openenv \
    "$REPO_DIR/.venv/bin/openenv" \
    "$HOME/.venv/bin/openenv" \
    "$(python3 -m site --user-base 2>/dev/null)/bin/openenv"; do
  if command -v "$candidate" &>/dev/null 2>&1 || [ -x "$candidate" ]; then
    OPENENV_CMD="$candidate"; break
  fi
done

if [ -z "$OPENENV_CMD" ]; then
  fail "openenv command not found"
  hint "Install with: pip install 'openenv-core[core]'"
  hint "Or: uv add 'openenv-core[core]' && uv run openenv validate"
  stop_at "Step 3"
fi

VALIDATE_OK=false
VALIDATE_OUT=$(cd "$REPO_DIR" && "$OPENENV_CMD" validate 2>&1) && VALIDATE_OK=true

if $VALIDATE_OK; then
  pass "openenv validate passed"
  [ -n "$VALIDATE_OUT" ] && log "  $VALIDATE_OUT"
else
  fail "openenv validate failed"
  printf "%s\n" "$VALIDATE_OUT"
  hint "Common causes: openenv.yaml tasks: must be a list (not dict), port mismatch"
  stop_at "Step 3"
fi

# ── Summary ───────────────────────────────────────────────────────────────
printf "\n${BOLD}==========================================${NC}\n"
printf "${GREEN}${BOLD}  All 3/3 checks passed!${NC}\n"
printf "${GREEN}${BOLD}  Your submission is ready.${NC}\n"
printf "${BOLD}==========================================${NC}\n\n"
exit 0