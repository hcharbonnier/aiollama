#!/usr/bin/env bash
#
# Build the default and/or ROCm Docker images locally using `act` to run the
# local-only workflow ci-local/local-build.yaml under Docker on your PC.
#
# This does NOT modify any existing CI scripts under .github/workflows/.
# It only consumes ci-local/local-build.yaml (a separate, local-only workflow
# that reproduces the default + ROCm paths of release.yaml).
#
# Docker-in-Docker: the host docker socket (/var/run/docker.sock) is
# bind-mounted into the `act` runner container, so the docker/build-push-action
# steps inside the workflow talk to the host Docker daemon. No separate DinD
# daemon is spawned.
#
# All built images are loaded into the local Docker daemon with stable tags:
#   - aiollama:local                  default amd64 image
#   - aiollama:local-rocm             ROCm amd64 image
#   - aiollama/dep:<target>-amd64     one per backend dep stage
#
# Usage:
#   ./ci-local/build-local.sh              # build both default and rocm images
#   ./ci-local/build-local.sh default       # only the default amd64 image
#   ./ci-local/build-local.sh rocm          # only the amd64-rocm image
#   ./ci-local/build-local.sh deps          # only the backend dep stages
#   ./ci-local/build-local.sh setup         # only install/check act
#
# Environment variables (optional):
#   ACT_VERSION     pin a specific act release tag (default: latest)
#   ACT_IMAGE       runner image for act (default: catthehacker/ubuntu:act-latest)
#   ACT_VERBOSE     set to "1" for verbose act output
#   DOCKER_SOCK     path to the docker socket (default: /var/run/docker.sock)
#
# Designed to run under WSL (Ubuntu) where Docker is installed.

set -euo pipefail

MODE="${1:-all}"
ACT_VERSION="${ACT_VERSION:-latest}"
ACT_IMAGE="${ACT_IMAGE:-catthehacker/ubuntu:act-latest}"
ACT_VERBOSE="${ACT_VERBOSE:-0}"
DOCKER_SOCK="${DOCKER_SOCK:-/var/run/docker.sock}"

# Resolve repo root (this script lives in ci-local/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKFLOW="${SCRIPT_DIR}/local-build.yaml"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

log()  { printf '\033[1;34m[ci-local]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[ci-local]\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[1;31m[ci-local]\033[0m %s\n' "$*" >&2; }

usage() {
  cat <<EOF
Usage: $0 [all|default|rocm|deps|setup]

  all       Build both the default amd64 and amd64-rocm images
  default   Build only the default amd64 image   (tagged aiollama:local)
  rocm      Build only the amd64-rocm image      (tagged aiollama:local-rocm)
  deps      Build only the backend dep stages in isolation
            (tagged aiollama/dep:<target>-amd64, one per backend)
  setup     Install/verify act, then exit

Environment:
  ACT_VERSION   act release tag (default: latest)
  ACT_IMAGE     runner image (default: ${ACT_IMAGE})
  ACT_VERBOSE   1 for verbose output (default: 0)
  DOCKER_SOCK   docker socket path (default: ${DOCKER_SOCK})
EOF
}

# -----------------------------------------------------------------------------
# Prerequisites
# -----------------------------------------------------------------------------

check_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    err "docker not found in PATH. Install Docker (or run under WSL where Docker is available)."
    return 1
  fi
  if ! docker info >/dev/null 2>&1; then
    err "docker daemon not reachable. Start Docker Desktop / dockerd first."
    return 1
  fi
  if [ ! -S "${DOCKER_SOCK}" ]; then
    err "docker socket not found at ${DOCKER_SOCK}. Set DOCKER_SOCK=<path> if it differs."
    return 1
  fi
  log "docker: $(docker version --format '{{.Server.Version}}')  socket: ${DOCKER_SOCK}"
}

install_act() {
  if command -v act >/dev/null 2>&1; then
    log "act already installed: $(act --version 2>&1 | head -1)"
    return 0
  fi

  log "act not found — installing latest release to ~/.local/bin"
  local arch tmpdir url
  arch="$(uname -m)"
  case "${arch}" in
    x86_64)  arch="x86_64" ;;
    aarch64) arch="arm64"  ;;
    *) err "unsupported arch: ${arch}"; return 1 ;;
  esac

  tmpdir="$(mktemp -d)"
  trap 'rm -rf "${tmpdir}"' RETURN

  if [ "${ACT_VERSION}" = "latest" ]; then
    url="https://github.com/nektos/act/releases/latest/download/act_Linux_${arch}.tar.gz"
  else
    url="https://github.com/nektos/act/releases/download/${ACT_VERSION}/act_Linux_${arch}.tar.gz"
  fi

  log "downloading: ${url}"
  if ! curl -fsSL "${url}" -o "${tmpdir}/act.tar.gz"; then
    err "failed to download act. Check ACT_VERSION='${ACT_VERSION}' or install manually:"
    err "  https://github.com/nektos/act#installation"
    return 1
  fi

  mkdir -p "${HOME}/.local/bin"
  tar -xzf "${tmpdir}/act.tar.gz" -C "${HOME}/.local/bin" act
  chmod +x "${HOME}/.local/bin/act"

  case ":${PATH}:" in
    *":${HOME}/.local/bin:"*) ;;
    *) warn "~/.local/bin is not in PATH for this shell. Add it or run:"
       warn "  export PATH=\"${HOME}/.local/bin:\${PATH}\"" ;;
  esac

  # Use the freshly installed binary for the rest of this script.
  export PATH="${HOME}/.local/bin:${PATH}"
  log "act installed: $(act --version 2>&1 | head -1)"
}

# -----------------------------------------------------------------------------
# act invocation
# -----------------------------------------------------------------------------

run_act() {
  local job="$1"

  local verbose_flag=()
  if [ "${ACT_VERBOSE}" = "1" ]; then
    verbose_flag=(-v)
  fi

  log "running act job='${job}'"
  log "  workflow: ${WORKFLOW}"
  log "  runner image: ${ACT_IMAGE}"
  log "  docker socket bind: ${DOCKER_SOCK}"
  log "  repo: ${REPO_ROOT}"

  (cd "${REPO_ROOT}" && exec act \
    --bind "${DOCKER_SOCK}:/var/run/docker.sock" \
    --workflows "${WORKFLOW}" \
    --job "${job}" \
    --container-architecture linux/amd64 \
    "${verbose_flag[@]}")
}

print_built_images() {
  log "locally available images:"
  if ! docker images --format "  {{.Repository}}:{{.Tag}}  {{.Size}}" \
      | grep -E '^  aiollama[:/]'; then
    echo "  (none)"
  fi
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

case "${MODE}" in
  all|default|rocm|deps|setup) ;;
  -h|--help|help) usage; exit 0 ;;
  *) err "unknown mode: '${MODE}'"; usage; exit 2 ;;
esac

check_docker
install_act

if [ "${MODE}" = "setup" ]; then
  log "setup complete (act installed/verified). Nothing to run."
  exit 0
fi

# The workflow declares:
#   - setup-environment
#   - deps             (matrix of all backend dep stages)
#   - default-image    (target=archive,        tagged aiollama:local)
#   - rocm-image       (target=image-archive, FLAVOR=rocm, tagged aiollama:local-rocm)
# `act` automatically follows the `needs:` graph when given a terminal job,
# so for a single-variant build we just point at that terminal job. For
# `all` we run the two terminal jobs sequentially; the underlying buildx
# layer cache is shared via the host docker.sock.

case "${MODE}" in
  deps)
    run_act deps
    ;;
  default)
    run_act default-image
    ;;
  rocm)
    run_act rocm-image
    ;;
  all)
    log "=== default amd64 image ==="
    run_act default-image
    log "=== amd64-rocm image ==="
    run_act rocm-image
    ;;
esac

log "done."
print_built_images
