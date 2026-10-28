#!/usr/bin/env bash
# Build this worktree's native lisatools backends (CPU + cuda12x) into an
# ISOLATED prefix, without touching the shared venv or its site-packages.
#
#   scripts/window_conv/build_native.sh            # configure (first time) + build + install
#   scripts/window_conv/build_native.sh --clean    # wipe build dir + prefix, full rebuild
#
# Env overrides: WCONV_JOBS (default 32), WCONV_CUDA_ARCH (default 80 = A100).
# Both backends are always built (_use_worktree.use_native() requires both).
# Do NOT run while a WCONV_NATIVE=1 process is alive: `cmake --install`
# overwrites the loaded pycppdetector*.so in place (can SIGBUS that process).
#
# Layout (all under scripts/window_conv/.wconv_env/, git-ignored):
#   buildtools/   nanobind==2.12.0 (uv pip --target; the pyproject pin, and the
#                 nanobind internals ABI "v19_..." embedded in the shared-venv
#                 pycppdetector/cgbgpu/cbbhx .so files. The shared venv has
#                 nanobind 3.1.0 = ABI v22, which must NOT be used.)
#   build/        CMake build dir (reused -> incremental rebuilds)
#   site/         install prefix; site/lisatools_backend_{cpu,cuda12x}/pycppdetector*.so
#                 is what _use_worktree.use_native() puts on sys.path.
#
# The shared venv interpreter is used READ-ONLY (numpy headers, gpubackendtools
# cmake helpers, nanobind is taken from buildtools/ via PYTHONPATH).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE="$(cd "$HERE/../.." && pwd)"
ENV_DIR="$HERE/.wconv_env"
BT="$ENV_DIR/buildtools"
BUILD="$ENV_DIR/build"
SITE="$ENV_DIR/site"
PY=/data/asantini/globalfit/erebor_org_setup/.venv/bin/python
NVCC=/usr/local/cuda-12.5/bin/nvcc
NB_VERSION=2.12.0
JOBS="${WCONV_JOBS:-32}"
CUDA_ARCH="${WCONV_CUDA_ARCH:-80}"
WITH_GPU=ON

for a in "$@"; do
  case "$a" in
    --clean) rm -rf "$BUILD" "$SITE" ;;
    *) echo "unknown option $a" >&2; exit 2 ;;
  esac
done

# 1. isolated nanobind (never installed into the shared venv)
if [[ ! -f "$BT/nanobind/__init__.py" ]] ||
   [[ "$(PYTHONPATH="$BT" "$PY" -c 'import nanobind; print(nanobind.__version__)')" != "$NB_VERSION" ]]; then
  rm -rf "$BT"; mkdir -p "$BT"
  (cd /tmp && uv pip install --python "$PY" --target "$BT" "nanobind==$NB_VERSION")
fi
export PYTHONPATH="$BT${PYTHONPATH:+:$PYTHONPATH}"   # CMakeLists runs `python -m nanobind --cmake_dir`
NB_DIR="$("$PY" -m nanobind --cmake_dir)"
[[ "$NB_DIR" == "$BT"/* ]] || { echo "nanobind resolved to $NB_DIR, not $BT" >&2; exit 1; }

# 2. configure (only when the build dir is new)
SRC_CMAKE="$WORKTREE/src/lisatools/cutils/lisatools_sources.cmake"
if [[ ! -f "$BUILD/CMakeCache.txt" ]] ||
   ! grep -q "^LISATOOLS_WITH_GPU:STRING=$WITH_GPU\$" "$BUILD/CMakeCache.txt"; then
  # Mirror what scikit-build-core passes (Release, project version, SKBUILD_STATE=wheel).
  cmake -S "$WORKTREE" -B "$BUILD" -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSKBUILD_STATE=wheel \
    -DSKBUILD_PROJECT_VERSION=1.2.8 \
    -DPython_EXECUTABLE="$PY" -DPYTHON_EXECUTABLE="$PY" \
    -DLISATOOLS_WITH_GPU="$WITH_GPU" \
    -DLISATOOLS_CUDA_ARCH="$CUDA_ARCH" \
    -DCMAKE_CUDA_COMPILER="$NVCC" \
    -DCMAKE_INSTALL_PREFIX="$SITE"
  # configure_file() rewrites this TRACKED file in the source tree; keep src/ pristine.
  git -C "$WORKTREE" diff --quiet -- "$SRC_CMAKE" ||
    { echo "WARNING: restoring $SRC_CMAKE (rewritten by configure)" >&2; git -C "$WORKTREE" checkout -- "$SRC_CMAKE"; }
fi

# 3. build + install into the prefix
cmake --build "$BUILD" -j "$JOBS"
cmake --install "$BUILD" --prefix "$SITE" > /dev/null

# 4. sanity: nanobind internals ABI tag must match the shared-venv builds (v19)
for so in "$SITE"/lisatools_backend_*/pycppdetector*.so; do
  tag="$(strings "$so" | grep -oE 'v[0-9]+_system_libstdcpp[a-z0-9_]*' | sort -u)"
  echo "built $so  [nanobind ABI: $tag]"
  [[ "$tag" == v19_* ]] || { echo "unexpected nanobind ABI tag in $so" >&2; exit 1; }
done
