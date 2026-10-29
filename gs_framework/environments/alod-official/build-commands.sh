#!/usr/bin/env bash
set -euo pipefail

ALOD_ENV=/home/luis/miniconda3/envs/alod_official
ALOD_SOURCE=/home/luis/tese/LoDOfGaussians
ALOD_WORKTREE=/home/luis/tese/alod_official_worktree_73f5473
ALOD_BUILD=/home/luis/tese/alod_official_build
ALOD_RECORDS=/home/luis/tese/gs_framework/environments/alod-official
ALOD_PATCH=/home/luis/tese/gs_framework/patches/alod/cmake-cuda-architectures-sm120.patch
ALOD_COMMIT=73f547341547ae021ff036ad74fdf73a1240b450
FUSED_SSIM_COMMIT=a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38

test ! -e "$ALOD_ENV"
test ! -e "$ALOD_WORKTREE"
test ! -e "$ALOD_BUILD"

/home/luis/miniconda3/bin/conda create -y -p "$ALOD_ENV" \
  --file "$ALOD_RECORDS/conda-explicit.txt"

"$ALOD_ENV/bin/python" -m pip install --no-cache-dir \
  torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128

git -C "$ALOD_SOURCE" worktree add --detach "$ALOD_WORKTREE" "$ALOD_COMMIT"
mkdir -p "$ALOD_BUILD/cmake" "$ALOD_BUILD/wheels"
git clone https://github.com/rahul-goel/fused-ssim.git "$ALOD_BUILD/fused-ssim"
git -C "$ALOD_BUILD/fused-ssim" checkout --detach "$FUSED_SSIM_COMMIT"

# The pinned repository tracks stale CPython-3.10 build products. Delete them
# only in the disposable worktree so every installed artifact is rebuilt.
rm -rf "$ALOD_WORKTREE/submodules/gaussianhierarchy/build"
rm -rf "$ALOD_WORKTREE/submodules/simple-knn/build"
git -C "$ALOD_WORKTREE" apply "$ALOD_PATCH"

export CUDA_HOME="$ALOD_ENV"
export CUDA_ARCHITECTURES=120
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=2
export CC="$ALOD_ENV/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$ALOD_ENV/bin/x86_64-conda-linux-gnu-g++"
export CUDACXX="$ALOD_ENV/bin/nvcc"
export CPATH="$ALOD_ENV/include:$ALOD_ENV/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LD_LIBRARY_PATH="$ALOD_ENV/lib:$ALOD_ENV/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$ALOD_ENV/bin/python" -m pip wheel --no-cache-dir --no-build-isolation \
  --no-deps --wheel-dir "$ALOD_BUILD/wheels" \
  "$ALOD_WORKTREE/submodules/simple-knn"
"$ALOD_ENV/bin/python" -m pip wheel --no-cache-dir --no-build-isolation \
  --no-deps --wheel-dir "$ALOD_BUILD/wheels" \
  "$ALOD_WORKTREE/submodules/gaussianhierarchy"
"$ALOD_ENV/bin/python" -m pip wheel --no-cache-dir --no-build-isolation \
  --no-deps --wheel-dir "$ALOD_BUILD/wheels" "$ALOD_BUILD/fused-ssim"

# Conda restores packaging/pip. Native wheel hashes are recorded provenance for
# this build, but a clean rebuild may have a different ELF build ID, so install
# those freshly built wheels separately after restoring every other pip pin.
sed '/^packaging @/d; /^pip @/d; /^fused-ssim @/d; /^gaussian_hierarchy @/d; /^simple_knn @/d' "$ALOD_RECORDS/pip-freeze.txt" \
  > /tmp/alod-official-pip-lock.txt
"$ALOD_ENV/bin/python" -m pip install --no-cache-dir \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r /tmp/alod-official-pip-lock.txt
"$ALOD_ENV/bin/python" -m pip install --no-index "$ALOD_BUILD"/wheels/*.whl

"$ALOD_ENV/bin/cmake" \
  -S "$ALOD_WORKTREE/submodules/gaussianhierarchy" \
  -B "$ALOD_BUILD/cmake" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
"$ALOD_ENV/bin/cmake" --build "$ALOD_BUILD/cmake" -j2 --config Release

"$ALOD_RECORDS/smoke-commands.sh"
