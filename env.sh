# Usage: source env.sh
PROJECT_ROOT="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
export PROJECT_ROOT
export PATH="$PROJECT_ROOT/.env/bin:$PATH"
export LD_LIBRARY_PATH="$PROJECT_ROOT/.env/lib:$LD_LIBRARY_PATH"
export TORCH_EXTENSIONS_DIR="$PROJECT_ROOT/.cache/torch_extensions"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
