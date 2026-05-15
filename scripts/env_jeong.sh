#!/usr/bin/env bash
# DeepMath training env setup — Blackwell (sm_120) variant
# Usage: bash env.sh
# Recreates conda env "dmath" with Blackwell-compatible torch 2.7 / cu128 / vllm 0.9+.
# WARNING: removes any existing "dmath" env to avoid leftover torch 2.5 packages.

set -euo pipefail

ENV_NAME="deepmath_train"
PY_VER="3.12"
DEEPMATH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- conda init ---
source "$(conda info --base)/etc/profile.d/conda.sh"

# --- 1. recreate env from scratch (old dmath had torch 2.5.1+cu124, incompatible with sm_120) ---
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[env.bash] removing existing conda env: $ENV_NAME"
    conda deactivate 2>/dev/null || true
    conda env remove -y -n "$ENV_NAME"
fi
echo "[env.bash] creating conda env: $ENV_NAME (python=$PY_VER)"
conda create -y -n "$ENV_NAME" python="$PY_VER"

conda activate "$ENV_NAME"
echo "[env.bash] active env: $(python -c 'import sys; print(sys.executable)')"

# --- 2. torch 2.7.0 (cu128) — Blackwell sm_120 official support ---
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

# --- 3. flash-attn (prebuilt wheel for torch2.7+cu12+py312, cxx11abiTRUE
#       — must match torch 2.7+cu128 default ABI) ---
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# --- 4. vllm — pin to 0.9.x so it doesn't pull torch 2.10+ over our 2.7 ---
pip install "vllm>=0.9.0,<0.10"

# --- 5. ray (vllm usually pulls compatible ray; this is a no-op if so) ---
pip install "ray[default]"

# --- 6. hydra/omega — stable versions (the dev-pins from the old env.sh
#       were needed by vllm 0.7.3; newer vllm accepts stable releases) ---
pip install omegaconf hydra-core

# --- 7. eval / training utils — pin transformers to 4.51.x because:
#       (a) verl 0.2.0.dev's fsdp_workers.py imports AutoModelForVision2Seq
#           which was removed in transformers 5.x;
#       (b) vllm 0.9.2 metadata is permissive enough that pip resolves to 5.x
#           by default. Pinning 4.51.3 keeps both happy. ---
pip install "math-verify[antlr4_11_0]==0.7.0" fire deepspeed \
    tensorboardX prettytable datasets "transformers==4.51.3"

# --- 8. verl (editable) — note: verl's setup pins tensordict<0.6, which is
#       incompatible with torch 2.7 (tensordict 0.5.x imports ForkingPickler
#       in a way that broke in torch 2.7). We let verl install its pinned
#       tensordict here, then force-upgrade it in step 10. ---
pip install -e "$DEEPMATH_DIR/verl"

# --- 9. extras (used by reward function / runtime) ---
# sympy: torch 2.7's dynamo requires >=1.13 (uses sympy.core.basic._args_sortkey).
# A previous pin to <1.13 broke vLLM inference. The math-verify edge cases on
# sympy 1.13+ are tolerable (caught by reward_func try/except).
pip install langdetect==1.0.9 pebble==5.1.0 word2number timeout_decorator

# --- 10. override verl's tensordict<0.6 pin — newer tensordict is needed for torch 2.7 ---
pip install --upgrade "tensordict>=0.7"

# --- 11. patch vllm 0.9.x ovis.py: 'aimv2' is already registered by
#       transformers >=4.51 so the call without exist_ok=True crashes at import. ---
OVIS_PY="$(python -c 'import vllm, os; print(os.path.join(os.path.dirname(vllm.__file__), "transformers_utils/configs/ovis.py"))')"
if [[ -f "$OVIS_PY" ]]; then
    python -c "
import re, pathlib
p = pathlib.Path('$OVIS_PY')
s = p.read_text()
new = re.sub(
    r'AutoConfig\.register\(\"aimv2\",\s*AIMv2Config\)',
    'AutoConfig.register(\"aimv2\", AIMv2Config, exist_ok=True)',
    s,
)
if new != s:
    p.write_text(new)
    print('[env.bash] patched ovis.py: AutoConfig.register aimv2 exist_ok=True')
else:
    print('[env.bash] ovis.py: no patch needed (already patched or pattern changed)')
"
fi

# --- 10. sanity check (incl. actual sm_120 kernel launch) ---
echo ""
echo "[env.bash] verifying installs..."
python - <<'PY'
import torch, ray, vllm, flash_attn, verl, math_verify
print(f"torch       {torch.__version__} | cuda {torch.version.cuda} | gpu {torch.cuda.is_available()}")
print(f"compiled    {torch.cuda.get_arch_list()}")
print(f"ray         {ray.__version__}")
print(f"vllm        {vllm.__version__}")
print(f"flash_attn  {flash_attn.__version__}")
print(f"verl        ok")
print(f"math_verify ok")

# critical: actual GPU kernel launch on the user's Blackwell card
x = torch.randn(8, device='cuda')
y = (x @ x).sum().item()
print(f"sm_120 kernel launch OK: y={y:.3f}")
PY

echo ""
echo "[env.bash] DONE. activate with:  conda activate $ENV_NAME"
