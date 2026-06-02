import os
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")
load_dotenv(_REPO_ROOT / "server" / ".env")

REPO_ROOT = _REPO_ROOT
DATA_DIR = Path(os.getenv("TRIPOSPLAT_DATA_DIR", str(_REPO_ROOT / "server_data"))).resolve()
JOBS_DIR = DATA_DIR / "jobs"

REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0").strip()
REDIS_QUEUE_KEY = os.getenv("TRIPOSPLAT_REDIS_QUEUE", "triposplat:queue").strip()
REDIS_JOB_KEY_PREFIX = os.getenv("TRIPOSPLAT_REDIS_JOB_PREFIX", "triposplat:job:").strip()
REDIS_JOB_TTL_SECONDS = int(os.getenv("TRIPOSPLAT_JOB_TTL_SECONDS", str(7 * 24 * 3600)))

CKPT_PATH = os.getenv(
    "TRIPOSPLAT_CKPT",
    str(_REPO_ROOT / "ckpts/diffusion_models/triposplat_fp16.safetensors"),
)
DECODER_PATH = os.getenv(
    "TRIPOSPLAT_DECODER",
    str(_REPO_ROOT / "ckpts/vae/triposplat_vae_decoder_fp16.safetensors"),
)
DINOV3_PATH = os.getenv(
    "TRIPOSPLAT_DINOV3",
    str(_REPO_ROOT / "ckpts/clip_vision/dino_v3_vit_h.safetensors"),
)
FLUX2_VAE_PATH = os.getenv(
    "TRIPOSPLAT_FLUX2_VAE",
    str(_REPO_ROOT / "ckpts/vae/flux2-vae.safetensors"),
)
RMBG_PATH = os.getenv(
    "TRIPOSPLAT_RMBG",
    str(_REPO_ROOT / "ckpts/background_removal/birefnet.safetensors"),
)
DEVICE = os.getenv("TRIPOSPLAT_DEVICE", "cuda")

DEFAULT_SEED = int(os.getenv("TRIPOSPLAT_SEED", "42"))
DEFAULT_STEPS = int(os.getenv("TRIPOSPLAT_STEPS", "20"))
DEFAULT_GUIDANCE_SCALE = float(os.getenv("TRIPOSPLAT_GUIDANCE_SCALE", "3.0"))
DEFAULT_SHIFT = float(os.getenv("TRIPOSPLAT_SHIFT", "3.0"))
NUM_GAUSSIANS = int(os.getenv("TRIPOSPLAT_NUM_GAUSSIANS", "32768"))

WORKER_BRPOP_TIMEOUT = int(os.getenv("TRIPOSPLAT_WORKER_BRPOP_TIMEOUT", "5"))
WORKER_LOCK_PATH = DATA_DIR / ".job-worker.lock"

QINIU_ACCESS_KEY = os.getenv("QINIU_ACCESS_KEY", "").strip()
QINIU_SECRET_KEY = os.getenv("QINIU_SECRET_KEY", "").strip()
QINIU_BUCKET = os.getenv("QINIU_BUCKET", "").strip()
QINIU_CDN_DOMAIN = os.getenv("QINIU_CDN_DOMAIN", "").strip()
QINIU_TRIPOSPLAT_PREFIX = os.getenv("QINIU_TRIPOSPLAT_PREFIX", "triposplat").strip().strip("/")
QINIU_UPLOAD_TOKEN_EXPIRES = int(os.getenv("QINIU_UPLOAD_TOKEN_EXPIRES", "3600"))
