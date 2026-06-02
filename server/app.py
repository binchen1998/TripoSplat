import logging
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from server import job_store  # noqa: E402
from server.config import (  # noqa: E402
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_SEED,
    DEFAULT_STEPS,
    NUM_GAUSSIANS,
    REPO_ROOT,
    REDIS_QUEUE_KEY,
)
from server.job_store import JobStoreError  # noqa: E402
from server.pipeline_runner import default_job_params  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("triposplat.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        job_store.init_storage()
        await job_store.ping_async()
    except JobStoreError as exc:
        logger.error("%s", exc)
        raise
    yield
    await job_store.close_async()


app = FastAPI(title="TripoSplat API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/")
async def index_page():
    index = _STATIC_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(404, "index.html not found")
    return FileResponse(index)


@app.post("/api/jobs")
async def submit_job(image: UploadFile = File(...)):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "请上传图片文件")

    params = default_job_params()
    job_id = uuid.uuid4().hex
    job_dir = job_store.job_input_dir(job_id)
    ext = Path(image.filename or "upload.png").suffix.lower() or ".png"
    if ext not in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}:
        ext = ".png"
    input_path = job_dir / f"input{ext}"

    data = await image.read()
    if not data:
        raise HTTPException(400, "空文件")
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(400, "图片不能超过 20MB")
    input_path.write_bytes(data)

    try:
        job = job_store.create_and_enqueue_job(
            job_id=job_id,
            input_path=str(input_path),
            seed=params["seed"],
            steps=params["steps"],
            guidance_scale=params["guidance_scale"],
            num_gaussians=params["num_gaussians"],
        )
    except JobStoreError as exc:
        raise HTTPException(503, str(exc)) from exc

    logger.info("Submitted job %s (%d bytes)", job["id"], len(data))

    return {
        "job_id": job["id"],
        "status": "pending",
        "stage_message": job["stage_message"],
        "poll_url": f"/api/jobs/{job['id']}",
        "defaults": {
            "seed": DEFAULT_SEED,
            "steps": DEFAULT_STEPS,
            "guidance_scale": DEFAULT_GUIDANCE_SCALE,
            "num_gaussians": NUM_GAUSSIANS,
        },
    }


@app.get("/api/jobs/{job_id}")
async def poll_job(job_id: str):
    job = await job_store.get_job_async(job_id)
    if job is None:
        raise HTTPException(404, "任务不存在")
    return job_store.job_public_view(job)


@app.get("/api/health")
async def health():
    try:
        await job_store.ping_async()
        qlen = job_store.queue_length()
    except Exception as exc:
        raise HTTPException(503, f"Redis unavailable: {exc}") from exc
    return {
        "ok": True,
        "repo": str(REPO_ROOT),
        "redis_queue": REDIS_QUEUE_KEY,
        "queue_length": qlen,
    }
