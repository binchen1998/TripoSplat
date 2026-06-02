import logging
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
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
from server.params import (  # noqa: E402
    GAUSSIAN_COUNT_CHOICES,
    GUIDANCE_MAX,
    GUIDANCE_MIN,
    InvalidJobParamsError,
    STEPS_MAX,
    STEPS_MIN,
    parse_job_params,
)
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
_VIEWER_DIR = REPO_ROOT / "static" / "viewer"
if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
if _VIEWER_DIR.is_dir():
    app.mount("/viewer", StaticFiles(directory=str(_VIEWER_DIR), html=True), name="viewer")


@app.get("/")
async def index_page():
    index = _STATIC_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(404, "index.html not found")
    return FileResponse(index)


@app.get("/api/options")
async def job_options():
    """与 Gradio Sampling settings 一致的可选参数。"""
    d = default_job_params()
    return {
        "num_gaussians_choices": list(GAUSSIAN_COUNT_CHOICES),
        "defaults": d,
        "steps": {"min": STEPS_MIN, "max": STEPS_MAX},
        "guidance_scale": {"min": GUIDANCE_MIN, "max": GUIDANCE_MAX, "step": 0.5},
    }


@app.post("/api/jobs")
async def submit_job(
    image: UploadFile = File(...),
    num_gaussians: int = Form(default=NUM_GAUSSIANS),
    seed: int = Form(default=DEFAULT_SEED),
    steps: int = Form(default=DEFAULT_STEPS),
    guidance_scale: float = Form(default=DEFAULT_GUIDANCE_SCALE),
):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "请上传图片文件")

    try:
        params = parse_job_params(
            num_gaussians=num_gaussians,
            seed=seed,
            steps=steps,
            guidance_scale=guidance_scale,
        )
    except InvalidJobParamsError as exc:
        raise HTTPException(400, str(exc)) from exc
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
        "params": params,
    }


@app.get("/api/jobs/{job_id}")
async def poll_job(job_id: str):
    job = await job_store.get_job_async(job_id)
    if job is None:
        raise HTTPException(404, "任务不存在")
    return job_store.job_public_view(job)


@app.api_route("/api/jobs/{job_id}/output.ply", methods=["GET", "HEAD"])
async def job_output_ply(job_id: str, request: Request):
    """同源 PLY，供页面内 Spark viewer 加载（避免七牛跨域）。"""
    job = await job_store.get_job_async(job_id)
    if job is None:
        raise HTTPException(404, "任务不存在")
    ply_path = job_store.job_input_dir(job_id) / "output.ply"
    if not ply_path.is_file():
        raise HTTPException(404, "PLY 尚未生成")
    if request.method == "HEAD":
        size = ply_path.stat().st_size
        return Response(
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Length": str(size),
                "Accept-Ranges": "bytes",
            }
        )
    return FileResponse(
        ply_path,
        media_type="application/octet-stream",
        filename="output.ply",
    )


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
