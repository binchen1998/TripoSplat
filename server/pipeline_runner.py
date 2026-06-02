import logging
import sys
from pathlib import Path

import torch
from PIL import Image

from server.config import (
    CKPT_PATH,
    DECODER_PATH,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_SEED,
    DEFAULT_SHIFT,
    DEFAULT_STEPS,
    DEVICE,
    DINOV3_PATH,
    FLUX2_VAE_PATH,
    NUM_GAUSSIANS,
    REPO_ROOT,
    RMBG_PATH,
)
from server import job_store
from server.qiniu_cdn import QiniuUploadError, upload_triposplat_ply, upload_triposplat_splat

logger = logging.getLogger(__name__)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from triposplat import TripoSplatPipeline  # noqa: E402

_pipeline: TripoSplatPipeline | None = None


def get_pipeline() -> TripoSplatPipeline:
    global _pipeline
    if _pipeline is None:
        logger.info("Loading TripoSplat pipeline on %s …", DEVICE)
        _pipeline = TripoSplatPipeline(
            ckpt_path=CKPT_PATH,
            decoder_path=DECODER_PATH,
            dinov3_path=DINOV3_PATH,
            flux2_vae_encoder_path=FLUX2_VAE_PATH,
            rmbg_path=RMBG_PATH,
            device=DEVICE,
        )
    return _pipeline


def _set_stage(job_id: str, *, progress: float, stage_message: str) -> None:
    job_store.update_job(
        job_id,
        progress=max(0.0, min(100.0, float(progress))),
        stage_message=stage_message,
    )


def run_job(job: dict) -> None:
    job_id = job["id"]
    input_path = Path(job["input_path"])
    seed = int(job["seed"])
    steps = int(job["steps"])
    guidance_scale = float(job["guidance_scale"])
    num_gaussians = int(job["num_gaussians"])

    try:
        pipe = get_pipeline()
        gen = torch.Generator(device=pipe._device).manual_seed(seed)

        _set_stage(job_id, progress=5, stage_message="正在预处理图片（抠图、裁剪、黑底合成）")
        prepared = pipe.preprocess_image(Image.open(input_path))
        out_dir = job_store.job_input_dir(job_id)
        prepared.save(out_dir / "preprocessed.jpg", quality=92)

        _set_stage(job_id, progress=12, stage_message="正在编码输入图片（DINOv3 + VAE）")
        cond = pipe.encode_image(prepared, generator=gen)

        def flow_callback(step: int, total: int) -> None:
            # Flow 采样约占 12%–78%
            frac = step / max(total, 1)
            progress = 12 + frac * 66
            _set_stage(
                job_id,
                progress=progress,
                stage_message=f"Flow 采样去噪中（第 {step}/{total} 步）",
            )

        _set_stage(job_id, progress=12, stage_message=f"Flow 采样去噪中（共 {steps} 步）")
        out = pipe.sample_latent(
            cond,
            steps=steps,
            guidance_scale=guidance_scale,
            shift=DEFAULT_SHIFT,
            generator=gen,
            show_progress=False,
            callback=flow_callback,
        )

        _set_stage(job_id, progress=82, stage_message=f"正在解码 3D 高斯（{num_gaussians:,} 颗）")
        gaussian = pipe.decode_latent(out["latent"], num_gaussians=num_gaussians)

        _set_stage(job_id, progress=88, stage_message="正在导出 PLY 文件")
        ply_bytes = gaussian.to_ply_bytes()
        (out_dir / "output.ply").write_bytes(ply_bytes)

        _set_stage(job_id, progress=94, stage_message="正在上传 PLY 到七牛 CDN")
        ply_key, ply_url = upload_triposplat_ply(job_id, ply_bytes)

        splat_key, splat_url = "", ""
        try:
            splat_bytes = gaussian.to_splat_bytes()
            (out_dir / "output.splat").write_bytes(splat_bytes)
            splat_key, splat_url = upload_triposplat_splat(job_id, splat_bytes)
        except QiniuUploadError as exc:
            logger.warning("Job %s splat upload skipped: %s", job_id, exc)

        job_store.update_job(
            job_id,
            status="completed",
            progress=100,
            stage_message="生成完成，PLY 已上传七牛",
            ply_cdn_key=ply_key,
            ply_cdn_url=ply_url,
            splat_cdn_key=splat_key or None,
            splat_cdn_url=splat_url or None,
            finished_at=job_store.utc_now(),
        )
        logger.info("Job %s completed ply_cdn_url=%s", job_id, ply_url)

    except QiniuUploadError as exc:
        logger.exception("Job %s qiniu failed", job_id)
        job_store.update_job(
            job_id,
            status="failed",
            stage_message="七牛上传失败",
            error=str(exc),
            finished_at=job_store.utc_now(),
        )
    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        job_store.update_job(
            job_id,
            status="failed",
            stage_message="任务处理失败",
            error=str(exc),
            finished_at=job_store.utc_now(),
        )


def default_job_params() -> dict:
    return {
        "seed": DEFAULT_SEED,
        "steps": DEFAULT_STEPS,
        "guidance_scale": DEFAULT_GUIDANCE_SCALE,
        "num_gaussians": NUM_GAUSSIANS,
    }
