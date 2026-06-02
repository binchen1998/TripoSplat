"""Redis 任务队列：List 排队 + Hash 存任务状态（与 littlebits 相同 REDIS_URL 配置）。"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import redis
import redis.asyncio as aioredis

from server.config import (
    JOBS_DIR,
    REDIS_JOB_KEY_PREFIX,
    REDIS_JOB_TTL_SECONDS,
    REDIS_QUEUE_KEY,
    REDIS_URL,
)

_sync_client: redis.Redis | None = None
_async_client: aioredis.Redis | None = None

_JOB_FIELDS = (
    "id",
    "status",
    "progress",
    "stage_message",
    "error",
    "ply_cdn_key",
    "ply_cdn_url",
    "splat_cdn_key",
    "splat_cdn_url",
    "input_path",
    "seed",
    "steps",
    "guidance_scale",
    "num_gaussians",
    "created_at",
    "updated_at",
    "started_at",
    "finished_at",
)


class JobStoreError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_key(job_id: str) -> str:
    return f"{REDIS_JOB_KEY_PREFIX}{job_id}"


def _require_redis_url() -> None:
    if not REDIS_URL:
        raise JobStoreError(
            "REDIS_URL 未配置。请在 .env 中设置，例如 REDIS_URL=redis://127.0.0.1:6379/0"
        )


def get_redis_sync() -> redis.Redis:
    global _sync_client
    _require_redis_url()
    if _sync_client is None:
        _sync_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    return _sync_client


async def get_redis_async() -> aioredis.Redis:
    global _async_client
    _require_redis_url()
    if _async_client is None:
        _async_client = aioredis.Redis.from_url(REDIS_URL, decode_responses=True)
    return _async_client


async def close_async() -> None:
    global _async_client
    if _async_client is not None:
        await _async_client.aclose()
        _async_client = None


def init_storage() -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    client = get_redis_sync()
    client.ping()


async def ping_async() -> None:
    client = await get_redis_async()
    await client.ping()


def _normalize_job(raw: dict[str, Any]) -> dict[str, Any]:
    if not raw:
        return {}
    job = dict(raw)
    if "id" not in job and "job_id" in job:
        job["id"] = job["job_id"]
    if "progress" in job and job["progress"] is not None:
        job["progress"] = float(job["progress"])
    for key in ("seed", "steps", "num_gaussians"):
        if key in job and job[key] not in (None, ""):
            job[key] = int(job[key])
    if "guidance_scale" in job and job["guidance_scale"] not in (None, ""):
        job["guidance_scale"] = float(job["guidance_scale"])
    return job


def _hash_from_job(job: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key in _JOB_FIELDS:
        val = job.get(key)
        if val is None:
            continue
        out[key] = str(val)
    return out


def _save_job_hash(client: redis.Redis, job_id: str, mapping: dict[str, Any]) -> None:
    key = _job_key(job_id)
    client.hset(key, mapping=_hash_from_job(mapping))
    if REDIS_JOB_TTL_SECONDS > 0:
        client.expire(key, REDIS_JOB_TTL_SECONDS)


def create_and_enqueue_job(
    *,
    input_path: str,
    seed: int,
    steps: int,
    guidance_scale: float,
    num_gaussians: int,
    job_id: str | None = None,
) -> dict[str, Any]:
    client = get_redis_sync()
    job_id = job_id or uuid.uuid4().hex
    now = utc_now()
    job: dict[str, Any] = {
        "id": job_id,
        "status": "pending",
        "progress": 0,
        "stage_message": "任务已提交，等待 worker 处理",
        "error": "",
        "input_path": input_path,
        "seed": seed,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "num_gaussians": num_gaussians,
        "created_at": now,
        "updated_at": now,
    }
    _save_job_hash(client, job_id, job)
    client.rpush(REDIS_QUEUE_KEY, job_id)
    return job


def get_job(job_id: str) -> dict[str, Any] | None:
    client = get_redis_sync()
    raw = client.hgetall(_job_key(job_id))
    if not raw:
        return None
    return _normalize_job(raw)


async def get_job_async(job_id: str) -> dict[str, Any] | None:
    client = await get_redis_async()
    raw = await client.hgetall(_job_key(job_id))
    if not raw:
        return None
    return _normalize_job(raw)


def update_job(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    fields["updated_at"] = utc_now()
    client = get_redis_sync()
    key = _job_key(job_id)
    client.hset(key, mapping=_hash_from_job(fields))
    if REDIS_JOB_TTL_SECONDS > 0:
        client.expire(key, REDIS_JOB_TTL_SECONDS)


def pop_job_from_queue(timeout: int) -> dict[str, Any] | None:
    """Worker：BLPOP 阻塞取队首 job_id，并标记为 running。"""
    client = get_redis_sync()
    item = client.blpop(REDIS_QUEUE_KEY, timeout=max(1, timeout))
    if not item:
        return None
    _, job_id = item
    job = get_job(job_id)
    if job is None:
        return None
    now = utc_now()
    update_job(
        job_id,
        status="running",
        progress=1,
        stage_message="Worker 已领取任务，准备加载模型",
        started_at=now,
    )
    return get_job(job_id)


def job_input_dir(job_id: str) -> Path:
    path = JOBS_DIR / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def job_public_view(job: dict[str, Any]) -> dict[str, Any]:
    error = job.get("error") or None
    if error == "":
        error = None
    return {
        "job_id": job["id"],
        "status": job["status"],
        "progress": float(job.get("progress") or 0),
        "stage_message": job.get("stage_message") or "",
        "error": error,
        "num_gaussians": job.get("num_gaussians"),
        "ply_url": job.get("ply_cdn_url") or None,
        "splat_url": job.get("splat_cdn_url") or None,
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "started_at": job.get("started_at") or None,
        "finished_at": job.get("finished_at") or None,
    }


def queue_length() -> int:
    return int(get_redis_sync().llen(REDIS_QUEUE_KEY))
