"""七牛上传（与 littlebits backend/project_cdn.py 同一套 env 与 put_data 流程）。"""

import logging
import os
import uuid
from datetime import datetime

try:
    from qiniu import Auth, put_data
except ImportError:
    Auth = None
    put_data = None

from server.config import (
    QINIU_ACCESS_KEY,
    QINIU_BUCKET,
    QINIU_CDN_DOMAIN,
    QINIU_SECRET_KEY,
    QINIU_TRIPOSPLAT_PREFIX,
    QINIU_UPLOAD_TOKEN_EXPIRES,
)

logger = logging.getLogger(__name__)


class QiniuUploadError(RuntimeError):
    pass


def qiniu_ready() -> tuple[bool, str | None]:
    if not all([QINIU_ACCESS_KEY, QINIU_SECRET_KEY, QINIU_BUCKET, QINIU_CDN_DOMAIN]):
        return False, "七牛未配置完整（需要 QINIU_ACCESS_KEY/SECRET_KEY/BUCKET/CDN_DOMAIN）"
    if Auth is None or put_data is None:
        return False, "缺少 qiniu Python SDK（pip install qiniu）"
    return True, None


def _normalized_domain() -> str:
    domain = QINIU_CDN_DOMAIN.rstrip("/")
    if domain and "://" not in domain:
        domain = f"https://{domain}"
    return domain


def build_unique_object_key(base_key: str) -> str:
    directory, filename = base_key.rsplit("/", 1) if "/" in base_key else ("", base_key)
    stem, ext = os.path.splitext(filename)
    version = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    suffix = uuid.uuid4().hex[:8]
    object_key = f"{stem}/{version}-{suffix}{ext}"
    return f"{directory}/{object_key}" if directory else object_key


def build_triposplat_ply_key(job_id: str) -> str:
    prefix = QINIU_TRIPOSPLAT_PREFIX or "triposplat"
    return f"{prefix}/{job_id}/splat.ply"


def build_triposplat_splat_key(job_id: str) -> str:
    prefix = QINIU_TRIPOSPLAT_PREFIX or "triposplat"
    return f"{prefix}/{job_id}/splat.splat"


def public_url_for_key(key: str) -> str:
    return f"{_normalized_domain()}/{key.lstrip('/')}"


def upload_bytes(
    *,
    key: str,
    data: bytes,
    error_label: str,
    mime_type: str | None = None,
) -> str:
    ok, reason = qiniu_ready()
    if not ok:
        raise QiniuUploadError(reason or "七牛未就绪")

    auth = Auth(QINIU_ACCESS_KEY, QINIU_SECRET_KEY)
    upload_token = auth.upload_token(QINIU_BUCKET, key, QINIU_UPLOAD_TOKEN_EXPIRES)
    if mime_type:
        ret, info = put_data(upload_token, key, data, mime_type=mime_type)
    else:
        ret, info = put_data(upload_token, key, data)

    status_code = getattr(info, "status_code", None)
    if status_code is not None and status_code >= 300:
        raise QiniuUploadError(f"{error_label}，status={status_code}, ret={ret}")
    if ret is None:
        raise QiniuUploadError(f"{error_label}，返回为空")

    url = public_url_for_key(key)
    logger.info("qiniu upload ok key=%s url=%s", key, url)
    return url


def upload_triposplat_outputs(job_id: str, ply_bytes: bytes, splat_bytes: bytes) -> tuple[str, str, str, str]:
    """返回 (ply_key, ply_url, splat_key, splat_url)。"""
    ply_key = build_unique_object_key(build_triposplat_ply_key(job_id))
    splat_key = build_unique_object_key(build_triposplat_splat_key(job_id))
    ply_url = upload_bytes(
        key=ply_key,
        data=ply_bytes,
        error_label="上传 TripoSplat PLY 失败",
        mime_type="application/octet-stream",
    )
    splat_url = upload_bytes(
        key=splat_key,
        data=splat_bytes,
        error_label="上传 TripoSplat SPLAT 失败",
        mime_type="application/octet-stream",
    )
    return ply_key, ply_url, splat_key, splat_url
