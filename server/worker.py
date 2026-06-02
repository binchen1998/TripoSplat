"""独立 GPU job worker：Redis BLPOP 取任务，执行 TripoSplat + 七牛上传。"""
import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from server.config import (  # noqa: E402
    REDIS_QUEUE_KEY,
    REPO_ROOT,
    WORKER_BRPOP_TIMEOUT,
    WORKER_LOCK_PATH,
)
from server import job_store  # noqa: E402
from server.pipeline_runner import get_pipeline, run_job  # noqa: E402
from server.qiniu_cdn import qiniu_ready  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("triposplat.worker")


class WorkerLock:
    def __init__(self, path: Path):
        self.path = path
        self.handle = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.path, "a+b")
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.handle.seek(0)
            self.handle.truncate()
            self.handle.write(str(os.getpid()).encode("ascii"))
            self.handle.flush()
            return True
        except OSError:
            self.release()
            return False

    def release(self) -> None:
        if not self.handle:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        self.handle.close()
        self.handle = None


def main() -> None:
    job_store.init_storage()
    ok, reason = qiniu_ready()
    if not ok:
        logger.error("七牛未就绪，worker 无法上传 PLY: %s", reason)
        sys.exit(1)

    lock = WorkerLock(WORKER_LOCK_PATH)
    if not lock.acquire():
        logger.error("Another job worker is already running (lock: %s)", WORKER_LOCK_PATH)
        sys.exit(1)

    logger.info(
        "TripoSplat job worker started (repo=%s, queue=%s)",
        REPO_ROOT,
        REDIS_QUEUE_KEY,
    )
    logger.info("Loading TripoSplat models at startup …")
    get_pipeline()
    logger.info("Models ready, waiting for jobs on Redis queue")
    try:
        while True:
            job = job_store.pop_job_from_queue(timeout=WORKER_BRPOP_TIMEOUT)
            if job is None:
                continue
            logger.info("Processing job %s", job["id"])
            run_job(job)
    except KeyboardInterrupt:
        logger.info("Worker stopped")
    finally:
        lock.release()


if __name__ == "__main__":
    main()
