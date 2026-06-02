"""与 Gradio 对齐的采样参数校验。"""

from server.config import (
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_SEED,
    DEFAULT_STEPS,
    NUM_GAUSSIANS,
)

GAUSSIAN_COUNT_CHOICES = (32768, 65536, 131072, 262144)

STEPS_MIN = 1
STEPS_MAX = 50
GUIDANCE_MIN = 1.0
GUIDANCE_MAX = 10.0


class InvalidJobParamsError(ValueError):
    pass


def validate_num_gaussians(value: int) -> int:
    n = int(value)
    if n not in GAUSSIAN_COUNT_CHOICES:
        allowed = ", ".join(str(x) for x in GAUSSIAN_COUNT_CHOICES)
        raise InvalidJobParamsError(f"num_gaussians 必须是 {allowed} 之一")
    return n


def validate_steps(value: int) -> int:
    s = int(value)
    if not STEPS_MIN <= s <= STEPS_MAX:
        raise InvalidJobParamsError(f"steps 必须在 {STEPS_MIN}–{STEPS_MAX} 之间")
    return s


def validate_guidance_scale(value: float) -> float:
    g = float(value)
    if not GUIDANCE_MIN <= g <= GUIDANCE_MAX:
        raise InvalidJobParamsError(f"guidance_scale 必须在 {GUIDANCE_MIN}–{GUIDANCE_MAX} 之间")
    return g


def validate_seed(value: int) -> int:
    return int(value)


def parse_job_params(
    *,
    num_gaussians: int | None = None,
    seed: int | None = None,
    steps: int | None = None,
    guidance_scale: float | None = None,
) -> dict:
    return {
        "num_gaussians": validate_num_gaussians(
            NUM_GAUSSIANS if num_gaussians is None else num_gaussians
        ),
        "seed": validate_seed(DEFAULT_SEED if seed is None else seed),
        "steps": validate_steps(DEFAULT_STEPS if steps is None else steps),
        "guidance_scale": validate_guidance_scale(
            DEFAULT_GUIDANCE_SCALE if guidance_scale is None else guidance_scale
        ),
    }
