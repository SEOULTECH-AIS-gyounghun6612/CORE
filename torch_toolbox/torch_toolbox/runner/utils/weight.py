from __future__ import annotations
from pathlib import Path


def Resolve_weight_path(
    resume_path: str | None,
    weight_path: str | None,
    workspace: Path,
    start_iter: int | None,
) -> str | None:
    """resume과 weight 경로를 해석하여 실제 파일 경로를 반환함. resume이 우선.

    **가중치를 요구했는데 못 찾으면 예외를 던진다.** 예전엔 경고만 찍고 None을 돌려줬는데,
    호출 측은 그 None을 "처음부터 학습"과 구분할 수 없어 초기 가중치로 조용히 진행한다 —
    resume이 재시작이 되고, export는 학습된 적 없는 모델을 산출물로 낸다. 실제로 그렇게
    나간 ONNX가 있었고(학습된 patch_embed·헤더가 통째로 빠진 채 배포됨), 학습 IoU 0.977과
    배포 결과가 갈리는 원인이 됐다. 경고는 파이프라인 로그에 묻히므로 방어선이 못 된다.

    **둘 다 지정되지 않은 경우(처음부터 학습)만 None이 정상**이며, 그때만 조용히 넘어간다.

    Args:
        resume_path: run 디렉터리명. 지정되면 ``workspace/checkpoints`` 에서 찾는다.
        weight_path: 가중치 파일 경로. ``resume_path`` 가 없을 때만 쓰인다.
        workspace: run 디렉터리. resume 시 그 하위 ``checkpoints`` 를 뒤진다.
        start_iter: 지목할 iteration. None이면 디렉터리의 마지막 체크포인트.

    Returns:
        체크포인트 파일 경로. 가중치를 요구하지 않은 경우에만 None.

    Raises:
        FileNotFoundError: 가중치를 요구했으나(resume_path 또는 weight_path 지정) 해석하지
            못한 경우. 초기 가중치로 진행하지 않고 여기서 멈춘다.
    """
    if resume_path is not None:
        if weight_path is not None:
            print("[WARN] --resume과 --weight 동시 지정됨. --weight는 무시함.")

        _ckpt_dir = workspace / "checkpoints"
        if not _ckpt_dir.exists():
            raise FileNotFoundError(
                f"resume을 요구했으나 checkpoint 디렉터리가 없음: {_ckpt_dir}. "
                f"--resume_path 에는 상위 경로 없이 run 디렉터리명만 준다."
            )

        if start_iter is not None:
            _target = _ckpt_dir / f"checkpoint_{start_iter}.pt"
            if _target.exists():
                return str(_target)
            _avail = sorted(_p.name for _p in _ckpt_dir.glob("checkpoint_*.pt"))
            raise FileNotFoundError(
                f"checkpoint 파일 없음: {_target}. "
                f"{_ckpt_dir} 에 있는 것: {_avail if _avail else '없음'}"
            )

        _ckpts = sorted(_ckpt_dir.glob("checkpoint_*.pt"))
        if _ckpts:
            return str(_ckpts[-1])
        raise FileNotFoundError(
            f"resume을 요구했으나 checkpoint 디렉터리가 비어 있음: {_ckpt_dir}"
        )

    if weight_path is not None:
        if Path(weight_path).exists():
            return weight_path
        raise FileNotFoundError(f"weight 파일 없음: {weight_path}")

    return None
