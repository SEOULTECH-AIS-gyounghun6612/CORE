from __future__ import annotations
from pathlib import Path


def Resolve_weight_path(
    resume_path: str | None,
    weight_path: str | None,
    workspace: Path,
    start_iter: int | None,
) -> str | None:
    """resume과 weight 경로를 해석하여 실제 파일 경로를 반환함. resume이 우선."""
    if resume_path is not None:
        if weight_path is not None:
            print("[WARN] --resume과 --weight 동시 지정됨. --weight는 무시함.")

        _ckpt_dir = workspace / "checkpoints"
        if not _ckpt_dir.exists():
            print(f"[WARN] Checkpoint 디렉토리 없음: {_ckpt_dir}")
            return None

        if start_iter is not None:
            _target = _ckpt_dir / f"checkpoint_{start_iter}.pt"
            if _target.exists():
                return str(_target)
            print(f"[WARN] Checkpoint 파일 없음: {_target}")
            return None

        _ckpts = sorted(_ckpt_dir.glob("checkpoint_*.pt"))
        if _ckpts:
            return str(_ckpts[-1])
        print(f"[WARN] Checkpoint 없음: {_ckpt_dir}")
        return None

    if weight_path is not None:
        if Path(weight_path).exists():
            return weight_path
        print(f"[WARN] Weight 파일 없음: {weight_path}")

    return None
