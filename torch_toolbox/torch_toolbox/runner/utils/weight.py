from __future__ import annotations
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from python_toolbox.file import Read_from

#: iter 별 점수에서 하나를 고르는 규칙. **방향이 곧 규칙**이라 higher_is_better 같은
#: 플래그를 따로 두지 않는다.
SELECT_RULE = Callable[[Mapping[int, float]], int]


def Max_of(scores: Mapping[int, float]) -> int:
    """점수가 가장 큰 iteration. 동률이면 나중 것(더 학습된 쪽)."""
    return max(sorted(scores, reverse=True), key=lambda _k: scores[_k])


def Min_of(scores: Mapping[int, float]) -> int:
    """점수가 가장 작은 iteration. 동률이면 나중 것."""
    return min(sorted(scores, reverse=True), key=lambda _k: scores[_k])


@dataclass(frozen=True)
class Iter_Selection:
    """복원할 iteration 을 고르는 방법. **규칙과 대상을 나눠 든다.**

    나눠 두면 조립된다 — 같은 규칙을 train/val 어느 지표에도 걸 수 있고, 새 규칙을 붙여도
    대상 쪽은 안 건드린다. ``Min_of`` + ``("train", "loss")`` 도 ``Max_of`` +
    ``("val", "accuracy")`` 도 같은 자리에 들어간다.

    **러너가 선언한다.** CLI 문자열로 받지 않는 이유는 도메인 상수이기 때문이다 — 오타가
    런타임까지 살아 있을 이유가 없고, "이 도메인에서 좋은 모델이 무엇인가"는 실행할 때마다
    고를 값이 아니다.

    Attributes:
        rule: 점수 dict 에서 iteration 하나를 고르는 함수.
        mode: 볼 로그의 mode (``train`` / ``val`` 등).
        metric: 볼 지표 이름 (``accuracy`` / ``loss`` 등).
    """

    rule: SELECT_RULE
    mode: str
    metric: str

    def __call__(self, workspace: Path) -> int:
        """학습 로그를 읽어 iteration 을 고른다.

        Raises:
            FileNotFoundError: 그 mode 의 로그 디렉터리가 없는 경우.
            ValueError: 그 지표를 가진 iteration 이 하나도 없는 경우.
        """
        _scores = Read_metric_log(workspace, self.mode, self.metric)
        _pick = self.rule(_scores)
        print(f"[INFO] iter {_pick} 선택 ({self.rule.__name__} of "
              f"{self.mode}/{self.metric}={_scores[_pick]:.6f}, 후보 {len(_scores)}개)")
        return _pick


def Read_metric_log(workspace: Path, mode: str, metric: str) -> dict[int, float]:
    """``avg/<mode>/<acc_name>/iter_<i>.json`` 에서 ``{iter: 값}`` 을 모은다.

    ``log_iter`` 가 학습 중 떨군 값이다.

    Args:
        workspace: run 디렉터리.
        mode: 로그 mode 이름.
        metric: 지표 이름.

    Returns:
        iteration → 값.

    Raises:
        FileNotFoundError: 그 mode 의 로그 디렉터리가 없는 경우 — 학습을 안 돌렸거나
            mode 이름이 틀렸다.
        ValueError: 지표가 없거나, 여러 accumulator 가 같은 이름을 내서 모호한 경우.
    """
    _dir = workspace / "avg" / mode
    if not _dir.exists():
        raise FileNotFoundError(
            f"학습 로그가 없다: {_dir}. 학습을 돌렸는지, mode 이름('{mode}')이 맞는지 "
            f"확인할 것."
        )

    _scores: dict[int, float] = {}
    _sources: set[str] = set()
    for _json in _dir.glob("*/iter_*.json"):
        _iter = _Iter_of(_json)
        if _iter is None:
            continue
        _ok, _data = Read_from(_json)
        if not _ok or not isinstance(_data, dict) or metric not in _data:
            continue
        _scores[_iter] = float(_data[metric])
        _sources.add(_json.parent.name)

    if len(_sources) > 1:
        raise ValueError(
            f"'{metric}' 를 내는 accumulator 가 둘 이상이다({sorted(_sources)}). "
            f"어느 것을 볼지 모호하다."
        )
    if not _scores:
        _seen = sorted({_p.parent.name for _p in _dir.glob("*/iter_*.json")})
        raise ValueError(
            f"로그에 '{metric}' 지표가 없다. {_dir} 의 accumulator: "
            f"{_seen if _seen else '없음'}"
        )
    return _scores


def Resolve_weight_path(
    resume_path: str | None,
    weight_path: str | None,
    workspace: Path,
    start_iter: int | None,
    selection: Iter_Selection | None = None,
) -> str | None:
    """resume과 weight 경로를 해석하여 실제 파일 경로를 반환함. resume이 우선.

    **가중치를 요구했는데 못 찾으면 예외를 던진다.** 예전엔 경고만 찍고 None을 돌려줬는데,
    호출 측은 그 None을 "처음부터 학습"과 구분할 수 없어 초기 가중치로 조용히 진행한다 —
    resume이 재시작이 되고, export는 학습된 적 없는 모델을 산출물로 낸다. 실제로 그렇게
    나간 ONNX가 있었고(학습된 patch_embed·헤더가 통째로 빠진 채 배포됨), 학습 IoU 0.977과
    배포 결과가 갈리는 원인이 됐다. 경고는 파이프라인 로그에 묻히므로 방어선이 못 된다.

    **둘 다 지정되지 않은 경우(처음부터 학습)만 None이 정상**이며, 그때만 조용히 넘어간다.

    iteration 을 고르는 순서는 ``start_iter`` → ``selection`` → 마지막 체크포인트다.
    ``selection`` 을 넘길지는 **호출 측이 정한다** — 학습 재개에 걸면 이미 지난 iteration
    부터 다시 돌면서 뒤 체크포인트를 덮어쓴다.

    Args:
        resume_path: run 디렉터리명. 지정되면 ``workspace/checkpoints`` 에서 찾는다.
        weight_path: 가중치 파일 경로. ``resume_path`` 가 없을 때만 쓰인다.
        workspace: run 디렉터리. resume 시 그 하위 ``checkpoints`` 를 뒤진다.
        start_iter: 지목할 iteration. 명시하면 ``selection`` 보다 우선한다.
        selection: 러너가 선언한 iteration 선택 방법. None이면 마지막 체크포인트.

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

        _iter = start_iter
        if _iter is None and selection is not None:
            _iter = selection(workspace)

        if _iter is not None:
            _target = _ckpt_dir / f"checkpoint_{_iter}.pt"
            if _target.exists():
                return str(_target)
            raise FileNotFoundError(
                f"checkpoint 파일 없음: {_target}. "
                f"{_ckpt_dir} 에 있는 것: {_Available(_ckpt_dir)}"
            )

        # **iter 순으로 고른다.** 파일명에 0 패딩이 없어 문자열 정렬은 틀린다 —
        # checkpoint_9.pt 가 checkpoint_48.pt 보다 뒤로 가서 조용히 옛 가중치를 잡는다.
        _ckpts = _Sorted_checkpoints(_ckpt_dir)
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


def _Iter_of(path: Path) -> int | None:
    """``checkpoint_48.pt`` / ``iter_48.json`` 에서 48 을 뽑는다. 못 뽑으면 None."""
    _, _, _num = path.stem.rpartition("_")
    return int(_num) if _num.isdigit() else None


def _Sorted_checkpoints(ckpt_dir: Path) -> list[Path]:
    """iter 오름차순 체크포인트 목록. iter 를 못 읽는 파일은 뺀다."""
    _pairs = [
        (_i, _p) for _p in ckpt_dir.glob("checkpoint_*.pt")
        if (_i := _Iter_of(_p)) is not None
    ]
    return [_p for _, _p in sorted(_pairs)]


def _Available(ckpt_dir: Path) -> list[str]:
    """오류 메시지용 — 실제로 있는 체크포인트를 iter 순으로."""
    _names = [_p.name for _p in _Sorted_checkpoints(ckpt_dir)]
    return _names if _names else ["없음"]
