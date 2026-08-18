from __future__ import annotations
from pathlib import Path

from python_toolbox.file import Read_from

#: ``start_iter`` 에 숫자 대신 줄 수 있는 키워드. 학습 로그에서 iter 를 고른다.
BEST_KEYWORD = "best"

#: ``(mode, metric, higher_is_better)``. 어떤 지표를 어느 방향으로 볼지는 도메인 지식이라
#: 러너가 선언한다 — 이름으로 max/min 을 추측하면 그게 조용한 오답이 된다.
BEST_METRIC = tuple[str, str, bool]


def Select_iter_from_log(workspace: Path, best_metric: BEST_METRIC) -> int:
    """``avg`` 로그를 훑어 지표가 가장 좋은 iteration 을 고른다.

    ``log_iter`` 가 iter 마다 ``avg/<mode>/<acc_name>/iter_<i>.json`` 으로 떨군 값을 읽는다.

    **못 고르면 마지막 체크포인트로 떨어지지 않고 실패한다.** "best 를 골랐다" 고 믿는데
    실제로는 아무거나 잡는 것이 제일 나쁘다.

    Args:
        workspace: run 디렉터리.
        best_metric: ``(mode, metric, higher_is_better)``.

    Returns:
        선택된 iteration.

    Raises:
        FileNotFoundError: 로그 디렉터리가 없는 경우(학습을 안 돌렸거나 mode 이름이 틀렸다).
        ValueError: 그 지표를 가진 iter 가 없거나, 여러 accumulator 가 같은 이름을 내서
            어느 것을 볼지 모호한 경우.
    """
    _mode, _metric, _higher = best_metric
    _dir = workspace / "avg" / _mode
    if not _dir.exists():
        raise FileNotFoundError(
            f"'{BEST_KEYWORD}' 를 요구했으나 로그가 없다: {_dir}. "
            f"학습을 돌렸는지, mode 이름('{_mode}')이 맞는지 확인할 것."
        )

    _scores: dict[int, float] = {}
    _sources: set[str] = set()
    for _json in _dir.glob(f"*/iter_*.json"):
        _iter = _Iter_of(_json)
        if _iter is None:
            continue
        _ok, _data = Read_from(_json)
        if not _ok or not isinstance(_data, dict) or _metric not in _data:
            continue
        _scores[_iter] = float(_data[_metric])
        _sources.add(_json.parent.name)

    if len(_sources) > 1:
        raise ValueError(
            f"'{_metric}' 를 내는 accumulator 가 둘 이상이다({sorted(_sources)}). "
            f"어느 것을 볼지 모호하다."
        )
    if not _scores:
        _seen = sorted({_p.parent.name for _p in _dir.glob("*/iter_*.json")})
        raise ValueError(
            f"로그에 '{_metric}' 지표가 없다. {_dir} 의 accumulator: "
            f"{_seen if _seen else '없음'}"
        )

    _pick = (max if _higher else min)(_scores, key=lambda _k: _scores[_k])
    print(f"[INFO] best iter={_pick} ({_mode}/{_metric}={_scores[_pick]:.6f}, "
          f"후보 {len(_scores)}개)")
    return _pick


def Resolve_weight_path(
    resume_path: str | None,
    weight_path: str | None,
    workspace: Path,
    start_iter: int | str | None,
    best_metric: BEST_METRIC | None = None,
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
        start_iter: 지목할 iteration. 숫자면 그 iter, ``"best"`` 면 학습 로그에서 고른다
            (``best_metric`` 필요). None이면 디렉터리의 마지막 체크포인트.
        best_metric: ``"best"`` 를 해석할 ``(mode, metric, higher_is_better)``.

    Returns:
        체크포인트 파일 경로. 가중치를 요구하지 않은 경우에만 None.

    Raises:
        FileNotFoundError: 가중치를 요구했으나(resume_path 또는 weight_path 지정) 해석하지
            못한 경우. 초기 가중치로 진행하지 않고 여기서 멈춘다.
        ValueError: ``start_iter`` 가 숫자도 알려진 키워드도 아닌 경우.
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

        _iter = _Resolve_start_iter(start_iter, workspace, best_metric)

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


def _Resolve_start_iter(
    start_iter: int | str | None,
    workspace: Path,
    best_metric: BEST_METRIC | None,
) -> int | None:
    """``start_iter`` 를 정수로 정규화한다. 키워드면 로그에서 고른다."""
    if start_iter is None:
        return None
    if isinstance(start_iter, int):
        return start_iter

    _text = str(start_iter).strip()
    if _text.lstrip("-").isdigit():
        return int(_text)
    if _text != BEST_KEYWORD:
        raise ValueError(
            f"start_iter 는 정수이거나 '{BEST_KEYWORD}' 여야 한다: {start_iter!r}"
        )
    if best_metric is None:
        raise ValueError(
            f"'{BEST_KEYWORD}' 를 요구했으나 판정 기준이 없다. 러너가 best_metric "
            f"(mode, metric, higher_is_better) 을 선언해야 한다 — 어떤 지표를 어느 "
            f"방향으로 볼지는 프레임워크가 알 수 없다."
        )
    return Select_iter_from_log(workspace, best_metric)


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
