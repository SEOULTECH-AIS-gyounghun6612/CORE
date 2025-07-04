# scene.py
"""
### 다양한 장면(scene) 구성을 지원하는 데이터 경로 수집 및 구성 유틸리티.

### Requirement
- None

### Structure
- Get_scene_block: 단일 장면 데이터 정보를 구성하는 함수.
- Get_multi_scene_block: 다중 장면 데이터 정보를 구성하는 함수.
- Get_multi_scene_block_in_subset: 하위 집합 내 다중 장면 데이터 정보를 구성하는 함수.
"""
from pathlib import Path


def Get_scene_block(data_dir: Path, use_target: bool = True):
    """ ### 단일 장면 데이터 정보를 구성하는 함수

    주어진 폴더의 장면 데이터 정보 구성.
    입력 이미지 경로 기본 포함, 필요시 정답 데이터 경로 추가 존재.

    ------------------------------------------------------------------
    ### Args
    - data_dir: 장면 데이터 위치 폴더.
    - use_target (optional): 정답 데이터 포함 여부. (default=True)

    ### Returns
    - dict: 장면 데이터 정보.
    """
    _block = {"input": sorted((data_dir / "rgb").glob("*"))}
    if use_target:
        _block.update({
            "gt": sorted((data_dir / "depth").glob("*.npz"))
        })

    return _block


def Get_multi_scene_block(
    data_dir: Path,
    term: int = 20,
    pickup: bool = True,
    use_target: bool = True,
    prefix: str = ""
):
    """ ### 다중 장면 데이터 정보를 구성하는 함수

    지정된 상위 폴더 내 여러 하위 장면 폴더로부터 장면 데이터 정보 수집.
    입력 인자를 사용하여 항상 동일한 길이의 데이터 그룹 생성 지원 

    ------------------------------------------------------------------
    ### Args
    - data_dir: 다수 개별 장면 폴더 포함 상위 폴더.
    - term (optional): 데이터 선택/제외 간격. 0 또는 None 시 모든 하위 폴더 처리. (default=20)
    - pickup (optional): `term` 간격으로 하위 폴더 '선택'(True) 또는 '제외'(False) 결정. `term`이 0 아닐 시 유효. (default=True)
    - use_target (optional): 각 장면 블록에 정답 데이터 포함 여부. (default=True)
    - prefix (optional): 반환 딕셔너리 키(하위 장면 폴더명) 추가 접두사. (default="")

    ### Returns
    - dict: 키는 (선택적 `prefix` 적용) 하위 장면 폴더명, 값은 해당 폴더의 `Get_scene_block` 결과 딕셔너리.
    """
    _prefix = f"{prefix}*" if prefix != "" else "*"
    _paths = sorted(data_dir.glob(_prefix))
    if term:
        return dict((
            f"{_p.name}",
            Get_scene_block(_p, use_target)
        ) for _ct, _p in enumerate(_paths) if pickup ^ bool(_ct % term))
    
    return dict((_p.name, Get_scene_block(_p, use_target) ) for _p in _paths)


def Get_multi_scene_block_in_subset(
    data_dir: Path,
    subsets: list[str],
    term: int = 20,
    pickup: bool = True,
    use_target: bool = True
):
    """ ### 하위 집합 내 다중 장면 데이터 정보를 구성하는 함수

    상위 데이터 폴더 내 명시된 여러 하위 집합(subset) 폴더에 대해 `Get_multi_scene_block` 각각 호출, 결과 단일 딕셔너리로 통합 반환.

    ------------------------------------------------------------------
    ### Args
    - data_dir: 하위 집합(subset) 폴더 포함 상위 데이터 폴더.
    - subsets: 처리 대상 하위 집합 폴더명 리스트. `data_dir` 내 실제 폴더여야 함.
    - term (optional): `Get_multi_scene_block` 전달 데이터 선택/제외 간격. (default=20)
    - pickup (optional): `Get_multi_scene_block` 전달 데이터 선택/제외 방식. (default=True)
    - use_target (optional): `Get_multi_scene_block` 전달 정답 데이터 포함 여부. (default=True)

    ### Returns
    - dict: 모든 지정 하위 집합 수집 장면 데이터 정보 통합 딕셔너리. 키는 `subset이름_하위장면폴더이름` 형식, 값은 각 하위 장면의 `Get_scene_block` 반환 값.
    """
    _total = {}

    for _subset in subsets:
        _subset_path = data_dir / _subset

        if not _subset_path.exists():
            continue

        _total.update(Get_multi_scene_block(
            _subset_path,
            term,
            pickup,
            use_target,
            prefix=_subset
        ))

    return _total
