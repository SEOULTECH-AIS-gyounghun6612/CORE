from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import yaml

from ...definition import Custom_Dataset, Dataset_Config


@dataclass
class Classification_Dataset_Config(Dataset_Config):
    """폴더 구조 기반 classification dataset의 공통 설정.

    Attributes:
        id_map_file: class_name → class_id 매핑 YAML 파일명.
        transform: 전처리 transform 레지스트리 키. None이면 원본 그대로 사용.
        extensions: 수집할 파일 확장자 목록.
    """

    id_map_file: str = "id_map.yaml"
    transform: str | None = None
    extensions: list[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp"])


class Classification_Dataset(Custom_Dataset):
    """폴더 구조 기반 classification dataset의 공통 기반 클래스.

    Builder가 id_map 로드 → 샘플 스캔 → (id_map 없으면) 자동 생성 → 재스캔
    순서로 초기화를 오케스트레이션한다. 서브클래스는 _Scan_samples()만 구현하면 된다.

    PK 샘플러는 class_ids 프로퍼티를 통해 샘플별 class를 참조한다.

    Attributes:
        samples: (item, class_id, category_id) 형식의 샘플 리스트.
        id_map: class_name → (class_id, category_id) 매핑.
    """

    samples: list[tuple[Any, int, int]]
    id_map: dict[str, tuple[int, int]]

    def Builder(
        self,
        data_dir: str,
        name: str,
        category: str,
        id_map_file: str = "id_map.yaml",
        **kwargs,
    ):
        """id_map 로드와 샘플 스캔을 순서대로 수행한다.

        Args:
            data_dir: 데이터셋 루트 디렉토리 경로.
            name: 데이터셋 이름. data_dir/name/ 이 실제 루트가 된다.
            category: 사용할 카테고리 서브셋.
            id_map_file: id_map YAML 파일명.
            **kwargs: 서브클래스의 _Scan_samples()에 전달할 추가 인자.
        """
        _root = Path(data_dir) / name
        self.id_map = self._Load_id_map(_root / id_map_file)
        self.samples = self._Scan_samples(_root, **kwargs)

        # id_map 없으면 폴더명으로 자동 생성 후 재스캔
        if not self.id_map:
            self.id_map = self._Auto_id_map(_root)
            self.samples = self._Scan_samples(_root, **kwargs)

    def _Scan_samples(
        self, root: Path, **kwargs
    ) -> list[tuple[Any, int, int]]:
        """데이터 루트를 스캔해 (item, class_id, category_id) 목록을 반환한다.

        id_map이 설정된 상태에서 호출된다.

        Args:
            root: 데이터셋 루트 디렉토리.
            **kwargs: 서브클래스별 추가 파라미터 (예: extensions).

        Returns:
            (item, class_id, category_id) 튜플 리스트.
        """
        raise NotImplementedError

    @property
    def class_ids(self) -> list[int]:
        return [_s[1] for _s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def _Load_id_map(self, map_path: Path) -> dict[str, tuple[int, int]]:
        """YAML id_map 파일로부터 class 매핑을 로드한다.

        형식은 dict 이며, **바깥 키는 호출번호**(항목 나열 순서)이고 ``class_id`` 는
        항목 안에 별도 필드로 들어간다::

            0: {class_id: 0, name: no_label,     category_id: 6}
            1: {class_id: 1, name: 10D132000NT9, category_id: 2}

        둘을 분리하는 이유: class_id 는 ArcFace 프로토타입의 행 인덱스이자 배포 ONNX
        출력 인덱스라서 조용히 밀리면 기존 체크포인트와 어긋난다. 호출번호와 별개 필드로
        두면 이름이 바뀌어도(9자리 -> 12자리 통일 등) 번호가 그대로임이 diff 에 드러난다.

        파싱 후 **마지막 호출번호와 항목 수를 비교**해 검산한다. 어긋나면 항목이 빠졌거나
        번호가 건너뛴 것이고, 그 결과는 학습 중 device-side assert 로만 드러나서 원인을
        찾기 어렵다.

        부수적으로 :attr:`class_map` 에 ``class_id -> (name, category_id)`` 역방향을 채운다.

        Args:
            map_path: id_map YAML 경로.

        Returns:
            class_name → (class_id, category_id) 딕셔너리.
            파일이 없으면 빈 딕셔너리를 반환한다.

        Raises:
            ValueError: 옛 dict 형식인 경우.
        """
        self.class_map = {}
        if not map_path.exists():
            return {}
        with open(map_path, "r", encoding="utf-8") as _f:
            _raw = yaml.safe_load(_f) or {}

        # 옛 형식은 키가 부품명(문자열), 새 형식은 키가 호출번호(정수)다.
        if _raw and not str(next(iter(_raw))).lstrip("-").isdigit():
            raise ValueError(
                f"{map_path}: 옛 형식({{name: {{class_id, category_id}}}})이다. "
                f"호출번호를 키로 하고 class_id 를 필드로 두는 형식으로 마이그레이션할 것 "
                f"(tools/migrate_id_map.py)."
            )

        _by_name: dict[str, tuple[int, int]] = {}
        for _e in _raw.values():
            _cid = int(_e["class_id"])                    # 라벨 정본
            _name, _kid = str(_e["name"]), int(_e.get("category_id", -1))
            self.class_map[_cid] = (_name, _kid)
            _by_name[_name] = (_cid, _kid)

        # 마지막 호출번호 + 1 이 항목 수와 같아야 한다 (번호가 건너뛰거나 항목이 빠지지 않았는가)
        if _raw:
            _keys = [int(_k) for _k in _raw]
            if max(_keys) + 1 != len(_raw):
                print(
                    f"[WARN] {map_path}: 마지막 호출번호 {max(_keys)} 인데 항목이 "
                    f"{len(_raw)}개다. 번호가 건너뛰었거나 항목이 누락됐다."
                )
        return _by_name

    def _Auto_id_map(self, root: Path) -> dict[str, tuple[int, int]]:
        """루트 디렉토리의 하위 폴더명으로 id_map을 자동 생성한다.

        Args:
            root: 클래스 폴더들이 위치한 루트 디렉토리.

        Returns:
            folder_name → (index, -1) 딕셔너리.
        """
        _names = sorted(d.name for d in root.iterdir() if d.is_dir())
        return {n: (i, -1) for i, n in enumerate(_names)}

