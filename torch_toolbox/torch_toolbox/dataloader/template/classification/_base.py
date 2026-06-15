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

        Args:
            map_path: {class_name: {class_id: int, category_id: int}} 형식의 YAML 경로.

        Returns:
            class_name → (class_id, category_id) 딕셔너리.
            파일이 없으면 빈 딕셔너리를 반환한다.
        """
        if not map_path.exists():
            return {}
        with open(map_path, "r", encoding="utf-8") as _f:
            _raw: dict[str, dict[str, int]] = yaml.safe_load(_f) or {}
        return {
            _cls: (_e["class_id"], _e.get("category_id", -1))
            for _cls, _e in _raw.items()
        }

    def _Auto_id_map(self, root: Path) -> dict[str, tuple[int, int]]:
        """루트 디렉토리의 하위 폴더명으로 id_map을 자동 생성한다.

        Args:
            root: 클래스 폴더들이 위치한 루트 디렉토리.

        Returns:
            folder_name → (index, -1) 딕셔너리.
        """
        _names = sorted(d.name for d in root.iterdir() if d.is_dir())
        return {n: (i, -1) for i, n in enumerate(_names)}

