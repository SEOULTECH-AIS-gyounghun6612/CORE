"""프로젝트 실행 파이프라인 베이스 템플릿 모듈.

타임스탬프 기반 고유 워크스페이스 할당과 멱등성 Setup을 제공하는
Project_Template를 제공함. 진입점 구조와 실행 흐름은 하위 클래스에 위임함.

Requirement:
    - Python >= 3.10
    - time, uuid, pathlib
"""
from __future__ import annotations
import time
import uuid
from pathlib import Path


RESULT_ROOT = "./result"


class Project_Template:
    """프로젝트 실행 파이프라인 관리를 위한 베이스 템플릿.

    Config 구조와의 결합을 제거한 독립 템플릿. 인스턴스 생성 시 타임스탬프
    + UUID 기반 고유 워크스페이스가 할당되어 동일 프로젝트의 중복 실행
    덮어쓰기를 방지함. 실제 진입점 구조는 하위 클래스가 정의함.
    """

    def __init__(self, project_name: str):
        """
        Args:
            project_name: 프로젝트 고유 식별 이름.

        Raises:
            ValueError: 빈 project_name이 주어진 경우.
        """
        if not project_name:
            raise ValueError("[ERROR] Project name cannot be empty.")

        self.project_name = project_name

        # 타임스탬프 + 짧은 UUID로 워크스페이스 고유성 보장
        _run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.workspace = Path(RESULT_ROOT) / project_name / _run_id

        self._is_setup_done = False

    def _Setup(self) -> bool:
        """멱등성 작업 환경 초기화. 워크스페이스 디렉토리 생성을 보장함.

        Returns:
            True: 이미 초기화된 경우 (중복 호출).
            False: 최초 초기화를 수행한 경우.
        """
        if self._is_setup_done:
            return True

        self.workspace.mkdir(parents=True, exist_ok=True)
        self._is_setup_done = True
        return False
