from typing import Iterator
from pathlib import Path

from python_toolbox.project import Project_Template
from vision_toolbox.pipeline.core import Base_Step
from vision_toolbox.pipeline.state import Base_State


class Engine_Template(Project_Template):
    """파이프라인 알고리즘을 반복 실행하고 제어하는 범용 엔진 뼈대.
    
    데이터(Iterator)를 받아 파이프라인의 수명 주기를 관리합니다.
    사용자는 이 클래스를 상속받아 데이터 I/O, 전/후처리, 평가, 조기 종료 등의 
    세부 로직을 커스텀(Overriding)하여 사용합니다.
    """

    def __init__(
        self,
        project_name: str,
        pipeline: Base_Step,
        result_path: Path,
        max_iter: int = 1,
        **_kwargs,  # Config.Get_args()를 통해 언패킹된 파라미터 수용
    ):
        super().__init__(project_name)

        # 핵심 의존성
        self.pipeline = pipeline
        self.result_path = result_path
        self.max_iter = max_iter

        # 실행 상태 추적용 내부 변수
        self.current_iter = 0

    def Run(self, data_loader: Iterator[tuple[str, Base_State]]) -> None:
        """데이터 로더로부터 초기 상태를 받아 파이프라인을 실행합니다.
        
        Args:
            data_loader: (데이터_식별자, 초기_상태_객체)를 튜플로 반환하는 이터레이터.
        """
        # 전체 실행 전 준비
        self.on_run_start()

        for data_id, initial_state in data_loader:
            # 1. 단일 데이터에 대한 전처리 및 초기화
            _state = self.on_data_start(data_id, initial_state)

            # 2. 파이프라인 반복 실행 루프
            for self.current_iter in range(self.max_iter):

                # 루프 시작 전 처리
                _state = self.on_iter_start(_state)

                # 순수 알고리즘 (Model Forward)
                _state = self.pipeline(_state)

                # 루프 종료 후 평가 및 기록
                self.on_iter_end(_state)

                # 조기 종료 조건 검사
                if self.check_early_stop(_state):
                    break

            # 3. 단일 데이터 처리 완료 후 평가 및 기록
            self.on_data_end(data_id, _state)

        # 전체 실행 완료 후 요약 및 종료 처리
        self.on_run_end()

    # -------------------------------------------------------------------------
    # Hook Methods (사용자가 상속받아 오버라이딩 할 뼈대들)
    # -------------------------------------------------------------------------

    def on_run_start(self) -> None:
        """전체 Run 루프 시작 전 한 번 실행. (예: 결과 저장 디렉토리 생성 등)"""
        self.result_path.mkdir(parents=True, exist_ok=True)

    def on_data_start(self, data_id: str, state: Base_State) -> Base_State:
        """새로운 데이터(이미지) 처리가 시작될 때 실행. (예: 캔버스 생성, GT 로드)"""
        return state

    def on_iter_start(self, state: Base_State) -> Base_State:
        """각 반복(Iteration) 시작 전 실행. (예: 이전 결과를 바탕으로 crop 수행)"""
        return state

    def on_iter_end(self, state: Base_State) -> None:
        """각 반복 종료 후 실행. (예: 현재 상태의 IoU 계산, 중간 시각화 기록)"""
        pass

    def check_early_stop(self, state: Base_State) -> bool:
        return False

    def on_data_end(self, data_id: str, final_state: Base_State) -> None:
        """단일 데이터 처리 완료 후 1회 실행. (예: 최종 메트릭 기록, 결과 이미지 저장)"""
        pass

    def on_run_end(self) -> None:
        """모든 데이터 처리가 완료된 후 1회 실행. (예: 전체 평균 계산, 요약 파일 저장)"""
        pass
