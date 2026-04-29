"""프로젝트 실행 파이프라인 — 설정 데이터(Config) + 파이프라인 템플릿(Template).

config: Data_Schema 기반 파일 I/O 진입점 (Base_Config + 팩토리 함수)
template: 워크스페이스 + 멱등성 Setup 관리 (Project_Template)
"""
from .config import Base_Config, Build_sub_config, Build_parser_from_config
from .template import Project_Template, RESULT_ROOT


__all__ = [
    "Base_Config",
    "Build_sub_config",
    "Build_parser_from_config",
    "Project_Template",
    "RESULT_ROOT",
]
