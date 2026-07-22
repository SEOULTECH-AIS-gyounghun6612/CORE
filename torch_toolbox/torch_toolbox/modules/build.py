from __future__ import annotations
from typing import Any, cast

from python_toolbox.registry import Registry

from .. import CFGS
from .definition import Composable_Config, Composable_Module, Module_Config_Template
from .model.definition import Trainable_Model, Trainable_Model_Config
from .model.backbone import BACKBONES, BACKBONE_CFGS
from .loss.definition import Assemble_Loss_Config, Assemble_Loss


MODULES_CONFIG = (
    Module_Config_Template
    | Assemble_Loss_Config
    | Trainable_Model_Config | BACKBONE_CFGS
)
MODULES = (
    Composable_Module
    | Assemble_Loss
    | Trainable_Model | BACKBONES
)


def _Resolve_term(term: Any, built: dict[str, Any], context: dict[str, Any] | None) -> int:
    """차원 표현식의 항 하나를 정수로 해석한다.

    지원 형태::

        768              상수
        "$feat_dim"      외부 context 값 (dataset 등 조립 밖에서 오는 차원)
        "backbone"       **같은 계층**에서 이미 만들어진 모듈의 Out_channels() 마지막 항목
        "backbone[0]"    같은 모듈의 특정 출력 단

    이름만 쓰면 마지막 항목인 이유: 다단 feature 를 내는 백본을 받는 쪽은 관례적으로
    마지막 단을 쓴다(모델 forward 의 ``feat[-1]`` 과 같은 규약).
    """
    if isinstance(term, int):
        return term
    if not isinstance(term, str):
        raise TypeError(f"차원 표현식 항은 int 또는 str이어야 한다: {term!r}")

    if term.startswith("$"):
        _key = term[1:]
        if not context or _key not in context:
            raise KeyError(
                f"차원 표현식이 context['{_key}']를 참조하는데 주입되지 않았다. "
                f"assembler의 _Build_context()가 이 키를 채우는지 확인할 것. "
                f"(현재 context 키: {sorted(context) if context else []})"
            )
        return int(context[_key])

    _name, _, _rest = term.partition("[")
    _idx = int(_rest.rstrip("]")) if _rest else -1
    if _name not in built:
        raise KeyError(
            f"차원 표현식이 '{_name}'을 참조하는데 같은 계층에서 아직 만들어지지 않았다. "
            f"sub_module_meta에서 '{_name}'이 참조하는 쪽보다 **먼저** 선언되어야 한다. "
            f"(현재까지 만들어진 것: {sorted(built)})"
        )
    return int(built[_name].Out_channels()[_idx])


def _Resolve_value(value: Any, built: dict[str, Any], context: dict[str, Any] | None) -> Any:
    """meta 값 하나를 해석한다.

    두 형태를 지원한다::

        {"sum": [...]}   항들을 더해 정수로 (차원 산술)
        "$key"           context 값을 **그대로** 치환 (타입 제한 없음 — 정수, 리스트 등)

    나머지는 손대지 않는다.
    """
    if isinstance(value, dict) and set(value) == {"sum"}:
        return sum(_Resolve_term(_t, built, context) for _t in value["sum"])
    if isinstance(value, str) and value.startswith("$"):
        _key = value[1:]
        if not context or _key not in context:
            raise KeyError(
                f"config가 context['{_key}']를 참조하는데 주입되지 않았다. "
                f"assembler의 _Build_context()가 이 키를 채우는지 확인할 것. "
                f"(현재 context 키: {sorted(context) if context else []})"
            )
        return context[_key]
    return value


def _Resolve_meta(
    meta: dict[str, Any], built: dict[str, Any], context: dict[str, Any] | None
) -> dict[str, Any]:
    """meta 의 각 값을 :func:`_Resolve_value` 로 해석한다.

    config 원본은 건드리지 않는다 — 표현식이 그대로 남아 있어야 "이 값이 어디서
    왔는지"가 config 에 기록으로 남고, resume 시에도 같은 규칙으로 다시 풀린다.
    """
    return {_k: _Resolve_value(_v, built, context) for _k, _v in meta.items()}


def Build_from_registry(
    config: MODULES_CONFIG, registry: Registry, context: dict[str, Any] | None = None,
) -> MODULES:
    """Config 트리를 재귀적으로 순회하며 모듈을 조립한다.

    Composable_Config이면 sub_module_meta의 각 항목을 CFGS로 인스턴스화한 뒤
    재귀 빌드하고, 결과를 상위 모듈의 Build(**sub_modules)에 키워드 인자로 주입한다.
    리프 Config는 재귀 없이 바로 인스턴스화한다.

    **동일 계층 차원 해석**: 형제 모듈을 선언 순서대로 만들면서, 뒤 형제의 차원 표현식이
    앞 형제의 ``Out_channels()`` 를 참조할 수 있다. 여기에 외부 값(``context``)을 더해
    ``in_channels: {sum: [backbone, $feat_dim]}`` 같은 선언이 성립한다. 차원을 config 에
    숫자로 중복 기입하지 않으면서, 어떻게 도출되는지는 config 에 남는다.

    Args:
        config: 빌드할 모듈의 Config.
        registry: 대상 도메인 레지스트리 (MODELS, LOSSES 등).
        context: 조립 밖에서 오는 차원 값 (예: ``{"feat_dim": 695}``). ``$키`` 로 참조한다.

    Returns:
        조립 완료된 Composable_Module 인스턴스.
    """
    _sub_kwargs = {}

    if isinstance(config, Composable_Config):
        # 자식 먼저 빌드: 부모의 Build()가 서브모듈을 **kwargs로 받기 때문
        for _k, _meta in config.sub_module_meta.items():
            # 같은 계층에서 이미 만들어진 형제(_sub_kwargs)와 context 로 차원 표현식을 푼다
            _meta = _Resolve_meta(_meta, _sub_kwargs, context)
            _sub_cfg = cast(Composable_Config, CFGS.Get(_meta["config_type"])(**_meta))
            _sub_kwargs[_k] = Build_from_registry(_sub_cfg, registry, context)

    # 평탄화된 하이퍼파라미터 + 재귀 빌드된 서브모듈을 함께 생성자에 주입
    return registry.Get(config.object_type)(**config.Extract(), **_sub_kwargs)