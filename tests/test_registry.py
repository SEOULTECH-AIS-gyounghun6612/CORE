"""Registry 단위 테스트."""
from typing import Callable

import pytest

from python_toolbox.registry import Registry


# =============================================================================
# 기반 클래스 & 레지스트리 픽스처
# =============================================================================

class _Base:
    pass


class _Other:
    pass


@pytest.fixture
def cls_reg():
    return Registry[type[_Base]]("test_cls", _Base)


@pytest.fixture
def fn_reg():
    return Registry[Callable[[int, str], bool]]("test_fn", Callable[[int, str], bool])


# =============================================================================
# 초기화
# =============================================================================

def test_registry_invalid_target_type_raises():
    """클래스도 Callable도 아닌 target_type → TypeError."""
    with pytest.raises(TypeError):
        Registry("bad", 42)


# =============================================================================
# 클래스 모드 등록
# =============================================================================

def test_register_class_success(cls_reg):
    """올바른 하위 클래스 등록 성공."""
    @cls_reg.Register_module("sub")
    class _Sub(_Base):
        pass
    assert cls_reg.Get("sub") is _Sub


def test_register_class_wrong_base_raises(cls_reg):
    """상속 관계 불일치 → TypeError."""
    with pytest.raises(TypeError):
        @cls_reg.Register_module("other")
        class _NotBase:
            pass


def test_register_class_with_function_raises(cls_reg):
    """클래스 전용 레지스트리에 함수 등록 → TypeError."""
    with pytest.raises(TypeError):
        @cls_reg.Register_module("fn")
        def _fn():
            pass


def test_register_class_auto_name(cls_reg):
    """name 생략 시 __name__ lowercase로 자동 등록."""
    @cls_reg.Register_module()
    class _AutoNamed(_Base):
        pass
    assert cls_reg.Get("_autonamed") is _AutoNamed


def test_register_class_duplicate_raises(cls_reg):
    """동일 이름 중복 등록 → KeyError."""
    @cls_reg.Register_module("dup")
    class _A(_Base):
        pass
    with pytest.raises(KeyError):
        @cls_reg.Register_module("dup")
        class _B(_Base):
            pass


# =============================================================================
# Callable 모드 등록
# =============================================================================

def test_register_callable_success(fn_reg):
    """파라미터 수 일치하는 함수 등록 성공."""
    @fn_reg.Register_module("ok_fn")
    def _fn(a: int, b: str) -> bool:
        return True
    assert fn_reg.Get("ok_fn") is _fn


def test_register_callable_wrong_param_count_raises(fn_reg):
    """파라미터 수 불일치 → TypeError."""
    with pytest.raises(TypeError):
        @fn_reg.Register_module("bad_fn")
        def _fn(x: int) -> bool:
            return True


def test_register_callable_with_class_raises(fn_reg):
    """Callable 전용 레지스트리에 클래스 등록 → TypeError."""
    with pytest.raises(TypeError):
        @fn_reg.Register_module("a_class")
        class _C:
            pass


def test_register_lambda_raises(cls_reg):
    """클래스 전용 레지스트리에 람다 등록 → TypeError."""
    with pytest.raises(TypeError):
        cls_reg.Register_module()(lambda: None)


# =============================================================================
# Get
# =============================================================================

def test_get_unregistered_raises(cls_reg):
    """미등록 키 → KeyError."""
    with pytest.raises(KeyError):
        cls_reg.Get("nonexistent")


def test_get_with_expected_success(cls_reg):
    """`expected` 지정 시 issubclass 검증 통과."""
    @cls_reg.Register_module("good")
    class _Good(_Base):
        pass
    result = cls_reg.Get("good", _Base)
    assert result is _Good


def test_get_with_expected_wrong_type_raises(cls_reg):
    """`expected` 지정 시 issubclass 실패 → TypeError."""
    @cls_reg.Register_module("mismatch")
    class _M(_Base):
        pass
    with pytest.raises(TypeError):
        cls_reg.Get("mismatch", _Other)
