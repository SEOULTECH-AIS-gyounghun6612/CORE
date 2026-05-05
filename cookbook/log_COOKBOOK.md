# log — 확장 가능한 구조적 로깅 사용 예시

`log` 모듈은 `Data_Schema`를 상속받은 레코드 객체를 활용하여,
단순 텍스트 출력뿐만 아니라 JSON/YAML 저장에 적합한 구조화된 데이터 트리를 생성합니다.

---

## 기본 사용 (시스템 로그)

가장 기본적인 `Log_Line` 스키마를 사용하는 방법입니다.

```python
from python_toolbox.log import Logger, Log_Line

# 1. 기본 Log_Line 스키마를 사용하는 로거 생성
logger = Logger(Log_Line)

# 2. 로깅 수행 (시스템 레벨별 헬퍼 메서드 제공)
logger.Info(msg="서버가 시작되었습니다.", port=8080)
logger.Warning(msg="메모리 사용량이 높습니다.", usage="85%")
logger.Error(msg="DB 연결 실패", reason="timeout")

# 3. 콘솔에 출력 (특정 라인 번호 또는 전체)
logger.Print_to_console()
# 출력 예시:
# [2026-04-24T15:30:12.345678] INF - msg=서버가 시작되었습니다., port=8080
# [2026-04-24T15:30:12.346123] WRN - msg=메모리 사용량이 높습니다., usage=85%
# [2026-04-24T15:30:12.347890] ERR - msg=DB 연결 실패, reason=timeout
```

---

## 커스텀 스키마 생성 및 활용 (AI 학습 로그 등)

학습 지표(Epoch, Loss)와 같이 정해진 규격이 있는 데이터를 로깅할 때는 `Log_Line`을 상속받아 커스텀 스키마를 정의합니다.

```python
from dataclasses import dataclass
from python_toolbox.log import Logger, Log_Line

# 1. 커스텀 스키마 정의
@dataclass
class AI_Log_Line(Log_Line):
    epoch: int = 0
    loss: float = 0.0

    # 출력 포맷 커스텀 (선택 사항)
    def Format(self) -> str:
        return f"[{self.timestamp}] Epoch {self.epoch} - Loss: {self.loss:.4f}"

# 2. 커스텀 스키마 전담 로거 생성
ai_logger = Logger(AI_Log_Line)

# 3. 정해진 필드는 명시적으로 매핑됨
ai_logger.Info(epoch=1, loss=0.523)
ai_logger.Info(epoch=2, loss=0.412)

ai_logger.Print_to_console()
# 출력 예시:
# [2026-04-24T15:30:12.123] Epoch 1 - Loss: 0.5230
# [2026-04-24T15:30:12.456] Epoch 2 - Loss: 0.4120
```

---

## 스마트 인자 분배 (Smart Dispatch)

로거는 스키마에 정의된 **명시적 필드**와 **잉여 데이터**를 똑똑하게 구분하여 할당합니다.
스키마에 없는 추가 데이터는 `info` 필드(딕셔너리) 안으로 자동으로 모입니다.

```python
ai_logger.Info(epoch=3, loss=0.350, lr=1e-4, note="학습률 감소됨")

# ai_logger.book[-1] 의 데이터 구조 확인 (Extract 활용):
# {
#   "level": 2, 
#   "timestamp": "...", 
#   "epoch": 3, 
#   "loss": 0.350, 
#   "info": {"lr": 0.0001, "note": "학습률 감소됨"}  <- 잉여 데이터가 모임
# }
```

이러한 구조화 덕분에, 나중에 전체 로그를 `ai_logger.Serialize()` 한 번만 호출하면 JSON/YAML 저장에 바로 사용할 수 있는 dict 트리를 얻을 수 있습니다.

---

## 동적 타입 주입 (다형성)

단일 로거 인스턴스가 여러 형태의 로그 스키마를 유연하게 수용할 수 있습니다.

```python
# 기본 스키마(Log_Line)를 기준으로 하는 메인 로거
main_logger = Logger(Log_Line)

# 평소엔 일반 시스템 로그 작성
main_logger.Info(msg="학습 시작")

# 특수한 순간에만 AI_Log_Line 스키마로 덮어씌워서 기록 (다형성)
main_logger.Info(line_type=AI_Log_Line, epoch=1, loss=0.99)
```

---

## 안전한 오류 방지 (Fail-safe Clamping)

실수로 비정상적인 로그 레벨(0 미만, 5 초과)을 주입하더라도, 시스템 오류를 뿜고 죽지 않고 조용히 정상 범위(0~5)로 보정(Clamping)됩니다.

```python
# 존재하지 않는 레벨 99 주입 시도
logger._Logging(level=99, line_type=Log_Line, msg="비정상 로그")

# 내부적으로 level=5(CRI)로 강제 고정되어 프로그램 중단을 방지함
logger.Print_to_console(-1)
# 출력 예시: [2026-04-24T...] CRI - msg=비정상 로그
```
