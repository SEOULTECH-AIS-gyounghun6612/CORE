# ROADMAP

> 최근 갱신: 2026-05-05

`python_toolbox`는 이미 코어 유틸리티 묶음으로 안정화되어 있음. 지금 필요한 일은 기능 추가보다 공개 API 정합성, 문서 최신화, 선택적 확장 포인트 정리임.

## 현재 상태

- [x] `Data_Schema` 직렬화/추출 코어 존재
- [x] `file.Read_from` / `Write_to` 확장자 디스패치 존재
- [x] `Registry` 타입/시그니처 검증 존재
- [x] `Base_Config`와 `Project_Template` 분리 존재
- [x] `Logger` / `Log_Line` 구조적 로깅 존재

## 단기 우선순위

### 공개 API와 문서 정합성

- [ ] 루트 문서와 하위 cookbook의 공개 API 명칭 일치 유지
- [ ] `project` 문서에서 오래된 helper 함수명 재등장 방지
- [ ] 각 cookbook 예제가 실제 `__init__.py` export 기준을 따르도록 유지

### file / project 경계 명확화

- [ ] 설정 파일 읽기 helper를 `project`에 둘지, `file.Read_from + Config(**data)` 조합으로 둘지 정책 확정
- [ ] `Base_Config` 읽기 보조 API 필요성 재검토
- [ ] `Write_to`와 `Base_Config.Write_to` 사용 경계 문서화

### 테스트 및 예제

- [ ] 문서 예제와 테스트 커버리지 매핑 정리
- [ ] 신규 포맷 추가 시 `file` cookbook과 테스트를 함께 갱신하는 규칙 명문화

## 중기 과제

### file

- [ ] `.toml` 등 포맷 추가 여부 결정
- [ ] `Handle_exp` 출력 방식 개선 검토

### registry

- [ ] 등록된 키 목록 조회 API 필요성 검토
- [ ] 파라미터 개수 외 Callable 타입 검증 범위 확대 여부 검토

### project

- [ ] `Build_sub_config` 사용 시나리오 예제 보강
- [ ] `Build_parser_from_config`의 bool/list/dict 규약 문서 보강

### logging

- [ ] `Logger` 저장 경로 예제 보강
- [ ] `Log_Line` schema 확장 패턴 정리
