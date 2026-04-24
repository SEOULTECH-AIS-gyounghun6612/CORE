# ROADMAP

진행 현황 및 향후 계획을 관리하는 문서.

> **코드 분석 및 업데이트 일자:** 2026-04-24

---

## 완료된 작업 (최근)

- [x] **구조적 로깅(Structured Logging) 모듈 도입** — `Data_Schema` 기반의 유연한 `Logger` 및 `Log_Line` 아키텍처 설계 완료 (다형성 지원, 스마트 인자 분배, 안전한 클램핑).
- [x] **COOKBOOK 문서화 개편** — 메인 `COOKBOOK.md`를 각 모듈별(`data_schema`, `registry`, `system`, `log`) 상세 문서로 분리하여 유지보수성 향상.

---

## 기능 개선

- [ ] `file.Read_from` — 확장자 기반 분기 시 `.json`/`.yaml` 외 포맷 지원 (예: `.toml`)
- [ ] `Project_Template` — `_Setup()` 이후 생성된 workspace 경로를 신규 `Logger`를 활용하여 출력하는 훅 추가
- [ ] `Registry` — `Get_all()` 메서드 추가 (등록된 키 목록 조회)
- [ ] `Base_Config.Write_to` — JSON/YAML 포맷 자동 선택 로직 고도화

---

## 로깅 표준화 (신규 Logger 적용)

- [ ] `file.Handle_exp` — 기존 `print` 기반 예외 출력을 신규 구조적 `Logger` 시스템으로 교체
